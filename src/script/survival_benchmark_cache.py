from __future__ import annotations

import shutil
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path

from joblib import hash as joblib_hash

from src.data import PatientBloodTestDataset, PatientLandmarkTable
from src.data.benchmark_landmarks import BenchmarkLandmarks
from src.data.dataset_config import PatientDatasetConfig
from src.model.blood_test_imputation_statistics import (
    BloodTestImputationStatistics,
    build_blood_test_imputation_statistics,
    build_blood_test_imputation_statistics_from_tables,
)
from src.model.imputation import FittedFeatureImputer
from src.model.survival_evaluation import (
    SharedEvaluationFeatureCache,
    SurvivalEvaluationCache,
)
from src.script.ebm_survival_config import write_training_config_snapshot
from src.script.survival_benchmark_artifacts import file_sha256
from src.script.survival_dataset_artifacts import write_survival_dataset_artifacts
from src.script.survival_model_input import (
    ModelInputArtifact,
    SharedModelInputFeatureCache,
    build_model_input_artifact,
    write_model_input_artifact,
)
from src.script.survival_training import (
    PreparedSurvivalTraining,
    SharedSurvivalFeatureCache,
    training_not_estimable_reason,
)


@dataclass
class SurvivalBenchmarkFeatureCache:
    """Feature work shared by compatible benchmark tasks.

    ``landmarks`` owns the benchmark-wide canonical rows and target labels.
    ``survival_features`` owns the fitted training imputer and compatible
    bootstrap transformations; ``model_input_features`` owns compatible
    non-bootstrapped split transformations. Evaluation caches retain each
    target's labels across models, keyed by dataset config and fitted imputer
    state so independently saved copies share only compatible preparations.
    """

    landmarks: BenchmarkLandmarks
    survival_features: SharedSurvivalFeatureCache = field(default_factory=SharedSurvivalFeatureCache)
    model_input_features: SharedModelInputFeatureCache = field(default_factory=SharedModelInputFeatureCache)
    _evaluation_caches: dict[tuple[PatientDatasetConfig, str], SurvivalEvaluationCache] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _evaluation_features_by_imputer: dict[str, SharedEvaluationFeatureCache] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _imputation_statistics: BloodTestImputationStatistics | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def evaluation_features(self) -> SharedEvaluationFeatureCache:
        """Return transformed bootstrap features for compatible tasks."""
        return self.survival_features.evaluation_features

    def evaluation_cache(
        self,
        config: PatientDatasetConfig,
        imputer: FittedFeatureImputer,
        *,
        imputer_key: str | None = None,
    ) -> SurvivalEvaluationCache:
        """Reuse target-specific preparations only for identical fitted imputers."""
        fingerprint = fitted_imputer_fingerprint(imputer) if imputer_key is None else imputer_key
        config = replace(config, split="train")
        key = (config, fingerprint)
        cached = self._evaluation_caches.get(key)
        if cached is not None:
            return cached
        shared = self._evaluation_features_by_imputer.get(fingerprint)
        model_inputs = self.model_input_features.prepared
        same_imputer = imputer is self.model_input_features.imputer
        if shared is None:
            shared = self.evaluation_features if same_imputer else SharedEvaluationFeatureCache()
            self._evaluation_features_by_imputer[fingerprint] = shared
        pool_features = (
            {}
            if model_inputs is None or not same_imputer
            else {
                split: (table, model_inputs.imputation_results[split]) for split, table in model_inputs.tables.items()
            }
        )
        cached = SurvivalEvaluationCache(config, imputer, self.landmarks, shared, pool_features)
        self._evaluation_caches[key] = cached
        return cached

    def prepare_training(
        self,
        dataset: PatientBloodTestDataset,
        train_table: PatientLandmarkTable,
        imputation_config: Mapping[str, object] | None,
        *,
        nonnegative_feature_names: tuple[str, ...],
    ) -> PreparedSurvivalTraining | None:
        """Prepare or reuse the cache's complete training feature schema."""
        return self.survival_features.prepare(
            dataset,
            train_table,
            imputation_config,
            nonnegative_feature_names=nonnegative_feature_names,
        )

    def model_input_artifact(
        self,
        config: PatientDatasetConfig,
        prepared: PreparedSurvivalTraining | None,
    ) -> ModelInputArtifact:
        """Build current Recurrence-Free Survival labels with cache-shared features."""
        split_tables = {
            split: self.landmarks.table(replace(config, split=split))
            for split in ("train", "val", "test")
        }
        return build_model_input_artifact(
            split_tables,
            None if prepared is None else prepared.imputer,
            feature_cache=self.model_input_features,
            train_imputation=None if prepared is None else prepared.train_imputation,
        )

    def imputation_statistics(
        self,
        dataset: PatientBloodTestDataset,
        prepared: PreparedSurvivalTraining | None,
    ) -> BloodTestImputationStatistics | None:
        """Return feature-compatible imputation-source counts."""
        if prepared is None:
            return None
        if self._imputation_statistics is None:
            model_input_features = self.model_input_features.prepared
            self._imputation_statistics = (
                build_blood_test_imputation_statistics(dataset, prepared.imputer)
                if model_input_features is None
                else build_blood_test_imputation_statistics_from_tables(
                    dataset,
                    prepared.imputer,
                    model_input_features.tables,
                    model_input_features.imputation_results,
                )
            )
        return self._imputation_statistics


@dataclass
class TaskModelInputCache:
    """Serialize one task model-input frame shared by its feature models.

    ``artifact`` is the in-memory frame and metadata. ``shared_path`` is its
    single serialized source. The cached SHA-256 fingerprint is reused by every
    model completion marker.
    """

    artifact: ModelInputArtifact
    shared_path: Path
    _metadata: dict[str, object] | None = field(default=None, init=False, repr=False)
    _fingerprint: str | None = field(default=None, init=False, repr=False)

    @property
    def fingerprint(self) -> str:
        """Return the shared serialized input fingerprint."""
        self._prepare()
        if self._fingerprint is None:
            raise RuntimeError("Task model-input fingerprint was not prepared.")
        return self._fingerprint

    def shared_metadata(self, path: str, **extra: object) -> dict[str, object]:
        """Describe the shared source using a consumer-relative ``path``."""
        metadata = self._prepare()
        return {
            **metadata,
            "path": path,
            "sha256": self.fingerprint,
            **extra,
        }

    def _prepare(self) -> dict[str, object]:
        if self._metadata is None:
            self.shared_path.parent.mkdir(parents=True, exist_ok=True)
            self._metadata = write_model_input_artifact(self.artifact, self.shared_path)
            self._fingerprint = file_sha256(self.shared_path)
        return self._metadata


@dataclass
class TaskDatasetArtifactCache:
    """Render task files once and copy them into each EBM run directory.

    ``raw_config`` and ``dataset`` define the artifact contents. ``config_dir``
    resolves config-relative paths, ``shared_dir`` stores task-level sources,
    and ``imputation_statistics`` supplies shared report attribution.
    """

    raw_config: Mapping[str, Mapping[str, object]]
    dataset: PatientBloodTestDataset
    config_dir: Path
    shared_dir: Path
    imputation_statistics: BloodTestImputationStatistics | None
    _metadata: dict[str, object] | None = field(default=None, init=False, repr=False)

    def materialize(self, model_dir: Path) -> dict[str, object]:
        """Copy the shared dataset and population reports into ``model_dir``."""
        metadata = self._prepare()
        model_dir.mkdir(parents=True, exist_ok=True)
        for relative_path in artifact_paths(metadata):
            copy_relative_artifact(self.shared_dir, model_dir, relative_path)
        return deepcopy(metadata)

    def _prepare(self) -> dict[str, object]:
        if self._metadata is None:
            self.shared_dir.mkdir(parents=True, exist_ok=True)
            self._metadata = {
                "training_config": write_training_config_snapshot(
                    self.raw_config,
                    self.dataset.config,
                    self.shared_dir,
                    self.config_dir,
                ),
                **write_survival_dataset_artifacts(
                    self.dataset,
                    self.shared_dir,
                    self.raw_config.get("population_statistics"),
                    self.imputation_statistics,
                    landmark_event_counts_config=self.raw_config.get("landmark_event_counts"),
                ),
            }
        return self._metadata


@dataclass(frozen=True)
class PreparedSurvivalBenchmarkTask:
    """Shared prepared inputs for one Endpoint Site and Censoring Horizon.

    ``dataset`` and ``train_table`` retain current Recurrence-Free Survival
    labels. ``prepared`` owns cache-compatible training features.
    ``evaluation_cache`` supplies model-shared bootstrap inputs. The two
    artifact caches preserve per-model layouts.
    """

    dataset: PatientBloodTestDataset
    train_table: PatientLandmarkTable
    prepared: PreparedSurvivalTraining | None
    evaluation_cache: SurvivalEvaluationCache | None
    model_input_cache: TaskModelInputCache
    dataset_artifact_cache: TaskDatasetArtifactCache
    not_estimable_reason: str | None


def prepare_survival_benchmark_task(
    raw_config: Mapping[str, Mapping[str, object]],
    config_dir: Path,
    dataset_config: PatientDatasetConfig,
    site_dataset: PatientBloodTestDataset,
    feature_cache: SurvivalBenchmarkFeatureCache,
    shared_artifact_dir: Path,
    model_input_path: Path,
    ebm_config: Mapping[str, object] | None,
) -> PreparedSurvivalBenchmarkTask:
    """Prepare one task while reusing every compatible feature artifact."""
    dataset = PatientBloodTestDataset(
        dataset_config,
        prepared_data=site_dataset.prepared_data,
    )
    train_table = feature_cache.landmarks.table(dataset_config)
    prepared = feature_cache.prepare_training(
        dataset,
        train_table,
        raw_config["imputation"],
        nonnegative_feature_names=tuple(dataset.selected_test_keys),
    )
    model_input = feature_cache.model_input_artifact(dataset_config, prepared)
    imputation_statistics = feature_cache.imputation_statistics(dataset, prepared)
    model_input_cache = TaskModelInputCache(model_input, model_input_path)
    evaluation_cache = None if prepared is None else feature_cache.evaluation_cache(dataset.config, prepared.imputer)
    artifact_cache = TaskDatasetArtifactCache(
        raw_config,
        dataset,
        config_dir,
        shared_artifact_dir,
        imputation_statistics,
    )
    return PreparedSurvivalBenchmarkTask(
        dataset,
        train_table,
        prepared,
        evaluation_cache,
        model_input_cache,
        artifact_cache,
        training_not_estimable_reason(train_table, ebm_config),
    )


def fitted_imputer_fingerprint(imputer: FittedFeatureImputer) -> str:
    """Identify saved copies by fitted state, including coefficients and change semantics."""
    return joblib_hash(imputer, hash_name="sha1")


def artifact_paths(value: object) -> Iterator[Path]:
    """Yield report paths recorded in nested artifact metadata."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {"path", "csv_path", "latex_path"} and isinstance(nested, str):
                yield Path(nested)
            else:
                yield from artifact_paths(nested)
    elif isinstance(value, list | tuple):
        for nested in value:
            yield from artifact_paths(nested)


def copy_relative_artifact(shared_dir: Path, model_dir: Path, path: Path) -> None:
    """Copy one output-relative artifact; absolute artifacts already share a target."""
    if path.is_absolute():
        return
    source = shared_dir / path
    if not source.is_file():
        raise FileNotFoundError(f"Shared task artifact does not exist: {source}")
    destination = model_dir / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
