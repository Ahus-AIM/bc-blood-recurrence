from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np


@dataclass(frozen=True)
class RecurrenceBloodTestPlotConfig:
    """Trajectory export layout, dimensions, typography and file formats.

    ``per_site`` shows every report test in ``columns`` columns per site.
    ``combined`` retains one row per site with at most ``top_n`` ranked tests.
    ``both`` exports the combined overview alongside every per-site figure.
    Both layouts rank by the earliest significant window whose significance
    persists through recurrence, then adjusted p in that starting window.
    Calculable tests without a qualifying window follow by minimum adjusted p;
    exact ties use display name and test key. The minimum is not an extra
    tie breaker for tests with a qualifying window. Finally,
    per-site pages also append tests without calculable comparisons alphabetically.

    Panel width and minimum uncompressed row height are in inches.
    graph_height_scale compresses only the graphs and removes the saved height
    from each site row, preserving table and heading space. Fonts come from
    publication_style.yml; font_scale is their maximum multiplier, reduced
    when needed to fit each panel. Numeric intervals wrap and rows grow when
    needed. Figure size grows with the layout's columns and rows.
    Formats control exports; enabled affects rendering only, never report
    statistics or adjustment.
    """

    enabled: bool = True
    top_n: int = 5
    panel_width: float = 6.4
    row_height: float = 5.6
    font_scale: float = 1.0
    graph_height_scale: float = 0.5
    dpi: int = 200
    formats: tuple[str, ...] = ("pdf", "png", "svg")
    layout: Literal["combined", "per_site", "both"] = "combined"
    columns: int = 4

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> RecurrenceBloodTestPlotConfig:
        for name in ("top_n", "dpi", "columns"):
            if name in values and type(values[name]) is not int:
                raise ValueError(f"plot.{name} must be an integer.")
        if "enabled" in values and type(values["enabled"]) is not bool:
            raise ValueError("plot.enabled must be a boolean.")
        formats = values.get("formats", ["pdf", "png", "svg"])
        if not isinstance(formats, list):
            raise ValueError("plot.formats must be a list of pdf, png, or svg.")
        layout = values.get("layout", "combined")
        if layout not in ("combined", "per_site", "both"):
            raise ValueError("plot.layout must be combined, per_site, or both.")
        result = cls(
            enabled=bool(values.get("enabled", True)),
            top_n=int(values.get("top_n", 5)),
            panel_width=float(values.get("panel_width", 6.4)),
            row_height=float(values.get("row_height", 5.6)),
            font_scale=float(values.get("font_scale", 1.0)),
            graph_height_scale=float(values.get("graph_height_scale", 0.5)),
            dpi=int(values.get("dpi", 200)),
            formats=tuple(str(value) for value in formats),
            layout=cast(Literal["combined", "per_site", "both"], layout),
            columns=int(values.get("columns", 4)),
        )
        validate_plot_config(result)
        return result


def validate_plot_config(config: RecurrenceBloodTestPlotConfig) -> None:
    if config.layout not in ("combined", "per_site", "both"):
        raise ValueError("plot.layout must be combined, per_site, or both.")
    if type(config.columns) is not int or config.columns < 1:
        raise ValueError("plot.columns must be a positive integer.")
    if config.top_n < 1 or config.dpi < 1:
        raise ValueError("plot.top_n and plot.dpi must be positive.")
    dimensions = (config.panel_width, config.row_height)
    if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
        raise ValueError("Plot dimensions must be finite and positive.")
    if not np.isfinite(config.font_scale) or not 0 < config.font_scale <= 1:
        raise ValueError("plot.font_scale must be greater than zero and at most one.")
    if not np.isfinite(config.graph_height_scale) or not 0 < config.graph_height_scale <= 1:
        raise ValueError("plot.graph_height_scale must be greater than zero and at most one.")
    if not config.formats or len(set(config.formats)) != len(config.formats):
        raise ValueError("plot.formats must be nonempty and unique.")
    if set(config.formats) - {"pdf", "png", "svg"}:
        raise ValueError("plot.formats supports pdf, png, and svg.")
