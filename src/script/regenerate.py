"""Regenerate configured study outputs: uv run python -m src.script.regenerate."""

from src.regeneration.runner import run_regeneration


def main() -> None:
    run_regeneration()


if __name__ == "__main__":
    main()
