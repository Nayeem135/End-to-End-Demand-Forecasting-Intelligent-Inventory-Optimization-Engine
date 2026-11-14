"""Human-readable leaderboard report (Markdown) for the report and MLflow."""

from __future__ import annotations

import math

import pandas as pd

METRIC_COLUMNS = ("wrmsse", "wape", "smape", "rmsse", "mae", "bias", "coverage")
_PERCENT = {"wape", "smape", "coverage"}


def _fmt(column: str, value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        return f"{value:.1%}" if column in _PERCENT else f"{value:.4f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    """Render ``columns`` of ``df`` as a GitHub-flavoured Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    align = (
        "|"
        + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in columns)
        + "|"
    )
    rows = ["| " + " | ".join(_fmt(c, r[c]) for c in columns) + " |" for r in df.to_dict("records")]
    return "\n".join([header, align, *rows])


def leaderboard_report(
    overall_validation: pd.DataFrame,
    overall_test: pd.DataFrame,
    by_class: pd.DataFrame,
    champion: str,
) -> str:
    """Markdown report: overall leaderboards and the test leaderboard per demand class.

    Args:
        overall_validation: Leaderboard on the validation folds (champion selection).
        overall_test: Leaderboard on the held-out test windows.
        by_class: Output of ``leaderboard_by_class`` with a ``split`` column.
        champion: Model selected on validation WRMSSE.
    """
    cols = ["rank", "model", *METRIC_COLUMNS]
    parts = [
        "# Model leaderboard",
        "",
        f"Champion (lowest validation WRMSSE): **{champion}**. WAPE and sMAPE are pooled over "
        "all forecast days; sMAPE scores any non-zero forecast on a zero-sales day as 200 %, "
        "so it favours forecasting zero for intermittent items.",
        "",
        "## Validation folds (used for model selection)",
        "",
        markdown_table(overall_validation, cols),
        "",
        "## Test windows (never used for any decision)",
        "",
        markdown_table(overall_test, cols),
        "",
        "## Test windows by demand class",
    ]
    test_classes = by_class[by_class["split"] == "test"]
    for cls, part in test_classes.groupby("demand_class", sort=True):
        parts += [
            "",
            f"### {cls} ({int(part['n_series'].iloc[0])} series)",
            "",
            markdown_table(part, ["rank", "model", *METRIC_COLUMNS]),
        ]
    return "\n".join(parts) + "\n"
