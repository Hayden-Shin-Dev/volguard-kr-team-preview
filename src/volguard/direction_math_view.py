"""Read-only presentation of completed financial-structure experiments."""

from .terminal_data import read_json


def view(settings):
    data = read_json(settings.data_dir / "reports/direction-math.json", {})
    if not data:
        return dict(status="pending")
    fields = [
        "status",
        "created_at",
        "selected",
        "feature_choice",
        "threshold",
        "baseline",
        "standalone",
        "comparison",
        "results",
        "range_selected",
        "range_comparator",
        "range_comparison",
        "combined",
        "confidence",
        "selected_confidence",
        "selected_confidence_ci",
        "by_symbol",
        "by_regime",
        "examples",
        "development_gate",
        "operational_probability_enabled",
        "independent_evaluation",
        "rows",
        "ablations",
        "original_comparison",
        "research_review",
    ]
    return {k: data[k] for k in fields if k in data}
