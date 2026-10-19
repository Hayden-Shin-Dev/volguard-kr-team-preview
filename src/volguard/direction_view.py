"""Read-only presentation of actual stored, time-separated direction predictions."""

from pathlib import Path

import pandas as pd

from .terminal_data import read_json


def integration(settings):
    report = read_json(settings.data_dir / "reports/direction-integration.json", {})
    if not report:
        return {"status": "pending"}
    keys = [
        "selected",
        "reference",
        "comparison",
        "range_ablations",
        "versus_original",
        "results",
        "external_results",
        "external_ablation",
        "gate",
        "development_pass",
        "matched_rows",
        "external_rows",
        "completed_at",
        "operational_probability_enabled",
    ]
    return {
        "status": "development",
        **{k: report[k] for k in keys},
        "oof": read_json(settings.data_dir / "reports/direction-range-oof.json", {}),
    }


def integration_history(settings, symbol):
    report = read_json(settings.data_dir / "reports/direction-integration.json", {})
    if not report:
        return {"rows": [], "status": "pending"}
    frame = pd.read_parquet(
        Path(report["artifact_dir"]) / "predictions.parquet",
        filters=[
            ("symbol", "=", symbol),
            ("candidate", "in", [report["selected"], report["reference"]]),
            ("target_date", ">=", pd.Timestamp("2023-01-01")),
        ],
    )
    candidate = frame[frame.candidate.eq(report["selected"])]
    reference = frame[frame.candidate.eq(report["reference"])]
    joined = candidate.merge(
        reference[["target_date", "p_up"]], on="target_date", suffixes=("", "_base"), validate="one_to_one"
    )
    rows = []
    for r in joined.sort_values("target_date").tail(120).itertuples():
        correct = bool(r.p_up != 0.5 and ((r.p_up > 0.5) == r.label))
        inside = bool(r.range_forecast_low <= r.y <= r.range_forecast_high)
        rows.append(
            dict(
                date=r.date.strftime("%Y-%m-%d"),
                target_date=r.target_date.strftime("%Y-%m-%d"),
                p_up=float(r.p_up),
                p_down=float(1 - r.p_up),
                baseline_p_up=float(r.p_up_base),
                actual_up=bool(r.label),
                actual_return=float(r.direction_return),
                correct=correct,
                low=float(r.range_forecast_low),
                median=float(r.range_forecast_median),
                high=float(r.range_forecast_high),
                actual_range=float(r.y),
                inside=inside,
                joint=correct and inside,
            )
        )
    return dict(status="retrospective", selected=report["selected"], rows=rows)


def comparison(settings):
    report = read_json(settings.data_dir / "reports/platform-direction-accuracy.json", {})
    if not report:
        return {"status": "pending", "results": []}
    return {
        "status": "development",
        "dataset": read_json(settings.data_dir / "reports/platform-direction-dataset.json", {}),
        **{k: report[k] for k in ["selected", "names", "folds", "excluded_flat", "note"]},
        "results": [{k: v for k, v in r.items() if k != "by_symbol"} for r in report["results"]],
    }


def history(settings, symbol, model=None):
    report = read_json(settings.data_dir / "reports/platform-direction-accuracy.json", {})
    if not report:
        return {"rows": [], "status": "pending"}
    model = model or report["selected"]
    if model not in report["names"]:
        raise ValueError("비교 기록에 없는 모델입니다")
    frame = pd.read_parquet(
        Path(report["artifact_dir"]) / "predictions.parquet",
        filters=[("symbol", "=", symbol), ("model", "=", model), ("fold", "=", 2023)],
    )
    rows = []
    for r in frame.sort_values("target_date").tail(120).itertuples():
        rows.append(
            dict(
                date=r.date.strftime("%Y-%m-%d"),
                target_date=r.target_date.strftime("%Y-%m-%d"),
                p_up=float(r.p_up),
                p_down=float(r.p_down),
                prediction=r.prediction,
                actual="up" if r.label else "down",
                correct=bool(r.correct),
                actual_return=float(r.direction_return),
            )
        )
    selected = next(r for r in report["results"] if r["model"] == model)
    return dict(
        status="retrospective",
        model=model,
        name=report["names"][model],
        rows=rows,
        symbol_score=selected["by_symbol"].get(symbol),
        note="2023년 개발 평가 기록입니다. 다음 날을 제외한 이전 자료로 계산한 상승·하락 확률이며 실제 운영 사전 발행은 아닙니다.",
    )
