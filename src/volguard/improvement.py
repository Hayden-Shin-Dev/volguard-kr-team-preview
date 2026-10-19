"""Development-only model comparisons; the consumed 2025-26 holdout is never read."""

import json
import uuid

import joblib
import numpy as np
import pandas as pd

from .data import HAR_FEATURES, Lake
from .evaluation import promotion_gate
from .history import retrospective_frame
from .models import RangeModels, sequential_predictions
from .research import data_fingerprint, source_hash
from .util import atomic_json, digest, utcnow

RANGE_FEATURES = HAR_FEATURES + [
    "volume_ratio",
    "turnover_ratio",
    "market_range",
    "range_ewm5",
    "range_std22",
    "range_ratio",
    "market_range5",
    "market_range22",
    "relative_range",
]
CANDIDATES = {
    "range_regime": {"features": RANGE_FEATURES, "n_estimators": 350, "num_leaves": 31},
    "range_scaled": {
        "features": RANGE_FEATURES,
        "n_estimators": 350,
        "num_leaves": 31,
        "scaled_conformal": True,
    },
    "range_recent": {
        "features": RANGE_FEATURES,
        "n_estimators": 350,
        "num_leaves": 31,
        "scaled_conformal": True,
        "recent_years": 6,
    },
}


def enrich(frame):
    out = frame.sort_values(["symbol", "date"]).copy()
    groups = out.groupby("symbol", sort=False)
    out["range_ewm5"] = groups.range_1.transform(lambda s: s.ewm(span=5, min_periods=5).mean())
    out["range_std22"] = groups.range_1.transform(lambda s: s.rolling(22).std())
    out["range_ratio"] = out.range_5 / out.range_22.replace(0, np.nan)
    market = out.groupby("date").market_range.first().sort_index()
    out["market_range5"] = out.date.map(market.rolling(5).mean())
    out["market_range22"] = out.date.map(market.rolling(22).mean())
    out["relative_range"] = out.range_1 / out.market_range.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).sort_values(["date", "symbol"]).reset_index(drop=True)


def run(settings, candidates=None, report_name="improvement"):
    candidates = candidates or CANDIDATES
    lake = Lake(settings.data_dir)
    output = lake.root / "runs" / f"improvement-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    # Bound BEFORE creating features: no holdout outcome participates in selection.
    frame = enrich(
        retrospective_frame(lake, verified_price_returns=True).query("target_date <= '2023-12-31'")
    )
    manifest = {
        "created_at": utcnow(),
        "candidates": candidates,
        "folds": [2020, 2022, 2023],
        "selection": "lowest pooled daily interval score among gates passing; otherwise lowest score shadow only",
        "data_hash": data_fingerprint(lake),
        "source_hash": source_hash(),
        "evaluation_role": "development, not a new final holdout",
    }
    atomic_json(output / "manifest.json", manifest)
    source = output / "source"
    source.mkdir()
    import shutil
    from pathlib import Path

    shutil.copytree(Path(__file__).parent, source / "volguard", ignore=shutil.ignore_patterns("__pycache__"))
    results = []
    baselines = {}
    for name, config in candidates.items():
        predictions, folds = [], []
        for year in manifest["folds"]:
            print(f"{name}: development {year}", flush=True)
            train = frame[frame.target_date.le(f"{year - 2}-12-31")]
            model = RangeModels(config).fit(train, baseline=baselines.get(year))
            baselines.setdefault(year, model)
            window = frame[frame.target_date.between(f"{year - 1}-01-01", f"{year}-12-31")]
            prediction = pd.concat(
                [
                    sequential_predictions(model, window, method, f"{year}-01-01")
                    for method in ("har", "ml", "empirical")
                ],
                ignore_index=True,
            )
            prediction = prediction[prediction.symbol.isin(model.har)]
            prediction.to_parquet(output / f"{name}-{year}.parquet", index=False)
            gate = promotion_gate(prediction)
            folds.append({"year": year, **gate})
            predictions.append(prediction)
            print(
                json.dumps(
                    {
                        "candidate": name,
                        "year": year,
                        "improvement": gate.get("improvement"),
                        "selected": gate["selected"],
                    }
                ),
                flush=True,
            )
        gate = promotion_gate(pd.concat(predictions, ignore_index=True))
        results.append({"candidate": name, "config": config, "folds": folds, **gate})
        atomic_json(output / "progress.json", results)
    # Rank by the same predeclared paired daily score used by the promotion gate.
    eligible = [r for r in results if r["selected"] == "ml"] or results
    champion = min(eligible, key=lambda r: r["daily_scores"]["ml"])
    result = {
        **manifest,
        "results": results,
        "champion": champion["candidate"],
        "artifact_dir": str(output.resolve()),
        "deployment_eligible": False,
        "reason": "development selection; forward validation required before ML promotion",
    }
    atomic_json(output / "report.json", result)
    atomic_json(lake.root / "reports" / f"{report_name}.json", result)
    return result


def freeze_shadow(settings):
    lake = Lake(settings.data_dir)
    report = json.loads((lake.root / "reports" / "improvement.json").read_text(encoding="utf8"))
    target = lake.root / "config" / "shadow-release.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf8"))
    config = next(r["config"] for r in report["results"] if r["candidate"] == report["champion"])
    frame = enrich(retrospective_frame(lake, verified_price_returns=True))
    model = RangeModels(config).fit(frame[frame.target_date.le("2025-12-31")])
    output = lake.root / "runs" / f"shadow-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    joblib.dump(model, output / "models.joblib")
    result = {
        "created_at": utcnow(),
        "candidate": report["champion"],
        "config": config,
        "artifact_dir": str(output.resolve()),
        "model_hash": digest((output / "models.joblib").read_bytes()),
        "development_report_hash": digest(report),
        "train_end": "2025-12-31",
        "calibration_start": "2026-01-01",
        "minimum_forward_sessions": 126,
        "promotion": "original coverage/score/bootstrap gates, human-reviewed new version",
        "deployment_eligible": False,
    }
    atomic_json(target, result)
    return result
