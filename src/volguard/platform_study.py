"""Append-only development diagnostics. Never changes a production release."""

import json
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Settings
from .data import Lake
from .terminal_data import read_json
from .util import atomic_json, digest, utcnow


def local_settings():
    s = Settings.load()
    return Settings(Path(read_json(Path(".service-paths.json"))["data_root"]) / "live", s.environment)


def measures(df):
    width = df.high - df.low
    score = width + 20 * np.maximum(np.maximum(df.low - df.y, df.y - df.high), 0)
    return dict(
        n=len(df),
        coverage=float(df.y.between(df.low, df.high).mean()),
        width=float(width.mean()),
        score=float(score.mean()),
    )


def run(settings):
    root = settings.data_dir
    source = read_json(root / "reports/ensemble-study.json")
    out = root / "runs" / f"range-diagnostics-{uuid.uuid4().hex[:10]}"
    out.mkdir()
    manifest = dict(
        created_at=utcnow(),
        source_report_hash=digest(source),
        folds=source["folds"],
        factors=[0.96, 0.98, 1.0, 1.02],
        rule="Reused development only; no release promotion; width and score must both improve, coverage 87–93%",
        locked_test_used=False,
        source_hash=digest(Path(__file__).read_bytes()),
    )
    atomic_json(out / "manifest.json", manifest)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    df = pd.concat(
        [
            pd.read_parquet(Path(source["artifact_dir"]) / f"pred-nested7_tail50-{y}.parquet")
            for y in source["folds"]
        ],
        ignore_index=True,
    )
    df = df[df.method.eq("ml")].copy()
    df = df[np.isfinite(df[["low", "median", "high", "y"]]).all(axis=1)].copy()
    assert df.target_date.max() < pd.Timestamp("2024-01-01")
    df["width"] = df.high - df.low
    # Prior observed range, not the target, determines the volatility bucket.
    bars = Lake(root).bars()
    bars = bars[bars.date.lt("2024-01-01")]
    df = df.merge(bars[["symbol", "date", "range_pct"]], on=["symbol", "date"], how="left")
    df["regime"] = pd.cut(
        df.range_pct, [-np.inf, 2, 5, np.inf], labels=["below_2pct", "2_to_5pct", "above_5pct"]
    )
    market = pd.read_parquet(
        root / "silver/market.parquet", filters=[("date", "<", pd.Timestamp("2024-01-01"))]
    )
    df["market_range"] = df.date.map(market.drop_duplicates("date").set_index("date").range_pct)
    df["market_regime"] = pd.cut(
        df.market_range, [-np.inf, 1, 2, np.inf], labels=["below_1pct", "1_to_2pct", "above_2pct"]
    )
    candidates = []
    for f in manifest["factors"]:
        c = df.copy()
        c["low"] = np.maximum(0, c["median"] + f * (c.low - c["median"]))
        c["high"] = c["median"] + f * (c.high - c["median"])
        candidates.append({"factor": f, **measures(c)})
    base = measures(df)
    components = []
    release = read_json(root / "config/forecast-release.json", {})
    for c in release.get("components", []):
        model = joblib.load(c["path"])
        importance = {}
        for regressor in model.ml:
            for feature, gain in zip(
                regressor.feature_name_, regressor.booster_.feature_importance(importance_type="gain")
            ):
                importance[feature] = importance.get(feature, 0.0) + float(gain)
        total = sum(importance.values()) or 1
        components.append(
            {
                "model_hash": c["hash"],
                "training": model.fit_summary,
                "config": model.config,
                "gain_share": dict(
                    sorted(((k, v / total) for k, v in importance.items()), key=lambda p: -p[1])
                ),
            }
        )
    report = {
        **manifest,
        "artifact_dir": str(out),
        "baseline": base,
        "candidates": candidates,
        "by_symbol": {str(k): measures(g) for k, g in df.groupby("symbol")},
        "by_regime": {str(k): measures(g) for k, g in df.groupby("regime", observed=True)},
        "by_market_regime": {str(k): measures(g) for k, g in df.groupby("market_regime", observed=True)},
        "sector_status": "unavailable: frozen universe has no verified historical industry membership",
        "interpretation": "Width reflects conditional range scale and calibrated tail error; contraction alone trades missed tails against width.",
        "selected": "nested7_tail50",
        "promoted": False,
        "model_diagnostics": components,
        "improving_candidates": [
            c
            for c in candidates
            if c["width"] < base["width"] and c["score"] < base["score"] and 0.87 <= c["coverage"] <= 0.93
        ],
    }
    atomic_json(out / "report.json", report)
    pointer = root / "reports/platform-range.json"
    # Each full report remains in its unique run directory; only the new UI pointer advances.
    atomic_json(pointer, report)
    print(
        json.dumps(
            {k: report[k] for k in ["artifact_dir", "baseline", "candidates", "improving_candidates"]},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    run(local_settings())
