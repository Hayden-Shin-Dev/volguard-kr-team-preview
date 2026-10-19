"""Chronological refits of the frozen nested7_tail50 recipe, never live weights."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .blend_study import blend
from .boosting_study import BoostedRangeModels
from .calibration_study import CALIBRATIONS, raw_predictions, replay
from .data import Lake
from .learning_study import LearnedRangeModels, TAIL_FEATURES, tail_features
from .models import RangeModels
from .physical_study import frozen_frame
from .platform_study import local_settings
from .util import atomic_json, digest, utcnow

RANGE_COLS = ["range_forecast_low", "range_forecast_median", "range_forecast_high", "range_forecast_width"]


def check_temporal(frame):
    """A fitted target must be known before calibration and every forecast issue."""
    if frame.duplicated(["symbol", "target_date"]).any():
        raise ValueError("Duplicate OOF key")
    if not (frame.fit_target_end < frame.calibration_start).all():
        raise ValueError("Fit overlaps calibration")
    if not (frame.calibration_start <= frame.date).all():
        raise ValueError("Calibration begins after forecast issue")
    if not (frame.fit_target_end < frame.date).all() or not (frame.date < frame.target_date).all():
        raise ValueError("Future fit/target in OOF inputs")
    if not np.isfinite(frame[RANGE_COLS]).all().all():
        raise ValueError("Invalid OOF values")
    if not (
        (frame.range_forecast_low >= 0)
        & (frame.range_forecast_low <= frame.range_forecast_median)
        & (frame.range_forecast_median <= frame.range_forecast_high)
    ).all():
        raise ValueError("Invalid OOF interval")


def run(settings):
    root = settings.data_dir
    control = json.loads((root / "reports/calibration-study.json").read_text(encoding="utf8"))
    source = Path(control["artifact_dir"])
    frame = tail_features(frozen_frame(Lake(root), source))
    assert frame.target_date.max() < pd.Timestamp("2024-01-01")
    identity = dict(
        recipe="nested7_tail50",
        years=list(range(2015, 2024)),
        feature_hash=digest((source / "development-official.parquet").read_bytes()),
        source_hash=digest(Path(__file__).read_bytes()),
        note="Frozen recipe retrospective OOF; architecture selected later; not historical issuance or independent holdout",
        fit_rule="target <= year-2 end; prior year calibration; predict year; past errors only",
    )
    out = root / "runs" / ("direction-range-oof-" + digest(identity)[:12])
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "manifest.json", identity)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    studies = {
        n: json.loads((root / f"reports/{n}-study.json").read_text(encoding="utf8"))
        for n in ["boosting", "learning"]
    }
    parts = []
    for year in identity["years"]:
        dest, meta_path = out / f"oof-{year}.parquet", out / f"oof-{year}.json"
        if dest.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf8"))
            if digest(dest.read_bytes()) != meta["hash"]:
                raise ValueError("OOF checkpoint hash mismatch")
            parts.append(pd.read_parquet(dest))
            continue
        train = frame[frame.target_date.le(f"{year - 2}-12-31")]
        window = frame[frame.target_date.between(f"{year - 1}-01-01", f"{year}-12-31")]
        components, provenance = [], []
        baseline = None
        for name, stage, cls, config in [
            ("leaves7_nested", "boosting", BoostedRangeModels, {"num_leaves": 7}),
            ("tail_features", "learning", LearnedRangeModels, {"features": TAIL_FEATURES}),
        ]:
            archive = Path(studies[stage]["artifact_dir"])
            model_path, checkpoint = archive / f"{name}-{year}.joblib", archive / f"{name}-{year}.json"
            local_model = out / f"{name}-{year}.joblib"
            if model_path.exists() and checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding="utf8"))
                if digest(model_path.read_bytes()) != saved["hashes"][model_path.name]:
                    raise ValueError("Archived component mismatch")
                model = joblib.load(model_path)
                if pd.Timestamp(model.fit_summary["train_end"]) >= pd.Timestamp(f"{year - 1}-01-01"):
                    raise ValueError("Archived model contains future targets")
                mode = "verified archived chronological fit"
            elif local_model.exists():
                model = joblib.load(local_model)
                model_path, mode = local_model, "resumed chronological refit"
            else:
                print(f"OOF {year} fit {name}", flush=True)
                if baseline is None:
                    baseline = RangeModels().fit(train)
                model = cls(config).fit(train, baseline)
                joblib.dump(model, local_model)
                model_path, mode = local_model, "new chronological refit"
            raw = raw_predictions(model, window, "ml")
            # Both calibration methods replay target availability before each issue.
            components.append(
                blend(*(replay(raw, CALIBRATIONS[c], f"{year}-01-01") for c in ["signed252", "tails252"]))
            )
            provenance.append(
                dict(
                    component=name,
                    model=str(model_path),
                    hash=digest(model_path.read_bytes()),
                    fit=model.fit_summary,
                    mode=mode,
                )
            )
        result = blend(*components).dropna(subset=["low", "median", "high"]).copy()
        result = result.rename(columns={c: "range_forecast_" + c for c in ["low", "median", "high"]})
        result["range_forecast_width"] = result.range_forecast_high - result.range_forecast_low
        result["fit_target_end"] = max(pd.Timestamp(p["fit"]["train_end"]) for p in provenance)
        result["calibration_start"] = pd.Timestamp(f"{year - 1}-01-01")
        result["range_fold"] = year
        check_temporal(result)
        result.to_parquet(dest, index=False)
        atomic_json(
            meta_path,
            dict(hash=digest(dest.read_bytes()), n=len(result), components=provenance, completed_at=utcnow()),
        )
        parts.append(result)
        print(json.dumps(dict(oof_year=year, rows=len(result), completed=True)), flush=True)
    combined = pd.concat(parts, ignore_index=True)
    check_temporal(combined)
    combined.to_parquet(out / "oof.parquet", index=False)
    report = dict(
        **identity,
        artifact_dir=str(out),
        rows=len(combined),
        oof_hash=digest((out / "oof.parquet").read_bytes()),
        completed_at=utcnow(),
    )
    atomic_json(out / "report.json", report)
    atomic_json(root / "reports/direction-range-oof.json", report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    run(local_settings())
