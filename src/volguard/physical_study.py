"""Development ablation: learn quantiles in range units, rather than log units."""

import json
import shutil
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .calibration_study import CALIBRATIONS, paired_comparison, raw_predictions, replay
from .config import Settings
from .data import HAR_FEATURES, ML_REQUIRED_FEATURES, Lake
from .evaluation import promotion_gate
from .history import retrospective_frame
from .improvement import enrich
from .models import QUANTILES, RangeModels
from .util import atomic_json, digest, utcnow


class PhysicalRangeModels(RangeModels):
    """Ablate only the logarithm; fixed features, training dates and parameters."""

    def fit(self, frame, baseline=None):
        if baseline is None:
            raise ValueError("A training-only HAR reference is required")
        train = frame.dropna(subset=HAR_FEATURES + ["y"])
        self.symbols = sorted(train.symbol.unique())
        self.har = baseline.har.copy()
        train = train.dropna(subset=ML_REQUIRED_FEATURES)
        if len(train) < 500 or not self.har:
            raise ValueError("Insufficient training observations")
        target = train.y
        if self.config.get("target_scale"):
            target = target / train[self.config["target_scale"]].clip(lower=0.1)
        self.ml = []
        for quantile in QUANTILES:
            model = LGBMRegressor(
                objective="quantile", alpha=quantile, n_estimators=350, num_leaves=15,
                learning_rate=0.035, min_child_samples=40, reg_lambda=1.0,
                verbosity=-1, random_state=731, deterministic=True, force_col_wise=True, n_jobs=2,
            )
            model.fit(self._ml_x(train), target, categorical_feature=["symbol_code"])
            self.ml.append(model)
        self.fit_summary = {"rows": len(train), "train_end": str(train.target_date.max()),
                            "target_transform": "identity, optional ex-ante range scale"}
        return self

    def predict(self, frame, method):
        if method != "ml":
            return super().predict(frame, method)
        result = np.full((len(frame), 3), np.nan)
        valid = frame[ML_REQUIRED_FEATURES].notna().all(axis=1).to_numpy() & frame.symbol.isin(self.symbols)
        if valid.any():
            result[valid] = np.column_stack([m.predict(self._ml_x(frame.loc[valid])) for m in self.ml])
        if self.config.get("target_scale"):
            result *= frame[self.config["target_scale"]].clip(lower=0.1).to_numpy()[:, None]
        result[~np.isfinite(result)] = np.nan
        return np.sort(np.maximum(0, result), axis=1)


def frozen_frame(lake, control_root):
    control = json.loads((control_root / "manifest.json").read_text(encoding="utf8"))
    path, meta = control_root / "development-official.parquet", control_root / "development-features.json"
    if meta.exists():
        evidence = json.loads(meta.read_text(encoding="utf8"))
        if evidence["file_hash"] != digest(path.read_bytes()) or evidence["data_hashes"] != control["data_hashes"]:
            raise ValueError("Frozen development snapshot changed")
        frame = pd.read_parquet(path)
    else:
        for name, expected in control["data_hashes"].items():
            if digest((lake.root / "silver" / name).read_bytes()) != expected:
                raise ValueError("Cannot create control snapshot after source data changed")
        frame = enrich(retrospective_frame(lake, verified_price_returns=True)
                       .query("target_date <= '2023-12-31'"))
        frame.to_parquet(path, index=False)
        atomic_json(meta, {"file_hash": digest(path.read_bytes()), "data_hashes": control["data_hashes"]})
    if not frame.target_date.le("2023-12-31").all():
        raise ValueError("Development snapshot contains later outcomes")
    return frame


def run(settings, control_root=None):
    lake = Lake(settings.data_dir)
    if control_root is None:
        control_report = json.loads((lake.root / "reports/calibration-study.json").read_text(encoding="utf8"))
        control_root = control_report["artifact_dir"]
    control_root = Path(control_root)
    control = json.loads((control_root / "manifest.json").read_text(encoding="utf8"))
    frame = frozen_frame(lake, control_root)
    config = control["model_configs"]["official_normalized"]
    configs = {"physical_full": {k: v for k, v in config.items() if k != "target_scale"},
               "physical_normalized": config.copy()}
    calibrations = {k: CALIBRATIONS[k] for k in ("current", "signed252", "tails252")}
    output = lake.root / "runs" / f"physical-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    manifest = {
        "created_at": utcnow(), "model_configs": configs, "calibrations": calibrations,
        "folds": control["folds"], "control_artifact": str(control_root),
        "control_hash": digest(control), "data_hashes": control["data_hashes"],
        "source_hash": digest(Path(__file__).read_bytes()),
        "feature_snapshot_hash": digest((control_root / "development-official.parquet").read_bytes()),
        "evaluation_role": "reused development; no final holdout",
        "selection": control["selection"],
    }
    atomic_json(output / "manifest.json", manifest)
    shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    print(f"Artifacts: {output}", flush=True)
    all_results, controls = {}, []
    for year in manifest["folds"]:
        baseline = joblib.load(control_root / f"original-{year}.joblib")
        original = pd.read_parquet(control_root / f"pred-original__current-{year}.parquet")
        controls.append(original)
        for name, model_config in configs.items():
            print(f"Fit {name} development {year}", flush=True)
            model = PhysicalRangeModels(model_config).fit(
                frame[frame.target_date.le(f"{year - 2}-12-31")], baseline=baseline)
            joblib.dump(model, output / f"{name}-{year}.joblib")
            window = frame[frame.target_date.between(f"{year - 1}-01-01", f"{year}-12-31")]
            raw = raw_predictions(model, window, "ml")
            raw.to_parquet(output / f"raw-{name}-{year}.parquet", index=False)
            for calibration, cal_config in calibrations.items():
                key = f"{name}__{calibration}"
                calibrated = replay(raw, {**cal_config, "scaled": bool(model_config.get("scaled_conformal"))},
                                    f"{year}-01-01")
                reference = pd.read_parquet(control_root / f"pred-original__{calibration}-{year}.parquet")
                prediction = pd.concat([calibrated, reference.query("method != 'ml'")], ignore_index=True)
                prediction.to_parquet(output / f"pred-{key}-{year}.parquet", index=False)
                all_results.setdefault(key, []).append(prediction)
                gate = promotion_gate(prediction)
                print(json.dumps({"candidate": key, "year": year, "improvement": gate["improvement"],
                                  "coverage": gate["paired_ml_coverage"]}), flush=True)
    original = pd.concat(controls, ignore_index=True)
    results = []
    for key, predictions in all_results.items():
        pred = pd.concat(predictions, ignore_index=True)
        ml = pred.query("method == 'ml'")
        frozen = promotion_gate(pd.concat([ml, original.query("method != 'ml'")], ignore_index=True))
        same = promotion_gate(pred)
        compare = paired_comparison(ml, original.query("method == 'ml'"))
        folds = [{"year": y, **promotion_gate(p)} for y, p in zip(manifest["folds"], predictions)]
        passed = (frozen["ml_pass"] and same["ml_pass"] and 0.89 <= frozen["paired_ml_coverage"] <= 0.91
                  and compare["width_reduction"] > 0 and compare["score_improvement"] > 0
                  and all(0.87 <= f["paired_ml_coverage"] <= 0.93 for f in folds))
        results.append({"candidate": key, "versus_frozen_har": frozen, "versus_same_calibration_har": same,
                        "versus_original_ml": compare, "folds": folds, "development_pass": bool(passed),
                        "empty_expanded": int(ml.empty_expanded.sum())})
    passed = [r for r in results if r["development_pass"]]
    best = min(passed or results, key=lambda r: r["versus_frozen_har"]["daily_scores"]["ml"])
    report = {**manifest, "results": results, "champion": best["candidate"],
              "development_pass": bool(passed), "deployment_eligible": False,
              "artifact_dir": str(output.resolve()), "completed_at": utcnow()}
    atomic_json(output / "report.json", report)
    atomic_json(lake.root / "reports/physical-study.json", report)
    print(json.dumps({"champion": best["candidate"], "result": best}), flush=True)
    return report


if __name__ == "__main__":
    import argparse

    from .physical_study import run as module_run

    parser = argparse.ArgumentParser()
    parser.add_argument("--control-root")
    args = parser.parse_args()
    module_run(Settings.load(), args.control_root)
