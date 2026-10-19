"""Development-only learning changes, with fixed calibration and archived controls.

The old final test is never evaluated. The reused development folds do not provide
independent confirmation after selection. Production code does not import this module.
"""

import json
import shutil
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .blend_study import blend
from .calibration_study import CALIBRATIONS, paired_comparison, raw_predictions, replay
from .config import Settings
from .data import HAR_FEATURES, ML_FEATURES, ML_REQUIRED_FEATURES, Lake
from .evaluation import promotion_gate
from .models import QUANTILES, RangeModels
from .physical_study import frozen_frame
from .util import atomic_json, digest, utcnow

TAIL_FEATURES = list(dict.fromkeys(ML_FEATURES + [
    "range_std22", "range_max5", "range_max22", "range_q90_22", "range_ewm5",
    "range_ratio", "market_range5", "market_range22", "body_pct", "close_location",
]))
CANDIDATES = {
    "har_residual": {"residual": True, "use_symbol": False, "strengths": [0.5, 1.0]},
    "har_residual_weighted": {"residual": True, "use_symbol": False, "features": TAIL_FEATURES,
                              "scale_weight": True, "num_leaves": 7, "min_child_samples": 200,
                              "reg_lambda": 5.0, "strengths": [0.5, 1.0]},
    "official_basic": {},
    "regularized": {"num_leaves": 7, "min_child_samples": 200, "reg_lambda": 5.0},
    "scale_weighted": {"scale_weight": True},
    "time_weighted": {"half_life_days": 1095},
    "tail_features": {"features": TAIL_FEATURES},
}


def tail_features(frame):
    frame = frame.sort_values(["symbol", "date"]).copy()
    group = frame.groupby("symbol", sort=False).range_1
    frame["range_max22"] = group.transform(lambda s: s.rolling(22, min_periods=22).max())
    frame["range_q90_22"] = group.transform(lambda s: s.rolling(22, min_periods=22).quantile(0.9))
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def training_weights(train, config):
    weights = np.ones(len(train))
    if config.get("scale_weight"):
        # Input-only importance weights; never weight by tomorrow's outcome.
        weights *= (1 + train.range_5).clip(lower=1, upper=20).to_numpy()
    if config.get("half_life_days"):
        age = (train.target_date.max() - train.target_date).dt.days.to_numpy()
        weights *= np.exp2(-age / config["half_life_days"])
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid training weights")
    return weights / weights.mean()


class LearnedRangeModels(RangeModels):
    def _x(self, frame, base=None):
        x = frame[self.config.get("features", ML_FEATURES)].copy()
        if self.config.get("use_symbol", True):
            x["symbol_code"] = frame.symbol.map({s: i for i, s in enumerate(self.symbols)}).fillna(-1).astype(int)
        if self.config.get("residual"):
            if base is None:
                base = super().predict(frame, "har")
            x[["har_low", "har_mid", "har_high"]] = np.log1p(base)
        return x

    def fit(self, frame, baseline=None):
        if baseline is None:
            raise ValueError("Training-only HAR reference required")
        self.har = baseline.har.copy()
        train = frame.dropna(subset=HAR_FEATURES + ML_REQUIRED_FEATURES + ["y"])
        if self.config.get("residual"):
            train = train[train.symbol.isin(self.har)]
        train = train.copy()
        if len(train) < 500:
            raise ValueError("Insufficient training data")
        self.symbols = sorted(train.symbol.unique())
        base = super().predict(train, "har") if self.config.get("residual") else None
        if base is not None:
            valid = np.isfinite(base).all(axis=1)
            train, base = train.loc[valid], base[valid]
        x = self._x(train, base)
        weights = training_weights(train, self.config)
        self.ml = []
        for index, quantile in enumerate(QUANTILES):
            target = np.log1p(train.y.to_numpy())
            if base is not None:
                target = target - np.log1p(base[:, index])
            model = LGBMRegressor(
                objective="quantile", alpha=quantile, n_estimators=200,
                num_leaves=self.config.get("num_leaves", 15), learning_rate=0.035,
                min_child_samples=self.config.get("min_child_samples", 40),
                reg_lambda=self.config.get("reg_lambda", 1.0), verbosity=-1,
                random_state=731, deterministic=True, force_col_wise=True, n_jobs=2,
            )
            categorical = ["symbol_code"] if "symbol_code" in x else []
            model.fit(x, target, sample_weight=weights, categorical_feature=categorical)
            self.ml.append(model)
        self.fit_summary = {"rows": len(train), "train_end": str(train.target_date.max()),
                            "weight_min": float(weights.min()), "weight_max": float(weights.max()),
                            "har_in_sample_residuals": bool(self.config.get("residual")),
                            "note": "HAR fitted only on this fold's training years; no validation labels used"}
        return self

    def predict(self, frame, method):
        if method != "ml":
            return super().predict(frame, method)
        result = np.full((len(frame), 3), np.nan)
        valid = frame[ML_REQUIRED_FEATURES].notna().all(axis=1) & frame.symbol.isin(self.symbols)
        if valid.any():
            selected = frame.loc[valid]
            base = super().predict(selected, "har") if self.config.get("residual") else None
            x = self._x(selected, base)
            predicted = np.column_stack([m.predict(x) for m in self.ml])
            if base is not None:
                predicted = np.log1p(base) + self.config.get("strength", 1.0) * predicted
            with np.errstate(over="ignore", invalid="ignore"):
                result[valid] = np.expm1(predicted)
        result[~np.isfinite(result)] = np.nan
        return np.sort(np.maximum(0, result), axis=1)


def evaluate_candidate(predictions, originals, previous, years):
    combined = pd.concat(predictions, ignore_index=True)
    ml = combined.query("method == 'ml'")
    frozen = promotion_gate(pd.concat([ml, originals.query("method != 'ml'")], ignore_index=True))
    same = promotion_gate(combined)
    versus_original = paired_comparison(ml, originals.query("method == 'ml'"))
    versus_previous = paired_comparison(ml, previous.query("method == 'ml'"))
    folds = [{"year": y, **promotion_gate(p)} for y, p in zip(years, predictions)]
    passed = (frozen["ml_pass"] and same["ml_pass"] and 0.89 <= frozen["paired_ml_coverage"] <= 0.91
              and versus_previous["width_reduction"] > 0 and versus_previous["score_improvement"] > 0
              and all(0.87 <= f["paired_ml_coverage"] <= 0.93 for f in folds))
    return {"versus_frozen_har": frozen, "versus_same_calibration_har": same,
            "versus_original_ml": versus_original, "versus_previous_champion": versus_previous,
            "folds": folds, "development_pass": bool(passed),
            "empty_expanded": int(ml.empty_expanded.sum())}


def run(settings, resume=None):
    control = json.loads((settings.data_dir / "reports/calibration-study.json").read_text(encoding="utf8"))
    previous = json.loads((settings.data_dir / "reports/forecast-improvement.json").read_text(encoding="utf8"))
    control_root, previous_root = Path(control["artifact_dir"]), Path(previous["artifact_dir"])
    frame = tail_features(frozen_frame(Lake(settings.data_dir), control_root))
    output = Path(resume) if resume else settings.data_dir / "runs" / f"learning-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": utcnow(), "candidates": CANDIDATES, "folds": control["folds"],
                "control_artifact": str(control_root), "control_hash": digest(control),
                "previous_artifact": str(previous_root), "previous_champion": previous["champion"],
                "previous_report_hash": digest(previous), "calibration": "signed252/tails252 equal interval blend; no scaling",
                "feature_snapshot_hash": digest((control_root / "development-official.parquet").read_bytes()),
                "source_hash": digest(Path(__file__).read_bytes()),
                "evaluation_role": "reused development only; no final-test access or promotion",
                "selection": "both HAR gates, pooled 89-91%, each fold 87-93%, width and score better than previous champion",
                "reference_hashes": {p.name: digest(p.read_bytes()) for y in control["folds"] for p in (
                    control_root / f"original-{y}.joblib", control_root / f"pred-original__current-{y}.parquet",
                    previous_root / f"pred-{previous['champion']}-{y}.parquet")}}
    if (output / "manifest.json").exists():
        saved = json.loads((output / "manifest.json").read_text(encoding="utf8"))
        for key in manifest.keys() - {"created_at"}:
            if manifest[key] != saved[key]:
                raise ValueError(f"Resume identity mismatch: {key}")
        manifest = saved
    else:
        atomic_json(output / "manifest.json", manifest)
        shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    print(f"Artifacts: {output}", flush=True)
    originals, previous_predictions, predictions_by_candidate = [], [], {}
    for year in manifest["folds"]:
        baseline = joblib.load(control_root / f"original-{year}.joblib")
        original = pd.read_parquet(control_root / f"pred-original__current-{year}.parquet")
        prior = pd.read_parquet(previous_root / f"pred-{previous['champion']}-{year}.parquet")
        originals.append(original)
        previous_predictions.append(prior)
        # Equal treatment: the same signed/tail blend is applied to the HAR comparison.
        har_reference = blend(*(pd.read_parquet(control_root / f"pred-original__{c}-{year}.parquet")
                                for c in ("signed252", "tails252"))).query("method != 'ml'")
        for name, config in CANDIDATES.items():
            print(f"Fit {name} development {year}", flush=True)
            checkpoint = output / f"{name}-{year}.json"
            if checkpoint.exists():
                info = json.loads(checkpoint.read_text(encoding="utf8"))
                for filename, expected in info["hashes"].items():
                    if digest((output / filename).read_bytes()) != expected:
                        raise ValueError(f"Checkpoint changed: {filename}")
                for key in info["keys"]:
                    predictions_by_candidate.setdefault(key, []).append(pd.read_parquet(output / f"pred-{key}-{year}.parquet"))
                continue
            model = LearnedRangeModels(config.copy()).fit(frame[frame.target_date.le(f"{year - 2}-12-31")], baseline)
            model_path = output / f"{name}-{year}.joblib"
            joblib.dump(model, model_path)
            paths, keys = [model_path], []
            window = frame[frame.target_date.between(f"{year - 1}-01-01", f"{year}-12-31")]
            for strength in config.get("strengths", [1.0]):
                model.config["strength"] = strength
                key = f"{name}_s{int(strength * 100)}"
                keys.append(key)
                raw = raw_predictions(model, window, "ml")
                raw_path = output / f"raw-{key}-{year}.parquet"
                raw.to_parquet(raw_path, index=False)
                prediction = blend(*(replay(raw, CALIBRATIONS[c], f"{year}-01-01")
                                     for c in ("signed252", "tails252")))
                prediction = pd.concat([prediction, har_reference], ignore_index=True)
                pred_path = output / f"pred-{key}-{year}.parquet"
                prediction.to_parquet(pred_path, index=False)
                paths.extend([raw_path, pred_path])
                predictions_by_candidate.setdefault(key, []).append(prediction)
                gate = promotion_gate(prediction)
                print(json.dumps({"candidate": key, "year": year, "improvement": gate["improvement"],
                                  "coverage": gate["paired_ml_coverage"], "width": gate["metrics"]["ml"]["width"]}), flush=True)
            atomic_json(checkpoint, {"keys": keys, "fit": model.fit_summary,
                                    "hashes": {p.name: digest(p.read_bytes()) for p in paths}})
    original, prior = pd.concat(originals, ignore_index=True), pd.concat(previous_predictions, ignore_index=True)
    results = [{"candidate": key, **evaluate_candidate(predictions, original, prior, manifest["folds"])}
               for key, predictions in predictions_by_candidate.items()]
    passing = [r for r in results if r["development_pass"]]
    best = min(passing or results, key=lambda r: r["versus_frozen_har"]["daily_scores"]["ml"])
    result = {**manifest, "results": results, "champion": best["candidate"],
              "development_pass": bool(passing), "deployment_eligible": False,
              "artifact_dir": str(output.resolve()), "completed_at": utcnow()}
    atomic_json(output / "report.json", result)
    atomic_json(settings.data_dir / "reports/learning-study.json", result)
    print(json.dumps({"champion": best["candidate"], "result": best}), flush=True)
    return result


if __name__ == "__main__":
    import argparse

    from .learning_study import run as module_run

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    args = parser.parse_args()
    module_run(Settings.load(), args.resume)
