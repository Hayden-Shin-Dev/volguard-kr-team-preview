"""Nested chronological early stopping; outer development labels never select trees."""

import json
import shutil
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

from .blend_study import blend
from .calibration_study import CALIBRATIONS, raw_predictions, replay
from .config import Settings
from .data import HAR_FEATURES, ML_REQUIRED_FEATURES, Lake
from .evaluation import promotion_gate
from .learning_study import LearnedRangeModels, evaluate_candidate
from .models import QUANTILES
from .physical_study import frozen_frame
from .util import atomic_json, digest, utcnow

CANDIDATES = {f"leaves{leaves}_nested": {"num_leaves": leaves} for leaves in (7, 15, 31)}


def inner_split(train, days=252):
    dates = sorted(train.target_date.unique())
    if len(dates) < days + 252:
        raise ValueError("Not enough dates for nested selection")
    boundary = dates[-days]
    validation = train[train.target_date.ge(boundary)]
    cutoff = (validation.date.min() + pd.Timedelta(hours=18, minutes=30)).tz_localize("Asia/Seoul")
    available = pd.to_datetime(train.target_available_at, utc=True)
    fitting = train[train.target_date.lt(boundary) & available.le(cutoff)]
    if fitting.empty or validation.empty:
        raise ValueError("Empty nested chronological split")
    return fitting, validation


def physical_pinball(quantile):
    def score(log_y, log_prediction):
        error = np.expm1(log_y) - np.maximum(0, np.expm1(log_prediction))
        value = np.maximum(quantile * error, (quantile - 1) * error).mean()
        return "range_pinball", float(value), False
    return score


class BoostedRangeModels(LearnedRangeModels):
    def fit(self, frame, baseline=None):
        if baseline is None:
            raise ValueError("Training-only HAR required")
        train = frame.dropna(subset=HAR_FEATURES + ML_REQUIRED_FEATURES + ["y"]).copy()
        self.har, self.symbols = baseline.har.copy(), sorted(train.symbol.unique())
        fitting, validation = inner_split(train)
        self.ml, selections = [], []
        for q in QUANTILES:
            params = dict(objective="quantile", alpha=q, num_leaves=self.config["num_leaves"],
                          learning_rate=.035, min_child_samples=40, reg_lambda=1., verbosity=-1,
                          random_state=731, deterministic=True, force_col_wise=True, n_jobs=2, metric="None")
            select = LGBMRegressor(**params, n_estimators=1600)
            select.fit(self._x(fitting), np.log1p(fitting.y), categorical_feature=["symbol_code"],
                       eval_set=[(self._x(validation), np.log1p(validation.y))],
                       eval_metric=physical_pinball(q), callbacks=[early_stopping(100, verbose=False)])
            trees = int(select.best_iteration_)
            model = LGBMRegressor(**params, n_estimators=trees)
            model.fit(self._x(train), np.log1p(train.y), categorical_feature=["symbol_code"])
            self.ml.append(model)
            selections.append({"quantile": q, "trees": trees,
                               "inner_pinball": select.best_score_["valid_0"]["range_pinball"]})
        self.fit_summary = {"rows": len(train), "train_end": str(train.target_date.max()),
                            "inner_fit_end": str(fitting.target_date.max()),
                            "inner_valid_start": str(validation.target_date.min()),
                            "inner_valid_end": str(validation.target_date.max()), "selection": selections}
        return self


def run(settings, resume=None):
    control = json.loads((settings.data_dir / "reports/calibration-study.json").read_text(encoding="utf8"))
    previous = json.loads((settings.data_dir / "reports/forecast-improvement.json").read_text(encoding="utf8"))
    source, previous_root = Path(control["artifact_dir"]), Path(previous["artifact_dir"])
    frame = frozen_frame(Lake(settings.data_dir), source)
    output = Path(resume) if resume else settings.data_dir / "runs" / f"boosting-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": utcnow(), "candidates": CANDIDATES, "folds": control["folds"],
                "control_artifact": str(source), "previous_artifact": str(previous_root),
                "previous_champion": previous["champion"], "previous_report_hash": digest(previous),
                "source_hash": digest(Path(__file__).read_bytes()),
                "feature_snapshot_hash": digest((source / "development-official.parquet").read_bytes()),
                "calibration": "signed252/tails252 equal blend",
                "inner_selection": "last 252 training target dates; physical pinball; max1600 trees; patience100; refit full train",
                "evaluation_role": "reused development only; no final test or production promotion",
                "selection": "both HAR gates, pooled89-91%, eachfold87-93%, width and score beat previous champion",
                "reference_hashes": {p.name: digest(p.read_bytes()) for y in control["folds"] for p in (
                    source / f"original-{y}.joblib", source / f"pred-original__current-{y}.parquet",
                    previous_root / f"pred-{previous['champion']}-{y}.parquet")}}
    if (output / "manifest.json").exists():
        saved = json.loads((output / "manifest.json").read_text(encoding="utf8"))
        if any(saved[k] != manifest[k] for k in manifest.keys() - {"created_at"}):
            raise ValueError("Resume identity mismatch")
        manifest = saved
    else:
        atomic_json(output / "manifest.json", manifest)
        shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    print(f"Artifacts: {output}", flush=True)
    originals, priors, predictions = [], [], {key: [] for key in CANDIDATES}
    for year in manifest["folds"]:
        baseline = joblib.load(source / f"original-{year}.joblib")
        originals.append(pd.read_parquet(source / f"pred-original__current-{year}.parquet"))
        prior = pd.read_parquet(previous_root / f"pred-{previous['champion']}-{year}.parquet")
        priors.append(prior)
        for key, config in CANDIDATES.items():
            checkpoint = output / f"{key}-{year}.json"
            pred_path = output / f"pred-{key}-{year}.parquet"
            if checkpoint.exists():
                info = json.loads(checkpoint.read_text(encoding="utf8"))
                if any(digest((output / name).read_bytes()) != h for name, h in info["hashes"].items()):
                    raise ValueError("Checkpoint changed")
                predictions[key].append(pd.read_parquet(pred_path))
                continue
            print(f"Fit {key} development {year}", flush=True)
            model = BoostedRangeModels(config.copy()).fit(frame[frame.target_date.le(f"{year - 2}-12-31")], baseline)
            model_path, raw_path = output / f"{key}-{year}.joblib", output / f"raw-{key}-{year}.parquet"
            joblib.dump(model, model_path)
            raw = raw_predictions(model, frame[frame.target_date.between(f"{year - 1}-01-01", f"{year}-12-31")], "ml")
            raw.to_parquet(raw_path, index=False)
            ml = blend(*(replay(raw, CALIBRATIONS[c], f"{year}-01-01") for c in ("signed252", "tails252")))
            pred = pd.concat([ml, prior.query("method != 'ml'")], ignore_index=True)
            pred.to_parquet(pred_path, index=False)
            predictions[key].append(pred)
            atomic_json(checkpoint, {"fit": model.fit_summary, "hashes": {
                p.name: digest(p.read_bytes()) for p in (model_path, raw_path, pred_path)}})
            gate = promotion_gate(pred)
            print(json.dumps({"candidate": key, "year": year, "fit": model.fit_summary,
                              "improvement": gate["improvement"], "coverage": gate["paired_ml_coverage"],
                              "width": gate["metrics"]["ml"]["width"]}), flush=True)
    original, prior = pd.concat(originals, ignore_index=True), pd.concat(priors, ignore_index=True)
    results = [{"candidate": key, **evaluate_candidate(parts, original, prior, manifest["folds"])}
               for key, parts in predictions.items()]
    passing = [r for r in results if r["development_pass"]]
    best = min(passing or results, key=lambda r: r["versus_frozen_har"]["daily_scores"]["ml"])
    report = {**manifest, "results": results, "champion": best["candidate"],
              "development_pass": bool(passing), "deployment_eligible": False,
              "artifact_dir": str(output.resolve()), "completed_at": utcnow()}
    atomic_json(output / "report.json", report)
    atomic_json(settings.data_dir / "reports/boosting-study.json", report)
    print(json.dumps({"champion": best["candidate"], "result": best}), flush=True)


if __name__ == "__main__":
    import argparse

    from .boosting_study import run as module_run

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    args = parser.parse_args()
    module_run(Settings.load(), args.resume)
