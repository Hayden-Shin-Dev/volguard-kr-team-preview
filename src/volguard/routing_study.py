"""Causal stock-specific HAR/ML routing on development folds only.

Uses settled past proper scores, never a retrospective best-model oracle.
Interval selection is not a conformal coverage guarantee; measure it explicitly.
"""

import heapq
import json
import shutil
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from .blend_study import blend
from .calibration_study import CALIBRATIONS, replay
from .config import Settings
from .evaluation import interval_score
from .learning_study import evaluate_candidate
from .util import atomic_json, digest, utcnow

CANDIDATES = {f"history{window}_har{int(weight * 100)}": {"window": window, "weight": weight}
              for window in (60, 120) for weight in (0.5, 1.0)}


def route(ml, har, window=60, weight=0.5, minimum=60):
    if not 1 <= minimum <= window or not 0 <= weight <= 1:
        raise ValueError("Invalid routing parameters")
    keys = ["symbol", "target_date"]
    ml, har = (frame.set_index(keys).sort_index() for frame in (ml, har))
    pd.testing.assert_index_equal(ml.index, har.index)
    for column in ("date", "y", "target_available_at"):
        pd.testing.assert_series_equal(ml[column], har[column])
    combined = ml.reset_index().copy()
    for column in ("low", "median", "high"):
        combined[f"har_{column}"] = har[column].to_numpy()
    blocks = []
    for _, group in combined.groupby("symbol", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        default_issue = (group.date + pd.Timedelta(hours=18, minutes=30)).dt.tz_localize("Asia/Seoul")
        issue = pd.to_datetime(group.get("issue_at", default_issue), utc=True).fillna(default_issue)
        cutoffs = issue.dt.as_unit("ns").astype("int64").to_numpy()
        close = (group.target_date + pd.Timedelta(hours=18, minutes=30)).dt.tz_localize("Asia/Seoul")
        available = pd.to_datetime(group.target_available_at, utc=True)
        release = np.maximum(available.dt.as_unit("ns").astype("int64").to_numpy(),
                             close.dt.as_unit("ns").astype("int64").to_numpy())
        a = group[["low", "median", "high"]].to_numpy(float)
        b = group[["har_low", "har_median", "har_high"]].to_numpy(float)
        valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        # Scores may be computed in advance, but are hidden behind the release heap.
        ml_scores = interval_score(group.y, a[:, 0], a[:, 2])
        har_scores = interval_score(group.y, b[:, 0], b[:, 2])
        ready = valid & np.isfinite(group.y) & available.notna() & group.target_date.notna()
        pending, history = [], deque(maxlen=window)
        weights, counts = np.zeros(len(group)), np.zeros(len(group), int)
        for i in range(len(group)):
            while pending and pending[0][0] <= cutoffs[i]:
                _, _, loss_ml, loss_har = heapq.heappop(pending)
                history.append((loss_ml, loss_har))
            counts[i] = len(history)
            if valid[i] and len(history) >= minimum and b[i, 2] - b[i, 0] < a[i, 2] - a[i, 0]:
                past = np.asarray(history)
                if past[:, 1].mean() < 0.95 * past[:, 0].mean():
                    weights[i] = weight
            if ready.iloc[i]:
                heapq.heappush(pending, (release[i], i, ml_scores[i], har_scores[i]))
        # A missing HAR must not erase a valid original ML interval (0 * NaN is NaN).
        routed = a.copy()
        changed = weights > 0
        routed[changed] = (1 - weights[changed, None]) * a[changed] + weights[changed, None] * b[changed]
        group[["low", "median", "high"]] = routed
        group["har_weight"], group["routing_n"] = weights, counts
        blocks.append(group.drop(columns=["har_low", "har_median", "har_high"]))
    return pd.concat(blocks, ignore_index=True)


def run(settings):
    control = json.loads((settings.data_dir / "reports/calibration-study.json").read_text(encoding="utf8"))
    previous = json.loads((settings.data_dir / "reports/forecast-improvement.json").read_text(encoding="utf8"))
    source, prior_root = Path(control["artifact_dir"]), Path(previous["artifact_dir"])
    output = settings.data_dir / "runs" / f"routing-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    manifest = {"created_at": utcnow(), "candidates": CANDIDATES, "folds": control["folds"],
                "control_artifact": str(source), "previous_artifact": str(prior_root),
                "previous_champion": previous["champion"], "previous_report_hash": digest(previous),
                "source_hash": digest(Path(__file__).read_bytes()),
                "selection": "same gates as learning study; improve both width and score over previous champion",
                "rule": "HAR narrower today and 5% better on last 60/120 settled paired scores; minimum 60",
                "evaluation_role": "reused development; no final test or production promotion",
                "model": "original frozen ML/HAR; signed252/tails252 equal blend",
                "raw_hashes": {p.name: digest(p.read_bytes()) for year in control["folds"] for p in (
                    source / f"raw-original-{year}-ml.parquet", source / f"raw-original-{year}-har.parquet")}}
    atomic_json(output / "manifest.json", manifest)
    shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    print(f"Artifacts: {output}", flush=True)
    originals, priors, predictions = [], [], {key: [] for key in CANDIDATES}
    for year in manifest["folds"]:
        originals.append(pd.read_parquet(source / f"pred-original__current-{year}.parquet"))
        prior = pd.read_parquet(prior_root / f"pred-{previous['champion']}-{year}.parquet")
        priors.append(prior)
        full = {}
        for method in ("ml", "har"):
            raw = pd.read_parquet(source / f"raw-original-{year}-{method}.parquet")
            full[method] = blend(*(replay(raw, CALIBRATIONS[c], f"{year - 1}-01-01")
                                   for c in ("signed252", "tails252")))
        for key, config in CANDIDATES.items():
            routed = route(full["ml"], full["har"], **config)
            routed = routed[routed.target_date.ge(f"{year}-01-01")]
            pred = pd.concat([routed, prior.query("method != 'ml'")], ignore_index=True)
            pred.to_parquet(output / f"pred-{key}-{year}.parquet", index=False)
            predictions[key].append(pred)
            print(json.dumps({"candidate": key, "year": year,
                              "routed_fraction": float(routed.har_weight.gt(0).mean())}), flush=True)
    original, prior = pd.concat(originals, ignore_index=True), pd.concat(priors, ignore_index=True)
    results = [{"candidate": key, **evaluate_candidate(parts, original, prior, manifest["folds"])}
               for key, parts in predictions.items()]
    passing = [r for r in results if r["development_pass"]]
    best = min(passing or results, key=lambda r: r["versus_frozen_har"]["daily_scores"]["ml"])
    report = {**manifest, "results": results, "champion": best["candidate"],
              "development_pass": bool(passing), "deployment_eligible": False,
              "artifact_dir": str(output.resolve()), "completed_at": utcnow()}
    atomic_json(output / "report.json", report)
    atomic_json(settings.data_dir / "reports/routing-study.json", report)
    print(json.dumps({"champion": best["candidate"], "result": best}), flush=True)


if __name__ == "__main__":
    run(Settings.load())
