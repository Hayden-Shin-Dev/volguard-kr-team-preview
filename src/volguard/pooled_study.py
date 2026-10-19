"""Development-only partial pooling of width-normalized conformal residuals."""

import heapq
import json
import shutil
import uuid
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration_study import corrected_quantile, paired_comparison
from .config import Settings
from .evaluation import promotion_gate
from .util import atomic_json, digest, utcnow


def pooled_replay(frame, weight, start_date):
    if not 0 <= weight <= 1:
        raise ValueError("Pooling weight must be in [0,1]")
    work = frame.copy()
    default = (work.date + pd.Timedelta(hours=18, minutes=30)).dt.tz_localize("Asia/Seoul")
    work["cutoff"] = pd.to_datetime(work.get("issue_at", default), utc=True).fillna(default)
    work = work.sort_values(["cutoff", "symbol"]).reset_index(drop=True)
    cutoffs = work.cutoff.dt.as_unit("ns").astype("int64").to_numpy()
    available = pd.to_datetime(work.target_available_at, utc=True).dt.as_unit("ns")
    closed = (work.target_date + pd.Timedelta(hours=18, minutes=30)).dt.tz_localize("Asia/Seoul")
    release = np.maximum(available.astype("int64").to_numpy(), closed.dt.as_unit("ns").astype("int64").to_numpy())
    raw, y, symbols = work[["raw_low", "raw_mid", "raw_high"]].to_numpy(), work.y.to_numpy(), work.symbol.to_numpy()
    scales = np.maximum(0.1, raw[:, 2] - raw[:, 0])
    valid = np.isfinite(raw).all(axis=1)
    ready = valid & np.isfinite(y) & available.notna().to_numpy() & work.target_date.notna().to_numpy()
    scores = np.maximum(raw[:, 0] - y, y - raw[:, 2]) / scales
    local, pooled, pending = defaultdict(lambda: deque(maxlen=252)), deque(maxlen=12600), []
    result, ns, pool_ns, empty = np.full((len(work), 3), np.nan), np.zeros(len(work), int), np.zeros(len(work), int), np.zeros(len(work), bool)
    for cutoff, positions in work.groupby(cutoffs, sort=True).indices.items():
        while pending and pending[0][0] <= cutoff:
            _, index = heapq.heappop(pending)
            local[symbols[index]].append(scores[index])
            pooled.append(scores[index])
        global_q = corrected_quantile(pooled, 0.9) if len(pooled) >= 1000 else None
        for i in positions:
            values = local[symbols[i]]
            ns[i], pool_ns[i] = len(values), len(pooled)
            if valid[i] and len(values) >= 120 and global_q is not None:
                q = (1 - weight) * corrected_quantile(values, 0.9) + weight * global_q
                low, high = max(0, raw[i, 0] - q * scales[i]), raw[i, 2] + q * scales[i]
                if low > high:
                    low = high = max(0, (low + high) / 2)
                    empty[i] = True
                result[i] = (low, min(high, max(low, raw[i, 1])), high)
            if ready[i]:
                heapq.heappush(pending, (release[i], i))
    out = work[["symbol", "date", "target_date", "y", "method", "target_available_at"]].copy()
    out[["low", "median", "high"]] = result
    out["calibration_n"], out["pool_n"], out["empty_expanded"] = ns, pool_ns, empty
    return out[out.target_date.ge(pd.Timestamp(start_date))].reset_index(drop=True)


def run(settings):
    control = json.loads((settings.data_dir / "reports/calibration-study.json").read_text(encoding="utf8"))
    root = Path(control["artifact_dir"])
    output = settings.data_dir / "runs" / f"pooled-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    manifest = {"created_at": utcnow(), "folds": control["folds"], "weights": [0.25, 0.5],
                "models": ["original", "official_normalized"], "stock_window": 252, "pool_window": 12600,
                "stock_minimum": 120, "pool_minimum": 1000, "scale": "raw interval width, floor 0.1",
                "control_artifact": str(root), "control_hash": digest(control),
                "source_hash": digest(Path(__file__).read_bytes()),
                "evaluation_role": "reused development; no coverage theorem for partial pooling",
                "selection": control["selection"]}
    atomic_json(output / "manifest.json", manifest)
    shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    print(f"Artifacts: {output}", flush=True)
    all_predictions, original = {}, []
    for year in manifest["folds"]:
        ref = pd.read_parquet(root / f"pred-original__current-{year}.parquet")
        original.append(ref)
        for model_name in manifest["models"]:
            raw = {m: pd.read_parquet(root / f"raw-{model_name}-{year}-{m}.parquet") for m in ("har", "ml")}
            for weight in manifest["weights"]:
                key = f"{model_name}__pool{int(weight * 100)}"
                predicted = pd.concat([pooled_replay(raw[m], weight, f"{year}-01-01") for m in ("har", "ml")]
                                      + [ref.query("method == 'empirical'")], ignore_index=True)
                predicted.to_parquet(output / f"pred-{key}-{year}.parquet", index=False)
                all_predictions.setdefault(key, []).append(predicted)
                gate = promotion_gate(predicted)
                print(json.dumps({"candidate": key, "year": year, "improvement": gate["improvement"],
                                  "coverage": gate["paired_ml_coverage"]}), flush=True)
    reference = pd.concat(original, ignore_index=True)
    results = []
    for key, predictions in all_predictions.items():
        combined = pd.concat(predictions, ignore_index=True)
        ml = combined.query("method == 'ml'")
        frozen = promotion_gate(pd.concat([ml, reference.query("method != 'ml'")], ignore_index=True))
        same = promotion_gate(combined)
        comparison = paired_comparison(ml, reference.query("method == 'ml'"))
        folds = [{"year": y, **promotion_gate(p)} for y, p in zip(manifest["folds"], predictions)]
        passed = (frozen["ml_pass"] and same["ml_pass"] and 0.89 <= frozen["paired_ml_coverage"] <= 0.91
                  and comparison["width_reduction"] > 0 and comparison["score_improvement"] > 0
                  and all(0.87 <= f["paired_ml_coverage"] <= 0.93 for f in folds))
        results.append({"candidate": key, "versus_frozen_har": frozen, "versus_same_calibration_har": same,
                        "versus_original_ml": comparison, "folds": folds, "development_pass": bool(passed),
                        "empty_expanded": int(ml.empty_expanded.sum())})
    passing = [r for r in results if r["development_pass"]]
    best = min(passing or results, key=lambda r: r["versus_frozen_har"]["daily_scores"]["ml"])
    report = {**manifest, "results": results, "champion": best["candidate"],
              "development_pass": bool(passing), "deployment_eligible": False,
              "artifact_dir": str(output.resolve()), "completed_at": utcnow()}
    atomic_json(output / "report.json", report)
    atomic_json(settings.data_dir / "reports/pooled-study.json", report)
    print(json.dumps({"champion": best["candidate"], "result": best}), flush=True)
    return report


if __name__ == "__main__":
    run(Settings.load())
