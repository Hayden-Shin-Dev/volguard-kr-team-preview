"""Versioned development experiments; never imported by the operating predictor.

Read only 2020/2022/2023 development outcomes. Replay labels at their availability
time, including adaptive updates. Signed correction can produce an empty interval;
we explicitly expand that empty set to its nonnegative midpoint and count it.
No finite-sample or conditional coverage guarantee is claimed for market data.
"""

import heapq
import json
import math
import shutil
import uuid
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Settings
from .data import Lake
from .evaluation import interval_score, metrics, promotion_gate
from .history import retrospective_frame
from .improvement import enrich
from .models import RangeModels
from .util import atomic_json, digest, utcnow

CALIBRATIONS = {
    "current": {"signed": False, "window": 252},
    "signed252": {"signed": True, "window": 252},
    "signed126": {"signed": True, "window": 126},
    "tails252": {"signed": True, "window": 252, "asymmetric": True},
    "tails126": {"signed": True, "window": 126, "asymmetric": True},
    "adaptive252": {"signed": True, "window": 252, "gamma": 0.005},
    "adaptive126": {"signed": True, "window": 126, "gamma": 0.005},
}


def corrected_quantile(values, probability):
    values = np.asarray(values, float)
    rank = math.ceil((len(values) + 1) * probability)
    if not len(values) or not np.isfinite(values).all() or not 1 <= rank <= len(values):
        return None
    return float(np.partition(values, rank - 1)[rank - 1])


def bounds(raw, lower_errors, upper_errors, config, alpha=0.1):
    raw = np.asarray(raw, float)
    if len(lower_errors) < 120 or not np.isfinite(raw).all():
        return None, False
    if config.get("asymmetric"):
        left = corrected_quantile(lower_errors, 1 - alpha / 2)
        right = corrected_quantile(upper_errors, 1 - alpha / 2)
    else:
        left = right = corrected_quantile(np.maximum(lower_errors, upper_errors), 1 - alpha)
    if left is None or right is None:
        return None, False
    if not config.get("signed"):
        left, right = max(0, left), max(0, right)
    low, high = max(0, raw[0] - left), raw[2] + right
    empty = low > high
    if empty:
        low = high = max(0, (low + high) / 2)
    return (low, float(np.clip(raw[1], low, high)), high), empty


def replay_reference(raw_frame, config, start_date):
    records = []
    for symbol, group in raw_frame.groupby("symbol", sort=True):
        lower, upper = deque(maxlen=config["window"]), deque(maxlen=config["window"])
        pending, alpha = [], 0.1
        for counter, row in enumerate(group.sort_values("date").itertuples()):
            cutoff = getattr(row, "issue_at", pd.NaT)
            if pd.isna(cutoff):
                cutoff = (row.date + pd.Timedelta(hours=18, minutes=30)).tz_localize("Asia/Seoul")
            while pending and pending[0][0] <= cutoff:
                _, _, left_error, right_error, miss = heapq.heappop(pending)
                lower.append(left_error)
                upper.append(right_error)
                if miss is not None and config.get("gamma"):
                    alpha = float(np.clip(alpha + config["gamma"] * (0.1 - miss), 0.02, 0.2))
            raw = np.array([row.raw_low, row.raw_mid, row.raw_high])
            scale = max(0.1, raw[2] - raw[0]) if config.get("scaled") else 1.0
            interval, empty = bounds(raw / scale, lower, upper, config, alpha)
            if interval is not None:
                interval = tuple(value * scale for value in interval)
            if pd.notna(row.target_date) and row.target_date >= pd.Timestamp(start_date):
                low, middle, high = interval if interval is not None else (np.nan, np.nan, np.nan)
                records.append({
                    "symbol": symbol, "date": row.date, "target_date": row.target_date,
                    "y": row.y, "low": low, "median": middle, "high": high,
                    "method": row.method, "calibration_n": len(lower), "alpha": alpha,
                    "empty_expanded": empty, "target_available_at": row.target_available_at,
                })
            if np.isfinite(raw).all() and pd.notna(row.y) and pd.notna(row.target_available_at):
                close = (row.target_date + pd.Timedelta(hours=18, minutes=30)).tz_localize("Asia/Seoul")
                release = max(row.target_available_at, close)
                miss = None if interval is None else float(not interval[0] <= row.y <= interval[2])
                heapq.heappush(pending, (
                    release, counter, (raw[0] - row.y) / scale, (row.y - raw[2]) / scale, miss,
                ))
    return pd.DataFrame(records)


def replay(raw_frame, config, start_date):
    """Same event replay with timezone conversion outside the per-row loop."""
    blocks = []
    for symbol, group in raw_frame.groupby("symbol", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        default_cutoff = (group.date + pd.Timedelta(hours=18, minutes=30)).dt.tz_localize("Asia/Seoul")
        cutoffs = pd.to_datetime(group.get("issue_at", default_cutoff), utc=True)
        cutoffs = cutoffs.fillna(default_cutoff).dt.as_unit("ns").astype("int64").to_numpy()
        closed = (group.target_date + pd.Timedelta(hours=18, minutes=30)).dt.tz_localize("Asia/Seoul")
        available = pd.to_datetime(group.target_available_at, utc=True).dt.as_unit("ns")
        releases = np.maximum(available.astype("int64").to_numpy(),
                              closed.dt.as_unit("ns").astype("int64").to_numpy())
        values = group[["raw_low", "raw_mid", "raw_high"]].to_numpy(float)
        ys = group.y.to_numpy(float)
        valid = np.isfinite(values).all(axis=1)
        ready = valid & np.isfinite(ys) & available.notna().to_numpy() & group.target_date.notna().to_numpy()
        scales = np.maximum(0.1, values[:, 2] - values[:, 0]) if config.get("scaled") else np.ones(len(group))
        normalized = values / scales[:, None]
        low_out, mid_out, high_out = (np.full(len(group), np.nan) for _ in range(3))
        ns, alphas, empties = np.zeros(len(group), int), np.zeros(len(group)), np.zeros(len(group), bool)
        window = config["window"]
        errors = np.zeros((window, 2))
        pending, count, alpha = [], 0, 0.1
        gamma = config.get("gamma", 0)
        asymmetric, signed = config.get("asymmetric", False), config.get("signed", False)
        for i in range(len(group)):
            while pending and pending[0][0] <= cutoffs[i]:
                _, _, left_error, right_error, miss = heapq.heappop(pending)
                errors[count % window] = (left_error, right_error)
                count += 1
                if miss is not None and gamma:
                    alpha = min(0.2, max(0.02, alpha + gamma * (0.1 - miss)))
            n = min(count, window)
            ns[i], alphas[i] = n, alpha
            if n >= 120 and valid[i]:
                rank = math.ceil((n + 1) * (1 - alpha / 2 if asymmetric else 1 - alpha))
                if rank <= n:
                    if asymmetric:
                        left = float(np.partition(errors[:n, 0], rank - 1)[rank - 1])
                        right = float(np.partition(errors[:n, 1], rank - 1)[rank - 1])
                    else:
                        scores = np.maximum(errors[:n, 0], errors[:n, 1])
                        left = right = float(np.partition(scores, rank - 1)[rank - 1])
                    if not signed:
                        left, right = max(0, left), max(0, right)
                    low, high = max(0, normalized[i, 0] - left), normalized[i, 2] + right
                    if low > high:
                        empties[i] = True
                        low = high = max(0, (low + high) / 2)
                    middle = min(high, max(low, normalized[i, 1]))
                    low_out[i], mid_out[i], high_out[i] = low * scales[i], middle * scales[i], high * scales[i]
            if ready[i]:
                miss = None if not np.isfinite(low_out[i]) else float(not low_out[i] <= ys[i] <= high_out[i])
                heapq.heappush(pending, (releases[i], i, (values[i, 0] - ys[i]) / scales[i],
                                        (ys[i] - values[i, 2]) / scales[i], miss))
        out = group[["symbol", "date", "target_date", "y"]].copy()
        out["low"], out["median"], out["high"] = low_out, mid_out, high_out
        out["method"], out["calibration_n"], out["alpha"] = group.method, ns, alphas
        out["empty_expanded"], out["target_available_at"] = empties, group.target_available_at
        blocks.append(out[out.target_date.ge(pd.Timestamp(start_date))])
    return pd.concat(blocks, ignore_index=True)


def raw_predictions(model, frame, method):
    out = frame[["symbol", "date", "target_date", "y", "target_available_at"]].copy()
    out[["raw_low", "raw_mid", "raw_high"]] = model.predict(frame, method)
    out["method"] = method
    return out[out.symbol.isin(model.har)].reset_index(drop=True)


def paired_comparison(candidate, reference):
    keys = ["symbol", "target_date"]
    part = candidate.dropna(subset=["y", "low", "high"])
    other = reference.dropna(subset=["y", "low", "high"])
    joined = part.merge(other[keys + ["low", "high"]], on=keys, suffixes=("", "_ref"), validate="one_to_one")
    score = interval_score(joined.y, joined.low, joined.high)
    ref = interval_score(joined.y, joined.low_ref, joined.high_ref)
    daily = pd.DataFrame({"day": joined.target_date, "score": score, "reference": ref}).groupby("day").mean()
    width, ref_width = (joined.high - joined.low).mean(), (joined.high_ref - joined.low_ref).mean()
    return {
        "paired_n": len(joined), "width_reduction": float(1 - width / ref_width),
        "score_improvement": float(1 - daily.score.mean() / daily.reference.mean()),
    }


def run(settings, output=None, raw_cache=None):
    lake = Lake(settings.data_dir)
    output = Path(output) if output else lake.root / "runs" / f"calibration-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((lake.root / "reports" / "improvement.json").read_text(encoding="utf8"))
    official = next(r["config"] for r in report["results"] if r["candidate"] == "official_normalized")
    model_configs = {"original": {}, "official_normalized": official}
    manifest = {
        "created_at": utcnow(), "folds": [2020, 2022, 2023], "model_configs": model_configs,
        "calibrations": CALIBRATIONS, "evaluation_role": "reused development only; not untouched test",
        "selection": "original gates plus pooled 89-91% coverage, narrower and lower score than original ML, and fold coverage 87-93%; no deployment",
        "limits": "survivorship bias; retrieved history not point-in-time archives; adaptive clipping has no asserted theorem",
        "empty_interval_rule": "expand to nonnegative midpoint, count and report",
        "data_hashes": {p.name: digest(p.read_bytes()) for p in (lake.root / "silver").glob("*.parquet")},
        "source_hash": digest(Path(__file__).read_bytes()),
    }
    if raw_cache:
        raw_cache = Path(raw_cache)
        cached = json.loads((raw_cache / "manifest.json").read_text(encoding="utf8"))
        for key in ("folds", "model_configs", "data_hashes"):
            if manifest[key] != cached[key]:
                raise ValueError(f"Raw cache mismatch: {key}")
        for name in ("models.py", "data.py", "history.py", "improvement.py"):
            if (raw_cache / "source" / name).read_bytes() != (Path(__file__).parent / name).read_bytes():
                raise ValueError(f"Raw cache upstream source mismatch: {name}")
        cache_files = list(raw_cache.glob("*.joblib")) + list(raw_cache.glob("raw-*.parquet"))
        manifest["raw_cache"] = {"path": str(raw_cache), "hashes": {
            p.name: digest(p.read_bytes()) for p in cache_files}}
        for path in cache_files:
            shutil.copy2(path, output / path.name)
    if (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text(encoding="utf8"))
        for key in ("folds", "model_configs", "calibrations", "data_hashes", "source_hash"):
            if manifest[key] != previous[key]:
                raise ValueError(f"Resume manifest changed: {key}")
        manifest = previous
    else:
        atomic_json(output / "manifest.json", manifest)
        shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    print(f"Artifacts: {output}", flush=True)
    # Feature generation is causal; outcome filter precedes all fitting and scoring.
    frames = {
        name: enrich(retrospective_frame(lake, verified_price_returns=name != "original")
                     .query("target_date <= '2023-12-31'"))
        for name in model_configs
    }
    # Later live ingestion must not change the feature inputs of follow-up studies.
    snapshot = output / "development-official.parquet"
    frames["official_normalized"].to_parquet(snapshot, index=False)
    atomic_json(output / "development-features.json", {
        "file_hash": digest(snapshot.read_bytes()), "data_hashes": manifest["data_hashes"],
    })
    all_results = {}
    for year in manifest["folds"]:
        baseline = None
        for name, config in model_configs.items():
            model_path = output / f"{name}-{year}.joblib"
            frame = frames[name]
            if model_path.exists():
                model = joblib.load(model_path)
            else:
                print(f"Fit {name} development {year}", flush=True)
                train = frame[frame.target_date.le(f"{year - 2}-12-31")]
                model = RangeModels(config).fit(train, baseline=baseline)
                joblib.dump(model, model_path)
            if name == "original":
                baseline = model
            window = frame[frame.target_date.between(f"{year - 1}-01-01", f"{year}-12-31")]
            raw = {}
            for method in ("har", "ml", "empirical"):
                path = output / f"raw-{name}-{year}-{method}.parquet"
                if path.exists():
                    raw[method] = pd.read_parquet(path)
                else:
                    raw[method] = raw_predictions(model, window, method)
                    raw[method].to_parquet(path, index=False)
            empirical = raw["empirical"].rename(columns={"raw_low": "low", "raw_mid": "median", "raw_high": "high"})
            empirical = empirical[empirical.target_date.ge(f"{year}-01-01")]
            for calibration, calibration_config in CALIBRATIONS.items():
                key = f"{name}__{calibration}"
                path = output / f"pred-{key}-{year}.parquet"
                if path.exists():
                    prediction = pd.read_parquet(path)
                else:
                    prediction = pd.concat([
                        replay(raw[method], {**calibration_config,
                            "scaled": bool(config.get("scaled_conformal")) if method == "ml" else False},
                            f"{year}-01-01") for method in ("har", "ml")
                    ] + [empirical], ignore_index=True)
                    prediction.to_parquet(path, index=False)
                all_results.setdefault(key, []).append(prediction)
                stat = {m: metrics(g) for m, g in prediction.groupby("method")}
                print(json.dumps({"year": year, "candidate": key, "metrics": stat}), flush=True)
        atomic_json(output / "progress.json", {"completed_year": year, "updated_at": utcnow()})
    pooled = {k: pd.concat(v, ignore_index=True) for k, v in all_results.items()}
    original_ml = pooled["original__current"].query("method == 'ml'")
    original_har = pooled["original__current"].query("method == 'har'")
    results = []
    for key, prediction in pooled.items():
        ml = prediction.query("method == 'ml'")
        emp = prediction.query("method == 'empirical'")
        same_rule = promotion_gate(prediction)
        frozen_har = promotion_gate(pd.concat([ml, emp, original_har], ignore_index=True))
        folds = [{"year": year, **promotion_gate(p)} for year, p in zip(manifest["folds"], all_results[key])]
        comparison = paired_comparison(ml, original_ml)
        candidate_valid = (
            frozen_har["ml_pass"] and same_rule["ml_pass"]
            and 0.89 <= frozen_har["paired_ml_coverage"] <= 0.91
            and comparison["width_reduction"] > 0 and comparison["score_improvement"] > 0
            and all(0.87 <= f["paired_ml_coverage"] <= 0.93 for f in folds)
        )
        results.append({"candidate": key, "versus_frozen_har": frozen_har,
                        "versus_same_calibration_har": same_rule, "versus_original_ml": comparison,
                        "folds": folds, "development_pass": bool(candidate_valid),
                        "empty_expanded": int(ml.empty_expanded.eq(True).sum())})
    passing = [r for r in results if r["development_pass"]]
    best = min(passing or results, key=lambda r: r["versus_frozen_har"]["daily_scores"]["ml"])
    result = {**manifest, "results": results, "champion": best["candidate"],
              "development_pass": bool(passing), "deployment_eligible": False,
              "artifact_dir": str(output.resolve()), "completed_at": utcnow()}
    atomic_json(output / "report.json", result)
    atomic_json(lake.root / "reports" / "calibration-study.json", result)
    print(json.dumps({"champion": best["candidate"], "development_pass": bool(passing),
                      "result": best}, default=str), flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    parser.add_argument("--raw-cache")
    args = parser.parse_args()
    run(Settings.load(), args.resume, args.raw_cache)
