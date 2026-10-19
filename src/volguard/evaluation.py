import numpy as np
from scipy.stats import norm


def interval_score(y, low, high, alpha=0.1):
    y, low, high = np.asarray(y), np.asarray(low), np.asarray(high)
    return high - low + 2 / alpha * np.maximum(low - y, 0) + 2 / alpha * np.maximum(y - high, 0)


def metrics(frame):
    valid = frame.dropna(subset=["y", "low", "high"])
    if valid.empty:
        return {"n": 0, "coverage": None, "width": None, "score": None}
    return {
        "n": len(valid),
        "coverage": float(((valid.y >= valid.low) & (valid.y <= valid.high)).mean()),
        "width": float((valid.high - valid.low).mean()),
        "score": float(interval_score(valid.y, valid.low, valid.high).mean()),
    }


def block_bootstrap_upper(differences, block=20, repeats=2000, seed=731):
    values = np.asarray(differences, float)
    if len(values) < 2 * block or not np.isfinite(values).all():
        return None
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(len(values) / block))
    means = np.empty(repeats)
    for i in range(repeats):
        starts = rng.integers(0, len(values) - block + 1, blocks)
        sample = np.concatenate([values[start : start + block] for start in starts])[: len(values)]
        means[i] = sample.mean()
    return float(np.quantile(means, 0.975))


def promotion_gate(predictions):
    by_method = {method: frame for method, frame in predictions.groupby("method")}
    result = {
        "metrics": {method: metrics(frame) for method, frame in by_method.items()},
        "har_pass": False,
        "ml_pass": False,
        "selected": "observation_only",
    }
    if not {"har", "empirical"}.issubset(by_method):
        result["reason"] = "missing model outputs"
        return result
    # Compare identical stock-date observations, never independently dropped samples.
    keys = ["symbol", "target_date"]
    parts = {}
    for method in by_method:
        part = by_method[method].dropna(subset=["y", "low", "high"]).copy()
        part[f"score_{method}"] = interval_score(part.y, part.low, part.high)
        part[f"hit_{method}"] = (part.y >= part.low) & (part.y <= part.high)
        part = part[keys + [f"score_{method}", f"hit_{method}"]]
        parts[method] = part
    expected = by_method["empirical"].dropna(subset=["y", "low", "high"])
    baseline_pairs = parts["har"].merge(parts["empirical"], on=keys, validate="one_to_one")
    baseline_daily = baseline_pairs.groupby("target_date")[["score_har", "score_empirical"]].mean()
    baseline_availability = len(baseline_pairs) / len(expected) if len(expected) else 0
    baseline_coverage = float(baseline_pairs.hit_har.mean()) if len(baseline_pairs) else None
    har_pass = bool(
        len(baseline_daily) >= 126
        and baseline_availability >= 0.95
        and baseline_coverage is not None
        and 0.87 <= baseline_coverage <= 0.93
        and baseline_daily.score_har.mean() <= baseline_daily.score_empirical.mean()
    )
    result.update(
        har_pass=har_pass,
        selected="har" if har_pass else "observation_only",
        har_availability=baseline_availability,
        baseline_coverage=baseline_coverage,
    )
    if "ml" not in parts:
        result["reason"] = "ML outputs unavailable; HAR assessed independently"
        return result
    joined = baseline_pairs.merge(parts["ml"], on=keys, validate="one_to_one")
    result["paired_n"] = len(joined)
    result["availability"] = len(joined) / len(expected) if len(expected) else 0
    if joined.empty:
        result["reason"] = "paired prediction availability below 95%"
        return result
    availability_pass = result["availability"] >= 0.95
    if not availability_pass:
        result["reason"] = "paired prediction availability below 95%"
    daily = joined.groupby("target_date")[["score_har", "score_ml", "score_empirical"]].mean()
    if len(daily) < 126:
        result["reason"] = "fewer than 126 out-of-sample trading days"
        return result
    baseline, ml, empirical = (float(daily[c].mean()) for c in daily.columns)
    har_coverage, ml_coverage = float(joined.hit_har.mean()), float(joined.hit_ml.mean())
    upper = block_bootstrap_upper(daily.score_ml - daily.score_har)
    ml_pass = (
        availability_pass
        and 0.87 <= ml_coverage <= 0.93
        and baseline > 0
        and ml <= 0.95 * baseline
        and ml <= empirical
        and upper is not None
        and upper < 0
    )
    result.update(
        har_pass=bool(har_pass),
        ml_pass=bool(ml_pass),
        bootstrap_upper=upper,
        paired_har_coverage=har_coverage,
        paired_ml_coverage=ml_coverage,
        daily_scores={"har": baseline, "ml": ml, "empirical": empirical},
        improvement=1 - ml / baseline if baseline else None,
        selected="ml" if ml_pass else "har" if har_pass else "observation_only",
    )
    return result


def coverage_health(hits):
    values = np.asarray(hits, dtype=bool)
    if len(values) < 60:
        return {"status": "insufficient", "n": len(values), "coverage": None, "ci": None}
    tail = values[-60:]
    p, n, z = float(tail.mean()), len(tail), norm.ppf(0.975)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    # Five consecutive fully populated rolling windows; no intraday repeated counts.
    bad = len(values) >= 64 and all(
        values[-60 - i : len(values) - i if i else None].mean() < 0.8 for i in range(5)
    )
    return {
        "status": "restricted" if bad else "normal",
        "n": n,
        "coverage": p,
        "ci": [float(center - half), float(center + half)],
        "ci_note": "Wilson interval is descriptive; serial dependence is not adjusted",
    }
