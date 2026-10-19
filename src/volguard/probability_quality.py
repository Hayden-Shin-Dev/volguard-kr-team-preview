"""Probability-first audit and prospective monitoring; no model/forecast mutation."""

from pathlib import Path

import numpy as np
import pandas as pd

from .direction_historical_reliability import draws
from .event_direction import freeze
from .platform_study import local_settings
from .terminal_data import read_json
from .util import atomic_json, digest, utcnow

BANDS = [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.8)]


def losses(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log1p(-p))


def decomposition(y, p, weights):
    """Fixed 5pp grouping; raw Brier != grouped decomposition in general."""
    w = weights / weights.sum()
    base = float(w @ y)
    resolution = reliability = ece = grouped_brier = 0.0
    ids = np.minimum((np.asarray(p) * 20).astype(int), 19)
    for k in np.unique(ids):
        mask = ids == k
        mass = w[mask].sum()
        actual = float(w[mask] @ y[mask] / mass)
        mean = float(w[mask] @ p[mask] / mass)
        resolution += mass * (actual - base) ** 2
        reliability += mass * (mean - actual) ** 2
        ece += mass * abs(mean - actual)
        grouped_brier += float(w[mask] @ ((y[mask] - mean) ** 2))
    raw = float(w @ ((p - y) ** 2))
    return dict(
        resolution=float(resolution),
        reliability=float(reliability),
        ece=float(ece),
        uncertainty=base * (1 - base),
        grouped_brier=grouped_brier,
        within_bin_remainder=raw - grouped_brier,
        probability_std=float(np.sqrt(w @ ((p - float(w @ p)) ** 2))),
    )


def band(frame, p, side, lo, hi, indices, dates):
    q = p if side == "up" else 1 - p
    y = frame.label.to_numpy(float) if side == "up" else 1 - frame.label.to_numpy(float)
    keep = (q >= lo) & ((q < hi) | ((hi == 1) & (q == 1)))
    g = pd.DataFrame(dict(date=frame.target_date.to_numpy()[keep], p=q[keep], y=y[keep], n=1))
    r = dict(
        side=side,
        low=lo,
        high=hi,
        n=len(g),
        dates=int(g.date.nunique()),
        coverage=float(keep.mean()),
        mean_probability=None,
        actual_frequency=None,
        frequency_ci95=None,
        gap_ci95=None,
        date_balanced_frequency=None,
        effective_dates=0.0,
        max_date_share=0.0,
        status="insufficient",
    )
    if g.empty:
        return r
    daily = g.groupby("date")[["p", "y", "n"]].sum().reindex(dates, fill_value=0)
    count = daily.n.to_numpy()
    eff = float(count.sum() ** 2 / (count @ count))
    sampled = daily.to_numpy()[indices].sum(axis=1)
    valid = sampled[:, 2] > 0
    gap = (sampled[valid, 1] - sampled[valid, 0]) / sampled[valid, 2]
    ci = np.quantile(gap, [0.025, 0.975]).tolist()
    daily_mean = daily[["p", "y"]].div(daily.n.replace(0, np.nan), axis=0)
    r.update(
        mean_probability=float(g.p.mean()),
        actual_frequency=float(g.y.mean()),
        frequency_ci95=np.quantile(sampled[valid, 1] / sampled[valid, 2], [0.025, 0.975]).tolist(),
        gap_ci95=ci,
        date_balanced_frequency=float(daily_mean.y.mean()),
        date_balanced_probability=float(daily_mean.p.mean()),
        effective_dates=eff,
        max_date_share=float(count.max() / count.sum()),
    )
    if len(dates) < 40 or r["dates"] < 20:
        r["frequency_ci95"] = None
        r["gap_ci95"] = None
    if len(g) >= 500 and r["dates"] >= 60:
        if eff < 30 or r["max_date_share"] > 0.1 or valid.mean() < 0.95:
            r["status"] = "date_concentration"
        elif ci[0] >= -0.05 and ci[1] <= 0.05:
            r["status"] = "compatible"
        else:
            r["status"] = "not_calibrated"
    return r


def evaluate(frame, p, prior):
    if len(frame) == 0:
        return dict(n=0, status="awaiting_outcomes", bins=[], reliability=[], qualified=False)
    p = np.asarray(p, dtype=float)
    if len(p) != len(frame) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Only scored non-flat outcomes may enter probability metrics")
    y = frame.label.to_numpy(float)
    dates = sorted(frame.target_date.unique())
    idx = draws(dates)
    b, ll = losses(y, p)
    pb, pl = losses(y, np.full(len(p), prior))
    daily = (
        pd.DataFrame(dict(date=frame.target_date, brier=b, ll=ll, prior_brier=pb, prior_ll=pl, n=1))
        .groupby("date")
        .sum()
    )
    totals = daily.to_numpy()[idx].sum(axis=1)
    means = daily.div(daily.n, axis=0).to_numpy()[idx].mean(axis=1)
    weight = 1 / frame.target_date.map(frame.target_date.value_counts()).to_numpy()

    def overall(weights, samples):
        w = weights / weights.sum()
        br, prior_br = float(w @ b), float(w @ pb)
        logloss, prior_ll = float(w @ ll), float(w @ pl)
        return dict(
            brier=br,
            log_loss=logloss,
            prior_brier=prior_br,
            prior_log_loss=prior_ll,
            brier_skill=1 - br / prior_br if prior_br > 0 else None,
            brier_skill_ci95=np.quantile(
                1 - samples[:, 0] / np.maximum(samples[:, 2], 1e-12), [0.025, 0.975]
            ).tolist(),
            log_loss_improvement=prior_ll - logloss,
            log_loss_improvement_ci95=np.quantile(
                (samples[:, 3] - samples[:, 1]) / samples[:, 4], [0.025, 0.975]
            ).tolist(),
            **decomposition(y, p, w),
        )

    row = overall(np.ones(len(p)), totals)
    balanced = overall(weight, means)
    bins = [band(frame, p, side, lo, hi, idx, dates) for side in ["up", "down"] for lo, hi in BANDS]
    reliability = [band(frame, p, "up", i / 20, (i + 1) / 20, idx, dates) for i in range(20)]
    supported = [r for r in bins if r["low"] >= 0.6 and r["status"] == "compatible" and r["coverage"] >= 0.01]
    skill = all(
        r["brier_skill_ci95"][0] > 0 and r["log_loss_improvement_ci95"][0] > 0 for r in [row, balanced]
    )
    resolution_pass = bool(
        balanced["resolution"] > 0
        and balanced["probability_std"] >= 0.02
        and np.quantile(p, 0.9) - np.quantile(p, 0.1) >= 0.05
    )
    return dict(
        n=len(frame),
        dates=len(dates),
        status="measured",
        row_weighted=row,
        date_balanced=balanced,
        bins=bins,
        reliability=reliability,
        high60_coverage=float(np.mean(np.maximum(p, 1 - p) >= 0.6)),
        high70_coverage=float(np.mean(np.maximum(p, 1 - p) >= 0.7)),
        supported_strong_bands=[dict(side=r["side"], low=r["low"], high=r["high"]) for r in supported],
        skill_pass=skill,
        resolution_pass=resolution_pass,
        qualified=bool(skill and supported and resolution_pass),
        note="Resolution is descriptive, grouping-dependent and not sufficient for qualification. Distinct dates are not guaranteed independent.",
    )


def install(settings):
    root = settings.data_dir
    path = root / "config/probability-quality-v2.json"
    if path.exists():
        saved = read_json(path)["payload"]
        if saved["code_hash"] != digest(Path(__file__).read_bytes()):
            raise ValueError("Frozen probability evaluation changed; create a new version")
        return saved
    f = pd.read_parquet(root / "runs/direction-calibration-study-20261017/features.parquet")
    train = f[f.target_date.lt("2022-01-01")]
    prior = float(train.label.mean())
    payload = dict(
        version="probability-quality-v2",
        revision_reason="Explicit nonconstant resolution gate and inclusive probability-1 bin; v1 retained, no model selection",
        at=utcnow(),
        prior_up=prior,
        prior_n=len(train),
        prior_cutoff="2021-12-31",
        prior_source_hash=digest(
            (root / "runs/direction-calibration-study-20261017/features.parquet").read_bytes()
        ),
        bands=BANDS,
        ece_bins="fixed 5 percentage point bins over [0,1]",
        bootstrap="20-date moving blocks; 2000 replicates; all same-date stocks together",
        gates=dict(
            min_rows=500,
            min_dates=60,
            min_effective_dates=30,
            max_date_share=0.1,
            gap_tolerance=0.05,
            min_strong_band_coverage=0.01,
            resolution="positive date-balanced grouped resolution; probability std >=2pp; p90-p10 >=5pp",
            skill="BSS and LogLoss-improvement lower95 >0 under both row and equal-date weighting",
        ),
        evaluation="Existing predictions only, no re-fit/re-selection on 2023-2024",
        availability=dict(
            unseen_historical_period="not_established",
            used_direction=["2023", "2024"],
            used_other_target="2025-01-01 to 2026-09-30 range audit",
            first_prospective_target="2026-10-20; only actual timely-issued forecasts",
        ),
        conditional_range=dict(
            contract="same target day, cutoff, stock and model versions; separate up/down conditional calibration required",
            enabled=False,
            reason="Current nested7_tail50 interval is unconditional; cannot relabel as directional conditional interval",
        ),
        auto_promote=False,
        code_hash=digest(Path(__file__).read_bytes()),
    )
    freeze(path, payload)
    return payload


def historical(settings):
    root = settings.data_dir
    policy = install(settings)
    out = root / "runs/probability-quality-v2-20261017"
    out.mkdir(exist_ok=True)
    freeze(out / "protocol.json", policy)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    old = root / "runs/direction-calibration-study-20261017"
    methods = {}
    for method in ["raw", "sigmoid", "beta", "isotonic", "market_sigmoid"]:
        f = pd.read_parquet(old / f"development-initial_refit__{method}.parquet")
        methods[method] = evaluate(f, f.p_up.to_numpy(), policy["prior_up"])
        print("AUDIT existing 2023, no selection:", method, flush=True)
    results = {}
    for name in ["initial_refit__sigmoid", "technical_recent__market_sigmoid"]:
        f = pd.read_parquet(old / f"final-{name}.parquet")
        results[name] = evaluate(f, f.p_up.to_numpy(), policy["prior_up"])
    result = dict(
        at=utcnow(),
        status="retrospective_audit_only",
        independent=False,
        policy=policy,
        calibration_comparison=methods,
        results=results,
        selection="none; no model chosen with reused periods",
        production_changed=False,
    )
    atomic_json(out / "report.json", result)
    atomic_json(root / "reports/probability-quality-history.json", result)
    print("Historical probability audit complete", flush=True)
    return result


def prospective(settings):
    policy = install(settings)
    root = settings.data_dir
    registry = read_json(root / "config/direction-forward.json", {})
    valid = {m["version"]: m for m in registry.get("models", [])}
    groups = {k: [] for k in valid}
    issued = {k: 0 for k in valid}
    flat = {k: 0 for k in valid}
    for path in (root / "direction-forward").glob("????-??-??/*.json"):
        r = read_json(path)
        version = r.get("model_version")
        if version not in valid:
            continue
        issued[version] += 1
        outcome = read_json(root / "direction-forward/outcomes" / r["target_date"] / path.name, {})
        if outcome.get("prediction_hash") != digest(path.read_bytes()):
            continue
        if outcome.get("direction") == "flat":
            flat[version] += 1
        if outcome.get("direction") not in ["up", "down"]:
            continue
        groups[version].append(
            dict(
                symbol=r["symbol"],
                target_date=pd.Timestamp(r["target_date"]),
                label=int(outcome["direction"] == "up"),
                p_up=r["p_up"],
            )
        )
    models = []
    for version, rows in groups.items():
        f = pd.DataFrame(rows)
        metrics = evaluate(f, f.p_up.to_numpy() if len(f) else [], policy["prior_up"])
        models.append(
            dict(
                id=valid[version]["id"],
                name=valid[version]["name"],
                version=version,
                issued=issued[version],
                flat=flat[version],
                pending=issued[version] - flat[version] - len(rows),
                metrics=metrics,
            )
        )
    result = dict(
        at=utcnow(),
        status="prospective_research",
        policy=policy,
        models=models,
        research_only=True,
        auto_promote=False,
    )
    atomic_json(root / "reports/probability-quality-forward.json", result)
    return result


def conditional_contract(direction, interval):
    if not direction.get("model_version") or not interval.get("model_version"):
        return dict(enabled=False, reason="model_version_missing")
    for key in ["symbol", "target_date", "feature_cutoff"]:
        if direction.get(key) is None or direction.get(key) != interval.get(key):
            return dict(enabled=False, reason=f"{key}_mismatch")
    if not direction.get("probability_qualified"):
        return dict(enabled=False, reason="direction_probability_unverified")
    if interval.get("scope") != "direction_conditioned" or not interval.get("conditional_coverage_verified"):
        return dict(enabled=False, reason="unconditional_range_cannot_be_relabelled")
    return dict(
        enabled=True,
        direction_version=direction.get("model_version"),
        range_version=interval.get("model_version"),
    )


if __name__ == "__main__":
    historical(local_settings())
    prospective(local_settings())
