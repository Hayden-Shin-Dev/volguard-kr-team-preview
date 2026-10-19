"""Two-sided prospective reliability; never mix retrospective examples into counts."""

import numpy as np
import pandas as pd

from .terminal_data import read_json
from .util import atomic_json, digest, utcnow


def summarize(rows, side, low, high, all_dates, total, protocol):
    key = "p_up" if side == "up" else "p_down"
    chosen = [r for r in rows if low <= r[key] and (r[key] < high or high == 1)]
    scored = [r for r in chosen if r.get("direction") in {"up", "down"}]
    result = dict(
        side=side,
        low=low,
        high=high,
        issued=len(chosen),
        n=len(scored),
        flat=sum(r.get("direction") == "flat" for r in chosen),
        pending=sum(r.get("direction") is None for r in chosen),
        coverage=len(chosen) / total if total else 0,
        dates=len({r["target_date"] for r in scored}),
        status="insufficient",
        mean_probability=None,
        observed_frequency=None,
        gap_ci95=None,
    )
    if not scored:
        return result
    p = np.array([r[key] for r in scored])
    y = np.array([r["direction"] == side for r in scored], dtype=float)
    result.update(
        mean_probability=float(p.mean()),
        observed_frequency=float(y.mean()),
        brier=float(((y - p) ** 2).mean()),
        log_loss=float(-(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))).mean()),
    )
    if len(scored) < protocol["min_rows"] or result["dates"] < protocol["min_dates"]:
        return result
    f = pd.DataFrame(dict(date=[r["target_date"] for r in scored], y=y, p=p, n=1))
    daily = f.groupby("date")[["y", "p", "n"]].sum().reindex(all_dates, fill_value=0).to_numpy()
    block = min(protocol["bootstrap_block"], len(daily))
    rng = np.random.default_rng(1718)
    starts = rng.integers(
        0, len(daily) - block + 1, (protocol["bootstrap_draws"], int(np.ceil(len(daily) / block)))
    )
    indices = (starts[:, :, None] + np.arange(block)).reshape(len(starts), -1)[:, : len(daily)]
    sums = daily[indices].sum(axis=1)
    valid = sums[:, 2] > 0
    if valid.mean() < 0.95:
        return result
    gaps = (sums[valid, 0] - sums[valid, 1]) / sums[valid, 2]
    ci = np.quantile(gaps, [0.025, 0.975]).tolist()
    tol = protocol["calibration_tolerance"]
    result.update(
        gap_ci95=ci,
        rate_ci95=np.quantile(sums[valid, 0] / sums[valid, 2], [0.025, 0.975]).tolist(),
        status="compatible_research" if ci[0] >= -tol and ci[1] <= tol else "not_confirmed",
    )
    return result


def report(settings):
    root = settings.data_dir / "direction-forward"
    protocol = read_json(settings.data_dir / "config/direction-forward.json", {})
    if not protocol:
        return dict(status="not_configured", models=[])
    rows = []
    for path in sorted(root.glob("????-??-??/*.json")):
        row = read_json(path)
        result = read_json(root / "outcomes" / row["target_date"] / path.name, {})
        if result and result.get("prediction_hash") != digest(path.read_bytes()):
            result = {}  # tampered/mismatched outcomes must not count as observations
        rows.append(dict(**row, direction=result.get("direction")))
    models = []
    edges = protocol["probability_edges"]
    for model in protocol["models"]:
        group = [r for r in rows if r["model_version"] == model["version"]]
        dates = sorted({r["target_date"] for r in group if r["direction"] in {"up", "down"}})
        bins = [
            summarize(group, side, lo, hi, dates, len(group), protocol)
            for side in ["up", "down"]
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        strong = [summarize(group, side, 0.6, 1.0, dates, len(group), protocol) for side in ["up", "down"]]
        models.append(
            dict(
                id=model["id"],
                name=model["name"],
                version=model["version"],
                fitted_history=model["fitted_history"],
                issued=len(group),
                scored=sum(r["direction"] in {"up", "down"} for r in group),
                flat=sum(r["direction"] == "flat" for r in group),
                pending=sum(r["direction"] is None for r in group),
                bins=bins,
                strong=strong,
            )
        )
    # Compare only the exact same prospective symbol/target pairs; no model selection here.
    maps = {
        m["id"]: {
            (r["symbol"], r["target_date"]): r
            for r in rows
            if r["model_version"] == m["version"] and r["direction"] in {"up", "down"}
        }
        for m in protocol["models"]
    }
    common = set.intersection(*(set(v) for v in maps.values())) if maps else set()
    comparison = []
    for key, values in maps.items():
        if common:
            p = np.array([values[k]["p_up"] for k in sorted(common)])
            y = np.array([values[k]["direction"] == "up" for k in sorted(common)])
            comparison.append(
                dict(
                    model=key,
                    n=len(common),
                    brier=float(np.mean((p - y) ** 2)),
                    log_loss=float(
                        -np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
                    ),
                )
            )
    result = dict(
        status="research_only",
        updated_at=utcnow(),
        contract=protocol["contract"],
        models=models,
        comparison=comparison,
        batch=read_json(root / "batch-status.json", {}),
        scoring=read_json(root / "scoring-status.json", {}),
        latest=[
            dict(
                symbol=r["symbol"],
                target_date=r["target_date"],
                issued_at=r["issued_at"],
                model=r["details"]["model"],
                p_up=r["p_up"],
                p_down=r["p_down"],
                direction=r["direction"],
            )
            for r in sorted(rows, key=lambda r: r["issued_at"])[-30:]
        ],
        qualification="500개 비보합 표본·60개 거래일 이상; 날짜 블록 95% 오차 구간이 ±5%p 안에 있어야 연구상 정합. 여러 구간 반복 검토 및 국면 변화에 대한 독립 검증은 별도.",
    )
    atomic_json(settings.data_dir / "reports/direction-forward.json", result)
    return result
