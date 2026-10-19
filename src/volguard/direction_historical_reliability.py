"""Matched, frozen-model probability audit. Reused 2024; no tuning or activation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Lake, features
from .direction import extend
from .direction_forward import infer
from .direction_study import enhance
from .event_direction import freeze
from .platform_study import local_settings
from .technical_features import build
from .terminal_data import read_json
from .util import atomic_json, digest, utcnow


def draws(dates, repeats=2000):
    n = len(dates)
    block = min(20, n)
    rng = np.random.default_rng(1720)
    starts = rng.integers(0, n - block + 1, (repeats, int(np.ceil(n / block))))
    return (starts[:, :, None] + np.arange(block)).reshape(repeats, -1)[:, :n]


def bucket(frame, probability, side, lo, hi, dates, indices):
    p = probability if side == "up" else 1 - probability
    y = frame.label.to_numpy() if side == "up" else 1 - frame.label.to_numpy()
    mask = (p >= lo) & ((p < hi) | ((hi == 1) & (p <= hi)))
    f = pd.DataFrame(dict(date=frame.target_date.to_numpy()[mask], p=p[mask], y=y[mask], n=1))
    r = dict(
        side=side,
        low=lo,
        high=hi,
        n=len(f),
        dates=int(f.date.nunique()),
        coverage=float(mask.mean()),
        mean_probability=None,
        actual_frequency=None,
        frequency_ci95=None,
        gap_ci95=None,
        status="insufficient",
    )
    if not len(f):
        return r
    r.update(
        mean_probability=float(f.p.mean()), actual_frequency=float(f.y.mean()), gap=float((f.y - f.p).mean())
    )
    daily = f.groupby("date")[["p", "y", "n"]].sum().reindex(dates, fill_value=0).to_numpy()
    summed = daily[indices].sum(axis=1)
    valid = summed[:, 2] > 0
    freq = summed[valid, 1] / summed[valid, 2]
    gap = (summed[valid, 1] - summed[valid, 0]) / summed[valid, 2]
    ci = np.quantile(gap, [0.025, 0.975]).tolist()
    r.update(frequency_ci95=np.quantile(freq, [0.025, 0.975]).tolist(), gap_ci95=ci)
    if len(f) >= 500 and f.date.nunique() >= 60 and valid.mean() >= 0.95:
        r["status"] = "compatible_retrospective" if ci[0] >= -0.05 and ci[1] <= 0.05 else "not_confirmed"
    return r


def scores(frame, p, dates, indices):
    y = frame.label.to_numpy()
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    bins = [
        bucket(frame, p, side, i / 10, (i + 1) / 10, dates, indices)
        for side in ["up", "down"]
        for i in range(10)
    ]
    return dict(
        n=len(frame),
        accuracy=float(np.mean((p >= 0.5) == y)),
        brier=float(np.mean((p - y) ** 2)),
        log_loss=float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        ece=float(sum(r["coverage"] * abs(r.get("gap", 0)) for r in bins if r["side"] == "up")),
        strong60=float(np.mean(np.maximum(p, 1 - p) >= 0.6)),
        strong70=float(np.mean(np.maximum(p, 1 - p) >= 0.7)),
        bins=bins,
        down60=bucket(frame, p, "down", 0.6, 1, dates, indices),
    )


def compare(frame, a, b, dates, indices):
    y = frame.label.to_numpy()

    def loss(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))

    f = pd.DataFrame(
        dict(date=frame.target_date, brier=(b - y) ** 2 - (a - y) ** 2, log_loss=loss(b) - loss(a), n=1)
    )
    daily = f.groupby("date")[["brier", "log_loss", "n"]].sum().reindex(dates).to_numpy()
    sampled = daily[indices].sum(axis=1)
    return {
        key: dict(
            delta=float(f[key].mean()),
            ci95=np.quantile(sampled[:, i] / sampled[:, 2], [0.025, 0.975]).tolist(),
        )
        for i, key in enumerate(["brier", "log_loss"])
    }


def run():
    settings = local_settings()
    root = settings.data_dir
    out = root / "runs/direction-historical-reliability-20261017"
    out.mkdir(exist_ok=True)
    forward = read_json(root / "config/direction-forward.json")
    specs = [m for m in forward["models"] if m["id"] in {"initial", "technical_expanding"}]
    protected = {
        str(root / "config/direction-forward.json"): digest(
            (root / "config/direction-forward.json").read_bytes()
        )
    }
    for p in (root / "forward").glob("*.json"):
        protected[str(p)] = digest(p.read_bytes())
    protocol = dict(
        evaluation=["2024-01-01", "2024-12-31"],
        independent=False,
        models=[dict(id=m["id"], hash=m["hash"], calibration=m["calibration"]) for m in specs],
        bins="[0,.1),...,[.9,1]; emphasize 50-60,60-70,70-80; also down >=60",
        resampling="same dates across stocks; 20-day blocks, 2000 draws, seed 1720",
        support="500 observations and 60 dates; 95% calibration gap CI entirely within +/-5pp",
        selection="none; use fixed actual prospective models without fitting",
        price_timing="reuse original historical daily availability assumption 18:00; not PIT verified",
        code_hash=digest(Path(__file__).read_bytes()),
        protected=protected,
    )
    freeze(out / "protocol.json", protocol)
    cached = out / "matched-predictions.parquet"
    if cached.exists():
        f = pd.read_parquet(cached)
        audit = read_json(out / "input-audit.json")
        if digest(cached.read_bytes()) != audit["prediction_hash"]:
            raise ValueError("Prediction audit mismatch")
    else:
        bars = pd.read_parquet(
            root / "silver/bars.parquet", filters=[("date", "<", pd.Timestamp("2025-01-01"))]
        )
        raw_hash = digest(pd.util.hash_pandas_object(bars, index=False).values.tobytes())
        bars["available_at"] = (
            (bars.date + pd.Timedelta(hours=18)).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
        )
        sessions = [s for s in Lake(root).calendar() if s["date"] < "2025-01-01"]
        print("causal base features through 2024", flush=True)
        frame = enhance(extend(features(bars, sessions)))
        market = pd.read_parquet(
            root / "silver/market.parquet", filters=[("date", "<", pd.Timestamp("2025-01-01"))]
        )
        frame["market_range"] = frame.date.map(market.drop_duplicates("date").set_index("date").range_pct)
        print("technical features through 2024", flush=True)
        ta = build(frame, bars, "standard")
        frame = frame.merge(ta, on=["symbol", "date"], how="left", validate="one_to_one")
        archived = pd.read_parquet(root / "runs/direction-v1/evaluation.parquet")
        frame = frame.merge(
            archived.rename(columns={"label": "old_label", "probability": "archived_initial"}),
            on=["symbol", "date", "target_date"],
            validate="one_to_one",
        )
        if not frame.label.eq(frame.old_label).all():
            raise ValueError("Historical outcome revision detected")
        cols = sorted(set(c for m in specs for c in m["cols"]))
        f = frame[frame[cols].notna().all(axis=1) & frame.ta_ready.eq(True)].copy()
        if not f.target_date.between("2024-01-01", "2024-12-31").all():
            raise ValueError("Wrong evaluation dates")
        for model in specs:
            f[model["id"]] = infer(f, model)
        max_diff = float(np.max(np.abs(f.initial - f.archived_initial)))
        if max_diff > 1e-9:
            raise ValueError(f"Initial archived inference mismatch: {max_diff}")
        f = f[["symbol", "date", "target_date", "label", "initial", "technical_expanding", "market_range"]]
        f.to_parquet(cached, index=False)
        audit = dict(
            source_bars_hash=raw_hash,
            archived_initial_n=len(archived),
            matched_n=len(f),
            excluded_n=len(archived) - len(f),
            symbols=int(f.symbol.nunique()),
            dates=int(f.target_date.nunique()),
            start=str(f.target_date.min().date()),
            end=str(f.target_date.max().date()),
            initial_max_difference=max_diff,
            prediction_hash=digest(cached.read_bytes()),
        )
        atomic_json(out / "input-audit.json", audit)
    dates = sorted(f.target_date.unique())
    indices = draws(dates)
    result = {m["id"]: scores(f, f[m["id"]].to_numpy(), dates, indices) for m in specs}
    comparison = compare(f, f.initial.to_numpy(), f.technical_expanding.to_numpy(), dates, indices)
    halves = {}
    for name, mask in {
        "first_half": f.target_date.dt.month.le(6),
        "second_half": f.target_date.dt.month.gt(6),
    }.items():
        g = f[mask]
        d = sorted(g.target_date.unique())
        halves[name] = {m["id"]: scores(g, g[m["id"]].to_numpy(), d, draws(d)) for m in specs}
    report = dict(
        status="retrospective_only",
        generated_at=utcnow(),
        protocol=protocol,
        audit=audit,
        models=result,
        comparison=comparison,
        halves=halves,
        conclusion="initial_better"
        if all(x["ci95"][0] > 0 for x in comparison.values())
        else "research_better"
        if all(x["ci95"][1] < 0 for x in comparison.values())
        else "difference_uncertain",
        limitations=[
            "2024년 재사용 회고 비교이며 새로운 독립 검증이 아닙니다.",
            "각 모델의 확률 구간에는 서로 다른 사례가 들어갑니다. 전체 비교는 동일 종목·거래일입니다.",
            "보합 제외 조건부 확률이며 생존편향과 과거 공개·수정 이력 한계가 남습니다.",
            "점별 신뢰구간은 다중 구간 동시 보장이 아니며 운영 승격 근거로 사용하지 않습니다.",
        ],
    )
    assert all(digest(Path(p).read_bytes()) == h for p, h in protected.items())
    atomic_json(out / "report.json", report)
    atomic_json(root / "reports/direction-historical-reliability.json", report)
    print(json.dumps(dict(audit=audit, comparison=comparison, conclusion=report["conclusion"])), flush=True)


if __name__ == "__main__":
    run()
