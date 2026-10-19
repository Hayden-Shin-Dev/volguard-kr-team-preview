"""Small predeclared condition grid versus nonlinear learners; no hindsight regimes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .direction_integration import calibrate
from .direction_math_features import BASE
from .direction_math_study import config, measure, paired_ci, run_raw, safe_json
from .event_direction import freeze, losses
from .new_information_sources import output
from .platform_study import local_settings
from .technical_features import GROUPS
from .util import atomic_json, digest, utcnow

EVENTS = {
    "rsi_exit_low": 1,
    "macd_cross_up": 1,
    "macd_cross_down": 0,
    "bb_lower_reentry": 1,
    "bb_upper_reentry": 0,
    "golden_cross": 1,
    "dead_cross": 0,
    "volume_breakout": 1,
    "volume_breakdown": 0,
}
CONTEXTS = [
    "any",
    "market_up",
    "market_down",
    "high_vol",
    "low_vol",
    "volume_surge",
    "trend_aligned",
    "other_indicator",
]


def conditions(f):
    masks = {}
    for event, direction in EVENTS.items():
        contexts = dict(
            any=pd.Series(True, index=f.index),
            market_up=f.market_momentum.gt(0),
            market_down=f.market_momentum.le(0),
            high_vol=f.markov_high.ge(0.7),
            low_vol=f.markov_high.le(0.3),
            volume_surge=f.prior_volume_ratio.ge(2),
            trend_aligned=f.adx.ge(25)
            & (f.plus_di.gt(f.minus_di) if direction else f.plus_di.lt(f.minus_di)),
            other_indicator=(f.stoch_cross_up.eq(1) if direction else f.stoch_cross_down.eq(1)),
        )
        for condition, mask in contexts.items():
            masks[event + "__" + condition] = f[event].eq(1) & mask
    return masks


def evidence(frame, train, draws=4000):
    dates = np.sort(frame.target_date.unique())
    rng = np.random.default_rng(171019)
    block = min(20, len(dates))
    starts = rng.integers(0, len(dates) - block + 1, (draws, int(np.ceil(len(dates) / block))))
    ids = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, : len(dates)]
    training = conditions(train)
    result = {}
    for key, mask in conditions(frame).items():
        g = frame[mask]
        fit = train[training[key]]
        p = float((fit.label.sum() + 100 * train.label.mean()) / (len(fit) + 100))
        direction = EVENTS[key.split("__")[0]]
        row = dict(
            n=len(g),
            dates=int(g.target_date.nunique()),
            coverage=len(g) / len(frame),
            day_coverage=g.target_date.nunique() / len(dates),
            p_up=p,
            p_down=1 - p,
            fit_n=len(fit),
            supported=False,
        )
        result[key] = row
        if g.empty:
            continue
        y, q = g.label.to_numpy(), g.baseline_p.to_numpy()
        b, ll = losses(y, p)
        qb, qll = losses(y, q)
        hit = ((p > 0.5) == y).astype(float)
        qhit = ((q > 0.5) == y).astype(float)
        inside = g.y.between(g.range_forecast_low, g.range_forecast_high).to_numpy()
        values = pd.DataFrame(
            dict(
                date=g.target_date.to_numpy(),
                n=1.0,
                up=y,
                interpretation=y if direction else 1 - y,
                accuracy=hit,
                gap=y - p,
                accuracy_delta=hit - qhit,
                brier_delta=b - qb,
                log_loss_delta=ll - qll,
            )
        )
        daily = values.groupby("date").sum().reindex(dates, fill_value=0).to_numpy()
        sums = daily[ids].sum(axis=1)
        valid = sums[:, 0] > 0
        samples = sums[valid, 1:] / sums[valid, 0, None]
        keys = list(values.columns)[2:]
        ci = dict(zip(keys, np.quantile(samples, [0.025, 0.975], axis=0).T.tolist()))
        a = 0.05 / (len(EVENTS) * len(CONTEXTS) * 3)
        simultaneous = dict(zip(keys, np.quantile(samples, [a / 2, 1 - a / 2], axis=0).T.tolist()))
        row.update(
            up_frequency=float(y.mean()),
            down_frequency=float(1 - y.mean()),
            interpretation_accuracy=float((y if direction else 1 - y).mean()),
            accuracy=float(hit.mean()),
            baseline_accuracy=float(qhit.mean()),
            brier=float(b.mean()),
            baseline_brier=float(qb.mean()),
            log_loss=float(ll.mean()),
            baseline_log_loss=float(qll.mean()),
            calibration_gap=float((y - p).mean()),
            ci95=ci,
            familywise_ci=simultaneous,
            joint_accuracy=float((hit.astype(bool) & inside).mean()),
            baseline_joint_accuracy=float((qhit.astype(bool) & inside).mean()),
            nonempty_fraction=float(valid.mean()),
            yearly={
                str(year): dict(
                    n=len(part),
                    dates=int(part.target_date.nunique()),
                    up=float(part.label.mean()),
                    brier_delta=float(((p - part.label) ** 2 - (part.baseline_p - part.label) ** 2).mean()),
                )
                for year, part in g.groupby(g.target_date.dt.year)
            },
        )
        stable = all(
            v["n"] >= 100 and v["dates"] >= 20 and v["brier_delta"] < 0 for v in row["yearly"].values()
        )
        row["supported"] = bool(
            len(g) >= 300
            and g.target_date.nunique() >= 60
            and valid.mean() >= 0.95
            and stable
            and simultaneous["accuracy_delta"][0] > 0
            and simultaneous["brier_delta"][1] < 0
            and simultaneous["log_loss_delta"][1] < 0
            and ci["gap"][0] >= -0.05
            and ci["gap"][1] <= 0.05
        )
    return result


def run():
    root = local_settings().data_dir
    out = root / "runs/technical-conditions-20261017"
    out.mkdir(exist_ok=True)
    previous = root / "runs/technical-direction-20261017"
    paths = [
        previous / "features-standard.parquet",
        previous / "features-strict.parquet",
        output() / "global-features.parquet",
    ]
    protocol = dict(
        sources={str(p): digest(p.read_bytes()) for p in paths},
        code_hash=digest(Path(__file__).read_bytes()),
        events=EVENTS,
        contexts=CONTEXTS,
        profiles="standard only; 72 predeclared event/context combinations, no recursive conjunctions",
        unavailable="Historical sector membership and sector index returns unverified; excluded, market proxy not relabeled sector",
        fitting="5y rolling, quarterly, half-life730; LightGBM7leaves/CatBoost4depth fixed, base versus all TA",
        selection="2021-22; accuracy no worse, Brier/LL improve, Brier improvement in EACH year; min Brier else logistic baseline",
        confirmation="2023 REUSED not independent; never promote automatically",
    )
    freeze(out / "protocol.json", protocol)
    standard, strict, match = [pd.read_parquet(p) for p in paths]
    cols = sum(GROUPS.values(), [])
    common = strict.loc[strict.ta_ready & strict[cols].notna().all(axis=1), ["symbol", "date"]]
    standard = standard[standard.ta_ready & standard[cols].notna().all(axis=1)]
    data = match.merge(common, on=["symbol", "date"], validate="one_to_one").merge(
        standard.drop(columns="ta_ready"), on=["symbol", "date"], validate="one_to_one"
    )
    baseline = pd.concat(
        [pd.read_parquet(p) for p in sorted(previous.glob("raw-baseline-*.parquet"))], ignore_index=True
    )
    if len(baseline.period.unique()) != 16:
        raise ValueError("Baseline training not finished")
    predictions = {
        "logistic_baseline": calibrate(baseline, "raw")
        .sort_values(["target_date", "symbol"])
        .reset_index(drop=True)
    }
    for kind in ["lightgbm", "catboost"]:
        for group, columns in [("base", BASE), ("technical", [c for c in BASE if c != "rsi14"] + cols)]:
            name = kind + "_" + group
            c = config(columns)
            c["kind"] = kind
            raw = run_raw(data, c, out, name, [2020, 2021, 2022, 2023])
            for method in ["raw", "sigmoid"]:
                predictions[name + "__" + method] = (
                    calibrate(raw, method).sort_values(["target_date", "symbol"]).reset_index(drop=True)
                )
    base = predictions["logistic_baseline"]
    development = {}
    for n, f in predictions.items():
        pd.testing.assert_frame_equal(
            f[["symbol", "target_date", "label"]], base[["symbol", "target_date", "label"]]
        )
        development[n] = measure(f[f.target_date.dt.year.lt(2023)])
    b = development["logistic_baseline"]
    candidates = [
        n
        for n, r in development.items()
        if r["accuracy"] >= b["accuracy"]
        and r["brier"] < b["brier"]
        and r["log_loss"] < b["log_loss"]
        and all(
            measure(predictions[n][predictions[n].target_date.dt.year.eq(y)])["brier"]
            <= measure(base[base.target_date.dt.year.eq(y)])["brier"]
            for y in [2021, 2022]
        )
    ]
    selected = min(candidates, key=lambda n: development[n]["brier"]) if candidates else "logistic_baseline"
    events = base.merge(
        data[["symbol", "date", "market_momentum", "markov_high"] + cols],
        on=["symbol", "date", "market_momentum", "markov_high"],
        validate="one_to_one",
    )
    events["baseline_p"] = events.p_up
    dev = evidence(events[events.target_date.dt.year.lt(2023)], data[data.target_date.lt("2021-01-01")])
    selected_conditions = [k for k, v in dev.items() if v["supported"]]
    freeze(
        out / "selection.json",
        safe_json(dict(selected=selected, conditions=selected_conditions, development=development)),
    )
    diagnostic = evidence(
        events[events.target_date.dt.year.eq(2023)], data[data.target_date.lt("2023-01-01")]
    )
    ref = base[base.target_date.dt.year.eq(2023)]
    results = {
        n: dict(
            **measure(f[f.target_date.dt.year.eq(2023)]),
            comparison=paired_ci(f[f.target_date.dt.year.eq(2023)], ref),
        )
        for n, f in predictions.items()
    }
    nonlinear_ablation = {}
    for kind in ["lightgbm", "catboost"]:
        for method in ["raw", "sigmoid"]:
            a = predictions[kind + "_technical__" + method]
            bframe = predictions[kind + "_base__" + method]
            nonlinear_ablation[kind + "__" + method] = paired_ci(
                a[a.target_date.dt.year.eq(2023)], bframe[bframe.target_date.dt.year.eq(2023)]
            )
    report = safe_json(
        dict(
            status="complete_research_only",
            at=utcnow(),
            selected=selected,
            selected_conditions=selected_conditions,
            development=development,
            results=results,
            nonlinear_ablation=nonlinear_ablation,
            condition_development=dev,
            conditions=diagnostic,
            protocol=protocol,
            operational_enabled=False,
            limitations=[
                "72개 사전 정의 조합이다. 두 조건 이상을 재귀적으로 조합해 고적중률을 찾지 않았다.",
                "업종 자료의 과거 정합성이 미확인이라 업종 상대수익률은 제외했다.",
                "2023년은 재사용 확인이며 미래 독립 검증이 아니다.",
                "조건별 경험확률과 투자자의 관행적 해석 방향의 적중률은 다르다.",
                "복수 비교 보정은 근사 날짜 블록 구간이다. 반복 연구의 모든 선택 편향을 제거하지 못한다.",
                "nested7_tail50의 기존 OOF 구간으로 동시 적중을 계산하며 운영 모델은 변경하지 않는다.",
            ],
        )
    )
    pd.concat([f.assign(candidate=n) for n, f in predictions.items()], ignore_index=True).to_parquet(
        out / "predictions.parquet", index=False
    )
    atomic_json(out / "report.json", report)
    atomic_json(root / "reports/technical-conditions.json", report)
    print(
        json.dumps(
            dict(selected=selected, conditions=selected_conditions, metrics=results[selected]),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
