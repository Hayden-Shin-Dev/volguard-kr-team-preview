"""Learning-history ablation independent of range OOF availability."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .direction_integration import calibrate
from .direction_math_features import BASE
from .direction_math_study import config, measure, paired_ci, safe_json
from .event_direction import freeze
from .new_information_sources import output
from .platform_study import local_settings
from .selective_direction import side_evidence
from .technical_features import GROUPS
from .util import atomic_json, digest, utcnow

STRATEGIES = {
    "old2015_weighted5": dict(years=5, half_life=730, scope="pooled", start="2015-01-01"),
    "expanding": dict(years=None, half_life=None, scope="pooled", start=None),
    "rolling3": dict(years=3, half_life=None, scope="pooled", start=None),
    "rolling5": dict(years=5, half_life=None, scope="pooled", start=None),
    "rolling10": dict(years=10, half_life=None, scope="pooled", start=None),
    "weighted_all": dict(years=None, half_life=730, scope="pooled", start=None),
    "regime_all": dict(years=None, half_life=730, scope="regime", start=None),
}


def eligible_history(context, technical):
    cols = sum(GROUPS.values(), [])
    joined = context.merge(technical.drop(columns="ta_ready"), on=["symbol", "date"], validate="one_to_one")
    keep = (
        joined[BASE + cols].notna().all(axis=1)
        & joined.direction_return.notna()
        & joined.direction_return.ne(0)
    )
    keep &= joined.target_date.lt("2024-01-01")
    # No range OOF presence condition: those columns are outputs for scoring only.
    return joined.loc[keep].copy()


def run():
    root = local_settings().data_dir
    out = root / "runs/technical-history-20261017"
    out.mkdir(exist_ok=True)
    source = root / "runs/direction-math-20261017/context.parquet"
    ta = root / "runs/technical-direction-20261017/features-standard.parquet"
    strict = root / "runs/technical-direction-20261017/features-strict.parquet"
    match = output() / "global-features.parquet"
    protocol = dict(
        source_hash=digest(source.read_bytes()),
        technical_hash=digest(ta.read_bytes()),
        strict_hash=digest(strict.read_bytes()),
        match_hash=digest(match.read_bytes()),
        code_hash=digest(Path(__file__).read_bytes()),
        strategies=STRATEGIES,
        features="original BASE; BASE without simple RSI + all standard TA groups",
        periods="2020 OOF calibration warmup; 2021-22 selection; 2023 reused diagnostic",
        selection="development pooled Brier/LL improve with accuracy not worse; Brier not worse in EACH 2021,2022; minimum Brier otherwise reference",
        threshold="60/65/70% fixed; threshold per side selected in development only by date-block calibration support",
        regime="existing annual causal filtered Markov volatility > .5; per-regime training, pooled fallback if <500 rows",
        operational_enabled=False,
    )
    freeze(out / "protocol.json", protocol)
    context = pd.read_parquet(source)
    technical = pd.read_parquet(ta)
    cols = sum(GROUPS.values(), [])
    technical = technical[technical.ta_ready & technical[cols].notna().all(axis=1)]
    st = pd.read_parquet(strict)
    common = st.loc[st.ta_ready & st[cols].notna().all(axis=1), ["symbol", "date"]]
    technical = technical.merge(common, on=["symbol", "date"], validate="one_to_one")
    data = eligible_history(context, technical)
    expected = pd.read_parquet(match)[["symbol", "date"]].merge(
        common, on=["symbol", "date"], validate="one_to_one"
    )
    # Restrict evaluation, not training: using run_raw needs an explicit evaluation flag.
    eval_keys = set(map(tuple, expected[["symbol", "date"]].to_numpy()))
    data["evaluation_eligible"] = [
        (s, d) in eval_keys for s, d in data[["symbol", "date"]].itertuples(index=False, name=None)
    ]
    audit = dict(
        context_start=str(context.date.min().date()),
        direction_only_start=str(
            context.loc[context[BASE].notna().all(axis=1) & context.direction_return.notna(), "date"]
            .min()
            .date()
        ),
        technical_start=str(data.date.min().date()),
        rows=len(data),
        pre2015=int(data.date.lt("2015-01-01").sum()),
        years={str(k): int(v) for k, v in data.groupby(data.date.dt.year).size().items()},
        reference_restriction="2015 was range OOF experiment boundary; not intrinsic to direction target",
        missing_range_does_not_exclude=int(data.range_forecast_low.isna().sum()),
    )
    atomic_json(out / "audit.json", audit)
    predictions = {}
    # Custom evaluation runner reuses the vetted fitting routine with only evaluation rows marked;
    # run_raw takes all rows for training, so temporarily map non-evaluation targets outside tested years.
    # Instead use a local wrapper below to keep every training target timestamp untouched.
    for features, columns in {"base": BASE, "technical": [c for c in BASE if c != "rsi14"] + cols}.items():
        for strategy, options in STRATEGIES.items():
            c = config(columns)
            c.update({k: v for k, v in options.items() if k != "start"})
            train_data = data[data.date.ge(options["start"])] if options["start"] else data
            name = f"{features}__{strategy}"
            raw = run_history(train_data, c, out, name)
            for method in ["raw", "sigmoid"]:
                predictions[name + "__" + method] = (
                    calibrate(raw, method).sort_values(["target_date", "symbol"]).reset_index(drop=True)
                )
    reference_name = "base__old2015_weighted5__raw"
    reference = predictions[reference_name]
    development = {}
    years = {}
    for n, f in predictions.items():
        pd.testing.assert_frame_equal(
            f[["symbol", "target_date", "label"]], reference[["symbol", "target_date", "label"]]
        )
        development[n] = measure(f[f.target_date.dt.year.lt(2023)])
        years[n] = {str(y): measure(f[f.target_date.dt.year.eq(y)]) for y in [2021, 2022]}
    b = development[reference_name]
    admissible = [
        n
        for n, r in development.items()
        if r["accuracy"] >= b["accuracy"]
        and r["brier"] < b["brier"]
        and r["log_loss"] < b["log_loss"]
        and all(years[n][str(y)]["brier"] <= years[reference_name][str(y)]["brier"] for y in [2021, 2022])
    ]
    selected = min(admissible, key=lambda n: development[n]["brier"]) if admissible else reference_name
    selected_dev = predictions[selected].query("target_date < '2023-01-01'").copy()
    selected_dev["baseline_p"] = reference.loc[reference.target_date.lt("2023-01-01"), "p_up"].to_numpy()
    signal_dev = [side_evidence(selected_dev, s, t) for s in ["up", "down"] for t in [0.6, 0.65, 0.7]]
    thresholds = {
        s: next((v["threshold"] for v in signal_dev if v["side"] == s and v["supported"]), None)
        for s in ["up", "down"]
    }
    freeze(
        out / "selection.json",
        safe_json(dict(selected=selected, thresholds=thresholds, development=development, years=years)),
    )
    results = {}
    ref = reference[reference.target_date.dt.year.eq(2023)]
    for n, f in predictions.items():
        f = f[f.target_date.dt.year.eq(2023)].copy()
        f["baseline_p"] = ref.p_up.to_numpy()
        results[n] = dict(
            **measure(f),
            comparison=paired_ci(f, ref),
            strong=[side_evidence(f, s, t) for s in ["up", "down"] for t in [0.6, 0.65, 0.7]],
        )
    report = safe_json(
        dict(
            status="complete_research_only",
            at=utcnow(),
            selected=selected,
            thresholds=thresholds,
            results=results,
            development=development,
            yearly_development=years,
            signal_development=signal_dev,
            protocol=protocol,
            audit=audit,
            operational_enabled=False,
            limitations=[
                "2023년 재사용 진단이며 새 독립 시험이 아니다.",
                "변동폭 OOF 누락은 방향 학습 제외 사유가 아니다.",
                "전체 이력 전략은 지표 준비·가격 연속성 검증을 통과한 가장 오래된 자료부터 사용한다.",
                "현재 구성 종목의 생존편향과 당시 공개·수정 이력의 한계가 있다.",
                "개발 두 해의 확률 오차 안정성을 요구하며 2023년 성적으로 전략을 다시 고르지 않는다.",
            ],
        )
    )
    pd.concat([f.assign(candidate=n) for n, f in predictions.items()], ignore_index=True).to_parquet(
        out / "predictions.parquet", index=False
    )
    atomic_json(out / "report.json", report)
    atomic_json(root / "reports/technical-history.json", report)
    print(
        json.dumps(
            dict(
                selected=selected,
                audit=audit,
                metrics={k: v for k, v in results[selected].items() if k != "strong"},
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_history(data, c, out, name):
    """Training on all eligible history; predict only frozen common evaluation keys."""
    import joblib
    from threadpoolctl import threadpool_limits

    from .direction_math_study import KEYS, factory, training_rows

    parts = []
    evaluation = data[data.evaluation_eligible & data.target_date.dt.year.between(2020, 2023)]
    for period, test in evaluation.groupby(evaluation.target_date.dt.to_period("Q"), sort=True):
        dest = out / f"raw-{name}-{period}.parquet"
        if dest.exists():
            metadata = json.loads(dest.with_suffix(".json").read_text("utf8"))
            if metadata["hash"] != digest(dest.read_bytes()) or metadata["config"] != c:
                raise ValueError("History checkpoint mismatch")
            parts.append(pd.read_parquet(dest))
            continue
        train = training_rows(data, test, c)

        def fit(tr, te):
            model = factory(c)
            weights = (
                np.ones(len(tr))
                if not c["half_life"]
                else np.exp2(-(test.date.min() - tr.target_date).dt.days / c["half_life"])
            )
            model.fit(
                tr[c["cols"]],
                tr.label,
                **{list(model.named_steps)[-1] + "__sample_weight": np.asarray(weights)},
            )
            return model, model.predict_proba(te[c["cols"]])[:, 1]

        with threadpool_limits(limits=3):
            model, p = fit(train, test)
            if c["scope"] == "regime":
                for state in [False, True]:
                    tr = train[train.markov_high.fillna(0.5).gt(0.5).eq(state)]
                    mask = test.markov_high.fillna(0.5).gt(0.5).eq(state)
                    if len(tr) >= 500 and tr.label.nunique() == 2 and mask.any():
                        _, local = fit(tr, test[mask])
                        p[np.flatnonzero(mask)] = local
        f = test[KEYS].copy()
        f["raw_p"], f["prior"], f["period"], f["fit_end"] = (
            np.clip(p, 1e-6, 1 - 1e-6),
            float(train.label.mean()),
            str(period),
            train.target_date.max(),
        )
        f.to_parquet(dest, index=False)
        atomic_json(
            dest.with_suffix(".json"),
            dict(
                hash=digest(dest.read_bytes()),
                config=c,
                train_n=len(train),
                train_start=str(train.date.min()),
                fit_end=str(train.target_date.max()),
            ),
        )
        if str(period) == "2023Q4":
            joblib.dump(dict(model=model, research_only=True), out / f"pooled-{name}.joblib")
        parts.append(f)
        print(name, period, len(train), len(test), flush=True)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    run()
