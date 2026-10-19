"""Paired TA-Lib ablations, chronological probability calibration, no live promotion."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import talib

from .direction_integration import calibrate
from .direction_math_features import BASE
from .direction_math_study import config, measure, paired_ci, run_raw, safe_json
from .event_direction import freeze, losses
from .new_information_sources import output
from .platform_study import local_settings
from .technical_features import GROUPS, PROFILES, SIGNALS, build
from .util import atomic_json, digest, utcnow


def signal_report(frame, training, draws=2000):
    dates = np.sort(frame.target_date.unique())
    rng = np.random.default_rng(171018)
    block = min(20, len(dates))
    starts = rng.integers(0, len(dates) - block + 1, (draws, int(np.ceil(len(dates) / block))))
    ids = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, : len(dates)]
    result = {}
    for name, (description, expected) in SIGNALS.items():
        fit = training[training[name].abs().gt(0)]
        p = float((fit.label.sum() + 100 * training.label.mean()) / (len(fit) + 100))
        g = frame[frame[name].abs().gt(0)]
        row = dict(
            description=description,
            expected="up" if expected else "down",
            p_up=p,
            p_down=1 - p,
            n=len(g),
            dates=int(g.target_date.nunique()),
            coverage=len(g) / len(frame),
            day_coverage=g.target_date.nunique() / len(dates),
            fit_n=len(fit),
            supported=False,
        )
        result[name] = row
        if g.empty:
            continue
        y, q = g.label.to_numpy(), g.baseline_p.to_numpy()
        b, ll = losses(y, p)
        qb, qll = losses(y, q)
        interpretation = y if expected else 1 - y
        hit = ((p > 0.5) == y).astype(float)
        qhit = ((q > 0.5) == y).astype(float)
        values = pd.DataFrame(
            dict(
                date=g.target_date.to_numpy(),
                n=1.0,
                up=y,
                interpretation=interpretation,
                hit=hit,
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
        # Bonferroni across both profiles and all signals and three improvement claims.
        a = 0.05 / (len(PROFILES) * len(SIGNALS) * 3)
        simultaneous = dict(zip(keys, np.quantile(samples, [a / 2, 1 - a / 2], axis=0).T.tolist()))
        row.update(
            up_frequency=float(y.mean()),
            down_frequency=float(1 - y.mean()),
            interpretation_accuracy=float(interpretation.mean()),
            probability_accuracy=float(hit.mean()),
            baseline_accuracy=float(qhit.mean()),
            brier=float(b.mean()),
            baseline_brier=float(qb.mean()),
            log_loss=float(ll.mean()),
            baseline_log_loss=float(qll.mean()),
            calibration_gap=float((y - p).mean()),
            ci95=ci,
            familywise_ci=simultaneous,
            nonempty_fraction=float(valid.mean()),
        )
        row["supported"] = bool(
            len(g) >= 300
            and g.target_date.nunique() >= 60
            and valid.mean() >= 0.95
            and simultaneous["accuracy_delta"][0] > 0
            and simultaneous["brier_delta"][1] < 0
            and simultaneous["log_loss_delta"][1] < 0
            and ci["gap"][0] >= -0.05
            and ci["gap"][1] <= 0.05
        )
    return result


def run():
    root = local_settings().data_dir
    out = root / "runs/technical-direction-20261017"
    out.mkdir(exist_ok=True)
    paths = dict(
        context=root / "runs/direction-math-20261017/context.parquet",
        bars=root / "silver/bars.parquet",
        match=output() / "global-features.parquet",
    )
    protocol = dict(
        created_version="technical-v1",
        sources={k: digest(v.read_bytes()) for k, v in paths.items()},
        code={
            p: digest(Path(p).read_bytes())
            for p in ["src/volguard/technical_features.py", "src/volguard/technical_study.py"]
        },
        talib=talib.__version__,
        profiles=PROFILES,
        groups=GROUPS,
        signals={k: list(v) for k, v in SIGNALS.items()},
        periods="2020 OOF warmup; 2021-22 development selection; 2023 reused diagnostic, NOT independent",
        training="same 5y rolling quarterly logistic C=.1, half-life730d; raw and prior1y OOF sigmoid",
        selection="development accuracy>=baseline, Brier and Log Loss strictly lower; choose lowest Brier else baseline",
        signal_selection="development familywise accuracy/Brier/logloss better, >=300 rows/60 days, gap CI within +/-5pp",
        data_policy="raw KRX price/volume; break at invalid or unverifiable continuity; common >=201 contiguous observations; no back adjustment",
        operational_enabled=False,
    )
    freeze(out / "protocol.json", protocol)
    context = pd.read_parquet(paths["context"])
    bars = pd.read_parquet(paths["bars"], filters=[("date", "<", pd.Timestamp("2024-01-01"))])
    match = pd.read_parquet(paths["match"])
    if match.target_date.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("Consumed locked period cannot be reopened")
    features = {}
    for profile in PROFILES:
        dest = out / f"features-{profile}.parquet"
        if dest.exists():
            f = pd.read_parquet(dest)
        else:
            f = build(context, bars, profile)
            f.to_parquet(dest, index=False)
        features[profile] = f
    cols = sum(GROUPS.values(), [])
    keys = ["symbol", "date"]
    common = features["standard"].loc[lambda x: x.ta_ready & x[cols].notna().all(axis=1), keys]
    common = common.merge(
        features["strict"].loc[lambda x: x.ta_ready & x[cols].notna().all(axis=1), keys],
        on=keys,
        validate="one_to_one",
    )
    data = {
        p: match.merge(common, on=keys, validate="one_to_one").merge(
            f.drop(columns="ta_ready"), on=keys, validate="one_to_one"
        )
        for p, f in features.items()
    }
    development = data["standard"][data["standard"].target_date.lt("2023-01-01")]
    corr = development[BASE + cols].corr().loc[cols, BASE]
    overlap = {
        c: dict(
            existing=corr.loc[c].abs().idxmax(), correlation=float(corr.loc[c, corr.loc[c].abs().idxmax()])
        )
        for c in cols
        if corr.loc[c].notna().any()
    }
    audit = dict(
        input_rows=len(match),
        common_rows=len(data["standard"]),
        input2023=int(match.target_date.dt.year.eq(2023).sum()),
        common2023=int(data["standard"].target_date.dt.year.eq(2023).sum()),
        overlap=overlap,
        features={p: digest((out / f"features-{p}.parquet").read_bytes()) for p in PROFILES},
        duplicates="existing rsi14 is simple return-gain ratio 0..1, not Wilder price RSI; replace it in RSI/full ablations; existing volume includes current day, new ratio excludes it",
        limitations="Current-universe survival bias; historical availability reconstructed; adjustment not fully verified; 201-observation common warmup changes comparison sample",
    )
    atomic_json(out / "audit.json", safe_json(audit))
    models = {"baseline": ("standard", BASE)}
    for group, extra in GROUPS.items():
        models[group] = ("standard", [c for c in BASE if group != "rsi" or c != "rsi14"] + extra)
    for p in PROFILES:
        models["all_" + p] = (p, [c for c in BASE if c != "rsi14"] + cols)
    predictions = {}
    for name, (profile, columns) in models.items():
        raw = run_raw(data[profile], config(columns), out, name, [2020, 2021, 2022, 2023])
        for method in ["raw", "sigmoid"]:
            predictions[name + "__" + method] = (
                calibrate(raw, method).sort_values(["target_date", "symbol"]).reset_index(drop=True)
            )
    baseline = predictions["baseline__raw"]
    dev_results = {}
    for name, f in predictions.items():
        pd.testing.assert_frame_equal(
            f[keys + ["target_date", "label"]], baseline[keys + ["target_date", "label"]]
        )
        dev_results[name] = measure(f[f.target_date.dt.year.lt(2023)])
    b = dev_results["baseline__raw"]
    valid = [
        n
        for n, r in dev_results.items()
        if r["accuracy"] >= b["accuracy"] and r["brier"] < b["brier"] and r["log_loss"] < b["log_loss"]
    ]
    selected = min(valid, key=lambda n: dev_results[n]["brier"]) if valid else "baseline__raw"
    signal_development = {}
    signal_data = {}
    for p in PROFILES:
        f = baseline.merge(data[p][keys + cols], on=keys, validate="one_to_one")
        f["baseline_p"] = f.p_up
        signal_data[p] = f
        signal_development[p] = signal_report(
            f[f.target_date.dt.year.lt(2023)], data[p][data[p].target_date.lt("2021-01-01")]
        )
    choices = {}
    for signal in SIGNALS:
        supported = [p for p in PROFILES if signal_development[p][signal]["supported"]]
        choices[signal] = (
            min(supported, key=lambda p: signal_development[p][signal]["brier"]) if supported else None
        )
    freeze(out / "selection.json", safe_json(dict(model=selected, signals=choices, development=dev_results)))
    confirmation, comparisons = {}, {}
    reference = baseline[baseline.target_date.dt.year.eq(2023)]
    for name, f in predictions.items():
        g = f[f.target_date.dt.year.eq(2023)]
        confirmation[name] = measure(g)
        comparisons[name] = paired_ci(g, reference)
    signal_results = {
        p: signal_report(f[f.target_date.dt.year.eq(2023)], data[p][data[p].target_date.lt("2023-01-01")])
        for p, f in signal_data.items()
    }
    report = safe_json(
        dict(
            status="complete_research_only",
            at=utcnow(),
            selected=selected,
            selected_signals=choices,
            development=dev_results,
            results=confirmation,
            comparisons=comparisons,
            signals=signal_results,
            signal_development=signal_development,
            audit=audit,
            protocol=protocol,
            operational_enabled=False,
            independent_evaluation="unavailable; 2023 reused",
            limitations=[
                "2023년은 재사용 진단이며 신규 독립 검증이 아닙니다.",
                "신호 해석의 정답률과 학습된 확률의 정답률을 구분합니다. 보합 제외입니다.",
                "현재 고정 구성 종목의 생존편향, 당시 발표·조정 이력의 미입증이 있습니다.",
                "기업행동·결측 경계에서 재시작하고 201행을 확보한 공통 표본만 비교했습니다.",
                "다이버전스는 과거 20일 극값만 사용한 정의입니다. 미래 봉으로 확정하는 피벗은 쓰지 않습니다.",
                "매매 수익률이나 거래비용 반영 성과가 아닌 다음 거래일 종가 방향 평가입니다.",
                "기존 nested7_tail50과 운영 확률 제한을 유지합니다.",
            ],
        )
    )
    pd.concat([f.assign(candidate=n) for n, f in predictions.items()], ignore_index=True).to_parquet(
        out / "predictions.parquet", index=False
    )
    atomic_json(out / "report.json", report)
    atomic_json(root / "reports/technical-direction.json", report)
    print(
        json.dumps(
            dict(selected=selected, selected_signals=choices, result=confirmation[selected]),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
