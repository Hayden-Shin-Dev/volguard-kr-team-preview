"""Market/relative-return joint distribution experiment; never changes releases."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .direction import COLS
from .direction_calibration_study import TECH, calibrated, fit_calibration
from .direction_historical_reliability import compare, draws, scores
from .event_direction import freeze
from .platform_study import local_settings
from .util import atomic_json, digest, utcnow

MKT = ["m_return", "m_momentum5", "m_momentum20", "m_vol", "m_range", "m_gap", "m_volume"]


def index_data(root, out):
    rows = []
    for p in (root / "bronze/krx-index-history").glob("*.raw"):
        for x in json.loads(p.read_bytes()).get("OutBlock_1", []):
            if x.get("IDX_NM") == "코스피 200" and x["BAS_DD"] < "20250101":
                rows.append(
                    dict(
                        date=pd.Timestamp(x["BAS_DD"]),
                        close=float(x["CLSPRC_IDX"]),
                        change=float(x["CMPPREVDD_IDX"]),
                        high=float(x["HGPRC_IDX"]),
                        low=float(x["LWPRC_IDX"]),
                        open=float(x["OPNPRC_IDX"]),
                        volume=float(x["ACC_TRDVOL"]),
                        source_hash=p.stem,
                    )
                )
    m = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    m["m_return"] = m.close / (m.close - m.change) - 1
    m["m_momentum5"] = m.close.pct_change(5)
    m["m_momentum20"] = m.close.pct_change(20)
    m["m_vol"] = m.m_return.rolling(20).std().clip(lower=0.003)
    m["m_range"] = m.high / m.low - 1
    m["m_gap"] = m.open / (m.close - m.change) - 1
    m["m_volume"] = m.volume / m.volume.rolling(20).mean()
    m["target_date"] = m.date.shift(-1)
    m["market_y"] = m.m_return.shift(-1)
    m.to_parquet(out / "kospi200.parquet", index=False)
    return m


def lift_interval(frame, p, low=0.6, high=0.7):
    """Paired date-block comparison with same-date breadth and index direction."""
    f = frame.assign(p=p, selected=(p >= low) & (p < high)).copy()
    dates = sorted(f.target_date.unique())
    g = f[f.selected]
    if g.empty:
        return dict(n=0, dates=0)
    a = pd.DataFrame(
        dict(
            date=g.target_date, n=1, breadth_lift=g.label - g.breadth, market_lift=g.label - g.market_y.gt(0)
        )
    )
    daily = (
        a.groupby("date")[["n", "breadth_lift", "market_lift"]].sum().reindex(dates, fill_value=0).to_numpy()
    )
    s = daily[draws(dates)].sum(axis=1)
    s = s[s[:, 0] > 0]
    return dict(
        n=len(g),
        dates=int(g.target_date.nunique()),
        coverage=float(len(g) / len(f)),
        predicted=float(g.p.mean()),
        actual=float(g.label.mean()),
        same_date_breadth=float(g.breadth.mean()),
        same_date_market_up=float(g.market_y.gt(0).mean()),
        breadth_lift=float(a.breadth_lift.mean()),
        breadth_lift_ci95=np.quantile(s[:, 1] / s[:, 0], [0.025, 0.975]).tolist(),
        market_lift=float(a.market_lift.mean()),
        market_lift_ci95=np.quantile(s[:, 2] / s[:, 0], [0.025, 0.975]).tolist(),
    )


def diagnostic(root, out, m):
    old = root / "runs/direction-calibration-study-20261017"
    p = pd.read_parquet(old / "final-technical_recent__market_sigmoid.parquet")
    b = pd.read_parquet(old / "final-initial_refit__sigmoid.parquet")
    p["baseline"] = b.p_up.to_numpy()
    p["breadth"] = p.target_date.map(p.groupby("target_date").label.mean())
    p = p.merge(m[["target_date", "market_y", "m_return"]], on="target_date", validate="many_to_one")
    p["signal"] = p.p_up.ge(0.6) & p.p_up.lt(0.7)
    rows = []
    for date, g in p.groupby("target_date"):
        s = g[g.signal]
        if s.empty:
            continue
        rows.append(
            dict(
                target_date=str(date.date()),
                signal_n=len(s),
                all_n=len(g),
                kospi200_return=float(g.market_y.iloc[0]),
                issue_day_kospi200_return=float(g.m_return.iloc[0]),
                all_up_frequency=float(g.label.mean()),
                signal_up_frequency=float(s.label.mean()),
                successes=int(s.label.sum()),
                failures=int(len(s) - s.label.sum()),
                mean_prediction=float(s.p_up.mean()),
                prediction_min=float(s.p_up.min()),
                prediction_max=float(s.p_up.max()),
                all_prediction_std=float(g.p_up.std()),
                baseline_same_signals_accuracy=float(((s.baseline >= 0.5) == s.label).mean()),
                baseline_all_accuracy=float(((g.baseline >= 0.5) == g.label).mean()),
                baseline_same_signals_mean=float(s.baseline.mean()),
            )
        )
    pd.DataFrame(rows).to_csv(out / "date-diagnostics.csv", index=False, encoding="utf-8-sig")
    p[p.target_date.isin(p.loc[p.signal, "target_date"])].to_parquet(
        out / "symbol-diagnostics.parquet", index=False
    )
    full = lift_interval(p, p.p_up.to_numpy())
    largest = max(rows, key=lambda x: x["signal_n"])["target_date"]
    leave = p[p.target_date.ne(pd.Timestamp(largest))]
    result = dict(
        dates=rows,
        pooled=full,
        largest_date=largest,
        without_largest=lift_interval(leave, leave.p_up.to_numpy()),
        equal_date_signal_up=float(np.mean([x["signal_up_frequency"] for x in rows])),
        universe="same 193 current-membership eligible stocks; not historical KOSPI200 membership breadth",
    )
    atomic_json(out / "concentration.json", result)
    print("DIAGNOSTIC", json.dumps({k: v for k, v in result.items() if k != "dates"}), flush=True)
    return result


def joint_probability(mu_m, mu_s, scale_m, scale_s, paired_errors):
    """P(market + excess > 0) from paired, dependent return innovations."""
    result = np.empty(len(mu_m))
    for i in range(0, len(mu_m), 256):
        j = slice(i, i + 256)
        total = (
            (mu_m[j] + mu_s[j])[:, None]
            + scale_m[j, None] * paired_errors[None, :, 0]
            + scale_s[j, None] * paired_errors[None, :, 1]
        )
        result[j] = (np.sum(total > 0, axis=1) + 0.5) / (len(paired_errors) + 1)
    return result


def metrics(f, p):
    ds = sorted(f.target_date.unique())
    r = scores(f, p, ds, draws(ds))
    r["roc_auc"] = float(roc_auc_score(f.label, p))
    r["up60_70"] = lift_interval(f, p)
    r["quarters"] = {
        str(q): lift_interval(g, p[f.index.get_indexer(g.index)])
        for q, g in f.groupby(f.target_date.dt.to_period("Q"))
    }
    return r


def conditions(f, volatility_cut):
    return {
        "all": np.ones(len(f), dtype=bool),
        "market_uptrend": f.m_momentum20.ge(0).to_numpy(),
        "market_downtrend": f.m_momentum20.lt(0).to_numpy(),
        "nonshock": f.m_return.abs().lt(0.02).to_numpy(),
        "high_vol": f.m_vol.ge(volatility_cut).to_numpy(),
    }


def condition_results(f, p, volatility_cut):
    result = {}
    for name, mask in conditions(f, volatility_cut).items():
        g, q = f.loc[mask], p[mask]
        s = metrics(g, q)
        band = next(b for b in s["bins"] if b["side"] == "up" and b["low"] == 0.6)
        quarters = sum(v["n"] > 0 for v in s["quarters"].values())
        result[name] = dict(
            band=band,
            lift=s["up60_70"],
            quarters=quarters,
            supported=band["status"] == "compatible_retrospective"
            and quarters >= 3
            and s["up60_70"].get("breadth_lift_ci95", [-1])[0] > 0,
        )
    return result


def run():
    root = local_settings().data_dir
    out = root / "runs/direction-joint-20261017"
    out.mkdir(exist_ok=True)
    protected = {str(p): digest(p.read_bytes()) for p in (root / "config").glob("*.json")}
    freeze(
        out / "protocol.json",
        dict(
            train="through 2020",
            innovation_distribution="2021 held-out paired residuals",
            calibration="2022",
            selection="2023",
            evaluation="2024 REUSED retrospective only",
            variants=["ridge", "lightgbm"],
            calibration_method="sigmoid C1000",
            residuals="4096 paired normalized market/excess residuals; dates sampled uniformly, then stocks within date",
            model_selection="lowest 2023 Brier then Log Loss; never choose using 2024",
            signal_selection="fixed up60-70 bins; development whole gap CI within +-5pp, >=500 rows and60dates, >=3quarters, same-date breadth lift CI>0",
            conditions=["all", "market_uptrend", "market_downtrend", "nonshock", "high_vol"],
            protected=protected,
            code_hash=digest(Path(__file__).read_bytes()),
        ),
    )
    m = index_data(root, out)
    diag = diagnostic(root, out, m)
    f = pd.read_parquet(root / "runs/direction-calibration-study-20261017/features.parquet")
    bars = pd.read_parquet(root / "silver/bars.parquet", filters=[("date", "<", pd.Timestamp("2025-01-01"))])
    price = bars[["symbol", "date", "close", "CMPPREVDD_PRC"]].copy()
    price["CMPPREVDD_PRC"] = pd.to_numeric(price.CMPPREVDD_PRC)
    f = f.merge(price[["symbol", "date", "close"]], on=["symbol", "date"], validate="one_to_one")
    f = f.merge(
        price.rename(columns={"date": "target_date", "close": "next_close"}),
        on=["symbol", "target_date"],
        validate="one_to_one",
    )
    f = f[(f.next_close - f.CMPPREVDD_PRC - f.close).abs().le(0.5)].copy()
    f["stock_y"] = f.next_close / f.close - 1
    assert f.stock_y.gt(0).astype(int).eq(f.label).all()
    f = (
        f.merge(
            m[["date", "target_date", "market_y"] + MKT], on=["date", "target_date"], validate="many_to_one"
        )
        .dropna(subset=MKT + ["market_y"])
        .reset_index(drop=True)
    )
    f["excess_y"] = f.stock_y - f.market_y
    f["s_scale"] = f.vol_22.clip(lower=0.005)
    f["breadth"] = f.target_date.map(f.groupby("target_date").label.mean())
    f.to_parquet(out / "features.parquet", index=False)
    market = m.dropna(subset=MKT + ["market_y"])
    train = f[f.target_date.lt("2021-01-01")]
    innovation = f[f.target_date.dt.year.eq(2021)]
    cal = f[f.target_date.dt.year.eq(2022)]
    dev = f[f.target_date.dt.year.eq(2023)]
    final = f[f.target_date.dt.year.eq(2024)]
    audit = {
        k: dict(
            n=len(g),
            dates=int(g.target_date.nunique()),
            start=str(g.target_date.min().date()),
            end=str(g.target_date.max().date()),
        )
        for k, g in dict(
            train=train, innovations=innovation, calibration=cal, selection=dev, final=final
        ).items()
    }
    atomic_json(out / "split-audit.json", audit)
    objects = {}
    development = {}
    with threadpool_limits(limits=3):
        baseline = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=500, random_state=17)
        ).fit(train[COLS], train.label)
        bp = baseline.predict_proba(cal[COLS])[:, 1]
        bc = fit_calibration(cal, bp, "sigmoid")
        objects["initial_refit"] = dict(model=baseline, calibrator=bc)
        development["initial_refit"] = metrics(
            dev, calibrated(bc, dev, baseline.predict_proba(dev[COLS])[:, 1], "sigmoid")
        )
        for variant in ["ridge", "lightgbm"]:

            def reg():
                if variant == "ridge":
                    return make_pipeline(StandardScaler(), Ridge(alpha=100))
                return LGBMRegressor(
                    n_estimators=100,
                    num_leaves=7,
                    max_depth=3,
                    min_child_samples=100,
                    learning_rate=0.03,
                    reg_lambda=20,
                    verbosity=-1,
                    n_jobs=3,
                    random_state=1730,
                )

            mt = market[market.target_date.lt("2021-01-01")]
            mm = reg().fit(mt[MKT], mt.market_y)
            sm = reg().fit(train[TECH + MKT], train.excess_y)
            e = innovation[["target_date"]].copy()
            e["em"] = (innovation.market_y - mm.predict(innovation[MKT])) / innovation.m_vol
            e["es"] = (innovation.excess_y - sm.predict(innovation[TECH + MKT])) / innovation.s_scale
            rng = np.random.default_rng(1730)
            groups = [g[["em", "es"]].to_numpy() for _, g in e.groupby("target_date")]
            pairs = np.array(
                [groups[j][rng.integers(len(groups[j]))] for j in rng.integers(len(groups), size=4096)]
            )

            def raw(g):
                return joint_probability(
                    mm.predict(g[MKT]),
                    sm.predict(g[TECH + MKT]),
                    g.m_vol.to_numpy(),
                    g.s_scale.to_numpy(),
                    pairs,
                )

            cp = raw(cal)
            calibrator = fit_calibration(cal, cp, "sigmoid")
            p = calibrated(calibrator, dev, raw(dev), "sigmoid")
            development[variant] = metrics(dev, p)
            dev[["symbol", "target_date", "label"]].assign(p_up=p).to_parquet(
                out / f"dev-{variant}.parquet", index=False
            )
            objects[variant] = dict(market=mm, stock=sm, pairs=pairs, calibrator=calibrator)
            print("TRAINED", variant, development[variant]["brier"], flush=True)
        selected = min(
            ["ridge", "lightgbm"], key=lambda k: (development[k]["brier"], development[k]["log_loss"])
        )
        volatility_cut = float(market.loc[market.target_date.lt("2021-01-01"), "m_vol"].median())
        dev_p = pd.read_parquet(out / f"dev-{selected}.parquet").p_up.to_numpy()
        conditional = condition_results(dev, dev_p, volatility_cut)
        qualified = [k for k, v in conditional.items() if v["supported"]]
        freeze(
            out / "selection.json",
            dict(
                selected=selected,
                development=development,
                conditional=conditional,
                qualified=qualified,
                volatility_cut=volatility_cut,
            ),
        )
        results = {}
        predictions = {}
        for key in ["initial_refit", selected]:
            obj = objects[key]
            if key == "initial_refit":
                raw = obj["model"].predict_proba(final[COLS])[:, 1]
            else:
                raw = joint_probability(
                    obj["market"].predict(final[MKT]),
                    obj["stock"].predict(final[TECH + MKT]),
                    final.m_vol.to_numpy(),
                    final.s_scale.to_numpy(),
                    obj["pairs"],
                )
            p = calibrated(obj["calibrator"], final, raw, "sigmoid")
            predictions[key] = p
            results[key] = metrics(final, p)
            final[["symbol", "date", "target_date", "label", "market_y", "breadth"]].assign(
                p_up=p
            ).to_parquet(out / f"final-{key}.parquet", index=False)
        for k, obj in objects.items():
            joblib.dump(obj, out / f"{k}.joblib")
    ds = sorted(final.target_date.unique())
    comparisons = compare(final, predictions["initial_refit"], predictions[selected], ds, draws(ds))
    # Archived initial predictions: same final rows, different historical training recipe.
    archive = pd.read_parquet(
        root / "runs/direction-historical-reliability-20261017/matched-predictions.parquet"
    )
    common = final.merge(
        archive[["symbol", "target_date", "initial", "technical_expanding"]],
        on=["symbol", "target_date"],
        validate="one_to_one",
    )
    archived = {k: metrics(common, common[k].to_numpy()) for k in ["initial", "technical_expanding"]}
    assert all(digest(Path(p).read_bytes()) == h for p, h in protected.items())
    report = dict(
        at=utcnow(),
        status="complete_research_only",
        production_changed=False,
        independent=False,
        diagnostic=diag,
        split_audit=audit,
        selected=selected,
        development=development,
        results=results,
        comparison=comparisons,
        archived=archived,
        development_conditions=conditional,
        qualified_conditions=qualified,
        final_conditions={
            k: v
            for k, v in condition_results(final, predictions[selected], volatility_cut).items()
            if k in qualified
        },
        limitations=[
            "2024 reused retrospective, not new independent evidence",
            "Conditional on non-flat stock close; current universe survivorship and historical timestamp limitations",
            "Joint residual distribution assumes scaled 2021 innovations transport to later years; not a validated full conditional distribution",
        ],
    )
    atomic_json(out / "report.json", report)
    atomic_json(root / "reports/direction-joint-study.json", report)
    print(
        "COMPLETE",
        json.dumps(
            dict(
                selected=selected,
                comparison=comparisons,
                results={
                    k: {x: v[x] for x in ["n", "accuracy", "brier", "log_loss", "roc_auc", "ece", "up60_70"]}
                    for k, v in results.items()
                },
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
