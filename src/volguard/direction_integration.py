"""Paired development ablations with genuine chronological range OOF inputs.

Never reads 2024+ outcomes or changes issued forecasts. Candidate/calibration
selection is frozen on 2021-22 before inspecting the reused 2023 confirmation.
"""

import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .direction import COLS, block_ci, metrics
from .direction_benchmark import decisions, load_development
from .direction_oof import RANGE_COLS, check_temporal
from .direction_sources import SYMBOLS
from .direction_study import EXTRA, logits
from .platform_study import local_settings
from .util import atomic_json, digest, utcnow

BASE = COLS + EXTRA
RICH = [
    "beta60",
    "correlation60",
    "relative_momentum5",
    "relative_momentum20",
    "signed_volume",
    "overnight5",
    "intraday5",
    "return_reversal",
    "range_return",
    "turnover_acceleration",
    "market_reversal",
    "weekday",
]
EXTERNAL = [
    "foreign1",
    "foreign5",
    "foreign20",
    "institution1",
    "institution5",
    "institution20",
    "disclosures1",
    "disclosures5",
    "disclosures20",
    "earnings20",
    "capital20",
]
CONFIGS = {
    "annual_base": dict(kind="logistic", features="base", cadence="Y", years=5, half_life=730, C=0.1),
    "monthly_unweighted": dict(kind="logistic", features="base", cadence="M", years=5, half_life=1e9, C=0.1),
    "monthly_base": dict(kind="logistic", features="base", cadence="M", years=5, half_life=730, C=0.1),
    "monthly_range": dict(kind="logistic", features="range", cadence="M", years=5, half_life=730, C=0.1),
    "monthly_rich": dict(kind="logistic", features="rich", cadence="M", years=5, half_life=730, C=0.1),
    "monthly_rich_short": dict(kind="logistic", features="rich", cadence="M", years=3, half_life=365, C=1.0),
    "lgb7": dict(
        kind="lgb", features="rich", cadence="Q", years=5, half_life=730, leaves=7, child=200, reg=5
    ),
    "lgb31": dict(
        kind="lgb", features="rich", cadence="Q", years=5, half_life=730, leaves=31, child=100, reg=10
    ),
    "cat4": dict(
        kind="cat", features="rich", cadence="Q", years=5, half_life=730, depth=4, iterations=250, reg=5
    ),
    "cat6": dict(
        kind="cat", features="rich", cadence="Q", years=5, half_life=730, depth=6, iterations=350, reg=10
    ),
}


def richer(data):
    """No target columns enter rolling features; grouping is chronological."""
    data = data.sort_values(["symbol", "date"]).copy()
    for _, g in data.groupby("symbol"):
        r, m = g.return_1, g.market_return
        var = m.rolling(60, min_periods=40).var()
        data.loc[g.index, "beta60"] = r.rolling(60, min_periods=40).cov(m) / var.replace(0, np.nan)
        data.loc[g.index, "correlation60"] = r.rolling(60, min_periods=40).corr(m)
        data.loc[g.index, "relative_momentum5"] = g.return_5 - np.expm1(np.log1p(m).rolling(5).sum())
        data.loc[g.index, "relative_momentum20"] = g.return_20 - np.expm1(np.log1p(m).rolling(20).sum())
        data.loc[g.index, "overnight5"] = g.gap.rolling(5).mean()
        data.loc[g.index, "intraday5"] = g.body_pct.rolling(5).mean()
    data["signed_volume"] = np.sign(data.return_1) * np.log1p(data.volume_ratio.clip(lower=0))
    data["return_reversal"] = data.return_1 * data.return_5
    data["range_return"] = data.scaled_return * data.range_forecast_median
    data["turnover_acceleration"] = data.turnover_ratio - data.volume_ratio
    data["market_reversal"] = data.market_return * data.market_momentum
    data["weekday"] = data.date.dt.dayofweek
    # Neutral missing values with explicitly fixed transform; no fitted future medians.
    data[RICH] = data[RICH].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-100, 100)
    return data.sort_values(["date", "symbol"]).reset_index(drop=True)


def external_features(data, root):
    parts = []
    for symbol in SYMBOLS:
        fp, dp = root / f"flow-{symbol}.parquet", root / f"dart-{symbol}.parquet"
        if not fp.exists() or not dp.exists():
            continue
        g = data[data.symbol.eq(symbol)].copy().sort_values("date")
        if g.empty:
            continue
        flow = pd.read_parquet(fp).sort_values("date").set_index("date")
        for prefix, col in [("foreign", "frgnr_invsr"), ("institution", "orgn")]:
            val = pd.to_numeric(flow[col].str.replace(",", ""), errors="coerce")
            amount = pd.to_numeric(flow.acc_trde_prica.str.replace(",", ""), errors="coerce")
            for window in [1, 5, 20]:
                ratio = val.rolling(window).sum() / amount.rolling(window).sum().replace(0, np.nan)
                # asof strictly before today: do not assume same-day release at 18:30.
                r = pd.DataFrame({"date": ratio.index, "value": ratio.values}).dropna()
                merged = pd.merge_asof(
                    g[["date"]],
                    r,
                    on="date",
                    direction="backward",
                    allow_exact_matches=False,
                    tolerance=pd.Timedelta(days=7),
                )
                g[f"{prefix}{window}"] = merged.value.to_numpy()
        dart = pd.read_parquet(dp)
        daily_index = pd.date_range("2014-01-01", "2023-12-31")
        for name, mask in [
            ("disclosures", np.ones(len(dart), bool)),
            ("earnings", dart.report_nm.str.contains("실적|사업보고서|분기보고서|반기보고서")),
            ("capital", dart.report_nm.str.contains("유상증자|전환사채|자기주식|배당|합병")),
        ]:
            # Date of receipt, never fiscal-period date or latest-restated accounts.
            daily = dart[mask].groupby("date").size().reindex(daily_index, fill_value=0)
            for window in [1, 5, 20] if name == "disclosures" else [20]:
                count = daily.rolling(window, min_periods=window).sum().shift(1)
                g[f"{name}{window}"] = g.date.map(count)
        parts.append(g.dropna(subset=EXTERNAL))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def factory(c):
    if c["kind"] == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=c["C"], max_iter=400, random_state=17))
    if c["kind"] == "lgb":
        return LGBMClassifier(
            n_estimators=300,
            num_leaves=c["leaves"],
            min_child_samples=c["child"],
            reg_lambda=c["reg"],
            learning_rate=0.03,
            n_jobs=3,
            random_state=17,
            verbosity=-1,
        )
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=c["iterations"],
        depth=c["depth"],
        learning_rate=0.03,
        l2_leaf_reg=c["reg"],
        thread_count=3,
        random_seed=17,
        verbose=False,
        allow_writing_files=False,
    )


def feature_cols(c):
    return (
        BASE
        + (RANGE_COLS if c["features"] != "base" else [])
        + (RICH if c["features"] in ["rich", "external"] else [])
        + (EXTERNAL if c["features"] == "external" else [])
    )


def temporal_train(data, test, c):
    cutoff = test.date.min()
    issue = (cutoff + pd.Timedelta(hours=18, minutes=30)).tz_localize("Asia/Seoul")
    train = data[(data.target_date <= cutoff) & (pd.to_datetime(data.target_available_at, utc=True) <= issue)]
    if c.get("years"):
        train = train[train.target_date > cutoff - pd.DateOffset(years=c["years"])]
    if train.empty or train.target_date.max() > test.date.min():
        raise ValueError("Temporal training boundary")
    return train


def rolling_raw(data, c, out, name):
    target = out / f"raw-{name}.parquet"
    if target.exists():
        return pd.read_parquet(target)
    parts, fits = [], []
    groups = data[data.target_date.between("2020-01-01", "2023-12-31")].groupby(
        data.target_date.dt.to_period(c["cadence"])
    )
    for period, test in groups:
        train = temporal_train(data, test, c)
        cols = feature_cols(c)
        weights = np.exp2(-(test.date.min() - train.target_date).dt.days.to_numpy() / c["half_life"])
        model = factory(c)
        with threadpool_limits(limits=3):
            options = (
                {"logisticregression__sample_weight": weights}
                if c["kind"] == "logistic"
                else {"sample_weight": weights}
            )
            model.fit(train[cols], train.label, **options)
            p = model.predict_proba(test[cols])[:, 1]
        row = test[
            ["symbol", "date", "target_date", "target_available_at", "label", "direction_return", "y"]
            + RANGE_COLS
        ].copy()
        row["raw_p"] = p
        row["prior"] = train.label.mean()
        row["fit_target_end"] = train.target_date.max()
        row["period"] = str(period)
        parts.append(row)
        fits.append(
            dict(
                period=str(period),
                train_n=len(train),
                fit_end=str(train.target_date.max()),
                test_start=str(test.target_date.min()),
                test_n=len(test),
            )
        )
        print(
            json.dumps(dict(candidate=name, period=str(period), train=len(train), test=len(test))), flush=True
        )
    raw = pd.concat(parts, ignore_index=True)
    raw.to_parquet(target, index=False)
    joblib.dump(dict(model=model, cols=cols, config=c, research_only=True), out / f"{name}.joblib")
    atomic_json(out / f"fits-{name}.json", fits)
    return raw


def calibrate(raw, method):
    parts = []
    for period, test in raw[raw.target_date.ge("2021-01-01")].groupby("period", sort=True):
        cutoff = test.date.min()
        issue = (cutoff + pd.Timedelta(hours=18, minutes=30)).tz_localize("Asia/Seoul")
        # Prior rolling out-of-sample predictions only; no fitting-model in-sample scores.
        cal = raw[
            (raw.target_date <= cutoff)
            & (raw.target_date > cutoff - pd.DateOffset(years=1))
            & (pd.to_datetime(raw.target_available_at, utc=True) <= issue)
        ]
        if len(cal) < 300 or cal.label.nunique() != 2:
            raise ValueError("Insufficient preceding OOF calibration")
        if method == "raw":
            p = test.raw_p.to_numpy()
        elif method == "sigmoid":
            m = LogisticRegression(C=1000, max_iter=300).fit(logits(cal.raw_p.to_numpy()), cal.label)
            p = m.predict_proba(logits(test.raw_p.to_numpy()))[:, 1]
        else:
            m = IsotonicRegression(out_of_bounds="clip").fit(cal.raw_p, cal.label)
            p = m.predict(test.raw_p)
        result = test.copy()
        result["p_up"] = np.clip(p, 1e-6, 1 - 1e-6)
        result["calibration_end"] = cal.target_date.max()
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def scores(frame):
    y, p = frame.label.to_numpy(), frame.p_up.to_numpy()
    hit = (p > 0.5) == y
    hit[p == 0.5] = False
    covered = frame.y.between(frame.range_forecast_low, frame.range_forecast_high).to_numpy()
    bins = []
    for low in np.arange(0, 1, 0.1):
        mask = (p >= low) & (p < (low + 0.1) if low < 0.9 else p <= 1)
        if mask.any():
            bins.append(
                dict(
                    low=float(low),
                    n=int(mask.sum()),
                    predicted=float(p[mask].mean()),
                    observed=float(y[mask].mean()),
                )
            )
    ece = sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in bins) / len(frame)
    return dict(
        **decisions(y, p),
        brier=float(np.mean((p - y) ** 2)),
        log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
        ece=ece,
        calibration=bins,
        range_coverage=float(covered.mean()),
        joint_accuracy=float((hit & covered).mean()),
    )


def paired(candidate, reference):
    keys = ["symbol", "target_date"]
    c = candidate.set_index(keys).sort_index()
    r = reference.set_index(keys).sort_index()
    pd.testing.assert_index_equal(c.index, r.index)
    np.testing.assert_array_equal(c.label, r.label)
    np.testing.assert_allclose(c.y, r.y)
    cc, rr = scores(c), scores(r)
    y = c.label.to_numpy()
    p = c.p_up.to_numpy()
    q = r.p_up.to_numpy()
    ph = ((p > 0.5) == y) & (p != 0.5)
    qh = ((q > 0.5) == y) & (q != 0.5)
    cov = c.y.between(c.range_forecast_low, c.range_forecast_high).to_numpy()
    series = dict(
        accuracy=ph.astype(float) - qh,
        brier=(p - y) ** 2 - (q - y) ** 2,
        log_loss=-(y * np.log(p) + (1 - y) * np.log1p(-p)) + (y * np.log(q) + (1 - y) * np.log1p(-q)),
        joint=(ph & cov).astype(float) - (qh & cov),
    )
    ci = {
        k: block_ci(pd.Series(v, index=c.index.get_level_values("target_date")).groupby(level=0).mean())
        for k, v in series.items()
    }
    return dict(
        candidate=cc,
        reference=rr,
        deltas=dict(
            accuracy=cc["accuracy"] - rr["accuracy"],
            brier=cc["brier"] - rr["brier"],
            log_loss=cc["log_loss"] - rr["log_loss"],
            ece=cc["ece"] - rr["ece"],
            joint=cc["joint_accuracy"] - rr["joint_accuracy"],
        ),
        ci=ci,
    )


def validate_resume_source(saved_source):
    """Permit reporting repairs, never silently reuse scores after training edits."""
    protected = {
        "richer",
        "external_features",
        "factory",
        "feature_cols",
        "temporal_train",
        "rolling_raw",
        "calibrate",
        "scores",
        "paired",
    }

    def signatures(source):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name in protected
        }

    if signatures(saved_source) != signatures(Path(__file__).read_text(encoding="utf8")):
        raise ValueError("Resume cannot change training, calibration or scoring functions")


def run(settings, resume=None):
    root = settings.data_dir
    oof_info = json.loads((root / "reports/direction-range-oof.json").read_text(encoding="utf8"))
    oof_path = Path(oof_info["artifact_dir"]) / "oof.parquet"
    if digest(oof_path.read_bytes()) != oof_info["oof_hash"]:
        raise ValueError("OOF hash mismatch")
    data, input_hash, _ = load_development(root)
    oof = pd.read_parquet(oof_path)
    check_temporal(oof)
    oof = oof[["symbol", "date", "target_date", "y"] + RANGE_COLS]
    data = data.merge(oof, on=["symbol", "date", "target_date"], validate="one_to_one", suffixes=("", "_oof"))
    np.testing.assert_allclose(data.y, data.y_oof, rtol=1e-8)
    data = richer(data.drop(columns=["y_oof"]))
    ext = external_features(data, root / "research-inputs/direction-external")
    manifest = dict(
        configs=CONFIGS,
        calibrations=["raw", "sigmoid", "isotonic"],
        input_hash=input_hash,
        oof_hash=oof_info["oof_hash"],
        source_hash=digest(Path(__file__).read_bytes()),
        range_features=RANGE_COLS,
        matched_rows=len(data),
        external_rows=len(ext),
        external_hashes={
            p.name: digest(p.read_bytes())
            for p in (root / "research-inputs/direction-external").glob("*.parquet")
        },
        selection="2021-22 highest accuracy, tie Brier; freeze before scoring 2023",
        confirmation="2023 reused development; 2024+ not read; no new untouched test claim",
        joint_definition="direction argmax correct AND observed range inside interval; same target date and rows",
        gate="accuracy and joint increase with paired CI lower>0; Brier/logloss improve with upper<0; ECE not worse; >=1% Brier/logloss gain vs prior and .5; no invalid external PIT",
        source_timing="OHLC history as-retrieved 18:00 availability assumption, not revision-vintage proof",
    )
    if resume:
        out = Path(resume)
        validate_resume_source((out / "source.py").read_text(encoding="utf8"))
        if (out / "artifact-hashes.json").exists():
            for name, expected in json.loads(
                (out / "artifact-hashes.json").read_text(encoding="utf8")
            ).items():
                if digest((out / name).read_bytes()) != expected:
                    raise ValueError("Resume artifact hash mismatch: " + name)
        saved = json.loads((out / "manifest.json").read_text(encoding="utf8"))
        for key in [
            "configs",
            "calibrations",
            "input_hash",
            "oof_hash",
            "external_hashes",
            "matched_rows",
            "external_rows",
            "selection",
            "confirmation",
        ]:
            if saved[key] != manifest[key]:
                raise ValueError("Resume identity mismatch: " + key)
        manifest = saved
        (out / "finalization-source.py").write_bytes(Path(__file__).read_bytes())
    else:
        out = root / "runs" / ("direction-integration-" + digest(manifest)[:12])
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "manifest.json", manifest)
        (out / "source.py").write_bytes(Path(__file__).read_bytes())
    candidates = {}
    for name, c in CONFIGS.items():
        raw = rolling_raw(data, c, out, name)
        for method in manifest["calibrations"]:
            candidates[name + "__" + method] = calibrate(raw, method)
    # Fixed equal-weight ensemble, without learning weights on the confirmation year.
    ensemble = candidates["monthly_rich__sigmoid"].copy()
    idx = ["symbol", "target_date"]
    for name in ["lgb7__sigmoid", "cat4__sigmoid"]:
        merged = ensemble[idx].merge(candidates[name][idx + ["p_up"]], on=idx, validate="one_to_one")
        ensemble["p_up"] += merged.p_up.to_numpy()
    ensemble["p_up"] /= 3
    candidates["ensemble__sigmoid"] = ensemble
    external = {}
    if len(ext):
        for name, feature in [
            ("external_base", "base"),
            ("external_range", "range"),
            ("external_rich", "rich"),
            ("external_full", "external"),
        ]:
            c = dict(CONFIGS["monthly_rich"], features=feature)
            raw = rolling_raw(ext, c, out, name)
            for method in ["raw", "sigmoid", "isotonic"]:
                external[name + "__" + method] = calibrate(raw, method)
    all_preds = []
    for name, df in {**candidates, **external}.items():
        all_preds.append(
            df.assign(candidate=name, scope="core3" if name.startswith("external") else "universe")
        )
    combined = pd.concat(all_preds, ignore_index=True)
    if (out / "predictions.parquet").exists():
        old_predictions = pd.read_parquet(out / "predictions.parquet")
        pd.testing.assert_frame_equal(old_predictions, combined)
    else:
        combined.to_parquet(out / "predictions.parquet", index=False)

    def selection(pool):
        options = [dict(candidate=n, **scores(d[d.target_date.lt("2023-01-01")])) for n, d in pool.items()]
        return sorted(options, key=lambda r: (-r["accuracy"], r["brier"]))

    base_names = [
        n
        for n in candidates
        if n.split("__")[0] in CONFIGS and CONFIGS[n.split("__")[0]]["features"] == "base"
    ]
    combined_names = [n for n in candidates if n not in base_names]
    selected = selection({n: candidates[n] for n in combined_names})[0]["candidate"]
    reference = selection({n: candidates[n] for n in base_names})[0]["candidate"]
    frozen = dict(
        selected=selected,
        reference=reference,
        ranking=selection(candidates),
        external_ranking=selection(external) if external else [],
        frozen_at=utcnow(),
    )
    if (out / "selection.json").exists():
        saved_selection = json.loads((out / "selection.json").read_text(encoding="utf8"))
        if any(saved_selection[k] != v for k, v in frozen.items() if k != "frozen_at"):
            raise ValueError("Frozen selection changed")
    else:
        atomic_json(out / "selection.json", frozen)
    confirm = lambda d: d[d.target_date.ge("2023-01-01")]
    results = [
        dict(
            candidate=n, selection=scores(d[d.target_date.lt("2023-01-01")]), confirmation=scores(confirm(d))
        )
        for n, d in candidates.items()
    ]
    comparison = paired(confirm(candidates[selected]), confirm(candidates[reference]))
    # Isolate the requested four OOF features with identical model/settings/calibration.
    ablations = {
        m: paired(confirm(candidates["monthly_range__" + m]), confirm(candidates["monthly_base__" + m]))
        for m in manifest["calibrations"]
    }
    archived = json.loads((root / "reports/platform-direction-accuracy.json").read_text(encoding="utf8"))
    old = pd.read_parquet(
        Path(archived["artifact_dir"]) / "predictions.parquet",
        filters=[("model", "=", "logistic"), ("fold", "=", 2023)],
    )
    matched = (
        confirm(candidates[selected])
        .drop(columns=["p_up"])
        .merge(old[["symbol", "target_date", "p_up"]], on=idx, validate="one_to_one")
    )
    old_comparison = paired(confirm(candidates[selected]), matched)
    c, r = comparison["candidate"], comparison["reference"]
    ci = comparison["ci"]
    prior = confirm(candidates[reference]).copy()
    prior["p_up"] = prior.prior
    prior_score = scores(prior)
    gate = dict(
        accuracy=ci["accuracy"][0] > 0,
        joint=ci["joint"][0] > 0,
        brier=ci["brier"][1] < 0,
        log_loss=ci["log_loss"][1] < 0,
        calibration=c["ece"] <= r["ece"],
        prior_brier=c["brier"] <= 0.99 * min(prior_score["brier"], 0.25),
        prior_log_loss=c["log_loss"] <= 0.99 * min(prior_score["log_loss"], float(np.log(2))),
        versus_original=old_comparison["ci"]["accuracy"][0] > 0,
    )
    external_results = [
        dict(
            candidate=n, selection=scores(d[d.target_date.lt("2023-01-01")]), confirmation=scores(confirm(d))
        )
        for n, d in external.items()
    ]
    external_ablation = (
        {
            m: paired(confirm(external["external_full__" + m]), confirm(external["external_rich__" + m]))
            for m in manifest["calibrations"]
        }
        if external
        else {}
    )
    report = dict(
        **{k: v for k, v in manifest.items() if k != "gate"},
        gate_rule=manifest["gate"],
        artifact_dir=str(out),
        selected=selected,
        reference=reference,
        results=results,
        comparison=comparison,
        range_ablations=ablations,
        versus_original=old_comparison,
        external_results=external_results,
        external_ablation=external_ablation,
        gate=gate,
        development_pass=all(gate.values()),
        operational_probability_enabled=False,
        completed_at=utcnow(),
    )
    atomic_json(out / "report.json", report)
    atomic_json(root / "reports/direction-integration.json", report)
    print(
        json.dumps(
            dict(
                artifact_dir=str(out),
                selected=selected,
                reference=reference,
                comparison=comparison,
                gate=gate,
            )
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    run(local_settings(), parser.parse_args().resume)
