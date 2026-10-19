"""Preregistered financial structure ablations, sequential model selection, frozen confirmation."""

import argparse
import ast
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .direction_integration import calibrate, scores
from .direction_math_features import BASE, GROUPS
from .direction_oof import RANGE_COLS
from .platform_study import local_settings
from .util import atomic_json, digest, utcnow

ALL = BASE + sum(GROUPS.values(), [])
RANGE = RANGE_COLS + ["range_relative"]
KEYS = ["symbol", "date", "target_date", "target_available_at", "label", "direction_return", "y"] + RANGE_COLS


def config(cols, **kw):
    return dict(
        cols=cols,
        kind="logistic",
        cadence="Q",
        years=5,
        half_life=730,
        scope="pooled",
        balanced=False,
        select=False,
        **kw,
    )


def configs():
    result = {"baseline": config(BASE), "all_math": config(ALL)}
    for group, cols in GROUPS.items():
        result["add_" + group] = config(BASE + cols)
        result["drop_" + group] = config([c for c in ALL if c not in cols])
    return result


def training_rows(data, test, c):
    cutoff = test.date.min()
    issue = (cutoff + pd.Timedelta(hours=18, minutes=30)).tz_localize("Asia/Seoul")
    train = data[(data.target_date <= cutoff) & (pd.to_datetime(data.target_available_at, utc=True) <= issue)]
    if c["years"]:
        train = train[train.target_date > cutoff - pd.DateOffset(years=c["years"])]
    if len(train) < 1000:
        raise ValueError("Insufficient chronological training rows")
    return train


def factory(c):
    if c["kind"] == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=200,
            num_leaves=7,
            min_child_samples=300,
            reg_lambda=10,
            learning_rate=0.03,
            n_jobs=3,
            random_state=1707,
            verbosity=-1,
        )
    elif c["kind"] == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=200,
            depth=4,
            l2_leaf_reg=10,
            thread_count=3,
            random_seed=1707,
            verbose=False,
            allow_writing_files=False,
        )
    else:
        model = LogisticRegression(
            C=0.1, max_iter=500, class_weight="balanced" if c["balanced"] else None, random_state=1707
        )
    steps = [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]
    if c["select"]:
        steps.append(SelectKBest(f_classif, k=min(30, len(c["cols"]))))
    return make_pipeline(*steps, model)


def run_raw(data, c, out, name, years):
    parts = []
    for period, test in data[data.target_date.dt.year.isin(years)].groupby(
        data.target_date.dt.to_period(c["cadence"]), sort=True
    ):
        dest = out / f"raw-{name}-{period}.parquet"
        stamp = out / f"raw-{name}-{period}.json"
        if dest.exists():
            saved = json.loads(stamp.read_text())
            if saved["hash"] != digest(dest.read_bytes()) or saved["config"] != c:
                raise ValueError("Checkpoint/config mismatch")
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
            last = list(model.named_steps)[-1]
            model.fit(tr[c["cols"]], tr.label, **{last + "__sample_weight": np.asarray(weights)})
            return model, model.predict_proba(te[c["cols"]])[:, 1]

        with threadpool_limits(limits=3):
            model, p = fit(train, test)
            specialized = 0
            if c["scope"] != "pooled":
                tr_group = train.symbol if c["scope"] == "stock" else train.markov_high.fillna(0.5).gt(0.5)
                te_group = test.symbol if c["scope"] == "stock" else test.markov_high.fillna(0.5).gt(0.5)
                for value in te_group.unique():
                    tr = train[tr_group.eq(value)]
                    mask = te_group.eq(value)
                    if len(tr) < 500 or tr.label.nunique() < 2:
                        continue
                    _, pp = fit(tr, test[mask])
                    p[np.flatnonzero(mask)] = pp
                    specialized += 1
        result = test[KEYS + ["markov_high", "market_momentum"]].copy()
        result["raw_p"] = np.clip(p, 1e-6, 1 - 1e-6)
        result["prior"] = float(train.label.mean())
        result["period"] = str(period)
        result["fit_end"] = train.target_date.max()
        result.to_parquet(dest, index=False)
        atomic_json(
            stamp,
            dict(
                hash=digest(dest.read_bytes()),
                config=c,
                train_n=len(train),
                fit_end=str(train.target_date.max()),
                specialized=specialized,
            ),
        )
        if years[-1] == 2023 and str(period).startswith("2023"):
            joblib.dump(dict(model=model, config=c, research_only=True), out / f"last-{name}.joblib")
        parts.append(result)
        print(name, period, "fit", len(train), "predict", len(test), flush=True)
    return pd.concat(parts, ignore_index=True)


def measure(g):
    result = scores(g)
    p, y = g.p_up.to_numpy(), g.label.to_numpy()
    active = p != 0.5
    result.update(
        balanced_accuracy=float(balanced_accuracy_score(y[active], p[active] > 0.5))
        if active.any()
        else None,
        roc_auc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
        mcc=float(matthews_corrcoef(y[active], p[active] > 0.5)) if active.any() else None,
    )
    width = g.range_forecast_high - g.range_forecast_low
    result["range_width"] = float(width.mean())
    result["interval_score"] = float(
        (
            width + 20 * np.maximum(np.maximum(g.range_forecast_low - g.y, g.y - g.range_forecast_high), 0)
        ).mean()
    )
    result["prediction_coverage"] = float(active.mean())
    return result


def subset(g, threshold):
    mask = np.maximum(g.p_up, 1 - g.p_up) >= threshold
    mask &= g.p_up.ne(0.5)
    chosen = g[mask]
    if chosen.empty:
        return dict(threshold=threshold, n=0, coverage=0, accuracy=None, accuracy_ci=None)
    hit = ((chosen.p_up > 0.5) == chosen.label).astype(float)
    n = len(hit)
    p = float(hit.mean())
    z = norm.ppf(0.975)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    rad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return dict(
        threshold=threshold,
        n=n,
        coverage=n / len(g),
        accuracy=p,
        wilson_iid_ci=[center - rad, center + rad],
        **{k: v for k, v in measure(chosen).items() if k not in ["n", "accuracy"]},
    )


def paired_ci(a, b):
    a = a.set_index(["target_date", "symbol"]).sort_index()
    b = b.set_index(["target_date", "symbol"]).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    np.testing.assert_array_equal(a.label, b.label)
    y = a.label.to_numpy()
    p = a.p_up.to_numpy()
    q = b.p_up.to_numpy()

    def hit(x):
        return ((x > 0.5) == y) & (x != 0.5)

    inside = a.y.between(a.range_forecast_low, a.range_forecast_high).to_numpy()

    def loss(x):
        return -(y * np.log(x) + (1 - y) * np.log1p(-x))

    delta = (
        pd.DataFrame(
            dict(
                accuracy=hit(p).astype(float) - hit(q),
                brier=(p - y) ** 2 - (q - y) ** 2,
                log_loss=loss(p) - loss(q),
                joint=(hit(p) & inside).astype(float) - (hit(q) & inside),
                n=1,
            ),
            index=a.index.get_level_values("target_date"),
        )
        .groupby(level=0)
        .sum()
    )
    values = delta.to_numpy()
    rng = np.random.default_rng(1707)
    draws = []
    for _ in range(2000):
        starts = rng.integers(0, max(1, len(values) - 19), size=int(np.ceil(len(values) / 20)))
        ids = np.concatenate([np.arange(s, min(s + 20, len(values))) for s in starts])[: len(values)]
        total = values[ids].sum(axis=0)
        draws.append(total[:-1] / total[-1])
    ci = np.quantile(draws, [0.025, 0.975], axis=0)
    est = values.sum(axis=0)
    return {
        k: dict(delta=float(est[i] / est[-1]), ci95=ci[:, i].tolist())
        for i, k in enumerate(delta.columns[:-1])
    }


def candidates(raw):
    return {method: calibrate(raw, method) for method in ["raw", "sigmoid", "isotonic"]}


def choose(results, base_brier):
    admissible = [r for r in results if r["brier"] <= base_brier]
    return sorted(admissible or results, key=lambda r: (-r["accuracy"], r["brier"], r["name"]))[0]["name"]


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def validate_identity(out):
    """Resume only the same inputs and fitted computation, not silently stale scores."""
    protected = {"config", "configs", "training_rows", "factory", "run_raw", "candidates", "choose"}
    tree = ast.parse(Path(__file__).read_text(encoding="utf8"))
    identity = dict(
        features_hash=digest((out / "features.parquet").read_bytes()),
        fitting_hash=digest(
            [
                ast.dump(n, include_attributes=False)
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name in protected
            ]
        ),
        calibration_hash=digest(Path(__file__).with_name("direction_integration.py").read_bytes()),
    )
    path = out / "experiment-identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Inputs or fitting logic changed; use a new research run, never stale checkpoints")
    if not path.exists():
        atomic_json(path, identity)
    return identity


def run(out):
    validate_identity(out)
    root = local_settings().data_dir
    data = pd.read_parquet(out / "features.parquet")
    data = data[data[RANGE].notna().all(axis=1)].sort_values(["date", "symbol"]).reset_index(drop=True)
    if data.target_date.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("Protected period")
    cs = configs()
    preds = {}
    results = []

    # All choices below are evaluated on development only, before confirmation.
    def develop(name, c):
        raw = run_raw(data, c, out, name, [2020, 2021, 2022])
        for method, g in candidates(raw).items():
            key = name + "__" + method
            preds[key] = g
            results.append(dict(name=key, **measure(g)))
        atomic_json(out / "development-progress.json", safe_json(dict(results=results, completed=name)))

    for name, c in cs.items():
        develop(name, c)
    baseline = next(r for r in results if r["name"] == "baseline__raw")
    feature_choice = choose(results, baseline["brier"]).split("__")[0]
    atomic_json(
        out / "feature-selection.json",
        dict(selected=feature_choice, selected_at=utcnow(), results=safe_json(results)),
    )
    selected_cols = cs[feature_choice]["cols"]
    for name, updates in {
        "monthly": dict(cadence="M"),
        "expanding": dict(years=None, half_life=None),
        "rolling_uniform": dict(half_life=None),
        "rolling3": dict(years=3),
        "recent1y": dict(half_life=365),
        "regime": dict(scope="regime"),
        "stock": dict(scope="stock"),
        "balanced": dict(balanced=True),
        "selection30": dict(select=True),
        "lightgbm": dict(kind="lightgbm"),
        "catboost": dict(kind="catboost"),
    }.items():
        cs[name] = dict(config(selected_cols), **updates)
        develop(name, cs[name])
    # Optional separately trained sequence research uses the exact same eligible keys.
    neural = out / "neural-predictions.parquet"
    if not neural.exists():
        raise RuntimeError(
            "Neural study pending; resume after its predictions are complete, before selection"
        )
    if neural.exists():
        nn = pd.read_parquet(neural)
        for name, raw in nn[nn.target_date.lt("2023-01-01")].groupby("model"):
            cs[name] = {"kind": "neural", "cols": []}
            for method, g in candidates(raw).items():
                key = name + "__" + method
                preds[key] = g
                results.append(dict(name=key, **measure(g)))
    # Fixed diversity ensemble; no test-fit weights.
    ensemble = preds[feature_choice + "__sigmoid"].copy()
    keys = ["symbol", "target_date"]
    ensemble = ensemble.set_index(keys).sort_index()
    for name in ["lightgbm__sigmoid", "catboost__sigmoid"]:
        p = preds[name].set_index(keys).sort_index()
        pd.testing.assert_index_equal(p.index, ensemble.index)
        ensemble.p_up += p.p_up
    ensemble.p_up /= 3
    preds["ensemble__sigmoid"] = ensemble.reset_index()
    results.append(dict(name="ensemble__sigmoid", **measure(preds["ensemble__sigmoid"])))
    selected = choose(results, baseline["brier"])
    selected_base = selected.split("__")[0]
    method = selected.split("__")[1]
    thresholds = [subset(preds[selected], t) for t in [0.5, 0.55, 0.6, 0.65, 0.7]]
    allowed = [r for r in thresholds if r["coverage"] >= 0.05 and r["n"] >= 1000]
    threshold = (
        sorted(allowed, key=lambda r: (-r["accuracy"], r["threshold"]))[0]["threshold"] if allowed else 0.5
    )
    choice = dict(
        selected=selected,
        feature_choice=feature_choice,
        threshold=threshold,
        development=safe_json(results),
        thresholds=safe_json(thresholds),
        frozen_at=utcnow(),
    )
    old = out / "standalone-selection.json"
    if old.exists():
        saved = json.loads(old.read_text())
        if saved["selected"] != selected or saved["threshold"] != threshold:
            raise ValueError("Frozen selection changed")
        choice = saved
    else:
        atomic_json(old, choice)
    print("STANDALONE_FROZEN", selected, "threshold", threshold, flush=True)
    # Range addition only after standalone development choice is fixed.
    if selected_base in ["ensemble", "lstm_numeric", "craft_adapted"]:
        # Late fusion logistic stacking of preceding OOF direction probabilities and range,
        # kept out of the classifier selection; never claims same-architecture ablation.
        integration_base = feature_choice
    else:
        integration_base = selected_base
    cs["with_range"] = dict(cs[integration_base], cols=cs[integration_base]["cols"] + RANGE)
    develop("with_range", cs["with_range"])
    range_methods = [r for r in results if r["name"].startswith("with_range__")]
    range_choice = choose(range_methods, float("inf"))
    atomic_json(
        out / "range-selection.json",
        dict(
            selected=range_choice,
            comparator=integration_base,
            frozen_at=utcnow(),
            note="same architecture comparator; standalone winner separately retained",
        ),
    )
    # Confirm all fixed studies without changing selection, including ablation failures.
    confirmations = {}
    for name, c in cs.items():
        if c["kind"] == "neural":
            raw = nn[nn.model.eq(name)]
        else:
            raw = run_raw(data, c, out, name, [2020, 2021, 2022, 2023])
        for m, g in candidates(raw).items():
            confirmations[name + "__" + m] = g[g.target_date.dt.year.eq(2023)].copy()
    en = confirmations[feature_choice + "__sigmoid"].set_index(keys).sort_index().copy()
    for name in ["lightgbm__sigmoid", "catboost__sigmoid"]:
        p = confirmations[name].set_index(keys).sort_index()
        pd.testing.assert_index_equal(p.index, en.index)
        en.p_up += p.p_up
    en.p_up /= 3
    confirmations["ensemble__sigmoid"] = en.reset_index()
    final = confirmations[selected]
    reference = confirmations["baseline__raw"]
    comp = paired_ci(final, reference)
    matched = confirmations[integration_base + "__" + range_choice.split("__")[1]]
    range_comp = paired_ci(confirmations[range_choice], matched)
    records = []
    for name, g in confirmations.items():
        records.append(dict(name=name, **measure(g)))
    by_symbol = {s: measure(g) for s, g in final.groupby("symbol")}
    regime = np.where(
        final.markov_high.isna(), "unknown", np.where(final.markov_high > 0.5, "high_vol", "low_vol")
    )
    by_regime = {str(k): measure(g) for k, g in final.groupby(regime)}
    confidence = [subset(final, t) for t in [0.5, 0.55, 0.6, 0.65, 0.7]]
    select_mask = np.maximum(final.p_up, 1 - final.p_up) >= threshold
    confident_ci = (
        paired_ci(
            final[select_mask],
            reference.set_index(keys).loc[final[select_mask].set_index(keys).index].reset_index(),
        )
        if select_mask.any()
        else None
    )
    examples = []
    for symbol in ["005930", "000660", "005380"]:
        sub = final[final.symbol.eq(symbol)].copy()
        sub["correct"] = (sub.p_up > 0.5) == sub.label
        for correct in [True, False]:
            for r in (
                sub[sub.correct.eq(correct)]
                .sort_values("p_up", key=lambda x: abs(x - 0.5), ascending=False)
                .head(2)
                .itertuples()
            ):
                examples.append(
                    dict(
                        symbol=symbol,
                        date=str(r.target_date.date()),
                        p_up=r.p_up,
                        actual_return=r.direction_return,
                        correct=correct,
                        range_inside=bool(r.range_forecast_low <= r.y <= r.range_forecast_high),
                    )
                )
    prior = reference.assign(p_up=reference.prior)
    gate = (
        comp["accuracy"]["ci95"][0] > 0 and comp["brier"]["ci95"][1] < 0 and comp["log_loss"]["ci95"][1] < 0
    )
    report = dict(
        created_at=utcnow(),
        artifact_dir=str(out),
        status="reused_development",
        selected=selected,
        feature_choice=feature_choice,
        threshold=threshold,
        baseline=measure(reference),
        standalone=measure(final),
        comparison=comp,
        results=records,
        development=results,
        range_selected=range_choice,
        range_comparator=integration_base + "__" + range_choice.split("__")[1],
        range_comparison=range_comp,
        combined=measure(confirmations[range_choice]),
        prior=measure(prior),
        confidence=confidence,
        selected_confidence=subset(final, threshold),
        selected_confidence_ci=confident_ci,
        by_symbol=by_symbol,
        by_regime=by_regime,
        examples=examples,
        development_gate=gate,
        operational_probability_enabled=False,
        independent_evaluation=dict(
            status="unavailable",
            reason="2023 and 2024 already examined; 2025-2026 locked range test preserved; prospective evaluation required",
        ),
        rows=len(final),
    )
    pd.concat([g.assign(candidate=n) for n, g in confirmations.items()], ignore_index=True).to_parquet(
        out / "confirmation-predictions.parquet", index=False
    )
    atomic_json(out / "report.json", safe_json(report))
    atomic_json(root / "reports/direction-math.json", safe_json(report))
    print("MATH_DONE", selected, measure(final)["accuracy"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run(args.out or local_settings().data_dir / "runs/direction-math-20261017")
