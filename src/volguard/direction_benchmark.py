"""Distinct classifiers compared by next-close argmax accuracy on development years only."""

import json
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import Lake, features
from .direction import COLS, block_ci, extend, metrics
from .direction_study import EXTRA, enhance, logits
from .platform_study import local_settings
from .util import atomic_json, digest, utcnow

NAMES = dict(
    logistic="로지스틱 회귀",
    lightgbm="LightGBM",
    catboost="CatBoost",
    random_forest="랜덤포레스트",
    extra_trees="Extra Trees",
    hist_gradient="히스토그램 부스팅",
    naive_bayes="나이브 베이즈",
    ensemble="균등 앙상블",
)


def factories():
    from catboost import CatBoostClassifier

    return dict(
        logistic=make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500, random_state=17)),
        lightgbm=LGBMClassifier(
            n_estimators=200,
            num_leaves=15,
            learning_rate=0.03,
            min_child_samples=500,
            reg_lambda=10,
            n_jobs=4,
            random_state=17,
            verbosity=-1,
        ),
        catboost=CatBoostClassifier(
            iterations=200,
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=10,
            thread_count=4,
            random_seed=17,
            verbose=False,
            allow_writing_files=False,
        ),
        random_forest=RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=300, max_samples=0.65, n_jobs=4, random_state=17
        ),
        extra_trees=ExtraTreesClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=300, n_jobs=4, random_state=17
        ),
        hist_gradient=HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=15,
            learning_rate=0.04,
            min_samples_leaf=500,
            l2_regularization=10,
            early_stopping=False,
            random_state=17,
        ),
        naive_bayes=make_pipeline(StandardScaler(), GaussianNB(var_smoothing=0.01)),
    )


def decisions(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid direction probability")
    predicted = p > 0.5
    active = p != 0.5
    correct = predicted == y

    def portion(mask):
        return dict(
            n=int(mask.sum()),
            correct=int(correct[mask].sum()),
            accuracy=float(correct[mask].mean()) if mask.any() else None,
        )

    confidence = np.maximum(p, 1 - p)
    return dict(
        n=len(y),
        predicted_n=int(active.sum()),
        ties=int((~active).sum()),
        correct=int(correct[active].sum()),
        accuracy=float(correct[active].mean()) if active.any() else None,
        up=portion(active & predicted),
        down=portion(active & ~predicted),
        confidence_bins=[
            dict(low=lo, high=hi, **portion(active & (confidence >= lo) & (confidence < hi)))
            for lo, hi in [(0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.000001)]
        ],
    )


def load_development(root):
    bars = pd.read_parquet(root / "silver/bars.parquet", filters=[("date", "<", pd.Timestamp("2024-01-01"))])
    input_hash = digest(pd.util.hash_pandas_object(bars, index=False).values.tobytes())
    bars["available_at"] = (
        (bars.date + pd.Timedelta(hours=18)).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
    )
    frame = features(bars, [s for s in Lake(root).calendar() if s["date"] < "2024-01-01"])
    market = pd.read_parquet(
        root / "silver/market.parquet", filters=[("date", "<", pd.Timestamp("2024-01-01"))]
    )
    frame["market_range"] = frame.date.map(market.drop_duplicates("date").set_index("date").range_pct)
    frame = enhance(extend(frame))
    eligible = frame.direction_return.notna() & frame.direction_return.ne(0) & frame.y.notna()
    eligible &= frame[COLS + EXTRA].notna().all(axis=1)
    return frame[eligible].copy(), input_hash, int(frame.direction_return.eq(0).sum())


def run(settings):
    root = settings.data_dir
    out = root / "runs" / ("direction-accuracy-" + uuid.uuid4().hex[:10])
    out.mkdir()
    configs = {k: str(m.get_params()) for k, m in factories().items()}
    manifest = dict(
        created_at=utcnow(),
        folds=[2021, 2022, 2023],
        names=NAMES,
        configurations=configs,
        selection="highest argmax accuracy on 2021-2022; ties by lower Brier; freeze before 2023",
        decision_rule="up if p_up > p_down, down if p_down > p_up; 50/50 abstain",
        label="next verified adjusted close return positive/negative; exact zero excluded",
        temporal_rule="train through fold-2; calibrate preceding year; evaluate fold",
        scope="reused development; 2024 and locked range test never opened",
        features=COLS + EXTRA,
        source_hash=digest(Path(__file__).read_bytes()),
    )
    atomic_json(out / "manifest.json", manifest)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    data, input_hash, flat_n = load_development(root)
    atomic_json(out / "input.json", dict(hash=input_hash, excluded_flat=flat_n))
    predictions, fold_results, selected = [], [], None
    for year in manifest["folds"]:
        if year == 2023:
            earlier = pd.concat(predictions, ignore_index=True)
            selection = []
            for name, g in earlier.groupby("model"):
                selection.append(
                    dict(
                        model=name,
                        **decisions(g.label, g.p_up),
                        brier=float(np.mean((g.p_up - g.label) ** 2)),
                    )
                )
            selected = sorted(selection, key=lambda r: (-r["accuracy"], r["brier"]))[0]["model"]
            atomic_json(
                out / "selection.json", dict(selected=selected, results=selection, frozen_at=utcnow())
            )
        train = data[data.target_date < f"{year - 1}-01-01"]
        cal = data[data.target_date.between(f"{year - 1}-01-01", f"{year - 1}-12-31")]
        test = data[data.target_date.between(f"{year}-01-01", f"{year}-12-31")]
        prior = float(train.label.mean())
        fold_probs = {}
        for name, model in factories().items():
            with threadpool_limits(limits=4):
                model.fit(train[COLS + EXTRA], train.label)
                calibrator = LogisticRegression(C=1000, max_iter=300).fit(
                    logits(model.predict_proba(cal[COLS + EXTRA])[:, 1]), cal.label
                )
                p = calibrator.predict_proba(logits(model.predict_proba(test[COLS + EXTRA])[:, 1]))[:, 1]
            fold_probs[name] = p
            if year == 2023:
                joblib.dump(
                    dict(model=model, calibrator=calibrator, cols=COLS + EXTRA, research_only=True),
                    out / f"{name}.joblib",
                )
            print(json.dumps(dict(fold=year, model=name, **decisions(test.label, p))), flush=True)
        fold_probs["ensemble"] = np.mean(list(fold_probs.values()), axis=0)
        for name, p in fold_probs.items():
            row = test[["symbol", "date", "target_date", "label", "direction_return"]].copy()
            row = row.assign(model=name, fold=year, p_up=p, p_down=1 - p, prior=prior)
            row["prediction"] = np.where(p > 0.5, "up", np.where(p < 0.5, "down", "tie"))
            row["correct"] = np.where(p == 0.5, False, (p > 0.5) == row.label)
            predictions.append(row)
            fold_results.append(dict(model=name, year=year, **decisions(test.label, p)))
        pd.concat(predictions, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    pred = pd.concat(predictions, ignore_index=True)
    results = []
    for name, g in pred.groupby("model"):
        confirm = g[g.fold.eq(2023)]
        primary = decisions(confirm.label, confirm.p_up)
        prior_correct = ((confirm.prior > 0.5) == confirm.label).astype(float)
        delta = confirm.correct.astype(float) - prior_correct
        ci = block_ci(delta.groupby(confirm.target_date).mean())
        results.append(
            dict(
                model=name,
                name=NAMES[name],
                overall=decisions(g.label, g.p_up),
                confirmation=primary,
                probability_metrics=metrics(confirm.label, confirm.p_up),
                accuracy_gain_ci=ci,
                prior_accuracy=float(prior_correct.mean()),
                always_up_accuracy=float(confirm.label.mean()),
                always_down_accuracy=float(1 - confirm.label.mean()),
                by_symbol={str(s): decisions(v.label, v.p_up) for s, v in confirm.groupby("symbol")},
            )
        )
    report = dict(
        **manifest,
        artifact_dir=str(out),
        selected=selected,
        results=results,
        fold_results=fold_results,
        excluded_flat=flat_n,
        completed_at=utcnow(),
        operational_probability_enabled=False,
        note="2023 is reused development confirmation, not an untouched final test. No automatic promotion.",
    )
    atomic_json(out / "report.json", report)
    atomic_json(root / "reports/platform-direction-accuracy.json", report)
    print(
        json.dumps(
            dict(
                selected=selected,
                results=[{k: r[k] for k in ["model", "confirmation", "prior_accuracy"]} for r in results],
                artifact_dir=str(out),
            )
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    run(local_settings())
