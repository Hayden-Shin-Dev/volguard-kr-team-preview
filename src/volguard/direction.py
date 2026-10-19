"""Independent direction research with a once-only 2024 retrospective evaluation.

Flat observations are excluded before fitting: probabilities are conditional on
a non-flat close. No 2025–2026 locked range outcomes are opened by this study.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Lake, features
from .platform_study import local_settings
from .terminal_data import read_json
from .util import atomic_json, digest, utcnow

COLS = [
    "return_1",
    "return_5",
    "return_10",
    "return_20",
    "vol_22",
    "vol_change",
    "ma_deviation",
    "range_1",
    "range_5",
    "range_22",
    "gap",
    "body_pct",
    "close_location",
    "volume_ratio",
    "turnover_ratio",
    "market_range",
]


def extend(frame):
    parts = []
    for _, g in frame.groupby("symbol", sort=True):
        g = g.sort_values("date").copy()
        r = g.return_1
        log = np.log1p(r)
        for n in [5, 10, 20]:
            g[f"return_{n}"] = np.expm1(log.rolling(n, min_periods=n).sum())
        g["vol_change"] = r.rolling(5).std() / g.vol_22.replace(0, np.nan)
        # Relative price chain resets after any unverified daily return.
        chain = log.groupby(r.isna().cumsum()).cumsum().pipe(np.exp)
        g["ma_deviation"] = chain / chain.rolling(20, min_periods=20).mean() - 1
        g.loc[r.isna().rolling(20).sum().gt(0), "ma_deviation"] = np.nan
        g["direction_return"] = r.shift(-1)
        g["label"] = g.direction_return.gt(0).astype(int)
        parts.append(g)
    return pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def metrics(y, p):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="uniform")
    return dict(
        n=len(y),
        accuracy=float(accuracy_score(y, p >= 0.5)),
        balanced_accuracy=float(balanced_accuracy_score(y, p >= 0.5)),
        roc_auc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
        pr_auc=float(average_precision_score(y, p)) if sum(y) else None,
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        calibration={"predicted": predicted.tolist(), "observed": observed.tolist()},
    )


def block_ci(differences, seed=1701):
    a = np.asarray(differences)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(2000):
        starts = rng.integers(0, max(1, len(a) - 19), size=int(np.ceil(len(a) / 20)))
        sample = np.concatenate([a[i : i + 20] for i in starts])[: len(a)]
        samples.append(sample.mean())
    return np.quantile(samples, [0.025, 0.975]).tolist()


def run(settings):
    root = settings.data_dir
    out = root / "runs/direction-v1"
    # Reservation is intentionally non-reentrant, including interrupted evaluations.
    out.mkdir(exist_ok=False)
    manifest = {
        "created_at": utcnow(),
        "train": ["2010-01-01", "2021-12-31"],
        "validation": ["2022-01-01", "2022-12-31"],
        "calibration": ["2023-01-01", "2023-12-31"],
        "evaluation": ["2024-01-01", "2024-12-31"],
        "flat_rule": "exclude exact zero verified return; P(up | nonflat)",
        "selection": "lowest validation Brier, then sigmoid calibration on 2023 only",
        "gate": "AUC > .52; Brier and log loss improve >=1% vs both priors; 20-day/2000 bootstrap Brier delta upper <0",
        "candidates": ["logistic", "lightgbm7", "lightgbm15"],
        "features": COLS,
        "limitations": [
            "Retrospective publication-time assumption; historical revision vintages unavailable",
            "Current universe survivorship bias",
            "2024 is new for this direction target, previously used by range calibration; not globally unseen market data",
            "External/sector and range-model outputs excluded: historical availability not proven",
        ],
        "source_hash": digest(Path(__file__).read_bytes()),
        "locked_range_test_used": False,
    }
    atomic_json(out / "manifest.json", manifest)
    lake = Lake(root)
    # Predicate pushdown avoids loading locked-period rows, even for preprocessing.
    bars = pd.read_parquet(root / "silver/bars.parquet", filters=[("date", "<", pd.Timestamp("2025-01-01"))])
    manifest["input_hash"] = digest(pd.util.hash_pandas_object(bars, index=False).values.tobytes())
    atomic_json(out / "manifest.json", manifest)
    bars["available_at"] = (
        (bars.date + pd.Timedelta(hours=18)).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
    )
    sessions = [s for s in lake.calendar() if s["date"] < "2025-01-01"]
    frame = features(bars, sessions)
    market_path = root / "silver/market.parquet"
    if market_path.exists():
        market = pd.read_parquet(market_path, filters=[("date", "<", pd.Timestamp("2025-01-01"))])
        # Match the live range feature's official market input, not a universe proxy.
        if "range_pct" in market:
            frame["market_range"] = frame.date.map(market.drop_duplicates("date").set_index("date").range_pct)
    frame = extend(frame)
    eligible = frame.direction_return.notna() & frame.direction_return.ne(0)
    eligible &= frame[COLS].notna().all(axis=1) & frame.y.notna()
    data = frame[eligible].copy()
    splits = {
        k: data[data.target_date.between(*manifest[k])].copy()
        for k in ["train", "validation", "calibration", "evaluation"]
    }
    for k, v in splits.items():
        if len(v) < 1000 or v.label.nunique() != 2:
            raise ValueError(f"Insufficient verified {k} samples: {len(v)}")
    train, val, cal, test = [splits[k] for k in ["train", "validation", "calibration", "evaluation"]]
    candidates = {
        "logistic": make_pipeline(
            SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=300, C=0.1, random_state=17)
        ),
        **{
            f"lightgbm{n}": LGBMClassifier(
                n_estimators=180,
                num_leaves=n,
                learning_rate=0.03,
                min_child_samples=500,
                reg_lambda=5,
                n_jobs=4,
                random_state=17,
                verbosity=-1,
            )
            for n in [7, 15]
        },
    }
    scores = {}
    for name, model in candidates.items():
        model.fit(train[COLS], train.label)
        scores[name] = metrics(val.label, model.predict_proba(val[COLS])[:, 1])
        print(json.dumps({"phase": "validation", "candidate": name, **scores[name]}), flush=True)
    selected = min(scores, key=lambda k: scores[k]["brier"])
    model = candidates[selected]

    # No refit after validation; 2023 is reserved exclusively for calibration.
    def logits(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    calibrator = LogisticRegression(C=1000, max_iter=300).fit(
        logits(model.predict_proba(cal[COLS])[:, 1]), cal.label
    )
    joblib.dump({"model": model, "calibrator": calibrator, "cols": COLS}, out / "model.joblib")
    atomic_json(out / "selection.json", {"selected": selected, "validation": scores, "frozen_at": utcnow()})
    p = calibrator.predict_proba(logits(model.predict_proba(test[COLS])[:, 1]))[:, 1]
    prior = float(train.label.mean())
    results = {
        "selected_calibrated": metrics(test.label, p),
        "training_prior": metrics(test.label, np.full(len(test), prior)),
        "fixed_50": metrics(test.label, np.full(len(test), 0.5)),
        "logistic": metrics(test.label, candidates["logistic"].predict_proba(test[COLS])[:, 1]),
    }
    test["probability"] = p
    test["brier_delta"] = (p - test.label) ** 2 - (prior - test.label) ** 2
    ci = block_ci(test.groupby("target_date").brier_delta.mean())
    m = results["selected_calibrated"]
    passed = (
        m["roc_auc"] > 0.52
        and ci[1] < 0
        and all(
            m[k] <= results[b][k] * 0.99
            for b in ["training_prior", "fixed_50"]
            for k in ["brier", "log_loss"]
        )
    )
    test[["symbol", "date", "target_date", "label", "probability"]].to_parquet(
        out / "evaluation.parquet", index=False
    )
    test["regime"] = pd.cut(test.market_range, [-np.inf, 1, 2, np.inf], labels=["calm", "normal", "volatile"])
    importance = (
        model.feature_importances_.tolist()
        if hasattr(model, "feature_importances_")
        else model[-1].coef_[0].tolist()
    )
    report = {
        **manifest,
        "artifact_dir": str(out),
        "selected": selected,
        "validation": scores,
        "evaluation": results,
        "brier_delta_ci": ci,
        "gate_pass": bool(passed),
        "operational_probability_enabled": False,
        "operational_reason": "새 미래 관측 및 시점 정보 검증 필요"
        if passed
        else "기준선 대비 확률 개선 관문 미통과",
        "samples": {k: len(v) for k, v in splits.items()},
        "excluded_flat": int(frame.direction_return.eq(0).sum()),
        "excluded_missing_features_or_labels": int((~eligible & ~frame.direction_return.eq(0)).sum()),
        "importance": dict(zip(COLS, importance)),
        "by_symbol": {str(k): metrics(g.label, g.probability) for k, g in test.groupby("symbol")},
        "by_regime": {
            str(k): metrics(g.label, g.probability) for k, g in test.groupby("regime", observed=True)
        },
        "completed_at": utcnow(),
    }
    atomic_json(out / "report.json", report)
    pointer = root / "reports/platform-direction.json"
    if not pointer.exists():
        atomic_json(pointer, report)
    print(
        json.dumps(
            {k: report[k] for k in ["selected", "evaluation", "gate_pass", "brier_delta_ci", "samples"]},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return report


def public_status(settings):
    r = read_json(settings.data_dir / "reports/platform-direction.json", {})
    return {
        "status": "research_only" if r else "not_evaluated",
        "probability": None,
        "reason": r.get("operational_reason", "과거 확률 검증 전입니다"),
        "metrics": r.get("evaluation", {}),
        "gate_pass": r.get("gate_pass", False),
        "evaluation_period": "2024년 · 보합 제외 · 회고 평가",
        "version": "direction-v1",
    }


if __name__ == "__main__":
    run(local_settings())
