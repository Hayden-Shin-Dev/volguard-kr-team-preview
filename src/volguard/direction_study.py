"""Direction development experiments. Never reads 2024+ or enables live probabilities."""

import json
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Lake, features
from .direction import COLS, block_ci, extend, metrics
from .platform_study import local_settings
from .util import atomic_json, digest, utcnow

EXTRA = [
    "scaled_return",
    "scaled_momentum",
    "rsi14",
    "up_fraction",
    "return_rank",
    "momentum_rank",
    "market_return",
    "market_breadth",
    "market_dispersion",
    "market_momentum",
    "relative_return",
    "market_volatility",
    "range_ratio",
]
CANDIDATES = {
    "original_logistic": {"kind": "logistic", "rich": False, "recent": False},
    "relative_logistic": {"kind": "logistic", "rich": True, "recent": False},
    "recent_logistic": {"kind": "logistic", "rich": True, "recent": True},
    "relative_tree7": {"kind": "tree", "leaves": 7, "rich": True, "recent": False},
    "recent_tree7": {"kind": "tree", "leaves": 7, "rich": True, "recent": True},
    "recent_tree15": {"kind": "tree", "leaves": 15, "rich": True, "recent": True},
}


def enhance(frame):
    """Only contemporaneously known returns; no labels enter market/rank aggregates."""
    frame = frame.copy()
    market = (
        frame.groupby("date")
        .return_1.agg(
            market_return="mean",
            market_dispersion="std",
            market_breadth=lambda r: r.dropna().gt(0).mean(),
        )
        .sort_index()
    )
    market["market_momentum"] = np.expm1(np.log1p(market.market_return).rolling(5).sum())
    market["market_volatility"] = market.market_return.rolling(22).std()
    for c in market:
        frame[c] = frame.date.map(market[c])
    frame["return_rank"] = frame.groupby("date").return_1.rank(pct=True)
    frame["momentum_rank"] = frame.groupby("date").return_20.rank(pct=True)
    frame["relative_return"] = frame.return_1 - frame.market_return
    frame["scaled_return"] = frame.return_1 / frame.vol_22.replace(0, np.nan)
    frame["scaled_momentum"] = frame.return_20 / (frame.vol_22.replace(0, np.nan) * np.sqrt(20))
    frame["range_ratio"] = frame.range_1 / frame.range_22.replace(0, np.nan)
    for _, g in frame.groupby("symbol"):
        g = g.sort_values("date")
        gain = g.return_1.clip(lower=0).rolling(14).mean()
        loss = -g.return_1.clip(upper=0).rolling(14).mean()
        frame.loc[g.index, "rsi14"] = gain / (gain + loss).replace(0, np.nan)
        signs = g.return_1.gt(0).astype(float).where(g.return_1.notna())
        frame.loc[g.index, "up_fraction"] = signs.rolling(10).mean()
    return frame.replace([np.inf, -np.inf], np.nan)


def logits(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibration_bins(y, p):
    table = pd.DataFrame({"y": y, "p": p})
    table["bin"] = pd.cut(table.p, np.linspace(0, 1, 11), include_lowest=True)
    return [
        dict(n=len(g), predicted=float(g.p.mean()), observed=float(g.y.mean()))
        for _, g in table.groupby("bin", observed=True)
    ]


def run(settings):
    root = settings.data_dir
    out = root / "runs" / ("direction-dev-" + uuid.uuid4().hex[:10])
    out.mkdir()
    manifest = dict(
        created_at=utcnow(),
        folds=[2021, 2022, 2023],
        candidates=CANDIDATES,
        features=COLS + EXTRA,
        source_hash=digest(Path(__file__).read_bytes()),
        evaluation="reused development only; 2024+ never opened",
        selection="minimum pooled Brier; selection-biased development comparison only",
        training="through fold-2; preceding full year sigmoid calibration; fold evaluation",
        recent="5 calendar years; exponential sample weight half-life 2 years",
        gate="AUC > .52; Brier and log loss >=1% better than both priors; block CI upper <0",
        operational_probability_enabled=False,
    )
    atomic_json(out / "manifest.json", manifest)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    bars = pd.read_parquet(root / "silver/bars.parquet", filters=[("date", "<", pd.Timestamp("2024-01-01"))])
    manifest["input_hash"] = digest(pd.util.hash_pandas_object(bars, index=False).values.tobytes())
    atomic_json(out / "manifest.json", manifest)
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
    # Identical evaluation rows for fair model comparisons.
    eligible &= frame[COLS + EXTRA].notna().all(axis=1)
    data = frame[eligible].copy()
    predictions, fold_metrics = [], []
    for year in manifest["folds"]:
        train = data[data.target_date < f"{year - 1}-01-01"]
        cal = data[data.target_date.between(f"{year - 1}-01-01", f"{year - 1}-12-31")]
        test = data[data.target_date.between(f"{year}-01-01", f"{year}-12-31")]
        if min(len(train), len(cal), len(test)) < 1000:
            raise ValueError("Insufficient fold samples")
        prior = float(train.label.mean())
        for name, spec in CANDIDATES.items():
            fit = train[train.target_date >= f"{year - 6}-01-01"] if spec["recent"] else train
            cols = COLS + EXTRA if spec["rich"] else COLS
            if spec["kind"] == "logistic":
                model = make_pipeline(
                    StandardScaler(), LogisticRegression(C=0.1, max_iter=500, random_state=17)
                )
            else:
                model = LGBMClassifier(
                    n_estimators=180,
                    num_leaves=spec["leaves"],
                    learning_rate=0.03,
                    min_child_samples=500,
                    reg_lambda=10,
                    n_jobs=4,
                    random_state=17,
                    verbosity=-1,
                )
            fit_args = {}
            if spec["recent"]:
                age = (fit.target_date.max() - fit.target_date).dt.days / 365.25
                fit_args[
                    "logisticregression__sample_weight" if spec["kind"] == "logistic" else "sample_weight"
                ] = np.exp2(-age / 2)
            model.fit(fit[cols], fit.label, **fit_args)
            calibrator = LogisticRegression(C=1000, max_iter=300).fit(
                logits(model.predict_proba(cal[cols])[:, 1]), cal.label
            )
            p = calibrator.predict_proba(logits(model.predict_proba(test[cols])[:, 1]))[:, 1]
            row = test[["symbol", "date", "target_date", "label"]].copy()
            row = row.assign(candidate=name, fold=year, probability=p, prior=prior)
            predictions.append(row)
            result = dict(
                candidate=name, fold=year, train_n=len(fit), calibration_n=len(cal), **metrics(test.label, p)
            )
            fold_metrics.append(result)
            print(json.dumps({k: result[k] for k in ["candidate", "fold", "brier", "roc_auc"]}), flush=True)
            if year == manifest["folds"][-1]:
                joblib.dump(
                    dict(model=model, calibrator=calibrator, cols=cols, research_only=True),
                    out / f"{name}.joblib",
                )
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_parquet(out / "predictions.parquet", index=False)
    baseline = pred[pred.candidate.eq("original_logistic")].copy()
    baseline["error"] = (baseline.probability - baseline.label) ** 2
    rows = []
    for name, g in pred.groupby("candidate"):
        m = metrics(g.label, g.probability)
        common = g.merge(
            baseline[["symbol", "target_date", "error"]], on=["symbol", "target_date"], validate="one_to_one"
        )
        delta = (common.probability - common.label) ** 2 - common.error
        ci = block_ci(delta.groupby(common.target_date).mean())
        prior_metrics = metrics(g.label, g.prior)
        fixed_metrics = metrics(g.label, np.full(len(g), 0.5))
        prior_delta = (g.probability - g.label) ** 2 - (g.prior - g.label) ** 2
        prior_ci = block_ci(prior_delta.groupby(g.target_date).mean())
        gate = (
            m["roc_auc"] > 0.52
            and prior_ci[1] < 0
            and all(
                m[k] <= b[k] * 0.99 for b in [prior_metrics, fixed_metrics] for k in ["brier", "log_loss"]
            )
        )
        rows.append(
            dict(
                candidate=name,
                **m,
                brier_delta_ci=ci,
                prior_delta_ci=prior_ci,
                development_gate_pass=bool(gate),
                calibration_bins=calibration_bins(g.label, g.probability),
            )
        )
    best = min(rows, key=lambda r: r["brier"])
    base = next(r for r in rows if r["candidate"] == "original_logistic")
    report = dict(
        **manifest,
        artifact_dir=str(out),
        results=rows,
        fold_results=fold_metrics,
        selected=best["candidate"],
        brier_improvement_pct=100 * (1 - best["brier"] / base["brier"]),
        baseline=base,
        completed_at=utcnow(),
        priors=dict(training_prior=prior_metrics, fixed_50=fixed_metrics),
        limitations=[
            "Reused development data and multiple candidates; no new final test or promotion",
            "Current-universe survivorship bias; retrospective availability and revision assumptions",
            "Direction conditional on nonflat close; cross-sectional returns use same-day known bars",
        ],
    )
    atomic_json(out / "report.json", report)
    atomic_json(root / "reports/platform-direction-development.json", report)
    print(
        json.dumps({k: report[k] for k in ["selected", "brier_improvement_pct", "artifact_dir"]}), flush=True
    )
    return report


if __name__ == "__main__":
    run(local_settings())
