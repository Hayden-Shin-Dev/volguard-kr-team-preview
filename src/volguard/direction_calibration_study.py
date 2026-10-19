"""Actual refits with disjoint train/calibration/selection/final retrospective years."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import Lake, features
from .direction import COLS, extend
from .direction_historical_reliability import compare, draws, scores
from .direction_study import EXTRA, enhance, logits
from .event_direction import freeze
from .platform_study import local_settings
from .technical_features import GROUPS, build
from .util import atomic_json, digest, utcnow

METHODS = ["raw", "sigmoid", "beta", "isotonic", "market_sigmoid"]
TECH = [c for c in COLS + EXTRA if c != "rsi14"] + sum(GROUPS.values(), [])
MODELS = {"initial_refit": COLS, "technical_full": TECH, "technical_recent": TECH}


def design(p, frame, method):
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    if method == "beta":
        return np.column_stack([np.log(p), -np.log1p(-p)])
    x = logits(p)
    if method == "market_sigmoid":
        state = frame.market_momentum.ge(0).to_numpy(dtype=float)
        return np.column_stack([x[:, 0], state, x[:, 0] * state])
    return x


def fit_calibration(frame, p, method):
    if not frame.target_date.dt.year.eq(2022).all():
        raise ValueError("Only 2022 outcomes may fit probability calibration")
    if method == "raw":
        return None
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(p, frame.label)
    return LogisticRegression(C=1000 if method == "sigmoid" else 1, max_iter=500, random_state=1721).fit(
        design(p, frame, method), frame.label
    )


def calibrated(model, frame, p, method):
    if method == "raw":
        return np.clip(p, 1e-6, 1 - 1e-6)
    if method == "isotonic":
        return np.clip(model.predict(p), 1e-6, 1 - 1e-6)
    return np.clip(model.predict_proba(design(p, frame, method))[:, 1], 1e-6, 1 - 1e-6)


def selection(results):
    baseline = "initial_refit__sigmoid"
    b = results[baseline]
    eligible = [k for k, r in results.items() if r["brier"] <= b["brier"] and r["log_loss"] <= b["log_loss"]]

    def support(key):
        return sum(
            x["n"]
            for x in results[key]["bins"]
            if x["side"] == "up" and x["low"] >= 0.6 and x["status"] == "compatible_retrospective"
        )

    strong = [k for k in eligible if support(k)]
    # Explicitly prefer credible stronger up probabilities only within non-worse proper scores.
    pool = strong or eligible
    chosen = min(pool, key=lambda k: (-support(k), results[k]["brier"], results[k]["log_loss"]))
    return dict(selected=chosen, baseline=baseline, eligible=eligible, strong_up_candidates=strong)


def dataset(root, out):
    path = out / "features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    bars = pd.read_parquet(root / "silver/bars.parquet", filters=[("date", "<", pd.Timestamp("2025-01-01"))])
    bars["available_at"] = (
        (bars.date + pd.Timedelta(hours=18)).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
    )
    sessions = [s for s in Lake(root).calendar() if s["date"] < "2025-01-01"]
    print("Building causal daily and technical features", flush=True)
    frame = enhance(extend(features(bars, sessions)))
    market = pd.read_parquet(
        root / "silver/market.parquet", filters=[("date", "<", pd.Timestamp("2025-01-01"))]
    )
    frame["market_range"] = frame.date.map(market.drop_duplicates("date").set_index("date").range_pct)
    frame = frame.merge(build(frame, bars, "standard"), on=["symbol", "date"], validate="one_to_one")
    columns = sorted(set(TECH + COLS + ["market_momentum"]))
    keep = frame[columns].notna().all(axis=1) & frame.ta_ready.eq(True)
    keep &= frame.direction_return.notna() & frame.direction_return.ne(0) & frame.y.notna()
    f = frame.loc[keep, ["symbol", "date", "target_date", "target_available_at", "label"] + columns].copy()
    f.to_parquet(path, index=False)
    return f


def run():
    root = local_settings().data_dir
    out = root / "runs/direction-calibration-study-20261017"
    out.mkdir(exist_ok=True)
    protected = {
        str(p): digest(p.read_bytes())
        for p in [
            root / "config/direction-forward.json",
            root / "config/forecast-release.json",
            *list((root / "forward").glob("*.json")),
        ]
    }
    freeze(
        out / "protocol.json",
        dict(
            train="2010-2021",
            calibration="2022",
            selection="2023",
            final="2024 reused retrospective, not a new independent test",
            methods=METHODS,
            models=MODELS,
            recent="2017-2021 train, 730-day half-life; all other recipes full history",
            baseline="same split and common training rows: original logistic C=.1 + sigmoid C=1000",
            selection_rule="Brier AND LL <= baseline; prefer supported up >=60 bins; max supported samples then minimum Brier/LL; otherwise minimum Brier/LL",
            support="fixed deciles; >=500 rows, >=60 dates; whole date-block 95% gap interval within +/-5pp",
            code_hash=digest(Path(__file__).read_bytes()),
            protected=protected,
            source_bars_hash=digest((root / "silver/bars.parquet").read_bytes()),
        ),
    )
    f = dataset(root, out)
    splits = {
        "train": f[f.target_date.lt("2022-01-01")],
        "calibration": f[f.target_date.dt.year.eq(2022)],
        "selection": f[f.target_date.dt.year.eq(2023)],
        "final": f[f.target_date.dt.year.eq(2024)],
    }
    if any(len(g) < 1000 for g in splits.values()):
        raise ValueError("Insufficient chronological split")
    if not (
        pd.to_datetime(splits["train"].target_available_at, utc=True)
        < pd.Timestamp("2022-01-01", tz="Asia/Seoul")
    ).all():
        raise ValueError("Training labels unavailable at training cutoff")
    audit = {
        k: dict(
            n=len(g),
            start=str(g.target_date.min().date()),
            end=str(g.target_date.max().date()),
            symbols=int(g.symbol.nunique()),
            dates=int(g.target_date.nunique()),
        )
        for k, g in splits.items()
    }
    atomic_json(out / "split-audit.json", audit)
    development, objects = {}, {}
    dev = splits["selection"]
    dev_dates = sorted(dev.target_date.unique())
    dev_indices = draws(dev_dates)
    for name, cols in MODELS.items():
        train = splits["train"]
        if name == "technical_recent":
            train = train[train.target_date.ge("2017-01-01")]
        model = make_pipeline(
            SimpleImputer(), StandardScaler(), LogisticRegression(C=0.1, max_iter=500, random_state=17)
        )
        weights = (
            np.ones(len(train))
            if name != "technical_recent"
            else np.exp2(-(pd.Timestamp("2022-01-01") - train.target_date).dt.days.to_numpy() / 730)
        )
        print("Training", name, len(train), len(cols), flush=True)
        with threadpool_limits(limits=3):
            model.fit(train[cols], train.label, logisticregression__sample_weight=weights)
            cal_raw = model.predict_proba(splits["calibration"][cols])[:, 1]
            dev_raw = model.predict_proba(dev[cols])[:, 1]
            for method in METHODS:
                calibrator = fit_calibration(splits["calibration"], cal_raw, method)
                p = calibrated(calibrator, dev, dev_raw, method)
                key = name + "__" + method
                development[key] = scores(dev, p, dev_dates, dev_indices)
                objects[key] = dict(model=model, calibrator=calibrator, method=method, cols=cols)
                joblib.dump(objects[key], out / f"{key}.joblib")
                dev[["symbol", "date", "target_date", "label"]].assign(p_up=p).to_parquet(
                    out / f"development-{key}.parquet", index=False
                )
                print("Calibrated", key, development[key]["brier"], flush=True)
    chosen = selection(development)
    freeze(out / "selection.json", dict(**chosen, development=development))
    print("FROZEN selection", chosen["selected"], flush=True)
    # No final-year predictions or metrics are computed before selection is saved.
    test = splits["final"]
    dates = sorted(test.target_date.unique())
    indices = draws(dates)
    results, predictions = {}, {}
    # Evaluate the baseline and chosen recipe only: no final-year candidate shopping.
    for key in dict.fromkeys([chosen["baseline"], chosen["selected"]]):
        obj = objects[key]
        with threadpool_limits(limits=3):
            p = calibrated(
                obj["calibrator"], test, obj["model"].predict_proba(test[obj["cols"]])[:, 1], obj["method"]
            )
        predictions[key] = p
        results[key] = scores(test, p, dates, indices)
        test[["symbol", "date", "target_date", "label"]].assign(p_up=p).to_parquet(
            out / f"final-{key}.parquet", index=False
        )
    base, picked = predictions[chosen["baseline"]], predictions[chosen["selected"]]
    paired = compare(test, base, picked, dates, indices)
    halves = {}
    for yearhalf, mask in {
        "first": test.target_date.dt.month.le(6),
        "second": test.target_date.dt.month.gt(6),
    }.items():
        g = test[mask]
        ds = sorted(g.target_date.unique())
        halves[yearhalf] = {k: scores(g, p[mask.to_numpy()], ds, draws(ds)) for k, p in predictions.items()}
    archived = pd.read_parquet(
        root / "runs/direction-historical-reliability-20261017/matched-predictions.parquet"
    )
    joined = (
        test[["symbol", "target_date", "label"]]
        .assign(selected=picked, baseline=base)
        .merge(
            archived[["symbol", "target_date", "initial", "technical_expanding", "label"]].rename(
                columns={"label": "archived_label"}
            ),
            on=["symbol", "target_date"],
            validate="one_to_one",
        )
    )
    if not joined.label.eq(joined.archived_label).all():
        raise ValueError("Archived label mismatch")
    ds = sorted(joined.target_date.unique())
    ids = draws(ds)
    references = {
        k: scores(joined, joined[k].to_numpy(), ds, ids)
        for k in ["selected", "baseline", "initial", "technical_expanding"]
    }
    report = dict(
        status="complete_research_only",
        at=utcnow(),
        selected=chosen["selected"],
        baseline=chosen["baseline"],
        split_audit=audit,
        development=development,
        selection=chosen,
        results=results,
        comparison=paired,
        halves=halves,
        archived_comparison=references,
        independent=False,
        production_changed=False,
        model_fit_count=3,
        evaluated_recipes=len(development),
        limitations=[
            "2024년은 학습·보정·선택과 분리됐지만 과거 다른 실험에서 재사용한 회고 평가입니다.",
            "2024년 결과를 보고 후보·보정·확률 구간을 변경하지 않았습니다.",
            "보합 제외 조건부 확률이며 일별 공개 시각 가정·수정 이력·생존편향의 한계가 남습니다.",
            "구간별 95% 구간은 다중 비교에 대한 동시 보장이 아닙니다. 운영 모델과 전향 예측은 유지합니다.",
        ],
    )
    assert all(digest(Path(p).read_bytes()) == h for p, h in protected.items())
    atomic_json(out / "report.json", report)
    atomic_json(root / "reports/direction-calibration-study.json", report)
    print(json.dumps(dict(selected=chosen["selected"], audit=audit, comparison=paired)), flush=True)


if __name__ == "__main__":
    run()
