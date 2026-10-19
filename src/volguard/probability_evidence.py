"""Chronological probability calibration with dated, auditable reliability evidence."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .new_information_sources import output
from .platform_study import local_settings
from .util import atomic_json, digest, utcnow

EDGES = [0, .4, .45, .5, .55, .6, .65, .7, .75, 1]


def fit_calibrator(frame, method, cutoff):
    available = pd.to_datetime(frame.target_available_at, utc=True)
    if (available >= pd.Timestamp(cutoff)).any():
        raise ValueError("Calibration outcome unavailable at cutoff")
    if method == "raw":
        return None
    x = np.clip(frame.raw_p.to_numpy(), 1e-6, 1-1e-6)
    if method == "sigmoid":
        return LogisticRegression(C=1, random_state=1707).fit(np.log(x/(1-x))[:, None], frame.label)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(x, frame.label)
    raise ValueError("Unknown calibration method")


def predict(frame, model, method):
    x = np.clip(frame.raw_p.to_numpy(), 1e-6, 1-1e-6)
    if method == "sigmoid":
        x = model.predict_proba(np.log(x/(1-x))[:, None])[:, 1]
    elif method == "isotonic":
        x = model.predict(x)
    return np.clip(x, 1e-6, 1-1e-6)


def metrics(frame, p):
    return dict(n=len(frame), brier=float(brier_score_loss(frame.label, p)),
                log_loss=float(log_loss(frame.label, p, labels=[0, 1])),
                accuracy=float(np.mean((p > .5) == frame.label)),
                auc=float(roc_auc_score(frame.label, p)))


def reliability(frame, draws=2000):
    """Resample contiguous dates jointly across stocks; preserve empty bin dates."""
    f = frame.copy()
    f["bin"] = np.minimum(np.searchsorted(EDGES, f.p_up, side="right")-1, len(EDGES)-2)
    dates = sorted(f.target_date.unique())
    rng = np.random.default_rng(1707)
    block = min(20, len(dates))
    starts = rng.integers(0, len(dates)-block+1, (draws, int(np.ceil(len(dates)/block))))
    indices = (starts[:, :, None]+np.arange(block)).reshape(draws, -1)[:, :len(dates)]
    results = []
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        g = f[f.bin == i]
        row = dict(id=f"P{i}", low=lo, high=hi, n=len(g), dates=int(g.target_date.nunique()),
                   coverage=len(g)/len(f), status="insufficient", examples=[])
        if len(g):
            daily = g.assign(count=1).groupby("target_date").agg(
                n=("count", "sum"), up=("label", "sum"), p=("p_up", "sum")
            ).reindex(dates, fill_value=0).to_numpy()
            sampled = daily[indices].sum(axis=1)
            valid = sampled[:, 0] > 0
            rates = sampled[valid, 1]/sampled[valid, 0]
            gaps = (sampled[valid, 1]-sampled[valid, 2])/sampled[valid, 0]
            ci = np.quantile(gaps, [.025, .975]).tolist()
            halves = [g[g.target_date.dt.month <= 6], g[g.target_date.dt.month > 6]]
            stable = all(len(h) >= 100 and h.target_date.nunique() >= 15 and
                         abs(float((h.label-h.p_up).mean())) <= .05 for h in halves)
            support = len(g) >= 500 and g.target_date.nunique() >= 60 and valid.mean() >= .95
            row.update(mean_probability=float(g.p_up.mean()), actual_up=float(g.label.mean()),
                       actual_down=float(1-g.label.mean()), gap=float((g.label-g.p_up).mean()),
                       rate_ci95=np.quantile(rates, [.025, .975]).tolist(), gap_ci95=ci,
                       status="supported_research" if support and stable and ci[0] >= -.05 and ci[1] <= .05
                       else "not_supported" if support else "insufficient",
                       halves=[dict(n=len(h), dates=int(h.target_date.nunique()),
                                    gap=float((h.label-h.p_up).mean()) if len(h) else None) for h in halves],
                       examples=[dict(symbol=str(r.symbol), feature_date=str(r.date.date()),
                                      target_date=str(r.target_date.date()), probability=float(r.p_up),
                                      actual="상승" if r.label else "하락",
                                      return_pct=float(r.direction_return)*100)
                                 for r in g.sort_values(["target_date", "symbol"]).tail(8).itertuples()])
        results.append(row)
    return results


def run():
    root = local_settings().data_dir
    folder = root / "runs/probability-evidence-20261017"
    folder.mkdir(exist_ok=True)
    paths = sorted((output()/"global").glob("raw-baseline-*.parquet"))
    if len(paths) != 12:
        raise ValueError("Require 2021-2023 quarterly chronological baseline predictions")
    manifest = dict(source={str(p): digest(p.read_bytes()) for p in paths},
                    code_hash=digest(Path(__file__).read_bytes()),
                    protocol="2021 calibration, 2022 selection, refit 2021-22, frozen 2023 reused development evaluation",
                    edges=EDGES, tolerance=.05, min_rows=500, min_dates=60,
                    block_days=20, bootstrap=2000, independent_test=False,
                    scope="P(up | next-day return is nonzero); unchanged baseline, no new features")
    reg = folder/"protocol.json"
    if reg.exists() and json.loads(reg.read_text("utf8"))["manifest"] != manifest:
        raise ValueError("Protocol changed: use a new run")
    if not reg.exists():
        atomic_json(reg, dict(created_at=utcnow(), manifest=manifest))
    data = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    if data.duplicated(["symbol", "target_date"]).any() or not (data.fit_end <= data.date).all():
        raise ValueError("Invalid chronological predictions")
    train = data[data.target_date.dt.year == 2021]
    select = data[data.target_date.dt.year == 2022]
    candidates = []
    for method in ["raw", "sigmoid", "isotonic"]:
        model = fit_calibrator(train, method, "2022-01-01T00:00:00+09:00")
        candidates.append(dict(method=method, **metrics(select, predict(select, model, method))))
    base = candidates[0]
    valid = [r for r in candidates if r["brier"] <= base["brier"] and r["log_loss"] <= base["log_loss"]]
    chosen = min(valid, key=lambda r: r["log_loss"])["method"]
    selection_path = folder/"selection.json"
    if selection_path.exists():
        old = json.loads(selection_path.read_text("utf8"))
        if old["selected"] != chosen or old["candidates"] != candidates:
            raise ValueError("Frozen selection changed")
    else:
        atomic_json(selection_path, dict(created_at=utcnow(), selected=chosen, candidates=candidates))
    calibration = data[data.target_date.dt.year < 2023]
    model = fit_calibrator(calibration, chosen, "2023-01-01T00:00:00+09:00")
    test = data[data.target_date.dt.year == 2023].copy()
    test["p_up"] = predict(test, model, chosen)
    test["p_down"] = 1-test.p_up
    test.to_parquet(folder/"predictions.parquet", index=False)
    joblib.dump(dict(model=model, method=chosen, research_only=True), folder/"calibrator.joblib")
    bins = reliability(test)
    report = dict(status="complete", created_at=utcnow(), selected=chosen,
                  evaluation="2023년 재사용 개발 평가 · 독립 시험 아님", scope="보합 제외 상승·하락 조건부 확률",
                  baseline=metrics(test, test.raw_p), calibrated=metrics(test, test.p_up),
                  prior=metrics(test, test.prior), selection=candidates, bins=bins,
                  raw_bins=reliability(test.assign(p_up=test.raw_p)),
                  supported_coverage=sum(r["coverage"] for r in bins if r["status"] == "supported_research"),
                  evidence_id="probability-evidence-20261017", input_hash=digest(manifest["source"]),
                  predictions_hash=digest((folder/"predictions.parquet").read_bytes()),
                  protocol=manifest,
                  limitations=["연구 표본의 보정 근거이며 개별 종목·미래 성능 보장이 아닙니다.",
                               "20거래일 블록으로 같은 날짜의 종목을 함께 재표집한 점별 95% 구간입니다. 다중 구간 동시 보장이 아닙니다.",
                               "보합은 제외했습니다. 상승+하락=100%는 보합 제외 조건부 확률입니다.",
                               "지원 구간은 오차 ±5%p·500건·60거래일·반기 안정성의 사전 고정 연구 기준입니다.",
                               "변동폭 모델의 90% 구간은 방향 비조건부입니다. 상승·하락별 검증된 구간으로 표시하지 않습니다.",
                               "실제 방향 확률 자동 활성화·모델 승격은 하지 않습니다. 독립 전향 검증은 별도입니다."])
    atomic_json(folder/"report.json", report)
    atomic_json(root/"reports/probability-evidence.json", report)
    print(json.dumps({k: report[k] for k in ["selected", "baseline", "calibrated", "supported_coverage"]}))
    lines = ["# 방향 확률 보정과 근거 검증", report["evaluation"],
             "2021년 확률 보정 → 2022년 Brier·Log Loss 동시 비악화 후보 중 Log Loss 선택 → 2021~2022년 재보정 → 2023년 고정 검증. 기존 모델·발행 예측은 변경하지 않는다.",
             f"선택: {chosen}. 기존: {report['baseline']}. 보정: {report['calibrated']}.",
             "| 상승확률 구간 | 표본 | 거래일 | 평균 예측 | 실제 상승 | 오차 95% 구간 | 판정 |",
             "|---|---:|---:|---:|---:|---|---|"]
    for r in bins:
        if r["n"]:
            lines.append(f"| {r['low']:.0%}~{r['high']:.0%} | {r['n']} | {r['dates']} | {r['mean_probability']:.2%} | {r['actual_up']:.2%} | {r['gap_ci95']} | {r['status']} |")
        else:
            lines.append(f"| {r['low']:.0%}~{r['high']:.0%} | 0 | 0 | — | — | — | 표본 부족 |")
    lines += ["", *report["limitations"], "", f"원본 예측·보정기·선택·프로토콜·보고서: {folder}"]
    Path("docs/probability-evidence-report.md").write_text("\n\n".join(lines).replace("|\n\n|", "|\n|"), encoding="utf8")


if __name__ == "__main__":
    run()
