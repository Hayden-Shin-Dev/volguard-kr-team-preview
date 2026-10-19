"""Fixed event-conditioned probabilities; research only, never promotes live models."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .platform_study import local_settings
from .util import atomic_json, digest, utcnow

EVENTS = {
    "market_down": "동일가중 시장 대용 수익률 ≤ −2%",
    "market_up": "동일가중 시장 대용 수익률 ≥ +2%",
    "abnormal_down": "과거 베타로 제거한 시장 잔차 ≤ 과거 표준편차의 −2배",
    "abnormal_up": "과거 베타로 제거한 시장 잔차 ≥ 과거 표준편차의 +2배",
    "volume_down": "직전 20일 평균 대비 거래량 ≥ 2배, 당일 수익률 < 0",
    "volume_up": "직전 20일 평균 대비 거래량 ≥ 2배, 당일 수익률 > 0",
    "gap_down": "전일 종가 대비 당일 시가 갭 ≤ −2%",
    "gap_up": "전일 종가 대비 당일 시가 갭 ≥ +2%",
}


def event_features(context):
    """Compute before outcome filtering; do not read targets or range forecasts."""
    columns = ["symbol", "date", "return_1", "market_return", "volume", "gap", "body_pct"]
    parts = []
    for _, raw in context[columns].groupby("symbol", sort=True):
        g = raw.sort_values("date").copy()
        r, m = g.return_1, g.market_return
        beta = (r.rolling(60, min_periods=60).cov(m) /
                m.rolling(60, min_periods=60).var().replace(0, np.nan)).shift(1)
        residual = r - beta*m
        g["event_residual_z"] = residual / residual.shift(1).rolling(60, min_periods=60).std().replace(0, np.nan)
        g["event_volume_ratio"] = g.volume / g.volume.shift(1).rolling(20, min_periods=20).mean().replace(0, np.nan)
        g["gap_valid"] = np.isclose((1+g.gap)*(1+g.body_pct/100)-1, r, rtol=1e-6, atol=1e-8)
        parts.append(g)
    return pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def event_masks(f):
    return {
        "market_down": f.market_return.le(-.02),
        "market_up": f.market_return.ge(.02),
        "abnormal_down": f.event_residual_z.le(-2),
        "abnormal_up": f.event_residual_z.ge(2),
        "volume_down": f.event_volume_ratio.ge(2) & f.return_1.lt(0),
        "volume_up": f.event_volume_ratio.ge(2) & f.return_1.gt(0),
        "gap_down": f.gap_valid & f.gap.le(-.02),
        "gap_up": f.gap_valid & f.gap.ge(.02),
    }


def fit_probabilities(frame):
    """100-observation prior fixed in advance, not fitted to confirmation results."""
    prior = float(frame.label.mean())
    return {name: dict(p_up=float((frame.loc[mask, "label"].sum()+100*prior)/(mask.sum()+100)),
                       fit_n=int(mask.sum()), fit_dates=int(frame.loc[mask, "target_date"].nunique()))
            for name, mask in event_masks(frame).items()}


def losses(y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return (p-y)**2, -(y*np.log(p)+(1-y)*np.log1p(-p))


def evaluate(frame, probabilities, draws=4000):
    dates = np.sort(frame.target_date.unique())
    rng = np.random.default_rng(171017)
    block = min(20, len(dates))
    starts = rng.integers(0, len(dates)-block+1, (draws, int(np.ceil(len(dates)/block))))
    indices = (starts[:, :, None]+np.arange(block)).reshape(draws, -1)[:, :len(dates)]
    result = {}
    for name, mask in event_masks(frame).items():
        g = frame.loc[mask].copy()
        p = probabilities[name]["p_up"]
        row = dict(definition=EVENTS[name], p_up=p, p_down=1-p, predicted="up" if p>.5 else "down",
                   n=len(g), dates=int(g.target_date.nunique()), row_coverage=len(g)/len(frame),
                   day_coverage=g.target_date.nunique()/len(dates), development_candidate=False)
        result[name] = row
        if g.empty:
            continue
        y, baseline = g.label.to_numpy(), g.baseline_p.to_numpy()
        hit = ((p>.5)==y).astype(float)
        bhit = ((baseline>.5)==y).astype(float)
        brier, logloss = losses(y, p)
        bbrier, blogloss = losses(y, baseline)
        gap = hit-max(p, 1-p)
        values = pd.DataFrame(dict(target_date=g.target_date.to_numpy(), n=1., up=y,
                                   hit=hit, accuracy_delta=hit-bhit, brier_delta=brier-bbrier,
                                   log_loss_delta=logloss-blogloss, calibration_gap=gap))
        daily = values.groupby("target_date").sum().reindex(dates, fill_value=0).to_numpy()
        sums = daily[indices].sum(axis=1)
        valid = sums[:, 0]>0
        samples = sums[valid, 1:]/sums[valid, 0, None]
        keys = list(values.columns)[2:]
        cis = dict(zip(keys, np.quantile(samples, [.025, .975], axis=0).T.tolist()))
        # Familywise correction for 8 events x 3 superiority claims, approximate bootstrap intervals.
        alpha = .05/(len(EVENTS)*3)
        corrected = dict(zip(keys, np.quantile(samples, [alpha/2, 1-alpha/2], axis=0).T.tolist()))
        range_valid = np.isfinite(g[["range_forecast_low", "range_forecast_high", "y"]]).all(axis=1)
        rg = g.loc[range_valid]
        covered = rg.y.between(rg.range_forecast_low, rg.range_forecast_high).to_numpy()
        row.update(up_frequency=float(y.mean()), down_frequency=float(1-y.mean()), accuracy=float(hit.mean()),
                   brier=float(brier.mean()), log_loss=float(logloss.mean()),
                   baseline_accuracy=float(bhit.mean()), baseline_brier=float(bbrier.mean()), baseline_log_loss=float(blogloss.mean()),
                   calibration_gap=float(gap.mean()), ci95=cis, familywise_ci=corrected,
                   bootstrap_nonempty_fraction=float(valid.mean()), range_n=len(rg),
                   range_coverage=float(covered.mean()) if len(rg) else None,
                   joint_accuracy=float((covered & ((p>.5)==rg.label)).mean()) if len(rg) else None,
                   baseline_joint_accuracy=float((covered & ((rg.baseline_p>.5)==rg.label)).mean()) if len(rg) else None)
        row["development_candidate"] = bool(len(g)>=300 and g.target_date.nunique()>=60 and valid.mean()>=.95
            and max(p, 1-p)>=.6 and corrected["accuracy_delta"][0]>0
            and corrected["brier_delta"][1]<0 and corrected["log_loss_delta"][1]<0
            and cis["calibration_gap"][0]>=-.05 and cis["calibration_gap"][1]<=.05)
    return result


def freeze(path, payload):
    """Never overwrite a protocol or selection after inspecting subsequent outcomes."""
    if path.exists():
        if json.loads(path.read_text("utf8"))["payload"] != payload:
            raise ValueError("Frozen event protocol differs; create a new research version")
    else:
        atomic_json(path, dict(created_at=utcnow(), payload=payload))


def run():
    root = local_settings().data_dir
    out = root/"runs/event-direction-20261017"
    out.mkdir(exist_ok=True)
    context_path = root/"runs/direction-math-20261017/context.parquet"
    baseline_path = root/"runs/selective-direction-20261017/predictions.parquet"
    protocol = dict(events=EVENTS, source_hash=digest(context_path.read_bytes()),
                    baseline_hash=digest(baseline_path.read_bytes()), code_hash=digest(Path(__file__).read_bytes()),
                    fit="2015-2020", selection="2021-2022", diagnostic="2023 REUSED, not independent",
                    probability="event empirical up frequency shrunk by 100 rows toward pooled fit frequency",
                    intervals="20-date moving blocks, 4000 resamples; 8 events x 3 claims Bonferroni approximate bounds",
                    gate="n>=300, dates>=60, confidence>=60%, familywise accuracy/Brier/logloss improvement; calibration gap CI within +/-5pp",
                    independent="pending future timely-issued observations after protocol freeze; no historical independent claims",
                    operational_enabled=False)
    freeze(out/"protocol.json", protocol)
    context = pd.read_parquet(context_path)
    if context.date.max()>=pd.Timestamp("2024-01-01"):
        raise ValueError("Consumed locked periods cannot be reopened")
    event = event_features(context)
    eligible = context[context.direction_return.notna() & context.direction_return.ne(0) & context.label.isin([0, 1])
                       & context.target_date.lt("2024-01-01") & context.date.ge("2015-01-01")]
    labels = ["symbol", "date", "target_date", "target_available_at", "label", "direction_return"]
    full = event.merge(eligible[labels], on=["symbol", "date"], validate="one_to_one")
    train = full[full.target_date.lt("2021-01-01")]
    initial = fit_probabilities(train)
    base = pd.read_parquet(baseline_path, filters=[("candidate", "==", "logistic__raw")])
    merged = base.merge(event, on=["symbol", "date"], validate="one_to_one")
    dev = merged[merged.target_date.lt("2023-01-01")]
    cutoff = (pd.Timestamp(dev.date.min())+pd.Timedelta(hours=18, minutes=30)).tz_localize("Asia/Seoul")
    if pd.to_datetime(train.target_available_at, utc=True).max()>cutoff:
        raise ValueError("Fit outcomes cross development prediction cutoff")
    development = evaluate(dev, initial)
    selected = [name for name, r in development.items() if r["development_candidate"]]
    final = fit_probabilities(full[full.target_date.lt("2023-01-01")])
    freeze(out/"selection.json", dict(initial=initial, final=final, selected=selected))
    # Only now consume the already-reused diagnostic period; never optimize on its outcomes.
    diagnostic = merged[merged.target_date.ge("2023-01-01")]
    results = evaluate(diagnostic, final)
    masks = event_masks(diagnostic)
    union = np.logical_or.reduce([masks[n].to_numpy() for n in selected]) if selected else np.zeros(len(diagnostic), dtype=bool)
    report = dict(status="development_complete_independent_pending", generated_at=utcnow(), protocol=protocol,
                  initial_probabilities=initial, frozen_probabilities=final, selected=selected,
                  development=development, diagnostic=results,
                  denominators=dict(development_rows=len(dev), development_dates=int(dev.target_date.nunique()),
                                    diagnostic_rows=len(diagnostic), diagnostic_dates=int(diagnostic.target_date.nunique())),
                  selected_union_coverage=float(union.mean()), operational_enabled=False,
                  independent=dict(status="not_available", n=0, earliest_target_date="2026-10-19",
                                   requires="prediction archived before target session; original source availability proof; frozen protocol; no backfill"),
                  limitations=["2023년 재사용 탐색 결과이며 독립 검증이 아니다. 2024·2025~2026년 시험도 이미 소비됐다.",
                               "상승/하락은 보합 제외 조건부이다. 행 비율과 사건 발생 거래일 비율을 분리한다.",
                               "시장 수익률은 현재 고정 종목군 동일가중 대용값이며 공식 코스피 지수가 아니다. 생존편향이 있다.",
                               "당시 원본 공개 시각·가격조정 검증이 부족한 재구성 자료다. 회고적 이용 가능 시각 가정이다.",
                               "거래량 급증은 주문 방향 불균형이 아니다. 사건은 겹칠 수 있어 개별 발생 비율을 더하지 않는다.",
                               "변동폭은 기존 시간순 OOF 예측을 그대로 사용한 비조건부 구간이다. 동시 적중률은 기술통계다.",
                               "강한 신호 관문은 개발 후보 선정용이며 미사용 기간 통과나 자동 운영 승격을 뜻하지 않는다."])
    atomic_json(out/"report.json", report)
    atomic_json(root/"reports/event-direction.json", report)
    for period, frame, probs in [("development", dev, initial), ("diagnostic_reused", diagnostic, final)]:
        records = []
        for name, mask in event_masks(frame).items():
            records.append(frame.loc[mask, list(base.columns)].assign(event=name, event_p_up=probs[name]["p_up"]))
        pd.concat(records, ignore_index=True).to_parquet(out/f"{period}.parquet", index=False)
    print(json.dumps(dict(selected=selected, diagnostic=results), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
