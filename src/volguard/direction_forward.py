"""Isolated research-only close-to-close inference, immutable issuance and scoring."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import KST
from .data import Lake, features, normalize
from .direction import extend
from .direction_contract import CONTRACT, aware, outcome, record, window
from .direction_study import enhance, logits
from .store import Store
from .terminal_data import read_json
from .util import atomic_json, digest, exclusive_run, utcnow


class Unavailable(ValueError):
    pass


def implementation_hash():
    names = [
        "direction_forward.py",
        "direction_contract.py",
        "direction_reliability.py",
        "data.py",
        "direction.py",
        "direction_study.py",
        "technical_features.py",
    ]
    return digest({name: digest((Path(__file__).parent / name).read_bytes()) for name in names})


def install(settings):
    """Pin existing models without retraining, reselection, or production activation."""
    root = settings.data_dir
    path = root / "config/direction-forward.json"
    if path.exists():
        existing = read_json(path)
        # Initial installation migration only, before any actual forecast exists.
        if "implementation_hash" not in existing and not list(
            (root / "direction-forward").glob("????-??-??/*.json")
        ):
            existing["implementation_hash"] = implementation_hash()
            atomic_json(path, existing)
        return existing
    specs = [
        (
            "technical_expanding",
            "기술적 지표 연구 모델",
            "technical-history-20261017/pooled-technical__expanding.joblib",
        ),
        (
            "baseline",
            "기존 방향 기준 모델",
            "technical-history-20261017/pooled-base__old2015_weighted5.joblib",
        ),
        ("initial", "초기 방향 모델 · 보정", "direction-v1/model.joblib"),
    ]
    models = []
    for key, name, relative in specs:
        source = root / "runs" / relative
        raw = source.read_bytes()
        h = digest(raw)
        target = root / "direction-forward/models" / f"{h}.joblib"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as f:
                f.write(raw)
        model = joblib.load(target)
        cols = list(model["model"].feature_names_in_)
        models.append(
            dict(
                id=key,
                name=name,
                hash=h,
                path=str(target),
                cols=cols,
                version=f"{key}-{h[:16]}",
                calibration="sigmoid" if "calibrator" in model else "raw",
                fitted_history="2021 학습·2023 확률 보정"
                if key == "initial"
                else "2023년 4분기 직전 고정 학습",
            )
        )
    protocol = dict(
        contract=CONTRACT,
        models=models,
        installed_at=utcnow(),
        probability_edges=[i / 10 for i in range(11)],
        min_rows=500,
        min_dates=60,
        calibration_tolerance=0.05,
        bootstrap_block=20,
        bootstrap_draws=1000,
        research_only=True,
        implementation_hash=implementation_hash(),
        automatic_promotion=False,
        automatic_retraining=False,
    )
    atomic_json(path, protocol)
    atomic_json(root / "config/direction-prospective-contract-v2.json", CONTRACT)
    return protocol


def fetch_day(settings, day):
    """Archive exact official exchange payload; never overwrite an issued input."""
    from .providers import KRX

    root = settings.data_dir / "direction-forward/sources"
    root.mkdir(parents=True, exist_ok=True)
    client = KRX()
    stock, index = client.daily(day), client.daily(day, index=True)
    received = datetime.now(KST).isoformat()
    if not stock.get("OutBlock_1"):
        raise Unavailable("KRX 공식 일봉 미도착: 당일 19:00·20:00 자동 재시도")
    # KRX.normalize checks the stock response date. Index must pass the same test.
    frame = normalize(KRX.normalize(stock, day))
    rows = [r for r in index.get("OutBlock_1", []) if r.get("IDX_NM") == "코스피"]
    if len(rows) != 1:
        raise Unavailable("코스피 지수 일봉 미도착: 다음 배치에서 재시도")
    idx = rows[0]
    if str(idx.get("BAS_DD", "")).replace("/", "").replace("-", "") != day.replace("-", ""):
        raise Unavailable("코스피 지수 기준일 오류: 요청 거래일과 응답 기준일 확인 필요")
    hi, lo = [float(str(idx[k]).replace(",", "")) for k in ["HGPRC_IDX", "LWPRC_IDX"]]
    if not np.isfinite([hi, lo]).all() or not 0 < lo <= hi:
        raise Unavailable("코스피 지수 고저가 오류: 공급자 데이터 확인 필요")
    payload = dict(day=day, received_at=received, stocks=stock, index=index)
    h = digest(payload)
    atomic_json(root / f"{h}.json", payload)
    frame["source_hash"] = h
    frame["available_at"] = pd.Timestamp(received)
    return frame, dict(day=day, hash=h, received_at=received, market_range=100 * (hi / lo - 1))


def prepare(settings, day, current, source):
    """Reconstruct only inputs observable at this close, retaining real receipt time."""
    from .technical_features import build

    lake = Lake(settings.data_dir)
    symbols = {m["symbol"] for m in lake.universe()["members"]}
    current = current[current.symbol.isin(symbols)].copy()
    missing = symbols - set(current.symbol)
    if missing:
        raise Unavailable(f"시장 단면 입력 {len(missing)}종목 누락: 고정 유니버스 전체 KRX 일봉 필요")
    # Preserve source vintage before feature computation. Historical data quality limits remain disclosed.
    historical = lake.bars()
    historical = historical[historical.date.lt(pd.Timestamp(day)) & historical.symbol.isin(symbols)]
    bars = pd.concat([historical, current], ignore_index=True).sort_values(["symbol", "date"])
    sessions = [s for s in lake.calendar() if s["date"] <= day]
    index = pd.DatetimeIndex(pd.to_datetime([s["date"] for s in sessions]))
    received = pd.Timestamp(source["received_at"])
    # Every historical row is used as known NOW, not claimed known on its historical date.
    cutoffs = pd.Series(received, index=index)
    frame = enhance(extend(features(bars, sessions, issue_cutoffs=cutoffs)))
    market = pd.read_parquet(lake.root / "silver/market.parquet")
    market = market[market.date.lt(pd.Timestamp(day))].copy()
    if "available_at" in market:
        market = market[pd.to_datetime(market.available_at, utc=True).le(received)]
    ranges = market.drop_duplicates("date").set_index("date").range_pct
    ranges.loc[pd.Timestamp(day)] = source["market_range"]
    frame["market_range"] = frame.date.map(ranges)
    ta = build(frame, bars, "standard")
    frame = frame.merge(ta, on=["symbol", "date"], how="left", validate="one_to_one")
    latest = frame[frame.date.eq(pd.Timestamp(day))].copy()
    # Durable, isolated source snapshot permits exact reconstruction without touching Silver.
    archive = lake.root / "direction-forward/inputs" / source["hash"]
    archive.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(archive / "bars.parquet", index=False)
    market.to_parquet(archive / "market.parquet", index=False)
    latest.to_parquet(archive / "features.parquet", index=False)
    h = digest((archive / "features.parquet").read_bytes())
    atomic_json(
        archive / "manifest.json",
        dict(
            source=source,
            feature_hash=h,
            calendar=sessions,
            bars_hash=digest((archive / "bars.parquet").read_bytes()),
            market_hash=digest((archive / "market.parquet").read_bytes()),
            historical_vintage_limitation=True,
            build_at=utcnow(),
        ),
    )
    return latest, dict(feature_hash=h, snapshot=str(archive))


def infer(frame, spec):
    path = Path(spec["path"])
    if digest(path.read_bytes()) != spec["hash"]:
        raise Unavailable("모델 파일 무결성 오류: 고정 연구 모델 파일 복구 필요")
    missing = [c for c in spec["cols"] if c not in frame or not np.isfinite(frame[c]).all()]
    if missing:
        raise Unavailable(
            "피처 미확보: " + ", ".join(missing) + ". 가격 연속성·거래정지·지표 준비 기간 확인 필요"
        )
    model = joblib.load(path)
    with threadpool_limits(limits=2):
        p = model["model"].predict_proba(frame[spec["cols"]])[:, 1]
        if spec["calibration"] == "sigmoid":
            p = model["calibrator"].predict_proba(logits(p))[:, 1]
    return p


def issue(settings, day, *, preview=False):
    now = datetime.now(KST)
    lake = Lake(settings.data_dir)
    calendar = {"sessions": lake.calendar()}
    dates = window(calendar, day)
    if not preview and (day != now.date().isoformat() or now < aware(dates["feature_cutoff"])):
        raise Unavailable("당일 정규장 마감 후에만 발행 가능: 과거 예측은 소급 발행하지 않음")
    protocol = read_json(settings.data_dir / "config/direction-forward.json")
    if not protocol or protocol["contract"] != CONTRACT:
        raise Unavailable("고정 연구 설정 없음 또는 계약 변경: direction_forward install 필요")
    if protocol.get("implementation_hash") != implementation_hash():
        raise Unavailable("고정 추론 코드 변경 감지: 기존 예측을 보존하고 별도 연구 버전 검토 필요")
    current, source = fetch_day(settings, day)
    frame, snapshot = prepare(settings, day, current, source)
    symbols = [m["symbol"] for m in Store(settings.db_path).watchlist()]
    results = []
    for symbol in symbols:
        rows = frame[frame.symbol.eq(symbol)]
        close = current[current.symbol.eq(symbol)]
        for spec in protocol["models"]:
            r = dict(symbol=symbol, model=spec["id"], version=spec["version"])
            try:
                if len(rows) != 1 or len(close) != 1 or not bool(close.valid_label.iloc[0]):
                    raise Unavailable("종목 정규장 유효 일봉 없음: 거래정지·결측 여부 확인 필요")
                probability = float(infer(rows, spec)[0])
                r.update(p_up=probability, p_down=1 - probability)
                if preview:
                    r["status"] = "preview_not_issued"
                else:
                    evidence = [
                        dict(
                            kind="krx_regular_daily",
                            session="regular",
                            source_hash=source["hash"],
                            observed_at=dates["feature_cutoff"],
                            available_at=source["received_at"],
                        )
                    ]
                    path = record(
                        settings.data_dir,
                        calendar,
                        day,
                        symbol,
                        spec["version"],
                        probability,
                        evidence,
                        details=dict(
                            model=spec["id"],
                            model_hash=spec["hash"],
                            anchor_close=float(close.close.iloc[0]),
                            **snapshot,
                        ),
                    )
                    r.update(status="issued", path=str(path))
            except FileExistsError:
                # Never report newly recomputed probabilities as the already-issued ones.
                r = dict(symbol=symbol, model=spec["id"], status="already_issued")
            except Unavailable as exc:
                r.update(status="unavailable", reason=str(exc))
            results.append(r)
    return dict(
        status="preview" if preview else "completed",
        anchor_date=day,
        target_date=dates["target_date"],
        at=utcnow(),
        results=results,
    )


def score_entry(entry, target, now):
    """Only raw continuous KRX closes; action discontinuities await verified adjustment."""
    anchor = entry["details"]["anchor_close"]
    change = pd.to_numeric(str(target.get("CMPPREVDD_PRC", "")).replace(",", ""), errors="coerce")
    comparable = bool(
        target.get("valid_label", False)
        and target.get("exchange") == "KRX"
        and np.isfinite(change)
        and abs(float(target["close"]) - float(change) - anchor) <= 0.5
    )
    return outcome(
        entry,
        anchor,
        target["close"],
        target_date=str(pd.Timestamp(target["date"]).date()),
        comparable=comparable,
        now=now,
    )


def evaluate(settings):
    now = datetime.now(KST)
    folder = settings.data_dir / "direction-forward"
    cache, pending = {}, []
    for path in sorted(folder.glob("????-??-??/*.json")):
        entry = read_json(path)
        dest = folder / "outcomes" / path.name
        # Model+symbol key repeats across dates: include target date in outcome path.
        dest = dest.parent / entry["target_date"] / dest.name
        if dest.exists() or now < aware(entry["target_close"]):
            continue
        try:
            day = entry["target_date"]
            if day not in cache:
                cache[day] = fetch_day(settings, day)
            frame, source = cache[day]
            target = frame[frame.symbol.eq(entry["symbol"])]
            if len(target) != 1:
                raise Unavailable("対象 종목 종가 없음: 거래정지 또는 공식 일봉 누락 확인")
            result = score_entry(entry, target.iloc[0], now)
            with exclusive_run(folder / "score.lock"):
                if not dest.exists():
                    atomic_json(
                        dest,
                        dict(
                            **result,
                            scored_at=utcnow(),
                            target_date=day,
                            prediction_hash=digest(path.read_bytes()),
                            source_hash=source["hash"],
                        ),
                    )
        except (ValueError, RuntimeError) as exc:
            pending.append(
                dict(
                    symbol=entry["symbol"],
                    target_date=entry["target_date"],
                    reason=str(exc)
                    if isinstance(exc, Unavailable)
                    else "종가 가격 기준·기업행동 또는 공급자 응답 확인 필요",
                )
            )
    atomic_json(folder / "scoring-status.json", dict(at=utcnow(), pending=pending))
    return pending


def run(settings, day):
    """Failure-isolated hook: direction research never interrupts range operations."""
    root = settings.data_dir / "direction-forward"
    with exclusive_run(root / "batch.lock"):
        try:
            evaluate(settings)
        except Exception as exc:
            atomic_json(root / "scoring-status.json", dict(at=utcnow(), error=type(exc).__name__))
        try:
            result = issue(settings, day)
        except Exception as exc:
            result = dict(
                status="unavailable",
                anchor_date=day,
                at=utcnow(),
                reason=str(exc)
                if isinstance(exc, Unavailable)
                else f"방향 연구 배치 오류({type(exc).__name__}): 공급자·모델·입력 자료 점검 필요",
            )
        atomic_json(root / "batch-status.json", result)
        from .direction_reliability import report

        return report(settings)


if __name__ == "__main__":
    from .platform_study import local_settings

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["install", "run", "preview", "evaluate"])
    parser.add_argument("--date", default=datetime.now(KST).date().isoformat())
    args = parser.parse_args()
    settings = local_settings()
    if args.command == "install":
        result = install(settings)
        print(json.dumps({"models": [m["id"] for m in result["models"]], "research_only": True}))
    elif args.command == "preview":
        result = issue(settings, args.date, preview=True)
        atomic_json(settings.data_dir / "reports/direction-forward-preview.json", result)
        print(json.dumps(result, ensure_ascii=False))
    elif args.command == "run":
        print(json.dumps(run(settings, args.date), ensure_ascii=False))
    else:
        evaluate(settings)
        from .direction_reliability import report

        print(json.dumps(report(settings), ensure_ascii=False))
