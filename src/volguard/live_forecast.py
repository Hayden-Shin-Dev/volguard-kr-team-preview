"""KRX-trained nested7_tail50 with dated Kiwoom KRX inputs and immutable issuance."""

import json
import math
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .blend_study import blend
from .calibration_study import CALIBRATIONS, raw_predictions, replay
from .config import KST
from .data import ML_REQUIRED_FEATURES, Lake, features, normalize
from .improvement import enrich
from .learning_study import tail_features
from .models import sequential_predictions
from .providers import Kiwoom, request_json
from .providers import signed_price as provider_price
from .store import Store
from .util import atomic_json, digest, exclusive_run, utcnow

_lock = threading.Lock()
SPEC_URL = "https://github.com/Kiwoom-Securities/Kiwoom-REST-API/blob/main/kiwoom/_data/kiwoom_api_spec.json"


class ForecastError(ValueError):
    def __init__(self, code, reason, remedy):
        self.code, self.reason, self.remedy = code, reason, remedy
        super().__init__(f"{reason} 해결: {remedy}")


def fail(code, reason, remedy):
    raise ForecastError(code, reason, remedy)


def signed_price(value):
    try:
        return provider_price(value)
    except (ValueError, TypeError):
        fail(
            "INVALID_PRICE",
            "키움 응답에 0·결측·비정상 가격이 있습니다.",
            "해당 일봉 가격 필드와 거래정지 여부를 확인하고 재조회하세요.",
        )


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def implementation_hash():
    names = [
        "live_forecast.py",
        "data.py",
        "models.py",
        "history.py",
        "improvement.py",
        "learning_study.py",
        "boosting_study.py",
        "calibration_study.py",
        "blend_study.py",
    ]
    return digest({n: digest((Path(__file__).parent / n).read_bytes()) for n in names})


def activate(settings):
    """Explicit user-authorized default; historical reports and artifacts stay intact."""
    root = settings.data_dir
    report = read(root / "reports/ensemble-study.json")
    if not next(r for r in report["results"] if r["candidate"] == "nested7_tail50")["development_pass"]:
        fail("MODEL_NOT_VALIDATED", "개발 검증 기록이 없습니다.", "nested7_tail50 원본 보고서를 복구하세요.")
    components = []
    for stage, name in [("boosting", "leaves7_nested"), ("learning", "tail_features")]:
        source = Path(read(root / f"reports/{stage}-study.json")["artifact_dir"])
        path = source / f"{name}-2023.joblib"
        checkpoint = read(source / f"{name}-2023.json")
        expected = checkpoint["hashes"][path.name]
        if digest(path.read_bytes()) != expected:
            fail("MODEL_HASH", f"{name} 보존 모델 해시가 다릅니다.", "원본 학습 산출물을 복구하세요.")
        components.append({"source": str(path), "name": name, "hash": expected, "fit": checkpoint["fit"]})
    baseline = read(root / "config/release.json")
    version = digest({"components": [c["hash"] for c in components], "code": implementation_hash()})[:20]
    output = root / "runs" / f"live-nested7-{version}"
    output.mkdir(parents=True, exist_ok=True)
    for c in components:
        dest = output / f"{c['name']}.joblib"
        if not dest.exists():
            shutil.copyfile(c["source"], dest)
        c["path"] = str(dest)
    manifest = {
        "model": "nested7_tail50",
        "version": version,
        "activated_at": utcnow(),
        "authorization": "explicit user request: default nested7_tail50; HAR fallback",
        "validation_basis": "development_pass; prospective performance not yet established",
        "development_report_hash": digest(report),
        "components": components,
        "weights": [0.5, 0.5],
        "calibration": ["signed252", "tails252"],
        "baseline_path": str(Path(baseline["artifact_dir"]) / "models.joblib"),
        "baseline_hash": baseline["model_hash"],
        "historical_end": baseline["historical_end"],
        "implementation_hash": implementation_hash(),
        "spec": SPEC_URL,
    }
    atomic_json(output / "manifest.json", manifest)
    atomic_json(root / "config/forecast-release.json", manifest)
    return manifest


def session_pair(sessions, now):
    closed = [
        s
        for s in sessions
        if s.get("hours_verified")
        and datetime.fromisoformat(f"{s['date']}T{s['close_time']}").replace(tzinfo=KST)
        + timedelta(minutes=10)
        <= now
    ]
    if not closed:
        fail("CALENDAR", "확정된 거래일을 찾지 못했습니다.", "KRX 거래일 달력을 갱신하세요.")
    anchor = closed[-1]
    following = next((s for s in sessions if s["date"] > anchor["date"]), None)
    if not following or not following.get("hours_verified"):
        fail(
            "CALENDAR_EXPIRED",
            "다음 거래일의 운영 시간이 확인되지 않았습니다.",
            "KRX 특별장 공지를 반영해 달력을 연장하세요.",
        )
    return anchor, following


def fetch(settings, client, code, symbol, day, adjusted="0"):
    store, lake = Store(settings.db_path), Lake(settings.data_dir)
    key = f"forecast_input:{code}:{symbol}:{day}:{adjusted}"
    cached = store.get("meta", key)
    if cached:
        path = lake.root / cached["path"]
        if path.exists() and digest(path.read_bytes()) == cached["file_hash"]:
            return read(path), cached
    body = (
        {"inds_cd": "001", "base_dt": day.replace("-", "")}
        if code == "ka20006"
        else {"stk_cd": symbol, "base_dt": day.replace("-", ""), "upd_stkpc_tp": adjusted}
    )
    try:
        if not client.token:
            client.authenticate()
        raw = request_json(
            "POST",
            f"https://{client.host}/api/dostk/chart",
            headers={"authorization": f"Bearer {client.token}", "api-id": code},
            json=body,
        )
    except Exception:
        fail(
            "BROKER_CONNECTION",
            "키움 일봉 조회에 실패했습니다.",
            "네트워크·키움 허용 IP·API 연결 상태를 확인하세요. 자동 재시도합니다.",
        )
    if str(raw.get("return_code", 0)) != "0":
        fail(
            "BROKER_REJECTED",
            f"키움 일봉 요청 거부(코드 {raw.get('return_code')}).",
            "API 권한·호출 제한을 확인하세요. 자동 재시도합니다.",
        )
    rows = raw.get("inds_dt_pole_qry" if code == "ka20006" else "stk_dt_pole_chart_qry", [])
    if not rows or rows[0].get("dt") != day.replace("-", ""):
        fail(
            "DAILY_NOT_FINAL",
            f"키움 응답에 {day} 확정 일봉이 없습니다.",
            "일봉 갱신 후 자동 재시도합니다. 거래정지 종목이면 거래 재개가 필요합니다.",
        )
    source_hash = lake.bronze(json.dumps(raw, ensure_ascii=False).encode(), "kiwoom-forecast", body)
    target = lake.root / "broker" / day / f"{code}-{symbol}-{adjusted}-{source_hash[:12]}.json"
    atomic_json(target, raw)
    meta = {
        "path": str(target.relative_to(lake.root)),
        "file_hash": digest(target.read_bytes()),
        "source_hash": source_hash,
        "received_at": utcnow(),
        "request": body,
        "api": code,
    }
    store.put("meta", key, meta)
    time.sleep(0.3)
    return raw, meta


def parse_bars(symbol, raw, adjusted, day, received_at):
    if raw.get("stk_cd") != symbol or adjusted.get("stk_cd") != symbol or len(symbol) != 6:
        fail(
            "SESSION_SYMBOL",
            "요청한 KRX 종목과 응답 종목이 다릅니다.",
            "KRX 코드로 조회하세요. NXT·통합 코드는 예측에 사용할 수 없습니다.",
        )
    records = raw.get("stk_dt_pole_chart_qry", [])
    if len({r["dt"] for r in records}) != len(records):
        fail("DUPLICATE_DATE", "키움 일봉에 중복 날짜가 있습니다.", "해당 일봉 응답을 재수집하세요.")
    adjusted_by_day = {r["dt"]: r for r in adjusted.get("stk_dt_pole_chart_qry", [])}
    out = []
    for r in records:
        if r["dt"] > day.replace("-", ""):
            fail(
                "FUTURE_INPUT",
                "미확정 미래 일봉이 응답에 섞였습니다.",
                "기준일을 마지막 확정 거래일로 제한하세요.",
            )
        prices = {
            k: signed_price(r[v])
            for k, v in {
                "open": "open_pric",
                "high": "high_pric",
                "low": "low_pric",
                "close": "cur_prc",
            }.items()
        }
        volume = float(r["trde_qty"].replace(",", ""))
        amount = float(r["trde_prica"].replace(",", "")) * 1_000_000
        if (
            not all(math.isfinite(v) and v > 0 for v in prices.values())
            or not prices["low"]
            <= min(prices["open"], prices["close"])
            <= max(prices["open"], prices["close"])
            <= prices["high"]
            or not math.isfinite(volume)
            or volume < 0
            or not math.isfinite(amount)
            or amount < 0
        ):
            fail(
                "INVALID_OHLC",
                f"{r['dt']} 가격·거래량 형식이 유효하지 않습니다.",
                "해당 날짜의 원본 일봉을 재조회하세요.",
            )
        a, factor = adjusted_by_day.get(r["dt"]), None
        if a:
            ratios = np.array(
                [
                    signed_price(a[k]) / signed_price(r[k])
                    for k in ["open_pric", "high_pric", "low_pric", "cur_prc"]
                ]
            )
            if (
                np.isfinite(ratios).all()
                and (ratios > 0).all()
                and np.ptp(ratios) <= max(0.002, ratios.mean() * 0.002)
            ):
                factor = float(np.median(ratios))
        out.append(
            dict(
                symbol=symbol,
                date=pd.Timestamp(r["dt"]),
                **prices,
                volume=volume,
                turnover=amount,
                available_at=received_at,
                exchange="KRX",
                halted=volume == 0,
                CMPPREVDD_PRC=r.get("pred_pre"),
                broker_adjustment_factor=factor,
            )
        )
    frame = pd.DataFrame(out).sort_values("date")
    current = frame[frame.date.eq(day)]
    if current.empty or current.iloc[0].volume == 0:
        fail(
            "HALTED",
            "대상 거래일의 거래량이 0이거나 일봉이 없습니다.",
            "거래 재개 후 확정 일봉이 필요합니다.",
        )
    if pd.isna(current.iloc[0].broker_adjustment_factor):
        fail(
            "PRICE_BASIS",
            "최신 원주가·수정주가의 OHLC 배율이 불일치합니다.",
            "수정주가 옵션 0/1 응답과 기업행동 자료를 확인하세요.",
        )
    return frame


def prepare_frame(settings, symbol, day, now, manifest, client):
    lake = Lake(settings.data_dir)
    raw, raw_meta = fetch(settings, client, "ka10081", symbol, day)
    adjusted, adj_meta = fetch(settings, client, "ka10081", symbol, day, "1")
    broker = parse_bars(symbol, raw, adjusted, day, raw_meta["received_at"])
    bars = lake.bars()
    selected = bars[bars.symbol.eq(symbol) & bars.date.le(day)].copy()
    if selected.empty:
        fail("HISTORY_MISSING", "KRX 학습 이력이 없는 종목입니다.", "해당 종목의 KRX 과거 일봉을 수집하세요.")
    historical = selected.date.le(manifest["historical_end"])
    selected.loc[historical, "available_at"] = (
        (selected.loc[historical, "date"] + pd.Timedelta(hours=18)).dt.tz_localize(KST).dt.tz_convert("UTC")
    )
    latest_official = selected.date.max()
    new = broker[broker.date.gt(latest_official)].copy()
    expected = [s["date"] for s in lake.calendar() if str(latest_official.date()) < s["date"] <= day]
    if new.date.dt.strftime("%Y-%m-%d").tolist() != expected:
        fail(
            "MISSING_SESSIONS",
            "KRX 이력과 키움 일봉 사이에 거래일이 누락됐습니다.",
            "누락 구간의 키움 연속 일봉을 수집하세요.",
        )
    factors = broker.tail(23).broker_adjustment_factor
    corporate_action = factors.isna().any() or factors.diff().abs().gt(0.002).any()
    if not new.empty:
        selected = pd.concat([selected, normalize(new)], ignore_index=True)
    day_row = selected[selected.date.eq(day)]
    if day_row.empty or not day_row.iloc[0].valid_label:
        fail(
            "INVALID_SESSION",
            "최신 일봉은 거래정지·결측·오류 상태입니다.",
            "해당 거래일의 거래 상태를 확인하세요.",
        )
    prior, last = selected[selected.date.lt(day)].tail(1).iloc[0], day_row.iloc[0]
    try:
        continuous = (
            abs(float(last.close) - float(str(last.CMPPREVDD_PRC).replace(",", "")) - float(prior.close))
            <= 0.5
        )
    except (AttributeError, TypeError, ValueError):
        continuous = False
    cutoffs = selected.groupby("date").available_at.max()
    scheduled = (cutoffs.index + pd.Timedelta(hours=18, minutes=30)).tz_localize(KST).tz_convert("UTC")
    cutoffs = pd.Series([max(a, b) for a, b in zip(cutoffs, scheduled)], index=cutoffs.index)
    cutoffs.loc[pd.Timestamp(day)] = max(pd.Timestamp(now), pd.Timestamp(raw_meta["received_at"]))
    frame = features(selected, lake.calendar(), issue_cutoffs=cutoffs)
    market = pd.read_parquet(lake.root / "silver/market.parquet")
    market = market[market.date.le(day)].copy()
    market_raw, market_meta = fetch(settings, client, "ka20006", "001", day)
    if market_raw.get("inds_cd") != "001":
        fail("MARKET_CODE", "시장 입력이 코스피 지수가 아닙니다.", "키움 업종코드 001로 조회하세요.")
    market_values = market.set_index("date").range_pct.to_dict()
    for r in market_raw["inds_dt_pole_qry"]:
        dt = pd.Timestamp(r["dt"])
        if dt > market.date.max() and dt <= pd.Timestamp(day):
            high, low = signed_price(r["high_pric"]), signed_price(r["low_pric"])
            if not math.isfinite(high + low) or not 0 < low <= high:
                fail("MARKET_PRICE", "코스피 지수 고저가가 유효하지 않습니다.", "시장 일봉을 재조회하세요.")
            market_values[dt] = 100 * (high / low - 1)
    frame["market_range"] = frame.date.map(market_values)
    frame = tail_features(enrich(frame))
    if corporate_action or not continuous:
        frame.loc[frame.date.eq(day), ["return_1", "gap", "vol_22"]] = np.nan
    if corporate_action:
        frame.loc[frame.date.eq(day), ["volume_ratio", "turnover_ratio"]] = np.nan
    current = frame[frame.date.eq(day)]
    missing = [c for c in ML_REQUIRED_FEATURES if current.empty or current[c].isna().any()]
    if missing:
        fail(
            "FEATURE_MISSING",
            f"필수 피처 계산 불가: {', '.join(missing)}.",
            "최근 22거래일 일봉 및 코스피 지수 누락을 확인하세요.",
        )
    evidence = {
        "source": "Kiwoom ka10081 KRX/raw",
        "session": "KRX regular daily chart",
        "price_adjustment": "raw OHLC; adjusted response checked; uncertain returns masked",
        "corporate_action_or_unknown": bool(corporate_action),
        "reference_continuous": bool(continuous),
        "masked_features": [
            c
            for c in ["return_1", "gap", "vol_22", "volume_ratio", "turnover_ratio"]
            if current[c].isna().any()
        ],
        "sources": [raw_meta, adj_meta, market_meta],
        "market_source": "KOSPI 001",
        "last_bar": {k: float(last[k]) for k in ["open", "high", "low", "close", "volume"]},
        "features": {
            k: None if pd.isna(v) else float(v)
            for k, v in current.iloc[0].items()
            if k
            in set(
                ML_REQUIRED_FEATURES
                + ["return_1", "gap", "volume_ratio", "turnover_ratio", "market_range5", "market_range22"]
            )
        },
    }
    return frame[frame.date.le(day) & frame.target_date.ge("2024-01-01")], evidence


def infer(manifest, frame, day, target):
    baseline_path = Path(manifest["baseline_path"])

    def values(rows):
        rows = rows[rows.date.eq(day)]
        if rows.empty or not np.isfinite(rows[["low", "median", "high"]].to_numpy()).all():
            return None
        r = rows.iloc[-1]
        return {**{k: float(r[k]) for k in ["low", "median", "high"]}, "calibration_n": int(r.calibration_n)}

    har_result, parts, fallback = None, [], None
    try:
        if digest(baseline_path.read_bytes()) == manifest["baseline_hash"]:
            har_result = values(sequential_predictions(joblib.load(baseline_path), frame, "har", target))
    except Exception:
        # A failed comparison model must not prevent a valid default ML forecast.
        pass
    try:
        for component in manifest["components"]:
            path = Path(component["path"])
            if digest(path.read_bytes()) != component["hash"]:
                raise ValueError("MODEL_COMPONENT_HASH")
            model = joblib.load(path)
            raw = raw_predictions(model, frame, "ml")
            raw = raw.merge(
                frame[["symbol", "date", "issue_at"]], on=["symbol", "date"], validate="one_to_one"
            )
            parts.append(blend(*(replay(raw, CALIBRATIONS[c], target) for c in manifest["calibration"])))
        ml_result = values(blend(*parts))
        if ml_result is None:
            raise ValueError("ML_CALIBRATION_SAMPLE_OR_SYMBOL")
    except Exception as exc:
        ml_result = None
        fallback = {
            "code": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
            "reason": "ML 추론·보정에 실패해 대체 계산을 사용합니다.",
            "remedy": "모델 해시·지원 종목·120개 이상 보정 표본을 점검하세요.",
        }
    if ml_result is None and har_result is None:
        fail(
            "NO_VALID_MODEL",
            "ML과 대체 모델 모두 유효한 구간을 만들지 못했습니다.",
            "모델 파일과 최근 252개 보정 오차를 확인하세요.",
        )
    return ml_result, har_result, fallback


def calculate(settings, symbol, now=None):
    now = now or datetime.now(KST)
    lake, store = Lake(settings.data_dir), Store(settings.db_path)
    if symbol not in {m["symbol"] for m in lake.universe()["members"]}:
        fail("UNKNOWN_SYMBOL", "지원하지 않는 종목입니다.", "고정 관심종목 목록에서 선택하세요.")
    anchor, following = session_pair(lake.calendar(), now)
    day, target = anchor["date"], following["date"]
    saved = store.prediction(symbol, target)
    if saved:
        return saved
    opened = datetime.fromisoformat(f"{target}T{following['open_time']}").replace(tzinfo=KST)
    if now >= opened:
        fail(
            "TARGET_ALREADY_OPEN",
            "오늘 장 시작 전에 발행된 예측이 없습니다.",
            "오늘 마감 후 다음 거래일 예측이 자동 생성됩니다.",
        )
    with _lock, exclusive_run(lake.root / "broker-forecast.lock"):
        saved = store.prediction(symbol, target)
        if saved:
            return saved
        manifest = read(lake.root / "config/forecast-release.json")
        if manifest["implementation_hash"] != implementation_hash():
            fail(
                "CODE_VERSION",
                "검증한 예측 코드와 실행 버전이 다릅니다.",
                "테스트 완료 후 모델 연결 버전을 다시 활성화하세요.",
            )
        forward_path = lake.root / "forward" / f"{target}-{manifest['version']}-{symbol}.json"
        if forward_path.exists():
            archived = read(forward_path).get("issued_prediction")
            if archived:
                store.publish(archived)
                return store.prediction(symbol, target)
        try:
            frame, evidence = prepare_frame(
                settings, symbol, day, now, manifest, Kiwoom(settings.environment)
            )
            ml, har, fallback = infer(manifest, frame, day, target)
            issued = datetime.now(KST)
            if issued >= opened:
                fail(
                    "ISSUANCE_TOO_LATE",
                    "계산 도중 대상 거래일이 시작됐습니다.",
                    "다음 마감 배치에서 새 예측을 발행합니다.",
                )
            result = {
                "symbol": symbol,
                "anchor_date": day,
                "issued_for": day,
                "target_date": target,
                **(ml or har),
                "method": "ml" if ml else "har",
                "model_name": "nested7_tail50" if ml else "HAR fallback",
                "model_version": manifest["version"],
                "generated_at": issued.isoformat(),
                "status": "ready",
                "coverage_target": 0.9,
                "nominal_coverage": 0.9,
                "demo": False,
                "data_version": digest(evidence),
                "validation_basis": manifest["validation_basis"],
                "fallback": fallback,
                "evidence": evidence,
                "comparison": {"ml": ml, "har": har},
            }
            paired = [
                {"symbol": symbol, "method": method, **value}
                for method, value in [("ml", ml), ("har", har)]
                if value
            ]
            empirical = frame[frame.date.eq(day)].iloc[0]
            if np.isfinite([empirical.emp_low, empirical.emp_mid, empirical.emp_high]).all():
                paired.append(
                    {
                        "symbol": symbol,
                        "method": "empirical",
                        "low": float(empirical.emp_low),
                        "median": float(empirical.emp_mid),
                        "high": float(empirical.emp_high),
                    }
                )
            if not forward_path.exists():
                atomic_json(
                    forward_path,
                    {
                        "target_date": target,
                        "issued_at": issued.isoformat(),
                        "model_hash": manifest["version"],
                        "baseline_model_hash": manifest["baseline_hash"],
                        "predictions": paired,
                        "issued_prediction": result,
                    },
                )
            store.publish(result)
            store.put("meta", f"forecast_error:{symbol}", None)
            return store.prediction(symbol, target)
        except ForecastError as exc:
            store.put(
                "meta",
                f"forecast_error:{symbol}",
                {"code": exc.code, "reason": exc.reason, "remedy": exc.remedy, "at": utcnow()},
            )
            if exc.code in {
                "SESSION_SYMBOL",
                "DUPLICATE_DATE",
                "FUTURE_INPUT",
                "INVALID_OHLC",
                "INVALID_PRICE",
                "PRICE_BASIS",
                "MARKET_CODE",
                "MARKET_PRICE",
            }:
                for code, item, adjustment in [
                    ("ka10081", symbol, "0"),
                    ("ka10081", symbol, "1"),
                    ("ka20006", "001", "0"),
                ]:
                    store.put("meta", f"forecast_input:{code}:{item}:{day}:{adjustment}", None)
            raise


def run_batch(settings, symbols=None):
    store = Store(settings.db_path)
    symbols = symbols or [r["symbol"] for r in store.watchlist()]
    results, failures = {}, {}
    for symbol in symbols:
        try:
            result = calculate(settings, symbol)
            results[symbol] = {k: result[k] for k in ["target_date", "model_name", "low", "median", "high"]}
        except ForecastError as exc:
            failures[symbol] = {"code": exc.code, "reason": exc.reason, "remedy": exc.remedy}
        except Exception as exc:
            failures[symbol] = {
                "code": type(exc).__name__,
                "reason": "예측 처리 중 내부 오류가 발생했습니다.",
                "remedy": "예측 실행 기록과 모델 파일을 점검하세요.",
            }
    report = {
        "generated_at": utcnow(),
        "status": "success" if not failures else "partial" if results else "failed",
        "data_ready": not failures,
        "predictions": results,
        "failures": failures,
    }
    atomic_json(settings.data_dir / "reports/broker-forecast.json", report)
    store.put("meta", "broker_forecast", report)
    return report
