"""Read-only views for the interactive terminal; no model fitting or issuance."""

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import KST
from .data import Lake
from .display_quotes import display_bar, display_current
from .portfolio import current_result
from .realtime import gauge, within_session
from .store import Store
from .util import digest, utcnow


@lru_cache(maxsize=4)
def parquet(path, modified):
    return pd.read_parquet(path)


def frame(path):
    # Visitors share a read-only market-data link; cache the actual file once.
    path = Path(path).resolve()
    return parquet(str(path), path.stat().st_mtime_ns) if path.exists() else pd.DataFrame()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else default


def clean(value):
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def history(settings, symbol, limit=180):
    bars = frame(settings.data_dir / "silver/bars.parquet")
    if bars.empty:
        return []
    rows = bars[bars.symbol.eq(symbol)].sort_values("date").tail(limit)
    rows = rows[(rows[["open", "high", "low", "close"]] > 0).all(axis=1)]
    result = []
    for r in rows.itertuples():
        if r.high < max(r.open, r.close) or r.low > min(r.open, r.close):
            continue
        result.append(
            {
                "time": r.date.strftime("%Y-%m-%d"),
                **{k: float(getattr(r, k)) for k in ("open", "high", "low", "close", "volume")},
            }
        )
    return result


def historical_prediction(settings, symbol):
    release = read_json(settings.data_dir / "config/release.json", {})
    if not release.get("artifact_dir"):
        return None
    path = Path(release["artifact_dir"]) / "predictions.parquet"
    data = frame(path)
    if data.empty:
        return None
    rows = data[data.symbol.eq(symbol) & data.method.eq("ml")].sort_values("target_date")
    if rows.empty:
        return None
    r = rows.iloc[-1]
    return {
        "low": float(r.low),
        "median": float(r["median"]),
        "high": float(r.high),
        "target_date": r.target_date.strftime("%Y-%m-%d"),
        "anchor_date": r.date.strftime("%Y-%m-%d"),
        "actual": float(r.y),
        "method": "ml",
        "status": "retrospective",
        "coverage_target": 0.9,
        "model_version": release.get("model_hash"),
        "calibration_n": int(r.calibration_n),
        "label": "잠금 평가에 보존된 과거 ML 예측 · 현재 거래에 사용 불가",
    }


def display_prediction(store, symbol, today, upcoming):
    with store.connect() as db:
        latest = db.execute(
            "SELECT payload FROM predictions WHERE symbol=? ORDER BY target_date DESC LIMIT 1", (symbol,)
        ).fetchone()
    latest = json.loads(latest[0]) if latest else None
    return latest if latest and latest["target_date"] in {today, upcoming} else None


def model_comparison(settings):
    original = read_json(settings.data_dir / "reports/test.json", {})
    improved = read_json(settings.data_dir / "reports/ensemble-study.json", {})
    candidate = next((r for r in improved.get("results", []) if r.get("candidate") == "nested7_tail50"), {})
    comparison = candidate.get("versus_frozen_har", {})
    metrics = comparison.get("metrics", {})
    har, ml = metrics.get("har", {}), metrics.get("ml", {})
    return {
        "original_improvement": original.get("improvement"),
        "original_coverage": original.get("metrics", {}).get("ml", {}).get("coverage"),
        "candidate": candidate.get("candidate"),
        "development_pass": improved.get("development_pass", False),
        "coverage": ml.get("coverage"),
        "width_reduction": 1 - ml["width"] / har["width"] if har.get("width") else None,
        "score_reduction": 1 - ml["score"] / har["score"] if har.get("score") else None,
        "evaluation_period": "2020·2022·2023년 개발 평가",
        "status": "기본 ML 적용 · 미래 성능 관찰 중"
        if (settings.data_dir / "config/forecast-release.json").exists()
        else "개발 검증 통과 · 별도 미래 검증·운영 승격 대기"
        if improved.get("development_pass")
        else "개발 검증 확인 필요",
    }


def stock(settings, symbol):
    lake, store = Lake(settings.data_dir), Store(settings.db_path)
    members = {m["symbol"]: m["name"] for m in lake.universe()["members"]}
    if symbol not in members:
        raise ValueError("고정 유니버스에 없는 종목입니다")
    now = datetime.now(KST)
    today = now.date().isoformat()
    sessions = lake.calendar()
    rows = history(settings, symbol, 540)
    quote = store.get("quotes", symbol)
    pred = store.prediction(symbol, today)
    upcoming = next((s["date"] for s in sessions if s["date"] > today), None)
    display = display_prediction(store, symbol, today, upcoming)
    from .live_forecast import ForecastError, session_pair

    try:
        _, next_forecast = session_pair(sessions, now)
        forecast_target = next_forecast["date"]
    except ForecastError:
        forecast_target = upcoming
    broker_bar = display_bar(store, symbol, now)
    chart_rows = list(rows)
    if broker_bar and (not rows or broker_bar["time"] > rows[-1]["time"]):
        chart_rows.append({k: broker_bar[k] for k in ("time", "open", "high", "low", "close", "volume")})
    else:
        broker_bar = None  # The official bar wins for an already available date.
    release = read_json(settings.data_dir / "config/release.json", {})
    report = read_json(settings.data_dir / "reports/test.json", {})
    active = read_json(settings.data_dir / "config/forecast-release.json", {})
    batch = store.get("meta", f"batch:{today}", {})
    state = gauge(quote, pred, store.get("meta", "connection", {}).get("confirmed_at"), now, sessions)
    live_candle = None
    if (
        quote
        and quote.get("open")
        and quote.get("complete")
        and quote.get("trade_date") == today
        and quote.get("exchange") == "KRX"
        and quote.get("session") == "regular"
        and state["price_freshness"]["status"] == "fresh"
        and state["connection_freshness"]["status"] == "fresh"
        and within_session(now, sessions)
    ):
        live_candle = {
            "time": today,
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["price"],
            "volume": quote["volume"],
        }
    return {
        "symbol": symbol,
        "name": members[symbol],
        "candles": rows,
        "chart_candles": chart_rows,
        "display_bar": broker_bar,
        "display_current": display_current(store, symbol, now),
        "display_refresh": store.get("meta", f"display_attempt:{symbol}", {}),
        "model_comparison": model_comparison(settings),
        "forecast_status": {
            "operating_model": active.get("model", release.get("selected", "observation_only")),
            "ml_approved": bool(active),
            "official_as_of": rows[-1]["time"] if rows else None,
            "batch_status": batch.get("status", "not_run"),
            "data_ready": batch.get("data_ready", False),
            "reason": (
                "키움 확정 일봉으로 AI 예측 자동 생성 중"
                if active and not display
                else "AI 예측 발행 완료"
                if active and display and display.get("status") == "ready"
                else "오늘 KRX 공식 일봉 미확보로 다음 거래일 예측 발행 대기"
                if batch and not batch.get("data_ready") and (not rows or rows[-1]["time"] < today)
                else "현재 대상 거래일에 발행된 예측 없음"
                if not display
                else "발행 예측의 검증 상태를 확인하세요"
                if display.get("status") != "ready"
                else "운영 예측 발행 완료"
            ),
            "ml_reason": "개선 ML 기본 적용 · 개발 검증 통과, 미래 성능 관찰 중"
            if active
            else "기존 ML 검증 관문 통과"
            if report.get("ml_pass")
            else "기존 ML 점수 개선 4.34% < 기준 5% · 개선 후보의 개발 평가와 별도",
        },
        "quote": quote,
        "live_candle": live_candle,
        "gauge": state,
        "gauge_prediction": pred,
        "prediction": display,
        "historical_prediction": historical_prediction(settings, symbol),
        "next_session": upcoming,
        "forecast_target": forecast_target,
        "forecast_error": store.get("meta", f"forecast_error:{symbol}"),
        "market_open": within_session(now, sessions),
        "as_of": rows[-1]["time"] if rows else None,
        "price_basis": (
            "KRX 과거 일봉 · 키움 KRX 최신 일봉" if broker_bar else "KRX 원주가 · 기업행동 시 가격 단절 가능"
        ),
        "generated_at": utcnow(),
    }


def overview(settings):
    lake, store = Lake(settings.data_dir), Store(settings.db_path)
    members = lake.universe()["members"]
    rows = []
    for item in store.watchlist():
        values = history(settings, item["symbol"], 25)
        broker_bar = display_bar(store, item["symbol"], datetime.now(KST))
        if broker_bar and (not values or broker_bar["time"] > values[-1]["time"]):
            values.append(broker_bar)
        last = values[-1] if values else None
        previous = values[-2] if len(values) > 1 else None
        previous_close = last.get("previous_close") if last else None
        previous_close = previous_close or (previous["close"] if previous else None)
        rows.append(
            {
                **item,
                "close": last["close"] if last else None,
                "change": 100 * (last["close"] / previous_close - 1) if previous_close else None,
                "as_of": last["time"] if last else None,
                "spark": [r["close"] for r in values],
            }
        )
    now = datetime.now(KST)
    return {
        "watchlist": rows,
        "universe": members,
        "market_open": within_session(now, lake.calendar()),
        "today": now.date().isoformat(),
        "generated_at": utcnow(),
        "operations": read_json(settings.data_dir / "reports/operations.json", {}),
        "alerts": [
            {k: r[k] for k in ("symbol", "trade_date", "threshold", "created_at")}
            for r in store.recent("alerts", 30)
        ],
        "operating_model": read_json(settings.data_dir / "config/forecast-release.json", {}).get(
            "model",
            read_json(settings.data_dir / "config/release.json", {}).get("selected", "observation_only"),
        ),
    }


def portfolio_view(settings):
    store = Store(settings.db_path)
    path = settings.data_dir / "silver/bars.parquet"
    return {
        "saved": store.get("meta", "portfolio", {"holdings": {}, "cash": 0}),
        "result": current_result(
            store,
            digest(path.read_bytes()) if path.exists() else "missing",
            Lake(settings.data_dir).calendar(),
        ),
    }
