"""Dated broker bars for display only. Never writes training bars or gauge quotes."""

import json
import math
import threading
import time
from datetime import datetime

from .config import KST
from .data import Lake
from .providers import Kiwoom, request_json, signed_price
from .store import Store

_lock = threading.Lock()
_current_lock = threading.Lock()
_clients = {}


def normalize_current(symbol, raw, now, executions=None):
    """SOR display quote. Retrieval time never stands in for a trade date."""
    price = signed_price(raw["cur_prc"])
    change = float(str(raw["pred_pre"]).replace(",", ""))
    volume = float(str(raw["trde_qty"]).replace(",", ""))
    values = {
        k: signed_price(raw[v])
        for k, v in {"open": "open_pric", "high": "high_pric", "low": "low_pric"}.items()
    }
    if not all(math.isfinite(v) for v in [price, change, volume, *values.values()]):
        raise ValueError("Invalid display quote")
    if price <= 0 or price - change <= 0 or volume < 0:
        raise ValueError("Invalid display quote")
    if not 0 < values["low"] <= min(price, values["open"]) <= max(price, values["open"]) <= values["high"]:
        raise ValueError("Invalid display OHLC")
    row = (executions or [{}])[0]
    clock = str(row.get("tm", ""))
    try:
        clock = datetime.strptime(clock, "%H%M%S").strftime("%H:%M:%S") if len(clock) == 6 else None
    except ValueError:
        clock = None
    return {
        "symbol": symbol,
        "close": price,
        **values,
        "volume": volume,
        "previous_close": price - change,
        "retrieved_at": now.isoformat(),
        "trade_clock": clock,
        "trade_date": None,
        "reported_exchange": row.get("stex_tp"),
        "exchange": "SOR",
        "source": "kiwoom_ka10001",
        "display_only": True,
    }


def refresh_current(settings, symbol):
    store = Store(settings.db_path)
    if symbol not in {m["symbol"] for m in Lake(settings.data_dir).universe()["members"]}:
        raise ValueError("지원하지 않는 종목입니다")
    with _current_lock:
        now = datetime.now(KST)
        attempt = store.get("meta", f"current_attempt:{symbol}", {})
        if attempt and 0 <= (now - datetime.fromisoformat(attempt["at"])).total_seconds() < 4:
            return
        try:
            client = _clients.setdefault(settings.environment, Kiwoom(settings.environment))
            expiry = (
                datetime.strptime(client.expiry, "%Y%m%d%H%M%S").replace(tzinfo=KST) if client.expiry else now
            )
            if not client.token or (expiry - now).total_seconds() < 60:
                client.authenticate()

            def fetch(api):
                body = request_json(
                    "POST",
                    f"https://{client.host}/api/dostk/stkinfo",
                    headers={"authorization": f"Bearer {client.token}", "api-id": api},
                    json={"stk_cd": f"{symbol}_AL"},
                )
                if str(body.get("return_code", 0)) != "0":
                    raise ValueError("Display quote unavailable")
                return body

            raw = fetch("ka10001")
            try:
                executions = fetch("ka10003").get("cntr_infr", [])
            except (ValueError, RuntimeError, OSError):
                executions = []
            quote = normalize_current(symbol, raw, datetime.now(KST), executions)
            store.put("meta", f"display_current:{symbol}", quote)
            state = "ready"
        except (ValueError, RuntimeError, KeyError, TypeError, OSError):
            state = "unavailable"
        store.put("meta", f"current_attempt:{symbol}", {"at": now.isoformat(), "status": state})


def display_current(store, symbol, now):
    quote = store.get("meta", f"display_current:{symbol}")
    if not quote:
        return None
    age = (now - datetime.fromisoformat(quote["retrieved_at"])).total_seconds()
    attempt = store.get("meta", f"current_attempt:{symbol}", {})
    return {**quote, "fresh": 0 <= age <= 15 and attempt.get("status") == "ready"}


def normalize_snapshot(symbol, raw, now, sessions):
    day = datetime.strptime(str(raw["date"]), "%Y%m%d").date().isoformat()
    session = next((s for s in sessions if s["date"] == day), None)
    if day > now.astimezone(KST).date().isoformat() or not session:
        raise ValueError("Invalid snapshot date")
    prices = {
        k: signed_price(raw[v])
        for k, v in {
            "open": "open_pric",
            "high": "high_pric",
            "low": "low_pric",
            "close": "close_pric",
        }.items()
    }
    volume = float(str(raw["trde_qty"]).replace(",", ""))
    if not (
        prices["low"]
        <= min(prices["open"], prices["close"])
        <= max(prices["open"], prices["close"])
        <= prices["high"]
    ):
        raise ValueError("Invalid snapshot OHLC")
    if not math.isfinite(volume) or volume < 0:
        raise ValueError("Invalid snapshot volume")
    closing = datetime.fromisoformat(f"{day}T{session['close_time']}").replace(tzinfo=KST)
    closed = session.get("hours_verified", False) and now >= closing
    previous_close = None
    try:
        previous_close = prices["close"] - float(str(raw["pre"]).replace(",", ""))
        if not math.isfinite(previous_close) or previous_close <= 0:
            previous_close = None
    except (KeyError, ValueError, TypeError):
        pass
    return {
        "symbol": symbol,
        "time": day,
        **prices,
        "volume": volume,
        "previous_close": previous_close,
        "retrieved_at": now.isoformat(),
        "source": "kiwoom_ka10005",
        "exchange": "KRX",
        "closed": closed,
        "display_only": True,
        "label": "키움 조회 일봉 · KRX 종목코드",
    }


def refresh(settings, symbols, force=False):
    store, lake = Store(settings.db_path), Lake(settings.data_dir)
    allowed = {m["symbol"] for m in lake.universe()["members"]}
    if not set(symbols).issubset(allowed) or len(symbols) > 30:
        raise ValueError("지원하지 않는 종목입니다")
    if not _lock.acquire(blocking=False):
        return {"status": "refreshing"}
    results = {}
    try:
        client = Kiwoom(settings.environment)
        for symbol in symbols:
            now = datetime.now(KST)
            previous = store.get("meta", f"display_attempt:{symbol}", {})
            age = (now - datetime.fromisoformat(previous["at"])).total_seconds() if previous else 9999
            if 0 <= age < (10 if force else 60):
                results[symbol] = "cached"
                continue
            try:
                raw = client.daily_snapshot(symbol)
                bar = normalize_snapshot(symbol, raw, now, lake.calendar())
                source_hash = lake.bronze(
                    json.dumps(raw, ensure_ascii=False).encode(),
                    "kiwoom-display-daily",
                    {"symbol": symbol, "date": bar["time"]},
                )
                bar["source_hash"] = source_hash
                store.put("meta", f"display_bar:{symbol}", bar)
                state = "ready"
            except (ValueError, RuntimeError, KeyError, TypeError, OSError):
                state = "unavailable"
            store.put("meta", f"display_attempt:{symbol}", {"at": now.isoformat(), "status": state})
            results[symbol] = state
            if len(symbols) > 1:
                time.sleep(0.3)
    finally:
        _lock.release()
    return {"status": "checked", "symbols": results}


def display_bar(store, symbol, now):
    bar = store.get("meta", f"display_bar:{symbol}")
    if not bar or bar["time"] > now.astimezone(KST).date().isoformat():
        return None
    # Intraday REST snapshots age out. Dated closed bars remain valid historical observations.
    age = (now - datetime.fromisoformat(bar["retrieved_at"])).total_seconds()
    return bar if bar.get("closed") or 0 <= age <= 60 else None
