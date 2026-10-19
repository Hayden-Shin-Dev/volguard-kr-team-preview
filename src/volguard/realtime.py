import asyncio
import json
import math
import time
from datetime import datetime

import websockets

from .config import KST
from .data import Lake
from .providers import Kiwoom, parse_tick, signed_price
from .store import Store
from .util import exclusive_run, utcnow


def aware(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("Timezone required")
    return stamp


def within_session(now, sessions):
    local = now.astimezone(KST)
    match = next((s for s in sessions if s["date"] == local.date().isoformat()), None)
    if not match or match.get("hours_verified") is False:
        return False
    open_at = datetime.fromisoformat(f"{match['date']}T{match['open_time']}").replace(tzinfo=KST)
    close_at = datetime.fromisoformat(f"{match['date']}T{match['close_time']}").replace(tzinfo=KST)
    return open_at <= local <= close_at


def gauge(quote, prediction, connection_at, now, sessions):
    result = {"status": "no_quote", "range_pct": None, "consumption_pct": None, "thresholds": []}
    def freshness(stamp, limit):
        try:
            age = (now - aware(stamp)).total_seconds() if stamp else None
        except (ValueError, TypeError):
            age = None
        return {"age_seconds": age, "status": "fresh" if age is not None and 0 <= age <= limit else "stale"}

    result["connection_freshness"] = freshness(connection_at, 15)
    result["price_freshness"] = freshness(quote.get("verified_at") if quote else None, 60)
    if not quote:
        return result
    if quote.get("exchange") != "KRX" or quote.get("session") != "regular":
        return {**result, "status": "wrong_session"}
    if quote.get("trade_date") != now.astimezone(KST).date().isoformat():
        return {**result, "status": "previous_day"}
    try:
        high, low = float(quote["high"]), float(quote["low"])
        if not all(math.isfinite(v) for v in (high, low)) or low <= 0 or high < low:
            return {**result, "status": "invalid_price"}
        result["range_pct"] = 100 * (high / low - 1)
        if not within_session(now, sessions):
            return {**result, "status": "market_closed"}
        connection_age = (now - aware(connection_at)).total_seconds() if connection_at else math.inf
        verified_age = (now - aware(quote["verified_at"])).total_seconds()
        if connection_age < 0 or connection_age > 15 or verified_age < 0 or verified_age > 60:
            return {**result, "status": "stale"}
    except (ValueError, KeyError, TypeError):
        return {**result, "status": "invalid_timestamp"}
    if not quote.get("complete"):
        return {**result, "status": "incomplete"}
    if not prediction or prediction.get("target_date") != quote["trade_date"]:
        return {**result, "status": "no_prediction"}
    if prediction.get("status") != "ready":
        return {**result, "status": "restricted"}
    high_bound = prediction.get("high")
    if high_bound is None or not math.isfinite(high_bound) or high_bound <= 0:
        return {**result, "status": "zero_or_invalid_bound"}
    consumption = 100 * result["range_pct"] / high_bound
    thresholds = ([80] if consumption >= 80 else []) + ([100] if consumption > 100 else [])
    return {**result, "status": "ready", "consumption_pct": consumption, "thresholds": thresholds}


class QuoteEngine:
    def __init__(self, store, sessions):
        self.store, self.sessions = store, sessions
        self.snapshots = {}
        self.last_observation = 0.0

    def disconnected(self):
        self.store.put("meta", "connection", {"confirmed_at": None, "status": "disconnected"})
        self.snapshots.clear()
        for item in self.store.watchlist():
            old = self.store.get("quotes", item["symbol"])
            if old:
                old["complete"] = False
                self.store.put("quotes", item["symbol"], old)

    def snapshot(self, symbol, payload, now):
        today = now.astimezone(KST).date().isoformat()
        date = datetime.strptime(str(payload["date"]), "%Y%m%d").date().isoformat()
        if date != today or not within_session(now, self.sessions):
            return False
        high, low = signed_price(payload["high_pric"]), signed_price(payload["low_pric"])
        if high < low:
            return False
        self.snapshots[symbol] = {"trade_date": date, "high": high, "low": low, "at": now.isoformat()}
        old = self.store.get("quotes", symbol)
        # A REST response alone cannot prove KRX regular-session scope. Prior 0B evidence required.
        if old and old.get("complete") and old["trade_date"] == today:
            if high >= old["high"] and low <= old["low"]:
                old.update(high=high, low=low, verified_at=now.isoformat())
                self.store.put("quotes", symbol, old)
        return True

    def accept(self, quote, now):
        if quote is None or not within_session(now, self.sessions):
            return False
        event = aware(quote["event_at"])
        if not within_session(event, self.sessions) or not -2 <= (now - event).total_seconds() <= 60:
            return False
        old = self.store.get("quotes", quote["symbol"])
        if old and old["trade_date"] == quote["trade_date"]:
            if event < aware(old["event_at"]) or quote["volume"] < old["volume"]:
                return False
            if quote["high"] < old["high"] or quote["low"] > old["low"]:
                return False  # Reject regressions; don't silently repair possible mixed-session data.
        snapshot = self.snapshots.get(quote["symbol"])
        complete = bool(
            snapshot
            and snapshot["trade_date"] == quote["trade_date"]
            and quote["high"] >= snapshot["high"]
            and quote["low"] <= snapshot["low"]
        )
        quote = {**quote, "complete": complete, "verified_at": now.isoformat()}
        self.store.put("quotes", quote["symbol"], quote)
        if quote["complete"] and time.monotonic() - self.last_observation >= 5:
            key = f"intraday:{quote['trade_date']}"
            observation = self.store.get("meta", key, {"first_seen": now.isoformat(), "samples": 0})
            observation.update(last_seen=now.isoformat(), samples=observation["samples"] + 1)
            self.store.put("meta", key, observation)
            self.last_observation = time.monotonic()
        self.emit(quote["symbol"], now)
        return True

    def emit(self, symbol, now):
        quote = self.store.get("quotes", symbol)
        prediction = self.store.prediction(symbol, now.astimezone(KST).date().isoformat())
        connection = self.store.get("meta", "connection", {})
        state = gauge(quote, prediction, connection.get("confirmed_at"), now, self.sessions)
        for threshold in state["thresholds"]:
            self.store.alert(symbol, quote["trade_date"], threshold, state)
        return state


async def websocket_probe(settings, symbols, seconds=15):
    """One bounded login/subscription probe. Only status/counts are persisted."""
    if not 1 <= len(set(symbols)) == len(symbols) <= 30:
        raise ValueError("1~30 unique symbols required")
    client = Kiwoom(settings.environment)
    await asyncio.to_thread(client.authenticate)
    result = {
        "status": "unverified",
        "requested_symbols": len(symbols),
        "ticks": 0,
        "regular_krx_ticks": 0,
        "checked_at": utcnow(),
        "during_regular_session": within_session(datetime.now(KST), Lake(settings.data_dir).calendar()),
    }
    async with websockets.connect(f"wss://{client.host}:10000/api/websocket", max_size=2**22) as ws:
        await ws.send(json.dumps({"trnm": "LOGIN", "token": client.token}))
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                body = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
                )
            except asyncio.TimeoutError:
                break
            if body.get("trnm") == "PING":
                await ws.send(json.dumps(body))
            elif body.get("trnm") == "LOGIN":
                if str(body.get("return_code")) != "0":
                    result["status"] = "login_rejected"
                    break
                await ws.send(
                    json.dumps(
                        {
                            "trnm": "REG",
                            "grp_no": "1",
                            "refresh": "0",
                            "data": [{"item": symbols, "type": ["0B"]}],
                        }
                    )
                )
            elif body.get("trnm") == "REG":
                result["status"] = "subscribed" if str(body.get("return_code")) == "0" else "rejected"
            elif body.get("trnm") == "REAL":
                for item in body.get("data", []):
                    result["ticks"] += 1
                    result["regular_krx_ticks"] += parse_tick(item, datetime.now(KST)) is not None
    from .util import atomic_json

    atomic_json(settings.data_dir / "reports" / "websocket.json", result)
    return result


async def collect(settings, stop_after=None):
    store, lake = Store(settings.db_path), Lake(settings.data_dir)
    engine = QuoteEngine(store, lake.calendar())
    end = time.monotonic() + stop_after if stop_after else math.inf
    with exclusive_run(settings.data_dir / "collector.lock"):
        delay = 1
        while time.monotonic() < end:
            engine.sessions = lake.calendar()
            symbols = [item["symbol"] for item in store.watchlist()]
            if not symbols:
                raise ValueError("관심종목을 먼저 설정하세요")
            if not within_session(datetime.now(KST), engine.sessions):
                engine.disconnected()
                await asyncio.sleep(min(5, max(0, end - time.monotonic())))
                continue
            refresh_task = None
            try:
                engine.disconnected()
                client = Kiwoom(settings.environment)
                await asyncio.to_thread(client.authenticate)
                async with websockets.connect(
                    f"wss://{client.host}:10000/api/websocket", max_size=2**22
                ) as ws:
                    await ws.send(json.dumps({"trnm": "LOGIN", "token": client.token}))
                    connected_at = time.monotonic()
                    registered = False

                    async def refresh_snapshots():
                        # REST is independent of WS receive: 30 snapshots must not block heartbeats.
                        while True:
                            for symbol in symbols:
                                try:
                                    snapshot = await asyncio.to_thread(client.daily_snapshot, symbol)
                                    engine.snapshot(symbol, snapshot, datetime.now(KST))
                                    engine.emit(symbol, datetime.now(KST))
                                except (ValueError, RuntimeError):
                                    old = store.get("quotes", symbol)
                                    if old:
                                        old["complete"] = False
                                        store.put("quotes", symbol, old)
                                    engine.snapshots.pop(symbol, None)
                                await asyncio.sleep(0.3)
                            await asyncio.sleep(20)

                    while time.monotonic() < end:
                        if not within_session(datetime.now(KST), engine.sessions):
                            break
                        if time.monotonic() - connected_at > 23 * 3600:
                            break  # Fresh token and subscription on next connection.
                        if [x["symbol"] for x in store.watchlist()] != symbols:
                            break  # Reconnect safely after subscription changes.
                        if not registered and time.monotonic() - connected_at > 20:
                            raise ValueError("LOGIN/REG response timeout")
                        try:
                            body = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                        except asyncio.TimeoutError:
                            pong = await ws.ping()
                            await asyncio.wait_for(pong, timeout=5)
                            if registered:
                                store.put(
                                    "meta", "connection", {"status": "connected", "confirmed_at": utcnow()}
                                )
                            body = {}
                        now = datetime.now(KST)
                        if body and registered:
                            store.put(
                                "meta", "connection", {"status": "connected", "confirmed_at": now.isoformat()}
                            )
                        kind = body.get("trnm")
                        if kind == "PING":
                            await ws.send(json.dumps(body))
                        elif kind == "LOGIN":
                            if str(body.get("return_code")) != "0":
                                raise ValueError("LOGIN rejected")
                            await ws.send(
                                json.dumps(
                                    {
                                        "trnm": "REG",
                                        "grp_no": "1",
                                        "refresh": "0",
                                        "data": [{"item": symbols, "type": ["0B"]}],
                                    }
                                )
                            )
                        elif kind == "REG":
                            if str(body.get("return_code")) != "0":
                                raise ValueError("Subscription rejected")
                            delay = 1
                            registered = True
                            if refresh_task is None:
                                refresh_task = asyncio.create_task(refresh_snapshots())
                        elif kind == "REAL" and registered:
                            for item in body.get("data", []):
                                tick = parse_tick(item, now)
                                if tick and item.get("values", {}).get("16"):
                                    try:
                                        opened = signed_price(item["values"]["16"])
                                        if tick["low"] <= opened <= tick["high"]:
                                            tick["open"] = opened  # Official KRX 0B opening-price field.
                                    except ValueError:
                                        pass
                                if tick and tick["symbol"] in symbols:
                                    engine.accept(tick, now)
                        if refresh_task is not None and refresh_task.done():
                            refresh_task.result()
            except (
                OSError,
                ValueError,
                RuntimeError,
                asyncio.TimeoutError,
                websockets.WebSocketException,
            ) as exc:
                store.put("meta", "collector_error", {"type": type(exc).__name__, "at": utcnow()})
            finally:
                if refresh_task is not None:
                    refresh_task.cancel()
                    await asyncio.gather(refresh_task, return_exceptions=True)
                engine.disconnected()
            await asyncio.sleep(min(delay, max(0, end - time.monotonic())))
            delay = min(30, delay * 2)
