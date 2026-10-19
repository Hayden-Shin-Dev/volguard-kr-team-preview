"""Deterministic synthetic data, isolated from live evidence and forecasts."""

from datetime import datetime

import numpy as np
import pandas as pd

from .config import KST
from .data import Lake
from .store import Store
from .util import atomic_json


def seed_demo(settings):
    lake = Lake(settings.data_dir)
    if (settings.data_dir / "config" / "universe.json").exists():
        return {"status": "already_seeded", "directory": str(settings.data_dir)}
    rng = np.random.default_rng(731)
    dates = pd.bdate_range("2010-01-04", "2026-10-30")
    today = pd.Timestamp(datetime.now(KST).date())
    end = min(today - pd.Timedelta(days=1), pd.Timestamp("2026-10-29"))
    observed = dates[dates <= end]
    members = [
        {"symbol": f"{i + 1:06d}", "name": name}
        for i, name in enumerate(["가상 알파", "가상 베타", "가상 감마"])
    ]
    portfolio = settings.data_dir / "config" / "demo-universe.csv"
    pd.DataFrame(members).to_csv(portfolio, index=False)
    lake.freeze_universe(portfolio, "2026-10-16", "SYNTHETIC: not actual KOSPI200", demo=True)
    calendar_path = settings.data_dir / "config" / "demo-calendar.csv"
    pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "open_time": "09:00:00", "close_time": "15:30:00"}
    ).to_csv(calendar_path, index=False)
    lake.set_calendar(calendar_path)
    market_return = rng.normal(0, 0.008, len(observed))
    rows = []
    for i, member in enumerate(members):
        price, previous_range = 50000 * (i + 1), 2.0
        for j, date in enumerate(observed):
            # Positive conditional range with AR signal; prices and low/high remain internally consistent.
            span = max(0.05, 0.6 + 0.62 * previous_range + rng.normal(0, 0.28))
            previous_range = span
            ret = 0.6 * market_return[j] + rng.normal(0, 0.004)
            close = price * np.exp(ret)
            low = min(price, close) / (1 + span / 250)
            high = max(price, close) * (1 + span / 250)
            volume = int(rng.lognormal(12, 0.3))
            rows.append(
                {
                    "symbol": member["symbol"],
                    "date": date,
                    "open": price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "turnover": volume * close,
                    "market_cap": close * 10000000,
                    "adjustment_factor": 1.0,
                    "adjustment_verified": True,
                    "pit_verified": True,
                    "available_at": (date + pd.Timedelta(hours=18)).tz_localize(KST),
                }
            )
            price = close
    frame = pd.DataFrame(rows)
    source = lake.bronze(frame.to_csv(index=False).encode(), "synthetic-seed-731")
    report = lake.ingest(frame, source)
    store = Store(settings.db_path)
    store.set_watchlist(members, {m["symbol"] for m in members})
    atomic_json(
        settings.data_dir / "reports" / "demo.json",
        {
            "synthetic": True,
            "seed": 731,
            "calendar": "weekdays only; not a KRX trading calendar",
            "report": report,
        },
    )
    return report


def replay_demo(settings, ticks=120, interval=1.0):
    import time
    from datetime import timedelta

    from .realtime import QuoteEngine

    store = Store(settings.db_path)
    # Replay clock is explicit and stays at a synthetic regular-session time even outside live market hours.
    now = datetime.now(KST).replace(hour=10, minute=0, second=0, microsecond=0)
    sessions = [{"date": now.date().isoformat(), "open_time": "09:00:00", "close_time": "15:30:00"}]
    engine = QuoteEngine(store, sessions)
    symbols = [x["symbol"] for x in store.watchlist()]
    for step in range(ticks):
        clock = now + timedelta(seconds=step * 5)
        store.put("meta", "demo_clock", clock.isoformat())
        store.put("meta", "connection", {"confirmed_at": clock.isoformat(), "status": "demo"})
        for i, symbol in enumerate(symbols):
            low = 50000.0 * (i + 1)
            high = low * (1 + 0.0004 * (step + 1))
            engine.snapshots[symbol] = {"trade_date": clock.date().isoformat(), "high": low, "low": low}
            engine.accept(
                {
                    "symbol": symbol,
                    "trade_date": clock.date().isoformat(),
                    "exchange": "KRX",
                    "session": "regular",
                    "event_at": clock.isoformat(),
                    "received_at": clock.isoformat(),
                    "price": high,
                    "high": high,
                    "low": low,
                    "volume": 1000 * (step + 1),
                },
                clock,
            )
        if interval:
            time.sleep(interval)
    return {"ticks": ticks, "synthetic": True}
