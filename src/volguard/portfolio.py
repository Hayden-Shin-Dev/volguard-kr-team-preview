"""Persist portfolio calculations with data and implementation identities."""

from pathlib import Path

import pandas as pd

from .risk import historical_var, portfolio_key, verified_returns, verified_stress
from .util import digest, utcnow


def cache_identity(holdings, cash, bars_hash, sessions):
    key, _ = portfolio_key(holdings, cash)
    return digest({"portfolio": key, "bars": bars_hash, "calendar": digest(sessions),
                   "risk_source": digest(Path(__file__).with_name("risk.py").read_bytes()),
                   "portfolio_source": digest(Path(__file__).read_bytes())})


def calculate(store, bars, sessions, holdings, cash, bars_hash):
    key, holdings = portfolio_key(holdings, cash)
    identity = cache_identity(holdings, cash, bars_hash, sessions)
    result = store.get("risk_cache", identity)
    if result is None:
        returns = verified_returns(bars, sessions, sorted(holdings)) if holdings else pd.DataFrame()
        result = historical_var(returns, holdings, cash,
                                as_of=bars.date.max() if not bars.empty else None, method="filtered_historical")
        result.update(stress=verified_stress(bars, sessions, holdings, cash),
                      holdings=holdings, cash=float(cash), calculated_at=utcnow(),
                      cache_key=identity, data_hash=bars_hash, portfolio_hash=key)
        total = sum(holdings.values()) + cash
        result["exposures"] = [{"symbol": s, "value_krw": v, "weight": v / total}
                               for s, v in sorted(holdings.items(), key=lambda pair: -pair[1])]
        result["cash_weight"] = cash / total
        store.put("risk_cache", identity, result)
    store.put("meta", "portfolio", {"holdings": holdings, "cash": float(cash)})
    store.put("meta", "portfolio_result", {"cache_key": identity, "portfolio_hash": key})
    return result


def current_result(store, bars_hash, sessions):
    saved = store.get("meta", "portfolio")
    pointer = store.get("meta", "portfolio_result")
    if not saved or not pointer:
        return None
    expected = cache_identity(saved["holdings"], saved["cash"], bars_hash, sessions)
    if expected != pointer["cache_key"]:
        return None  # Old results remain in the archive, not presented as current.
    return store.get("risk_cache", expected)
