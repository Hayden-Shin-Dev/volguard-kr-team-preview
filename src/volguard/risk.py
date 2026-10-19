import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

from .util import digest


def portfolio_key(holdings, cash):
    values = {str(k): float(v) for k, v in holdings.items() if v != 0}
    if len(values) > 30 or any(not np.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("최대 30개, 유한한 양수 평가금액만 허용합니다")
    if not np.isfinite(cash) or cash < 0 or sum(values.values()) + cash <= 0:
        raise ValueError("현금은 음수일 수 없고 전체 평가금액은 양수여야 합니다")
    return digest({"holdings": values, "cash": float(cash)}), values


def _ll(success, n, p):
    return float(xlogy(success, p) + xlogy(n - success, 1 - p))


def exception_tests(exceptions, alpha=0.05):
    hits = np.asarray(exceptions, dtype=int)
    n, failures = len(hits), int(hits.sum())
    if not n:
        return {"kupiec_p": None, "christoffersen_p": None, "failures": 0}
    lr_uc = max(0.0, -2 * (_ll(failures, n, alpha) - _ll(failures, n, failures / n)))
    kupiec = float(chi2.sf(lr_uc, 1))
    counts = np.zeros((2, 2), dtype=int)
    for previous, current in zip(hits[:-1], hits[1:]):
        counts[previous, current] += 1
    n0, n1 = counts[0].sum(), counts[1].sum()
    if not n0 or not n1:
        return {"kupiec_p": kupiec, "christoffersen_p": None, "failures": failures}
    pooled = (counts[0, 1] + counts[1, 1]) / (n0 + n1)
    ll_ind = _ll(counts[0, 1] + counts[1, 1], n0 + n1, pooled)
    ll_markov = _ll(counts[0, 1], n0, counts[0, 1] / n0) + _ll(counts[1, 1], n1, counts[1, 1] / n1)
    lr_ind = max(0.0, -2 * (ll_ind - ll_markov))
    # Conditional coverage includes both unconditional coverage and independence.
    return {
        "kupiec_p": kupiec,
        "christoffersen_p": float(chi2.sf(lr_uc + lr_ind, 2)),
        "independence_p": float(chi2.sf(lr_ind, 1)),
        "failures": failures,
    }


def historical_var(
    returns,
    holdings,
    cash=0.0,
    as_of=None,
    window=500,
    minimum_oos=500,
    method="historical",
    decay=0.94,
    evaluation_start=None,
):
    if method not in {"historical", "filtered_historical"} or not 0 < decay < 1:
        raise ValueError("Unknown VaR method or invalid EWMA decay")
    key, holdings = portfolio_key(holdings, cash)
    report = {
        "portfolio_hash": key,
        "method": method,
        "confidence": 0.95,
        "horizon": "1 trading day",
        "var_krw": None,
        "status": "insufficient",
        "n": 0,
    }
    if not holdings:
        return {**report, "status": "cash_only", "var_krw": 0.0, "reason": "주식 노출 없음"}
    if not set(holdings).issubset(returns.columns):
        return {**report, "reason": "종목 수익률 없음"}
    frame = returns.sort_index().loc[:, sorted(holdings)]
    if as_of is not None:
        frame = frame.loc[: pd.Timestamp(as_of)]
    report["as_of"] = str(frame.index[-1].date()) if len(frame) else None
    warmup = 60 if method == "filtered_historical" else 0
    if len(frame) < window + minimum_oos + warmup:
        return {**report, "reason": "학습 500일 + 외표본 500일 필요"}
    # Missing sessions are never compressed away: that would turn adjacent-day tests into gap tests.
    matrix = frame.to_numpy(float)
    if not np.isfinite(matrix[-(window + minimum_oos + warmup) :]).all():
        return {**report, "reason": "최근 검증 구간의 동시 수익률 또는 수정주가 불완전"}
    # Find the contiguous complete suffix; disjoint periods are not concatenated.
    missing = np.flatnonzero(~np.isfinite(matrix).all(axis=1))
    start = int(missing[-1] + 1) if len(missing) else 0
    matrix = matrix[start:]
    weights = np.array([holdings[s] for s in frame.columns], float)
    losses = -(matrix @ weights)
    sigma = np.ones((len(matrix) + 1, matrix.shape[1]))
    if warmup:
        variance = np.maximum(np.mean(matrix[:warmup] ** 2, axis=0), 1e-12)
        for i in range(warmup, len(matrix) + 1):
            sigma[i] = np.sqrt(variance)
            if i < len(matrix):
                variance = np.maximum(decay * variance + (1 - decay) * matrix[i] ** 2, 1e-12)

    def threshold_at(i):
        if method == "historical":
            scenarios = losses[i - window : i]
        else:
            # Each residual uses volatility forecast BEFORE that return, and all
            # assets retain their common scenario date. No look-ahead fitting.
            scenarios = -((matrix[i - window : i] / sigma[i - window : i] * sigma[i]) @ weights)
        return max(0.0, float(np.quantile(scenarios, 0.95, method="higher")))

    exceptions, thresholds = [], []
    dates = frame.index[start:]
    evaluated_dates = []
    for i in range(window + warmup, len(losses)):
        if evaluation_start is not None and dates[i] < pd.Timestamp(evaluation_start):
            continue
        threshold = threshold_at(i)
        thresholds.append(threshold)
        exceptions.append(bool(losses[i] > threshold))
        evaluated_dates.append(str(dates[i].date()))
    tests = exception_tests(exceptions)
    passed = len(exceptions) >= minimum_oos and all(
        tests[k] is not None and tests[k] >= 0.05 for k in ("kupiec_p", "christoffersen_p")
    )
    report.update(
        n=len(exceptions),
        tests=tests,
        exception_rate=float(np.mean(exceptions)) if exceptions else None,
        status="passed" if passed else "failed",
        total_value=sum(holdings.values()) + cash,
        note="현재 비중을 과거 일별 수익률에 적용한 가상 포트폴리오 검증",
        evaluation_start=evaluated_dates[0] if evaluated_dates else None,
        evaluation_end=evaluated_dates[-1] if evaluated_dates else None,
        decay=decay if warmup else None,
    )
    if passed:
        report["var_krw"] = threshold_at(len(losses))
    return report


def verified_returns(bars, sessions, symbols, as_of=None):
    """Price returns: verified adjustments or uninterrupted official KRX reference.

    Unknown corporate-action boundaries remain NaN; never infer a split factor.
    """
    calendar = pd.DatetimeIndex(pd.to_datetime([s["date"] for s in sessions]))
    output = pd.DataFrame(index=calendar, columns=symbols, dtype=float)
    for symbol in symbols:
        part = bars[bars.symbol.eq(symbol) & bars.exchange.eq("KRX")].set_index("date").reindex(calendar)
        valid = part.valid_label.eq(True)
        adjusted = part.adjusted_close.where(part.adjustment_verified.eq(True) & valid)
        returns = adjusted.pct_change(fill_method=None)
        if "CMPPREVDD_PRC" in part:
            change = pd.to_numeric(part.CMPPREVDD_PRC.astype(str).str.replace(",", ""), errors="coerce")
            reference = part.close - change
            prior = part.close.shift(1)
            continuity = valid & valid.shift(1, fill_value=False) & (reference - prior).abs().le(0.5)
            raw_return = part.close / prior - 1
            returns = returns.where(returns.notna(), raw_return.where(continuity))
        output[symbol] = returns
    return output.loc[: pd.Timestamp(as_of)] if as_of is not None else output


def verified_stress(bars, sessions, holdings, cash=0):
    _, holdings = portfolio_key(holdings, cash)
    scenarios = stress_tests(bars, holdings, cash)
    returns = verified_returns(bars, sessions, sorted(holdings))
    window = returns.loc["2020-02-21":"2020-03-23"]
    calendar = {s["date"] for s in sessions}
    complete = (not holdings or (
        {"2020-02-20", "2020-02-21", "2020-03-23"}.issubset(calendar)
        and len(window) > 0 and np.isfinite(window.to_numpy()).all()
    ))
    contributions = {s: float(holdings[s] * ((1 + window[s]).prod() - 1)) for s in holdings} if complete else {}
    scenarios[-1] = {
        "scenario": "2020-02-20 종가 → 2020-03-23 종가",
        "pnl_krw": float(sum(contributions.values())) if complete else None,
        "status": "historical" if complete else "incomplete",
        "reason": None if complete else "시작·종료 거래일 또는 구간 전체의 검증된 동시 수익률 부족",
        "contributions": contributions,
    }
    for scenario, shock in zip(scenarios[:3], (-.05, -.10, -.20)):
        scenario["contributions"] = {s: amount * shock for s, amount in holdings.items()}
    return scenarios


def stress_tests(bars, holdings, cash=0.0):
    _, holdings = portfolio_key(holdings, cash)
    scenarios = [
        {
            "scenario": f"주식 동시 {shock:.0%}",
            "pnl_krw": sum(holdings.values()) * shock,
            "status": "hypothetical",
        }
        for shock in (-0.05, -0.10, -0.20)
    ]
    pnl = 0.0
    complete = True
    for symbol, amount in holdings.items():
        part = bars[bars.symbol.eq(symbol)].set_index("date")
        try:
            first, last = (
                float(part.loc[pd.Timestamp(d), "adjusted_close"]) for d in ("2020-02-20", "2020-03-23")
            )
            if not np.isfinite([first, last]).all() or first <= 0:
                complete = False
                break
            pnl += amount * (last / first - 1)
        except (KeyError, TypeError):
            complete = False
            break
    scenarios.append(
        {
            "scenario": "2020-02-20 종가 → 2020-03-23 종가",
            "pnl_krw": pnl if complete else None,
            "status": "historical" if complete else "incomplete",
        }
    )
    return scenarios
