"""TA-Lib indicators on contiguous KRX history; never interpolate corporate actions."""

import numpy as np
import pandas as pd
import talib

PROFILES = {
    "standard": dict(rsi_low=30, rsi_high=70, ma_fast=20, ma_slow=60, bb_sigma=2.0, volume=2.0),
    "strict": dict(rsi_low=20, rsi_high=80, ma_fast=50, ma_slow=200, bb_sigma=2.5, volume=3.0),
}
GROUPS = {
    "rsi": [
        "ta_rsi",
        "rsi_enter_low",
        "rsi_exit_low",
        "rsi_enter_high",
        "rsi_exit_high",
        "rsi_bull_div",
        "rsi_bear_div",
    ],
    "macd": ["macd_pct", "macd_hist_pct", "macd_cross_up", "macd_cross_down", "macd_age"],
    "moving_average": ["ma_spread", "golden_cross", "dead_cross", "ma_cross_age"],
    "bands": [
        "bb_position",
        "bb_width",
        "bb_lower_break",
        "bb_lower_reentry",
        "bb_upper_break",
        "bb_upper_reentry",
    ],
    "stochastic": ["stoch_k", "stoch_d", "stoch_cross_up", "stoch_cross_down"],
    "trend_atr": ["adx", "plus_di", "minus_di", "dmi_cross_up", "dmi_cross_down", "atr_pct"],
    "volume": ["obv_change20", "prior_volume_ratio", "obv_bull_div", "volume_breakout", "volume_breakdown"],
    "candles": ["engulfing", "hammer", "shooting_star", "morning_star", "evening_star", "doji"],
    "interactions": ["rsi_exit_in_trend", "lower_break_rangebound", "breakout_trending", "macd_high_vol"],
}


def cross(a, b, up=True):
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    valid = a.notna() & a.shift(1).notna() & b.notna() & b.shift(1).notna()
    result = (a.gt(b) & a.shift(1).le(b.shift(1))) if up else (a.lt(b) & a.shift(1).ge(b.shift(1)))
    return result.astype(float).where(valid)


def age(signal):
    indices = pd.Series(np.arange(len(signal)), index=signal.index, dtype=float)
    return (indices - indices.where(signal.eq(1)).ffill()).clip(upper=200).fillna(200)


def segment_features(g, profile):
    p = PROFILES[profile]
    f = pd.DataFrame(index=g.index)
    o, h, low, c, v = [g[k].to_numpy(dtype=float) for k in ["open", "high", "low", "close", "volume"]]
    close = g.close
    f["ta_rsi"] = talib.RSI(c, timeperiod=14)
    r = f.ta_rsi
    f["rsi_enter_low"], f["rsi_exit_low"] = cross(r, p["rsi_low"], False), cross(r, p["rsi_low"])
    f["rsi_enter_high"], f["rsi_exit_high"] = cross(r, p["rsi_high"]), cross(r, p["rsi_high"], False)
    # Causal trailing-extreme divergence, not hindsight pivots that need right-side candles.
    low20, high20 = close.shift(1).rolling(20).min(), close.shift(1).rolling(20).max()
    f["rsi_bull_div"] = (close.lt(low20) & r.gt(r.shift(1).rolling(20).min() + 5)).astype(float)
    f["rsi_bear_div"] = (close.gt(high20) & r.lt(r.shift(1).rolling(20).max() - 5)).astype(float)
    macd, sig, hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    f["macd_pct"], f["macd_hist_pct"] = macd / c, hist / c
    m, s = pd.Series(macd, index=g.index), pd.Series(sig, index=g.index)
    f["macd_cross_up"], f["macd_cross_down"] = cross(m, s), cross(m, s, False)
    f["macd_age"] = age(f.macd_cross_up + f.macd_cross_down)
    fast, slow = close.rolling(p["ma_fast"]).mean(), close.rolling(p["ma_slow"]).mean()
    f["ma_spread"] = fast / slow - 1
    f["golden_cross"], f["dead_cross"] = cross(fast, slow), cross(fast, slow, False)
    f["ma_cross_age"] = age(f.golden_cross + f.dead_cross)
    upper, middle, lower = talib.BBANDS(
        c, timeperiod=20, nbdevup=p["bb_sigma"], nbdevdn=p["bb_sigma"], matype=0
    )
    upper, lower = pd.Series(upper, index=g.index), pd.Series(lower, index=g.index)
    f["bb_position"] = (close - lower) / (upper - lower).replace(0, np.nan)
    f["bb_width"] = (upper - lower) / middle
    f["bb_lower_break"], f["bb_lower_reentry"] = cross(close, lower, False), cross(close, lower)
    f["bb_upper_break"], f["bb_upper_reentry"] = cross(close, upper), cross(close, upper, False)
    k, d = talib.STOCH(
        h, low, c, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
    )
    f["stoch_k"], f["stoch_d"] = k, d
    f["stoch_cross_up"], f["stoch_cross_down"] = (
        cross(f.stoch_k, f.stoch_d),
        cross(f.stoch_k, f.stoch_d, False),
    )
    f["adx"] = talib.ADX(h, low, c, timeperiod=14)
    f["plus_di"], f["minus_di"] = (
        talib.PLUS_DI(h, low, c, timeperiod=14),
        talib.MINUS_DI(h, low, c, timeperiod=14),
    )
    f["dmi_cross_up"], f["dmi_cross_down"] = cross(f.plus_di, f.minus_di), cross(f.plus_di, f.minus_di, False)
    f["atr_pct"] = talib.ATR(h, low, c, timeperiod=14) / c
    obv = pd.Series(talib.OBV(c, v), index=g.index)
    f["obv_change20"] = obv.diff(20) / g.volume.rolling(20).sum().replace(0, np.nan)
    f["prior_volume_ratio"] = g.volume / g.volume.shift(1).rolling(20).mean().replace(0, np.nan)
    f["obv_bull_div"] = (close.lt(low20) & obv.gt(obv.shift(1).rolling(20).min())).astype(float)
    f["volume_breakout"] = (
        close.gt(g.high.shift(1).rolling(20).max()) & f.prior_volume_ratio.ge(p["volume"])
    ).astype(float)
    f["volume_breakdown"] = (
        close.lt(g.low.shift(1).rolling(20).min()) & f.prior_volume_ratio.ge(p["volume"])
    ).astype(float)
    for name, func in [
        ("engulfing", talib.CDLENGULFING),
        ("hammer", talib.CDLHAMMER),
        ("shooting_star", talib.CDLSHOOTINGSTAR),
        ("morning_star", talib.CDLMORNINGSTAR),
        ("evening_star", talib.CDLEVENINGSTAR),
        ("doji", talib.CDLDOJI),
    ]:
        f[name] = func(o, h, low, c) / 100.0
    f["rsi_exit_in_trend"] = f.rsi_exit_low * (f.adx.ge(25) & f.plus_di.gt(f.minus_di))
    f["lower_break_rangebound"] = f.bb_lower_break * f.adx.lt(20)
    f["breakout_trending"] = f.volume_breakout * f.adx.ge(25)
    f["macd_high_vol"] = f.macd_cross_up * f.atr_pct.gt(f.atr_pct.shift(1).rolling(60).median())
    f["ta_ready"] = np.arange(len(g)) >= 200
    return f.replace([np.inf, -np.inf], np.nan)


def build(context, bars, profile):
    keys = ["symbol", "date"]
    source = context[keys + ["return_1"]].merge(
        bars[keys + ["open", "high", "low", "close", "volume", "valid_label"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    parts = []
    for _, g in source.groupby("symbol", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        valid = (
            g.valid_label.eq(True) & g[["open", "high", "low", "close"]].gt(0).all(axis=1) & g.volume.gt(0)
        )
        valid &= g.high.ge(g[["open", "close", "low"]].max(axis=1)) & g.low.le(
            g[["open", "close", "high"]].min(axis=1)
        )
        continuous = np.isclose(g.close / g.close.shift(1) - 1, g.return_1, rtol=1e-6, atol=1e-8)
        boundary = ~(valid & valid.shift(1, fill_value=False) & continuous)
        for _, segment in g.groupby(boundary.cumsum(), sort=True):
            if not valid.loc[segment.index].all() or len(segment) < 201:
                continue
            calculated = segment_features(segment, profile)
            parts.append(pd.concat([segment[keys], calculated], axis=1))
    return pd.concat(parts, ignore_index=True)


# Interpretation direction is specified before observing outcomes, not chosen retrospectively.
SIGNALS = {
    "rsi_enter_low": ("RSI 과매도 진입 후 상승", 1),
    "rsi_exit_low": ("RSI 과매도 이탈 후 상승", 1),
    "rsi_enter_high": ("RSI 과매수 진입 후 하락", 0),
    "rsi_exit_high": ("RSI 과매수 이탈 후 하락", 0),
    "golden_cross": ("골든크로스 후 상승", 1),
    "dead_cross": ("데드크로스 후 하락", 0),
    "bb_lower_break": ("볼린저 하단 이탈 후 상승", 1),
    "bb_lower_reentry": ("볼린저 하단 재진입 후 상승", 1),
    "bb_upper_break": ("볼린저 상단 이탈 후 하락", 0),
    "macd_cross_up": ("MACD 상향 교차 후 상승", 1),
    "stoch_cross_up": ("스토캐스틱 상향 교차 후 상승", 1),
    "dmi_cross_up": ("DMI 상향 교차 후 상승", 1),
    "rsi_bull_div": ("RSI 상승 다이버전스 후 상승", 1),
    "obv_bull_div": ("OBV 상승 다이버전스 후 상승", 1),
    "volume_breakout": ("거래량 동반 상단 돌파 후 상승", 1),
    "volume_breakdown": ("거래량 동반 하단 돌파 후 하락", 0),
    "hammer": ("망치형 후 상승", 1),
    "morning_star": ("샛별형 후 상승", 1),
    "shooting_star": ("유성형 후 하락", 0),
    "evening_star": ("저녁별형 후 하락", 0),
    "rsi_exit_in_trend": ("상승 추세 중 RSI 과매도 이탈", 1),
    "lower_break_rangebound": ("비추세 국면 볼린저 하단 이탈 후 상승", 1),
    "breakout_trending": ("추세 국면 거래량 동반 돌파 후 상승", 1),
}
