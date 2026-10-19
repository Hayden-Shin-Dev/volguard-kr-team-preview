"""Immutable analysis context, explicitly hypothetical paths and dated replay."""

import json
import math
import re
import uuid

import numpy as np
import pandas as pd

from .chart_history import chart_history
from .investigation import capture
from .terminal_data import portfolio_view, stock
from .util import digest, utcnow


def path_for(anchor, range_pct, direction, seed):
    if (
        direction not in {"up", "flat", "down"}
        or not math.isfinite(anchor + range_pct)
        or anchor <= 0
        or range_pct < 0
    ):
        raise ValueError("시나리오의 가격·변동폭·방향을 확인하세요")
    rng = np.random.default_rng(seed)
    n = 96
    t = np.linspace(0, 1, n + 1)
    noise = np.r_[0, rng.normal(0, 0.14, n).cumsum()]
    bridge = noise - t * noise[-1]
    sign = {"up": 1, "flat": 0, "down": -1}[direction]
    x = bridge + t * sign * 0.8
    span = x.max() - x.min()
    x = x / (span or 1) * np.log1p(range_pct / 100)
    prices = anchor * np.exp(x)
    candles = [
        {
            "step": 1,
            "open": float(prices[0]),
            "high": float(prices.max()),
            "low": float(prices.min()),
            "close": float(prices[-1]),
        }
    ]
    drawdown = float(np.min(prices / np.maximum.accumulate(prices) - 1))
    return {
        "kind": "illustrative_only",
        "direction": direction,
        "candles": candles,
        "range_pct": float(100 * (prices.max() / prices.min() - 1)),
        "close_return": float(prices[-1] / anchor - 1),
        "max_drawdown": drawdown,
        "probability": None,
        "seed": seed,
        "assumptions": "다음 거래일 일봉 한 개. 시가=직전 종가·갭 0을 가정하고 선택 변동폭과 방향에 맞춘 예시입니다. 시가·종가·고저의 위치는 모델 예측이 아닙니다.",
        "validation": "미검증 가상 경로 · 분봉 원자료 부족으로 경로 적합도·포함 확률 미산출",
    }


def impact(saved, symbol, scenario):
    holdings, cash = saved.get("holdings", {}), float(saved.get("cash", 0))
    total = sum(holdings.values()) + cash
    rows = [
        {
            "symbol": s,
            "value_krw": v,
            "weight": v / total if total else 0,
            "pnl_krw": v * scenario["close_return"] if s == symbol else 0,
        }
        for s, v in sorted(holdings.items(), key=lambda x: -x[1])
    ]
    return {
        "total_krw": total,
        "pnl_krw": sum(r["pnl_krw"] for r in rows),
        "contributions": rows,
        "path_pnl": [
            holdings.get(symbol, 0) * (c["close"] / scenario["candles"][0]["open"] - 1)
            for c in scenario["candles"]
        ],
        "concentration": rows[0] if rows else None,
        "note": "선택 종목만 시나리오대로 변하고 다른 종목·현금은 그대로라는 가정. 항목별 시나리오 손익 기여이며 VaR 기여도가 아닙니다.",
    }


def save(settings, result):
    directory = settings.data_dir / "analyses"
    directory.mkdir(exist_ok=True)
    result["id"] = uuid.uuid4().hex
    result["snapshot_hash"] = digest(result)
    with (directory / f"{result['id']}.json").open("x", encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False)
    return result


def read(settings, key):
    if not re.fullmatch("[a-f0-9]{32}", key):
        raise ValueError("잘못된 분석 ID입니다")
    path = settings.data_dir / "analyses" / f"{key}.json"
    if not path.exists():
        raise ValueError("저장된 분석을 찾을 수 없습니다")
    return json.loads(path.read_text(encoding="utf8"))


def create(settings, symbol, direction="up", magnitude="median", parent=None):
    if magnitude not in {"low", "median", "high"}:
        raise ValueError("지원하지 않는 변동폭입니다")
    if parent:
        old = read(settings, parent)
        if old["symbol"] != symbol:
            raise ValueError("분석 종목이 다릅니다")
        context, market, portfolio, pred = old["context"], old["market"], old["portfolio"], old["prediction"]
        direction_status = old["direction"]
    else:
        market = stock(settings, symbol)
        pred = market.get("prediction")
        if not pred or pred.get("target_date") != market.get("forecast_target"):
            from .live_forecast import calculate

            calculate(settings, symbol)  # Existing immutable issuance and session gates remain authoritative.
            market = stock(settings, symbol)
            pred = market.get("prediction")
        if (
            not pred
            or pred.get("status") != "ready"
            or pred.get("target_date") != market.get("forecast_target")
        ):
            err = market.get("forecast_error") or {}
            raise ValueError(
                err.get("reason", "다음 거래일 예측이 아직 없습니다.")
                + " "
                + err.get("remedy", "확정 일봉 수집 후 다시 분석하세요.")
            )
        # Display the exact closed KRX bar used by inference, even before Silver catches up.
        last_bar = pred.get("evidence", {}).get("last_bar")
        chart = market.get("chart_candles", market.get("candles", []))
        if last_bar and (not chart or chart[-1]["time"] < pred["anchor_date"]):
            market["chart_candles"] = [*chart, {"time": pred["anchor_date"], **last_bar}]
        context, portfolio = capture(settings, [symbol]), portfolio_view(settings)
        from .scenario_direction import for_prediction
        direction_status = for_prediction(settings, symbol, pred)
        # Current quote/gauge remains dated separately from next-session prediction.
        context["stocks"][0]["prediction"] = pred
    anchor = pred.get("evidence", {}).get("last_bar", {}).get("close")
    if not anchor:
        raise ValueError("예측에 사용한 확정 종가가 없습니다. 키움 확정 일봉 추론을 확인하세요.")
    seed = int(
        digest({"symbol": symbol, "prediction": pred, "direction": direction, "magnitude": magnitude})[:8], 16
    )
    scenario = path_for(anchor, float(pred[magnitude]), direction, seed)
    scenario.update(target_date=pred["target_date"], magnitude=magnitude, anchor_price=anchor)
    alternatives = {d: path_for(anchor, float(pred[magnitude]), d, seed) for d in ["up", "down"]}
    for alternative in alternatives.values():
        alternative.update(target_date=pred["target_date"], magnitude=magnitude, anchor_price=anchor)
    return save(
        settings,
        {
            "created_at": utcnow(),
            "parent_id": parent,
            "symbol": symbol,
            "anchor_date": pred["anchor_date"],
            "target_date": pred["target_date"],
            "market": market,
            "prediction": pred,
            "direction": direction_status,
            "scenario": scenario,
            "alternatives": alternatives,
            "portfolio": portfolio,
            "impact": impact(portfolio["saved"], symbol, scenario),
            "context": context,
        },
    )


def replay(settings, symbol, day):
    archive = chart_history(settings, symbol)
    record, basis = None, None
    for name in ["forward", "development"]:
        record = next((r for r in archive[name]["rows"] if r["time"] == day), None)
        if record:
            basis = name
            break
    if record is None:
        raise ValueError("그 날짜에는 보존된 예측이 없습니다. 제공된 날짜 목록에서 선택하세요.")
    # Only preceding observations are inputs; future observations travel in a separately labeled payload.
    bars = pd.read_parquet(settings.data_dir / "silver/bars.parquet", filters=[("symbol", "=", symbol)])
    bars = bars.sort_values("date")
    before = bars[bars.date.lt(day)].tail(45)
    after = bars[bars.date.ge(day) & bars.valid_label.eq(True)].head(6)

    def candles(df):
        return [
            {
                "time": r.date.strftime("%Y-%m-%d"),
                **{k: float(getattr(r, k)) for k in ["open", "high", "low", "close", "volume"]},
            }
            for r in df.itertuples()
        ]

    if before.empty or after.empty or after.iloc[0].date != pd.Timestamp(day):
        raise ValueError("선택일의 실제 일봉 또는 직전 일봉이 없습니다")
    pred = {k: record[k] for k in ["low", "median", "high"]}
    pred["target_date"] = day
    return {
        "symbol": symbol,
        "target_date": day,
        "basis": basis,
        "label": archive[basis]["label"],
        "prediction": pred,
        "before": candles(before),
        "actual_after": candles(after),
        "score": {k: record[k] for k in ["actual", "inside", "score"]},
        "note": "실제 일봉을 하루씩 공개합니다. 당일 분봉 경로는 재구성하지 않습니다. 개발 기록은 당시 운영 발행 기록이 아닙니다.",
    }
