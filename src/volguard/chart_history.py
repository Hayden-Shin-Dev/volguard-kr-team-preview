"""Read archived forecasts for charts; never fit models or backfill issuance."""

import math

import pandas as pd

from .config import KST
from .data import Lake
from .terminal_data import frame, read_json


def chart_history(settings, symbol):
    lake = Lake(settings.data_dir)
    if symbol not in {m["symbol"] for m in lake.universe()["members"]}:
        raise ValueError("지원하지 않는 종목입니다")
    bars = frame(lake.root / "silver/bars.parquet")
    own = bars[bars.symbol.eq(symbol)].sort_values("date").copy()
    actual = own.set_index("date")

    def package(rows, label):
        valid = []
        for r in rows:
            if not all(math.isfinite(r[k]) for k in ("low", "median", "high", "actual")):
                continue
            if not 0 <= r["low"] <= r["median"] <= r["high"] or r["actual"] < 0:
                continue
            r["inside"] = r["low"] <= r["actual"] <= r["high"]
            r["score"] = r["high"] - r["low"] + 20 * max(r["low"] - r["actual"], r["actual"] - r["high"], 0)
            valid.append(r)
        valid = sorted(valid, key=lambda r: r["time"])[-120:]
        dates = {r["time"] for r in valid}
        candles = [
            {
                "time": r.date.strftime("%Y-%m-%d"),
                **{k: float(getattr(r, k)) for k in ("open", "high", "low", "close", "volume")},
            }
            for r in own.itertuples()
            if r.date.strftime("%Y-%m-%d") in dates
        ]
        n = len(valid)
        return {
            "label": label,
            "rows": valid,
            "candles": candles,
            "n": n,
            "coverage": sum(r["inside"] for r in valid) / n if n else None,
            "score": sum(r["score"] for r in valid) / n if n else None,
        }

    report = read_json(lake.root / "reports/ensemble-study.json", {})
    development = []
    if report.get("artifact_dir"):
        from pathlib import Path

        for path in sorted(Path(report["artifact_dir"]).glob("pred-nested7_tail50-*.parquet")):
            data = frame(path)
            for r in data[data.symbol.eq(symbol) & data.method.eq("ml")].itertuples():
                development.append(
                    {
                        "time": r.target_date.strftime("%Y-%m-%d"),
                        "low": float(r.low),
                        "median": float(r.median),
                        "high": float(r.high),
                        "actual": float(r.y),
                        "model": "nested7_tail50",
                    }
                )
    forward = []
    sessions = {s["date"]: s for s in lake.calendar()}
    now = pd.Timestamp.now(tz="UTC")
    for path in sorted((lake.root / "forward").glob("*.json")):
        saved = read_json(path, {})
        day = saved.get("target_date")
        s = sessions.get(day)
        if not s or pd.Timestamp(saved["issued_at"]) >= pd.Timestamp(f"{day} {s['open_time']}", tz=KST):
            continue
        if now <= pd.Timestamp(f"{day} {s['close_time']}", tz=KST) or pd.Timestamp(day) not in actual.index:
            continue
        bar = actual.loc[pd.Timestamp(day)]
        if not bar.valid_label or pd.Timestamp(bar.available_at) > now:
            continue
        # Only the actually issued model, never a shadow comparison or regenerated result.
        issued = saved.get("issued_prediction", {})
        if issued.get("symbol") != symbol or issued.get("method") != "ml":
            continue
        forward.append(
            {
                "time": day,
                **{k: float(issued[k]) for k in ("low", "median", "high")},
                "actual": float(bar.range_pct),
                "model": issued.get("model_name"),
                "version": issued.get("model_version"),
            }
        )
    return {
        "development": package(development, "개발 검증 기록 · 실제 운영 성적과 별도"),
        "forward": package(forward, "실제 사전 발행 예측 · 장 마감 후 채점"),
    }
