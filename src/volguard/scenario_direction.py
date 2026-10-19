"""Dated demo inference from the frozen research candidate; never operational issuance."""
import json
import math
from pathlib import Path

from .terminal_data import read_json
from .util import atomic_json, digest, utcnow


def for_prediction(settings, symbol, prediction):
    report = read_json(settings.data_dir / "reports/scenario-direction-preview.json", {})
    unavailable = {"status": "unavailable", "probability": None, "research_only": True,
                   "reason": "해당 기준일의 연구 확률이 없습니다. 변동폭 시나리오만 표시합니다."}
    if any(report.get(k) != prediction.get(k) for k in ("anchor_date", "target_date")):
        return unavailable
    row = next((r for r in report.get("results", []) if r["symbol"] == symbol), None)
    if not row or row.get("status") != "ready":
        return {**unavailable, "reason": (row or {}).get("reason", unavailable["reason"])}
    p, down = row.get("p_up"), row.get("p_down")
    if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
        return unavailable
    if not isinstance(down, (int, float)) or not math.isclose(p + down, 1, abs_tol=1e-8):
        return unavailable
    return {"status": "research_only", "probability": p, "p_up": p, "p_down": down,
            "research_only": True, "gate_pass": False, "version": report["model_version"],
            "name": report["model_name"], "anchor_date": report["anchor_date"],
            "target_date": report["target_date"], "computed_at": report["computed_at"],
            "input_hash": row["input_hash"], "conditional_on": "nonflat",
            "reason": "보합을 제외한 다음 거래일 종가 방향의 연구 확률입니다. 운영 성능은 미검증입니다.",
            "limitations": report["limitations"]}


def build(settings):
    """Use the audited snapshot, not latest unaudited or fabricated bars."""
    import pandas as pd
    import torch

    from .daily_sequence_candidate import FrozenDailyCandidate, make_windows
    from .direction_data_audit import build_features

    root = settings.data_dir
    config = read_json(root / "daily-sequence-forward/config.json")
    for key in ("history", "manifest"):
        if digest(Path(config[key]).read_bytes()) != config[key + "_hash"]:
            raise ValueError("고정 연구 자료 해시가 일치하지 않습니다")
    bars = pd.read_parquet(config["history"])
    anchor = bars.date.max()
    if anchor > pd.Timestamp.now().normalize():
        raise ValueError("미래 입력 날짜가 포함되었습니다")
    sessions = read_json(root / "config/calendar.json")["sessions"]
    target = next(s["date"] for s in sessions if s["date"] > anchor.strftime("%Y-%m-%d"))
    start = anchor - pd.Timedelta(days=220)
    bars = bars[bars.date.ge(start) & bars.date.le(anchor)]
    calendar = [s["date"] for s in sessions if start <= pd.Timestamp(s["date"]) <= anchor]
    features = build_features(bars, pd.to_datetime(calendar))
    names, windows, rejected = make_windows(features, anchor, calendar)
    torch.set_num_threads(2)
    model = FrozenDailyCandidate(config["manifest"])
    probabilities = model.predict(windows)
    results = [dict(symbol=s, p_up=float(p), p_down=float(1-p), status="ready",
                    input_hash=digest(x.tolist())) for s, p, x in zip(names, probabilities, windows)]
    results += [dict(symbol=s, status="unavailable", reason="정규장 가격 연속성 또는 연속 20거래일 피처를 확인하지 못했습니다.")
                for s in rejected]
    report = dict(kind="research_preview_not_operational_issuance", anchor_date=anchor.strftime("%Y-%m-%d"),
                  target_date=target, computed_at=utcnow(), model_name="GRU 일봉 시계열 앙상블",
                  model_version=model.version, data_hash=config["history_hash"], results=results,
                  conditional_on="nonflat", research_only=True, automatic_promotion=False,
                  limitations=["보합일을 제외하고 학습·평가한 조건부 방향 확률",
                               "2023~2026년은 반복 사용한 회고 평가 기간",
                               "상승 신호의 날짜 집중과 70% 이상 확률의 신뢰성은 미해결",
                               "방향별 변동폭 또는 개별 가격 경로 확률을 검증한 결과가 아님"])
    atomic_json(root / "reports/scenario-direction-preview.json", report)
    return report


if __name__ == "__main__":
    from .platform_study import local_settings
    result = build(local_settings())
    print(json.dumps({"anchor_date": result["anchor_date"], "target_date": result["target_date"],
                      "results": [r for r in result["results"] if r["symbol"] in {"005930", "000660", "005380"}],
                      "accepted": sum(r["status"] == "ready" for r in result["results"])}, ensure_ascii=False))
