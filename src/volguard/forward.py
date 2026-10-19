"""Score only forecasts actually saved before the target session opened."""

import json

import pandas as pd

from .config import KST
from .data import Lake
from .evaluation import promotion_gate
from .util import atomic_json, utcnow


def evaluate(settings):
    lake = Lake(settings.data_dir)
    bars = lake.bars()
    actual = bars[bars.valid_label].set_index(["symbol", "date"])
    sessions = {s["date"]: s for s in lake.calendar()}
    now = pd.Timestamp.now(tz="UTC")
    records, rejected, versions = [], [], set()
    for path in sorted((lake.root / "forward").glob("*.json")):
        saved = json.loads(path.read_text(encoding="utf8"))
        day = saved["target_date"]
        session = sessions.get(day)
        if session is None:
            rejected.append({"date": day, "reason": "calendar missing"})
            continue
        opened = pd.Timestamp(f"{day} {session['open_time']}", tz=KST)
        closed = pd.Timestamp(f"{day} {session['close_time']}", tz=KST)
        if pd.Timestamp(saved["issued_at"]) >= opened:
            rejected.append({"date": day, "reason": "late issuance"})
            continue
        if now <= closed:
            continue
        versions.add((saved["model_hash"], saved["baseline_model_hash"]))
        for row in saved["predictions"]:
            key = (row["symbol"], pd.Timestamp(day))
            if key not in actual.index:
                continue
            target = actual.loc[key]
            if target.available_at > now:
                continue
            records.append({**row, "target_date": pd.Timestamp(day), "y": float(target.range_pct)})
    result = {
        "created_at": utcnow(),
        "role": "actual forward predictions only",
        "rejected": rejected,
        "n_sessions": len({r["target_date"] for r in records}),
        "n_predictions": len(records),
        "status": "pending",
        "automatic_promotion": False,
    }
    if len(versions) > 1:
        result.update(status="blocked", reason="multiple model versions; evaluate separately")
    elif records:
        gate = promotion_gate(pd.DataFrame(records))
        result["gate"] = gate
        if result["n_sessions"] >= 126:
            result["status"] = "review_required" if gate["ml_pass"] else "failed"
    atomic_json(lake.root / "reports" / "forward.json", result)
    return result
