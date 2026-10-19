import numpy as np

from .util import atomic_json, utcnow


def report(store, settings):
    observations = [
        r
        for r in store.batch_observations()
        if not r["demo"] and r.get("observed_at", "")[:10] == r["input_date"]
    ]
    observations = sorted(observations, key=lambda r: r["input_date"])[-20:]
    intraday = [r for r in observations if store.get("meta", f"intraday:{r['input_date']}")]
    runs = [r for r in observations if r["data_ready"]]
    completed = [r for r in runs if r["status"] == "success"]
    latencies = store.get("meta", "render_latencies", [])
    p95 = float(np.quantile(latencies, 0.95)) if latencies else None
    result = {
        "generated_at": utcnow(),
        "observed_days": len(observations),
        "eligible_days": len(runs),
        "intraday_observed_days": len(intraday),
        "input_incomplete_days": len(observations) - len(runs),
        "batch_success_rate": len(completed) / len(runs) if runs else None,
        "render_delay_p95_seconds": p95,
        "render_sample_count": len(latencies),
        "status": "pending"
        if len(observations) < 20 or len(intraday) < 20 or not runs or p95 is None
        else "passed"
        if len(completed) / len(runs) >= 0.95 and p95 <= 10
        else "failed",
        "note": "20거래일 관찰은 실제 운영 증거가 필요합니다. 단위 테스트로 대체하지 않습니다.",
    }
    atomic_json(settings.data_dir / "reports" / "operations.json", result)
    return result
