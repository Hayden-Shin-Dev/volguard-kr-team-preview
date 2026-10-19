"""Independent evidence worker; UI closure does not stop queued investigations."""

import json
import time
from datetime import datetime
from threading import Event, Thread

from .config import KST, Settings
from .evidence import collect
from .investigation import Investigations, build_report, capture, local_evidence
from .report_ai import synthesize
from .store import Store
from .util import exclusive_run, utcnow


def enqueue_alerts(settings, jobs):
    store = Store(settings.db_path)
    if not store.get("meta", "investigation_preferences", {}).get("auto_alerts", True):
        return 0
    today = datetime.now(KST).date().isoformat()
    count = 0
    for alert in store.recent("alerts", 100):
        if alert["trade_date"] != today:
            continue
        key = f"alert:{alert['symbol']}:{alert['trade_date']}:{alert['threshold']}"
        if jobs.has_dedup(key):
            continue
        trigger = {k: alert[k] for k in ("symbol", "trade_date", "threshold", "created_at")}
        trigger["observed_state"] = json.loads(alert["payload"])
        snapshot = capture(settings, [alert["symbol"]], trigger=trigger)
        jobs.enqueue(snapshot, use_ai=False, dedup=key)
        count += 1
    return count


def process_one(settings, jobs, collector=collect, synthesizer=synthesize):
    job = jobs.claim()
    if job is None:
        return False
    try:
        request = job["request"]
        external = collector(settings, request["snapshot"])
        ai = synthesizer(local_evidence(request["snapshot"]) + external["sources"]) if request["use_ai"] else None
        report = build_report(request, external, ai)
        report["job_id"] = job["id"]
        jobs.finish(job["id"], report)
    except Exception as exc:
        jobs.fail(job["id"], type(exc).__name__)  # Never persist secret-bearing HTTP exception text.
    return True


def run(settings, once=False):
    store, jobs = Store(settings.db_path), Investigations(settings.data_dir)
    with exclusive_run(settings.data_dir / "report-worker.lock"):
        stop = Event()

        def heartbeat():
            while not stop.is_set():
                store.put("meta", "report_worker", {"heartbeat": utcnow(), "status": "running"})
                stop.wait(15)

        pulse = Thread(target=heartbeat, daemon=True)
        pulse.start()
        try:
            while True:
                try:
                    enqueue_alerts(settings, jobs)
                    active = process_one(settings, jobs)
                except Exception as exc:
                    store.put("meta", "report_worker_error", {"at": utcnow(), "type": type(exc).__name__})
                    active = False
                if once:
                    return
                if not active:
                    time.sleep(5)
        finally:
            stop.set()
            pulse.join(timeout=5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run(Settings.load(args.demo).prepare(), args.once)
