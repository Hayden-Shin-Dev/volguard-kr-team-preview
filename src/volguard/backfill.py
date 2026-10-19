"""Resumable official KRX history download, independent of model eligibility.

Checkpoints point to immutable Bronze objects. Empty exchange holidays are
recorded too. Nothing in this downloader attests historical publication time
or adjusted prices merely because a request succeeded.
"""

import concurrent.futures
import json
import os
import threading
import time
from datetime import datetime

import httpx
import pandas as pd

from .config import KST
from .data import Lake
from .providers import KRX, ProviderError
from .util import atomic_json, exclusive_run, utcnow


def download_history(settings, start="2010-01-04", end="2026-10-15", budget=9000):
    lake = Lake(settings.data_dir)
    directory = lake.root / "config" / "downloads"
    directory.mkdir(parents=True, exist_ok=True)
    # Weekdays are candidate request dates, never assumed exchange sessions.
    dates = pd.bdate_range(start, end).strftime("%Y-%m-%d").tolist()
    if pd.Timestamp(end).date() >= datetime.now(KST).date():
        raise ValueError("Historical backfill ends before today")
    dates.sort(key=lambda day: (day < "2020-01-01", day))
    tasks = [(day, kind) for day in dates for kind in ("stock", "index")]
    tasks = [(day, kind) for day, kind in tasks if not (directory / f"{day}-{kind}.json").exists()]
    key = os.environ.get("KRX_API_KEY", "").strip()
    if not key:
        raise ProviderError("KRX_KEY_MISSING")
    limiter = threading.Lock()
    accounting = {"requests": 0, "next": 0.0}
    progress = {
        "started_at": utcnow(),
        "start": start,
        "end": end,
        "pending_at_start": len(tasks),
        "completed": 0,
        "failed": 0,
        "rows": 0,
        "status": "running",
    }
    report_path = lake.root / "reports" / "backfill.json"
    daily_path = directory / f"usage-{datetime.now(KST).date()}.json"
    previous = json.loads(daily_path.read_text()) if daily_path.exists() else {"requests": 0}
    accounting["requests"] = previous["requests"]

    # Reserve room below the documented 10,000/key/day maximum for other probes.
    def fetch(task):
        day, kind = task
        endpoint = "sto/stk_bydd_trd" if kind == "stock" else "idx/kospi_dd_trd"
        for attempt in range(3):
            with limiter:
                if accounting["requests"] >= budget:
                    return {"day": day, "kind": kind, "error": "DAILY_BUDGET"}
                time.sleep(max(0, accounting["next"] - time.monotonic()))
                accounting["next"] = time.monotonic() + 0.125
                accounting["requests"] += 1
                atomic_json(daily_path, {"requests": accounting["requests"]})
            try:
                response = client.get(
                    f"{KRX.BASE}/{endpoint}",
                    headers={"AUTH_KEY": key},
                    params={"basDd": day.replace("-", "")},
                )
                if response.status_code in (401, 403):
                    return {"day": day, "kind": kind, "error": "AUTH_OR_SERVICE_PERMISSION"}
                response.raise_for_status()
                body = response.json()
                rows = body.get("OutBlock_1")
                if not isinstance(rows, list):
                    raise ValueError("SCHEMA")
                version = lake.bronze(response.content, f"krx-{kind}-history", {"date": day})
                result = {
                    "day": day,
                    "kind": kind,
                    "rows": len(rows),
                    "hash": version,
                    "fetched_at": utcnow(),
                }
                atomic_json(directory / f"{day}-{kind}.json", result)
                return result
            except (httpx.HTTPError, ValueError):
                if attempt < 2:
                    time.sleep(2**attempt)
        return {"day": day, "kind": kind, "error": "TRANSPORT_HTTP_OR_SCHEMA"}

    with (
        exclusive_run(lake.root / "backfill.lock"),
        httpx.Client(timeout=30, follow_redirects=False) as client,
    ):
        atomic_json(report_path, progress)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for result in pool.map(fetch, tasks):
                if "error" in result:
                    progress["failed"] += 1
                    progress["last_error"] = result
                else:
                    progress["completed"] += 1
                    progress["rows"] += result["rows"]
                total = progress["completed"] + progress["failed"]
                if total % 100 == 0 or total == len(tasks):
                    progress.update(
                        updated_at=utcnow(), last_date=result["day"], requests=accounting["requests"]
                    )
                    atomic_json(report_path, progress)
                    print(json.dumps(progress, ensure_ascii=False), flush=True)
        progress.update(status="complete" if not progress["failed"] else "incomplete", finished_at=utcnow())
        atomic_json(report_path, progress)
    return progress


if __name__ == "__main__":
    from .config import Settings

    download_history(Settings.load().prepare())
