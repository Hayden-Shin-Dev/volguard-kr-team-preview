"""Local process supervision and durable same-day batch retries; no trading API."""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

from .config import KST
from .data import Lake
from .operations import report
from .store import Store
from .util import exclusive_run, utcnow


def due_slot(now, sessions, attempts):
    day = now.astimezone(KST).date().isoformat()
    if day not in {s["date"] for s in sessions}:
        return None
    clock = now.astimezone(KST).strftime("%H:%M")
    if clock > "21:00":
        return None
    due = [slot for slot in ("18:30", "19:00", "20:00") if slot <= clock and f"{day}:{slot}" not in attempts]
    return due[-1] if due else None


def serve(settings):
    root = Path(__file__).resolve().parents[2]
    logs = settings.data_dir / "logs"
    logs.mkdir(exist_ok=True)
    store, lake = Store(settings.db_path), Lake(settings.data_dir)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root / ".venv" / "Lib" / "site-packages"))),
        "PYTHONIOENCODING": "utf-8",
    }
    children, streams = {}, {}
    retry_at = {}

    def start(name, args):
        path = logs / f"{name}.log"
        if path.exists() and path.stat().st_size > 5_000_000:
            os.replace(path, path.with_suffix(".previous.log"))
        if name in streams:
            streams[name].close()
        streams[name] = path.open("a", encoding="utf8")
        children[name] = subprocess.Popen(
            [sys.executable, "-X", "utf8", *args],
            cwd=root,
            env=env,
            stdout=streams[name],
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    with exclusive_run(settings.data_dir / "supervisor.lock"):
        try:
            while True:
                now = datetime.now(KST)
                sessions = lake.calendar()
                for name, args in {
                    "collector": ["-m", "volguard.cli", "collect"],
                    "report_worker": ["-m", "volguard.report_worker"],
                    "terminal": ["-m", "volguard.web_api"],
                    "dashboard": [
                        "-m",
                        "streamlit",
                        "run",
                        str(root / "app.py"),
                        "--server.address",
                        "127.0.0.1",
                        "--server.port",
                        "8501",
                        "--server.headless",
                        "true",
                        "--browser.gatherUsageStats",
                        "false",
                    ],
                }.items():
                    process = children.get(name)
                    if process is not None and process.poll() is None:
                        continue
                    if process is not None:
                        store.put(
                            "meta", f"process_exit:{name}", {"at": utcnow(), "code": process.returncode}
                        )
                        children.pop(name)
                        retry_at[name] = time.monotonic() + 30
                    if time.monotonic() < retry_at.get(name, 0):
                        continue
                    if name == "report_worker" and name not in retry_at:
                        heartbeat = store.get("meta", "report_worker", {}).get("heartbeat")
                        if heartbeat and 0 <= (now - datetime.fromisoformat(heartbeat)).total_seconds() < 180:
                            continue
                    if name in {"dashboard", "terminal"}:
                        try:
                            if (
                                httpx.get("http://127.0.0.1:8765/api/health" if name == "terminal" else "http://127.0.0.1:8501/_stcore/health", timeout=2).status_code
                                == 200
                            ):
                                continue
                        except httpx.HTTPError:
                            pass
                    start(name, args)
                attempts = store.get("meta", "batch_attempts", {})
                batch = children.get("daily")
                if batch is not None and batch.poll() is not None:
                    store.put("meta", "last_batch_exit", {"at": utcnow(), "code": batch.returncode})
                    children.pop("daily")
                slot = due_slot(now, sessions, attempts)
                if slot and "daily" not in children:
                    key = f"{now.date().isoformat()}:{slot}"
                    attempts[key] = {"started_at": utcnow()}
                    store.put("meta", "batch_attempts", dict(list(attempts.items())[-90:]))
                    start("daily", ["-m", "volguard.cli", "daily", "--date", now.date().isoformat()])
                store.put(
                    "meta",
                    "supervisor",
                    {
                        "status": "running",
                        "heartbeat": utcnow(),
                        "pid": os.getpid(),
                        "children": {k: p.pid for k, p in children.items()},
                        "calendar_end": sessions[-1]["date"],
                        "dashboard": "http://127.0.0.1:8501",
                        "terminal": "http://127.0.0.1:8765",
                    },
                )
                report(store, settings)
                time.sleep(5)
        finally:
            for process in children.values():
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
            for file in streams.values():
                file.close()
            store.put("meta", "supervisor", {"status": "stopped", "at": utcnow()})


if __name__ == "__main__":
    from .config import Settings

    serve(Settings.load().prepare())
