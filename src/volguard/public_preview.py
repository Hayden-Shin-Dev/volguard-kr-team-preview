"""Temporary, isolated team preview. Never expose the single-user API directly."""

import asyncio
import json
import os
import re
import secrets
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .config import KST, ROOT, Settings
from .intelligence import DeskJobs, execute
from .store import Store
from .terminal_data import stock
from .web_api import create_app

COOKIE = "vg_preview"
DIST = ROOT / "frontend/dist"
SAFE_REPORTS = {"test", "ensemble-study", "platform-direction", "platform-range",
                "platform-range-adaptation", "platform-agent-quality", "platform-direction-development",
                "forward", "broker-forecast"}
SAFE_REPORTS.update({"direction-integration", "direction-range-oof", "platform-direction-accuracy",
                     "platform-direction-dataset", "direction-math", "new-information",
                     "direction-calibration-study", "direction-joint-study", "probability-quality-history",
                     "probability-quality-forward", "free-information-probability",
                     "direction-historical-reliability", "direction-forward", "probability-evidence",
                     "selective-direction", "event-direction", "technical-direction", "technical-history",
                     "technical-conditions", "paper-signals"})
SAFE_REPORTS.add("scenario-direction-preview")
META_PREFIXES = ("display_current:", "current_attempt:", "display_bar:", "display_attempt:",
                 "forecast_error:")


def redact(value):
    """Keep local paths and credentials out of the public JSON and downloads."""
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()
                if k.lower() not in {"api_key", "app_key", "app_secret", "access_token", "authorization"}}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str) and re.search(r"[A-Za-z]:[\\/]|file://|/Users/|/home/", value):
        return "서버 내부 자료"
    return value


def link_directory(source, target):
    if not source.exists() or target.exists():
        return
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(source), str(target))
    else:
        target.symlink_to(source, target_is_directory=True)


@dataclass
class Visitor:
    settings: Settings
    app: FastAPI
    csrf: str = ""
    synced: float = 0


class Preview:
    def __init__(self, source, root, broker_refresh=False):
        self.source, self.root = source, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.visitors = {}
        self.lock = asyncio.Lock()
        self.slots = asyncio.Semaphore(4)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview-ai")
        self.rate = {}
        self.broker_refresh = broker_refresh
        self.quote_lock = threading.Lock()
        self.last_quote_call = 0
        self.market = Settings(self.root / "market", source.environment).prepare()
        shutil.copy2(source.data_dir / "config/universe.json", self.market.data_dir / "config/universe.json")
        # Inference uses shared read-only history and frozen models; all new output stays in preview.
        for path in (source.data_dir / "silver").glob("*.parquet"):
            target = self.market.data_dir / "silver" / path.name
            if not target.exists():
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copy2(path, target)
        for path in (source.data_dir / "config").glob("*.json"):
            shutil.copy2(path, self.market.data_dir / "config" / path.name)
        self.forecast_lock = threading.Lock()
        with sqlite3.connect(self.root / "limits.sqlite") as db:
            db.execute("CREATE TABLE IF NOT EXISTS calls(day TEXT, visitor TEXT, job TEXT, PRIMARY KEY(visitor,job))")

    def allowed_host(self, host):
        urlfile = self.root / "url.txt"
        public = urlsplit(urlfile.read_text().strip()).netloc if urlfile.exists() else ""
        render_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
        return host in {"127.0.0.1:8766", "localhost:8766", "testserver", public, render_host} and bool(host)

    def limited(self, key, maximum=120):
        now = time.monotonic()
        entries = [x for x in self.rate.get(key, []) if now - x < 60]
        self.rate[key] = entries
        if len(entries) >= maximum:
            return True
        entries.append(now)
        return False

    def reserve_ai(self, sid, job):
        today = datetime.now(KST).date().isoformat()
        with sqlite3.connect(self.root / "limits.sqlite") as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM calls WHERE visitor=? AND job=?", (sid, job)).fetchone():
                return True
            total = db.execute("SELECT count(*) FROM calls WHERE day=?", (today,)).fetchone()[0]
            own = db.execute("SELECT count(*) FROM calls WHERE day=? AND visitor=?", (today, sid)).fetchone()[0]
            if total >= 8 or own >= 2:
                return False
            db.execute("INSERT INTO calls VALUES(?,?,?)", (today, sid, job))
        return True

    def prepare(self, sid):
        root = self.root / "visitors" / sid
        root.mkdir(parents=True, exist_ok=True)
        # These endpoints only read market data. Broker refresh and issuance are intercepted below.
        for name in ("silver", "forward"):
            link_directory(self.source.data_dir / name, root / name)
        settings = Settings(root, self.source.environment).prepare()
        returning = settings.db_path.exists()
        for path in (self.source.data_dir / "config").glob("*.json"):
            shutil.copy2(path, root / "config" / path.name)
        for name in SAFE_REPORTS:
            path = self.source.data_dir / "reports" / f"{name}.json"
            if path.exists():
                shutil.copy2(path, root / "reports" / path.name)
        # No owner watchlist, holdings, risk cache, analyses, jobs, or operational logs are copied.
        universe = json.loads((root / "config/universe.json").read_text(encoding="utf8"))["members"]
        defaults = [m for m in universe if m["symbol"] in {"005930", "000660", "005380"}]
        if not returning:
            Store(settings.db_path).set_watchlist(defaults or universe[:3], {m["symbol"] for m in universe})
        return Visitor(settings, create_app(settings, run_worker=False))

    def sync(self, visitor):
        if time.monotonic() - visitor.synced < 5:
            return
        with Store(self.source.db_path).connect() as source, Store(visitor.settings.db_path).connect() as dest:
            for table in ("quotes", "predictions", "alerts"):
                rows = source.execute(f"SELECT * FROM {table}").fetchall()
                if rows:
                    marks = ",".join("?" for _ in rows[0])
                    dest.executemany(f"INSERT OR REPLACE INTO {table} VALUES ({marks})", [tuple(r) for r in rows])
            rows = source.execute("SELECT key,payload FROM meta").fetchall()
            safe = [(r[0], r[1]) for r in rows if r[0] == "connection" or r[0].startswith(META_PREFIXES)]
            dest.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)", safe)
            # Broker display prices use a shared preview cache, never the owner's DB.
            with Store(self.market.db_path).connect() as market:
                for row in market.execute("SELECT key,payload FROM meta"):
                    if row[0].startswith(META_PREFIXES):
                        dest.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", tuple(row))
                rows = market.execute("SELECT * FROM predictions").fetchall()
                if rows:
                    marks = ",".join("?" for _ in rows[0])
                    dest.executemany(f"INSERT OR IGNORE INTO predictions VALUES ({marks})", [tuple(r) for r in rows])
        visitor.synced = time.monotonic()

    def ensure_forecast(self, visitor, symbol):
        if not self.broker_refresh:
            return
        members = json.loads((self.market.data_dir / "config/universe.json").read_text(encoding="utf8"))["members"]
        if symbol not in {m["symbol"] for m in members}:
            raise ValueError("지원하지 않는 종목입니다. 종목 목록에서 선택하세요.")
        from .live_forecast import calculate

        with self.forecast_lock:
            try:
                calculate(self.market, symbol)
            finally:
                visitor.synced = 0
                self.sync(visitor)

    def refresh_quote(self, visitor, symbol):
        if not self.broker_refresh or not self.quote_lock.acquire(blocking=False):
            return
        try:
            if time.monotonic() - self.last_quote_call >= 1:
                from .display_quotes import refresh_current
                self.last_quote_call = time.monotonic()
                refresh_current(self.market, symbol)
            visitor.synced = 0
            self.sync(visitor)
        finally:
            self.quote_lock.release()

    async def visitor(self, request):
        sid = request.cookies.get(COOKIE)
        async with self.lock:
            if sid not in self.visitors:
                if len(self.visitors) >= 32:
                    return None, None
                if not (isinstance(sid, str) and re.fullmatch("[a-f0-9]{64}", sid)
                        and (self.root / "visitors" / sid / "state.sqlite").exists()):
                    sid = secrets.token_hex(32)
                visitor = await asyncio.to_thread(self.prepare, sid)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=visitor.app), base_url="http://localhost:8765") as client:
                    visitor.csrf = (await client.get("/api/session")).json()["csrf"]
                self.visitors[sid] = visitor
            visitor = self.visitors[sid]
            await asyncio.to_thread(self.sync, visitor)
        return sid, visitor

    def run_job(self, visitor):
        jobs = DeskJobs(visitor.settings.data_dir)
        job = jobs.claim()
        if job:
            execute(visitor.settings, jobs, job)


def create_preview(source, root, broker_refresh=False):
    service = Preview(source, root, broker_refresh)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.preview = service

    @app.middleware("http")
    async def boundary(request, call_next):
        host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        if not service.allowed_host(host):
            return JSONResponse({"detail": "허용되지 않은 주소입니다"}, status_code=400)
        if origin and origin not in {f"https://{host}", f"http://{host}"}:
            return JSONResponse({"detail": "허용되지 않은 요청 출처입니다"}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site" and request.method != "GET":
            return JSONResponse({"detail": "같은 체험 화면에서 요청하세요"}, status_code=403)
        if service.limited("global", 600):
            return JSONResponse({"detail": "잠시 후 다시 시도하세요"}, status_code=429)
        try:
            response = await call_next(request)
        except Exception:
            return JSONResponse({"detail": "체험 서버 처리에 실패했습니다. 잠시 후 다시 시도하세요."}, status_code=503)
        response.headers.update({"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
                                 "X-Frame-Options": "DENY", "X-Robots-Tag": "noindex, nofollow",
                                 "Cache-Control": "no-store"})
        return response

    @app.get("/preview.js")
    def script():
        content = (ROOT / "scripts/public-preview.js").read_text(encoding="utf8")
        if os.environ.get("RENDER_EXTERNAL_HOSTNAME"):
            content = content.replace("호스트 PC가 켜져 있을 때 접속 가능", "Render 서버에서 제공")
        return Response(content, media_type="text/javascript")

    @app.get("/healthz")
    def health():
        ready = (source.data_dir / "silver/bars.parquet").is_file()
        return JSONResponse({"ready": ready}, status_code=200 if ready else 503)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def dispatch(path, request: Request):
        if path.startswith("assets/") and request.method == "GET":
            target = (DIST / path).resolve()
            if target.is_relative_to((DIST / "assets").resolve()) and target.is_file():
                return FileResponse(target)
            return Response(status_code=404)
        if not path.startswith("api/"):
            if path != "" or request.method != "GET":
                return Response(status_code=404)
            sid, visitor = await service.visitor(request)
            if visitor is None:
                return Response("체험 서버 방문자 한도에 도달했습니다", status_code=503)
            html = (DIST / "index.html").read_text(encoding="utf8")
            response = HTMLResponse(html.replace("<head>", '<head><script src="/preview.js"></script>'))
            response.set_cookie(COOKIE, sid, httponly=True, secure=bool(os.environ.get("RENDER_EXTERNAL_HOSTNAME")) or "trycloudflare.com" in request.headers.get("host", ""), samesite="lax", max_age=86400)
            return response
        if service.limited("api", 400):
            return JSONResponse({"detail": "접속이 많습니다. 잠시 후 다시 시도하세요."}, status_code=429)
        sid, visitor = await service.visitor(request)
        if visitor is None:
            return JSONResponse({"detail": "체험 서버의 방문자 한도에 도달했습니다. 관리자에게 알려주세요."}, status_code=503)

        async def action():
            if service.limited(sid):
                return JSONResponse({"detail": "요청이 많습니다. 잠시 후 다시 시도하세요."}, status_code=429)
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 32768:
                    return JSONResponse({"detail": "입력 크기를 줄여주세요"}, status_code=413)
            mutation = request.method != "GET"
            if mutation and not secrets.compare_digest(request.headers.get("x-vg-csrf", ""), visitor.csrf):
                return JSONResponse({"detail": "화면을 새로고침하세요"}, status_code=403)
            method = request.method
            forward_path = path
            if path.startswith("api/stream/"):
                symbol = path.rsplit("/", 1)[-1]
                value = await asyncio.to_thread(stock, visitor.settings, symbol)
                return JSONResponse(redact({k: value.get(k) for k in ("quote", "prediction", "gauge", "market_open")}))
            if re.fullmatch(r"api/investigations/[a-f0-9]{32}/events", path):
                jobs = DeskJobs(visitor.settings.data_dir)
                job = jobs.read(path.split("/")[2])
                if not job:
                    return JSONResponse({"detail": "조사 기록이 없습니다"}, status_code=404)
                return JSONResponse(redact(job))
            if re.fullmatch(r"api/stocks/[0-9A-Z]{6}/(refresh|forecast)", path):
                symbol = path.split("/")[2]
                if path.endswith("/refresh") and method == "POST":
                    await asyncio.to_thread(service.refresh_quote, visitor, symbol)
                value = await asyncio.to_thread(stock, visitor.settings, symbol)
                if path.endswith("/forecast"):
                    value = value.get("prediction")
                    if not value:
                        try:
                            await asyncio.to_thread(service.ensure_forecast, visitor, symbol)
                        except ValueError as exc:
                            return JSONResponse({"detail": str(exc)}, status_code=409)
                        value = (await asyncio.to_thread(stock, visitor.settings, symbol)).get("prediction")
                    if not value:
                        return JSONResponse({"detail": "예측이 미발행된 종목입니다. 관리자에게 데이터 수집 상태 확인을 요청하세요."}, status_code=409)
                return JSONResponse(redact(value))
            allowed_mutations = {("POST", "api/analyses"), ("POST", "api/investigations"),
                                 ("POST", "api/portfolio"), ("PUT", "api/watchlist"), ("POST", "api/render")}
            if mutation and (method, path) not in allowed_mutations:
                return Response(status_code=405)
            if path == "api/render":
                return JSONResponse({"recorded": False})
            if path == "api/analyses" and mutation:
                try:
                    data = json.loads(body)
                except ValueError:
                    return JSONResponse({"detail": "입력을 확인하세요"}, status_code=422)
                if not data.get("parent_id"):
                    value = await asyncio.to_thread(stock, visitor.settings, data.get("symbol", ""))
                    pred = value.get("prediction")
                    if not pred or pred.get("target_date") != value.get("forecast_target"):
                        try:
                            await asyncio.to_thread(service.ensure_forecast, visitor, data.get("symbol", ""))
                        except ValueError as exc:
                            return JSONResponse({"detail": str(exc)}, status_code=409)
                        value = await asyncio.to_thread(stock, visitor.settings, data.get("symbol", ""))
                        if not value.get("prediction") or value["prediction"].get("target_date") != value.get("forecast_target"):
                            return JSONResponse({"detail": "예측이 미발행된 종목입니다. 관리자에게 데이터 수집 상태 확인을 요청하세요."}, status_code=409)
            if path == "api/investigations" and mutation:
                try:
                    data = json.loads(body)
                except ValueError:
                    return JSONResponse({"detail": "입력을 확인하세요"}, status_code=422)
                job = data.get("request_id", "")
                if not re.fullmatch("[a-f0-9]{32}", job):
                    return JSONResponse({"detail": "잘못된 조사 요청입니다"}, status_code=422)
                if not service.reserve_ai(sid, job):
                    return JSONResponse({"detail": "체험판 AI 한도에 도달했습니다. 브라우저당 하루 2회, 전체 하루 8회입니다."}, status_code=429)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=visitor.app), base_url="http://localhost:8765", timeout=90) as client:
                upstream = await client.request(method, "/" + forward_path,
                    params=request.query_params, content=bytes(body),
                    headers={"content-type": "application/json", "x-vg-csrf": visitor.csrf})
            if path == "api/investigations" and mutation and upstream.is_success:
                service.pool.submit(service.run_job, visitor)
            if "application/json" in upstream.headers.get("content-type", ""):
                value = upstream.json()
                if path == "api/session":
                    value["preview"] = {"ai_per_browser": 2, "ai_total": 8, "temporary": True}
                return JSONResponse(redact(value), status_code=upstream.status_code)
            # Only report text may pass through; unknown API routes stay errors.
            return Response(redact(upstream.text), status_code=upstream.status_code,
                            media_type=upstream.headers.get("content-type", "text/plain"))

        async with service.slots:
            response = await action()
        response.set_cookie(COOKIE, sid, httponly=True, secure=bool(os.environ.get("RENDER_EXTERNAL_HOSTNAME")) or "trycloudflare.com" in request.headers.get("host", ""), samesite="lax", max_age=86400)
        return response

    return app


if __name__ == "__main__":
    import uvicorn
    defaults = Settings.load()
    paths = json.loads((ROOT / ".service-paths.json").read_text(encoding="utf8"))
    base = Path(paths["data_root"])
    uvicorn.run(create_preview(Settings(base / "live", defaults.environment), base / "public-preview", broker_refresh=True),
                host="127.0.0.1", port=8766, access_log=False)
