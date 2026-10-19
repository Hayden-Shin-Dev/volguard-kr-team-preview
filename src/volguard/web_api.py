"""Loopback-only React API. Existing forecast and portfolio engines remain authoritative."""

import asyncio
import json
import secrets
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import KST, ROOT, Settings
from .data import Lake
from .display_quotes import refresh as refresh_display
from .display_quotes import refresh_current
from .intelligence import ROLES, DeskJobs, execute, report_markdown, settings_public
from .investigation import Investigations, markdown_report
from .portfolio import calculate
from .realtime import gauge, within_session
from .store import Store
from .terminal_data import (
    display_prediction,
    frame,
    model_comparison,
    overview,
    portfolio_view,
    read_json,
    stock,
)
from .terminal_telemetry import acknowledge, ticket
from .util import digest, utcnow


class InvestigationInput(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=5)
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    analysis_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class AnalysisInput(BaseModel):
    symbol: str = Field(pattern=r"^[0-9A-Z]{6}$")
    direction: str = Field(default="up", pattern=r"^(up|flat|down)$")
    magnitude: str = Field(default="median", pattern=r"^(low|median|high)$")
    parent_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class HoldingsInput(BaseModel):
    holdings: dict[str, float]
    cash: float = Field(ge=0, allow_inf_nan=False)


class WatchInput(BaseModel):
    symbols: list[str] = Field(max_length=30)


class PaintInput(BaseModel):
    ticket: str = Field(max_length=1000)
    visible: bool


def create_app(settings=None, run_worker=True):
    if settings is None:
        defaults = Settings.load()
        paths = read_json(ROOT / ".service-paths.json", {})
        settings = (
            Settings(Path(paths["data_root"]) / "live", defaults.environment)
            if paths.get("data_root")
            else defaults
        )
    settings = settings.prepare()
    jobs, store, lake = DeskJobs(settings.data_dir), Store(settings.db_path), Lake(settings.data_dir)
    csrf = secrets.token_urlsafe(32)
    stop = threading.Event()

    def worker():
        jobs.interrupt_running()
        while not stop.is_set():
            store.put("meta", "intelligence_worker", {"heartbeat": utcnow(), "status": "running"})
            job = jobs.claim()
            if job:
                execute(settings, jobs, job)
            else:
                stop.wait(1)

    def price_worker():
        while not stop.is_set():
            try:
                refresh_display(settings, [x["symbol"] for x in store.watchlist()])
            except (ValueError, RuntimeError, OSError):
                store.put("meta", "display_worker", {"status": "unavailable", "at": utcnow()})
            stop.wait(60)

    def forecast_worker():
        from .live_forecast import run_batch, session_pair

        while not stop.is_set():
            try:
                now = datetime.now(KST)
                anchor, following = session_pair(lake.calendar(), now)
                if (
                    settings.data_dir / "config/forecast-release.json"
                ).exists() and now < datetime.fromisoformat(
                    f"{following['date']}T{following['open_time']}"
                ).replace(tzinfo=KST):
                    run_batch(settings)
            except Exception as exc:
                store.put(
                    "meta",
                    "forecast_worker",
                    {"status": "retrying", "error_type": type(exc).__name__, "at": utcnow()},
                )
            stop.wait(60)

    @asynccontextmanager
    async def lifespan(app):
        thread = threading.Thread(target=worker, daemon=True)
        prices = threading.Thread(target=price_worker, daemon=True)
        forecasts = threading.Thread(target=forecast_worker, daemon=True)
        if run_worker:
            thread.start()
            prices.start()
            forecasts.start()
        yield
        stop.set()
        if run_worker:
            thread.join(timeout=2)
            prices.join(timeout=2)
            forecasts.join(timeout=2)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_boundary(request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in {
            "http://127.0.0.1:8765",
            "http://localhost:8765",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        }:
            return JSONResponse({"detail": "허용되지 않은 요청 출처입니다"}, status_code=403)
        if request.method in {"POST", "PUT", "DELETE", "PATCH"} and not secrets.compare_digest(
            request.headers.get("x-vg-csrf", ""), csrf
        ):
            return JSONResponse({"detail": "화면을 새로고침한 후 다시 시도하세요"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def validation_error(request, exc):
        return JSONResponse(
            {
                "detail": str(exc)
                if len(str(exc)) < 150 and "http" not in str(exc)
                else "입력 또는 자료를 확인할 수 없습니다"
            },
            status_code=422,
        )

    def members():
        return {m["symbol"]: m["name"] for m in lake.universe()["members"]}

    @app.get("/api/session")
    def session():
        return {"csrf": csrf, "ai": settings_public(), "roles": ROLES, "server_time": utcnow()}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "time": utcnow()}

    @app.post("/api/render")
    def painted(body: PaintInput):
        return {
            "recorded": acknowledge(
                store, csrf, body.ticket, body.visible, datetime.now(KST), lake.calendar()
            )
        }

    @app.get("/api/overview")
    def summary():
        return overview(settings)

    @app.get("/api/stocks/{symbol}")
    def stock_view(symbol: str):
        return stock(settings, symbol)

    @app.get("/api/stocks/{symbol}/prediction-history")
    def prediction_history(symbol: str):
        from .chart_history import chart_history

        if symbol not in members():
            raise HTTPException(404, "지원하지 않는 종목입니다")
        return chart_history(settings, symbol)

    @app.post("/api/stocks/{symbol}/refresh")
    async def refresh_stock(symbol: str):
        if symbol not in members():
            raise HTTPException(404, "지원하지 않는 종목입니다")
        await asyncio.to_thread(refresh_current, settings, symbol)
        await asyncio.to_thread(refresh_display, settings, [symbol])
        return stock(settings, symbol)

    @app.post("/api/stocks/{symbol}/forecast")
    async def chart_forecast(symbol: str):
        from .live_forecast import calculate

        if symbol not in members():
            raise HTTPException(404, "지원하지 않는 종목입니다")
        return await asyncio.to_thread(calculate, settings, symbol)

    @app.put("/api/watchlist")
    def save_watchlist(body: WatchInput):
        universe = members()
        if len(set(body.symbols)) != len(body.symbols) or not set(body.symbols).issubset(universe):
            raise ValueError("중복되거나 지원하지 않는 종목입니다")
        store.set_watchlist([{"symbol": s, "name": universe[s]} for s in body.symbols], set(universe))
        return {"saved": True}

    @app.get("/api/portfolio")
    def portfolio():
        return portfolio_view(settings)

    @app.post("/api/portfolio")
    def save_portfolio(body: HoldingsInput):
        if not set(body.holdings).issubset(members()):
            raise ValueError("지원하지 않는 보유 종목입니다")
        path = settings.data_dir / "silver/bars.parquet"
        return calculate(
            store, frame(path), lake.calendar(), body.holdings, body.cash, digest(path.read_bytes())
        )

    @app.get("/api/validation")
    def validation():
        report = read_json(settings.data_dir / "reports/test.json", {})
        return {
            "test": {
                k: report[k]
                for k in (
                    "metrics",
                    "har_pass",
                    "ml_pass",
                    "test_start",
                    "test_end",
                    "kind",
                    "limitations",
                    "improvement",
                )
                if k in report
            },
            "operations": read_json(settings.data_dir / "reports/operations.json", {}),
            "forward": read_json(settings.data_dir / "reports/forward.json", {}),
            "model_comparison": model_comparison(settings),
            "active_model": read_json(settings.data_dir / "config/forecast-release.json", {}).get("model"),
        }

    @app.post("/api/analyses")
    def analysis(body: AnalysisInput):
        from .analysis_flow import create

        return create(settings, body.symbol, body.direction, body.magnitude, body.parent_id)

    @app.get("/api/analyses/{key}")
    def analysis_read(key: str):
        from .analysis_flow import read

        return read(settings, key)

    @app.get("/api/stocks/{symbol}/replay/{day}")
    def replay_day(symbol: str, day: str):
        from .analysis_flow import replay

        return replay(settings, symbol, day)

    @app.get("/api/platform-validation")
    def platform_validation():
        from .direction import public_status

        r = read_json(settings.data_dir / "reports/platform-range.json", {})
        return {
            "direction": public_status(settings),
            "direction_development": read_json(settings.data_dir / "reports/platform-direction-development.json", {}),
            "range_adaptation": read_json(settings.data_dir / "reports/platform-range-adaptation.json", {}),
            "agent_quality": read_json(settings.data_dir / "reports/platform-agent-quality.json", {}),
            "range": {
                k: r.get(k)
                for k in ["baseline", "candidates", "by_regime", "improving_candidates", "sector_status"]
            },
        }

    @app.get("/api/direction-comparison")
    def direction_comparison():
        from .direction_view import comparison

        return comparison(settings)

    @app.get("/api/direction-integration")
    def direction_integration():
        from .direction_view import integration

        return integration(settings)

    @app.get("/api/direction-math")
    def direction_math():
        from .direction_math_view import view

        return view(settings)

    @app.get("/api/new-information")
    def new_information():
        from .terminal_data import read_json

        return read_json(settings.data_dir / "reports/new-information.json", {"status": "pending"})

    @app.get("/api/direction-calibration-study")
    def direction_calibration_study():
        return read_json(settings.data_dir / "reports/direction-calibration-study.json", {"status": "pending"})

    @app.get("/api/direction-joint-study")
    def direction_joint_study():
        return read_json(settings.data_dir / "reports/direction-joint-study.json", {"status": "pending"})

    @app.get("/api/probability-quality")
    def probability_quality_report():
        return {
            "history": read_json(settings.data_dir / "reports/probability-quality-history.json", {}),
            "forward": read_json(settings.data_dir / "reports/probability-quality-forward.json", {}),
            "free": read_json(settings.data_dir / "reports/free-information-probability.json", {}),
        }

    @app.get("/api/direction-historical-reliability")
    def direction_historical_reliability():
        from .terminal_data import read_json

        return read_json(settings.data_dir / "reports/direction-historical-reliability.json", {"status": "pending"})

    @app.get("/api/direction-forward")
    def direction_forward_report():
        from .terminal_data import read_json

        return read_json(settings.data_dir / "reports/direction-forward.json", {"status": "pending", "models": []})

    @app.get("/api/probability-evidence")
    def probability_evidence():
        from .terminal_data import read_json

        return read_json(settings.data_dir / "reports/probability-evidence.json", {"status": "pending"})

    @app.get("/api/selective-direction")
    def selective_direction():
        from .terminal_data import read_json

        return read_json(settings.data_dir / "reports/selective-direction.json", {"status": "pending"})

    @app.get("/api/event-direction")
    def event_direction():
        from .terminal_data import read_json

        return read_json(settings.data_dir / "reports/event-direction.json", {"status": "pending"})

    @app.get("/api/technical-research/{study}")
    def technical_research(study: str):
        from .terminal_data import read_json

        names = {"indicators": "technical-direction", "history": "technical-history", "conditions": "technical-conditions"}
        if study not in names:
            raise ValueError("지원하지 않는 연구 결과입니다")
        return read_json(settings.data_dir / "reports" / f"{names[study]}.json", {"status": "pending"})

    @app.get("/api/paper-signals")
    def paper_signals():
        from .terminal_data import read_json

        return read_json(settings.data_dir / "reports/paper-signals.json", {"status": "pending"})

    @app.get("/api/stocks/{symbol}/direction-integration-history")
    def direction_integration_history(symbol: str):
        from .direction_view import integration_history

        if symbol not in members():
            raise ValueError("지원하지 않는 종목입니다")
        return integration_history(settings, symbol)

    @app.get("/api/stocks/{symbol}/direction-history")
    def direction_history(symbol: str, model: str | None = None):
        from .direction_view import history

        if symbol not in members():
            raise ValueError("지원하지 않는 종목입니다")
        return history(settings, symbol, model)

    @app.post("/api/investigations")
    def investigate(body: InvestigationInput):
        if not settings_public()["ready"]:
            raise HTTPException(409, "OpenAI API 연결 설정이 필요합니다")
        if len(set(body.symbols)) != len(body.symbols) or not set(body.symbols).issubset(members()):
            raise ValueError("지원하지 않거나 중복된 조사 종목입니다")
        if body.analysis_id:
            from .analysis_flow import read

            snapshot = read(settings, body.analysis_id)
            if body.symbols != [snapshot["symbol"]]:
                raise ValueError("조사 종목과 분석 스냅샷이 다릅니다")
        return {"id": jobs.enqueue(body.symbols, body.request_id, body.analysis_id)}

    @app.get("/api/investigations")
    def runs():
        return jobs.recent()

    @app.get("/api/investigations/{key}")
    def run(key: str):
        value = jobs.read(key)
        if not value:
            raise HTTPException(404, "조사 기록이 없습니다")
        return value

    @app.get("/api/investigations/{key}/download")
    def download_report(key: str, format: str = "md"):
        value = run(key)
        content = (
            json.dumps(value, ensure_ascii=False, indent=2) if format == "json" else report_markdown(value)
        )
        suffix = "json" if format == "json" else "md"
        return Response(
            content,
            media_type="application/json" if suffix == "json" else "text/markdown",
            headers={"Content-Disposition": f'attachment; filename="volguard-{key[:12]}.{suffix}"'},
        )

    @app.get("/api/investigations/{key}/events")
    async def events(key: str, request: Request, after: int = 0):
        run(key)
        try:
            cursor = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            cursor = after

        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                rows = jobs.events(key, cursor)
                for row in rows:
                    cursor = row["seq"]
                    yield f"id: {cursor}\ndata: {json.dumps(row, ensure_ascii=False)}\n\n"
                if jobs.read(key)["status"] in {"completed", "partial", "failed", "interrupted"}:
                    return
                yield ": keepalive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/stream/{symbol}")
    async def quotes(symbol: str, request: Request):
        if symbol not in members():
            raise HTTPException(404, "지원하지 않는 종목입니다")

        async def stream():
            while not await request.is_disconnected():
                now = datetime.now(KST)
                q = store.get("quotes", symbol)
                today = now.date().isoformat()
                upcoming = next((s["date"] for s in lake.calendar() if s["date"] > today), None)
                state = gauge(
                    q,
                    store.prediction(symbol, now.date().isoformat()),
                    store.get("meta", "connection", {}).get("confirmed_at"),
                    now,
                    lake.calendar(),
                )
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "quote": q,
                            "prediction": display_prediction(store, symbol, today, upcoming),
                            "gauge": state,
                            "market_open": within_session(now, lake.calendar()),
                            "server_time": utcnow(),
                            "render_ticket": ticket(csrf, q, now, lake.calendar()),
                            "alerts": [
                                {k: r[k] for k in ("symbol", "trade_date", "threshold", "created_at")}
                                for r in store.recent("alerts", 30)
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
                await asyncio.sleep(5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/reports/legacy")
    def old_reports():
        return Investigations(settings.data_dir).recent()

    @app.get("/api/reports/legacy/{key}")
    def old_report(key: str):
        r = Investigations(settings.data_dir).read(key)
        if not r:
            raise HTTPException(404, "보고서가 없습니다")
        return Response(
            markdown_report(r),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="volguard-legacy-{key[:12]}.md"'},
        )

    dist = Path(__file__).resolve().parents[2] / "frontend/dist"
    if (dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}")
    def frontend(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        if not (dist / "index.html").exists():
            return Response("Frontend build required", status_code=503)
        return FileResponse(dist / "index.html")

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8765, access_log=False)
