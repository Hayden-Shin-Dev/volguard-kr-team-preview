"""Durable risk investigation jobs, immutable snapshots and auditable evidence."""

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from .config import KST
from .data import Lake
from .portfolio import current_result
from .realtime import gauge
from .store import Store
from .util import atomic_json, digest, utcnow


class Investigations:
    def __init__(self, root):
        self.root = Path(root) / "investigations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "jobs.sqlite"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs(
                  id TEXT PRIMARY KEY, dedup TEXT UNIQUE, created_at TEXT, updated_at TEXT,
                  status TEXT, request TEXT, error TEXT);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def has_dedup(self, key):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM jobs WHERE dedup=?", (key,)).fetchone() is not None

    def enqueue(self, snapshot, use_ai=False, dedup=None):
        request = {"snapshot": snapshot, "use_ai": bool(use_ai)}
        job_id, now = uuid.uuid4().hex, utcnow()
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?,?,?,?,?,NULL)",
                       (job_id, dedup or job_id, now, now, "queued", json.dumps(request, ensure_ascii=False)))
            row = db.execute("SELECT id FROM jobs WHERE dedup=?", (dedup or job_id,)).fetchone()
        return row["id"]

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # A crashed network call must not be automatically billed twice.
            old = (datetime.now(KST) - timedelta(minutes=10)).isoformat()
            db.execute("UPDATE jobs SET status='interrupted',error='worker_interrupted' "
                       "WHERE status='running' AND julianday(updated_at)<julianday(?)", (old,))
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',updated_at=? WHERE id=?", (utcnow(), row["id"]))
        return {**dict(row), "request": json.loads(row["request"])}

    def finish(self, job_id, report):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Invalid report ID")
        path = self.root / f"{job_id}.json"
        if path.exists():
            raise ValueError("Report is immutable")
        atomic_json(path, report)
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?,updated_at=? WHERE id=?",
                       (report["status"], utcnow(), job_id))

    def fail(self, job_id, error_type):
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='failed',error=?,updated_at=? WHERE id=?",
                       (error_type, utcnow(), job_id))

    def recent(self, limit=30):
        with self.connect() as db:
            return [dict(r) for r in db.execute(
                "SELECT id,created_at,updated_at,status,error FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))]

    def read(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Invalid report ID")
        path = self.root / f"{job_id}.json"
        return json.loads(path.read_text(encoding="utf8")) if path.exists() else None


def capture(settings, symbols=None, trigger=None, now=None):
    now = now or datetime.now(KST)
    store, lake = Store(settings.db_path), Lake(settings.data_dir)
    members = {m["symbol"]: m["name"] for m in lake.universe()["members"]}
    saved = store.get("meta", "portfolio", {"holdings": {}, "cash": 0})
    if symbols is None:
        symbols = sorted(set(saved["holdings"]) | {w["symbol"] for w in store.watchlist()})
    if len(symbols) > 30 or not set(symbols).issubset(members):
        raise ValueError("고정 유니버스 안의 최대 30종목만 조사할 수 있습니다")
    sessions = lake.calendar()
    connection = store.get("meta", "connection", {})
    today = now.date().isoformat()
    rows = []
    for symbol in symbols:
        quote = store.get("quotes", symbol)
        prediction = store.prediction(symbol, today)
        rows.append({"symbol": symbol, "name": members[symbol], "prediction": prediction,
                     "quote": quote, "gauge": gauge(quote, prediction, connection.get("confirmed_at"), now, sessions)})
    bars_path = settings.data_dir / "silver/bars.parquet"
    bars_hash = digest(bars_path.read_bytes()) if bars_path.exists() else "missing"
    risk = current_result(store, bars_hash, sessions)
    market_path = settings.data_dir / "silver/market.parquet"
    market = None
    if market_path.exists():
        import pandas as pd

        frame = pd.read_parquet(market_path)
        if "date" in frame:
            frame = frame[frame.date.le(pd.Timestamp(today))].sort_values("date")
            if not frame.empty:
                market = json.loads(frame.tail(5).to_json(orient="records", date_format="iso"))
    return {"captured_at": now.isoformat(), "mode": "demo" if settings.data_dir.name == "demo" else "live",
            "trigger": trigger, "stocks": rows, "risk": risk, "market": market,
            "portfolio_state": "current" if risk else "not_calculated_or_stale",
            "bars_hash": bars_hash, "calendar_hash": digest(sessions)}


def local_evidence(snapshot):
    """LLM-facing facts omit holdings, cash and absolute portfolio amounts."""
    sources = []
    if snapshot.get("trigger"):
        sources.append({"id": "A1", "kind": "recorded_alert", "title": "실제 기록된 위험 알림",
                        "text": json.dumps(snapshot["trigger"], ensure_ascii=False), "url": None,
                        "retrieved_at": snapshot["captured_at"], "content_hash": digest(snapshot["trigger"])})
    for row in snapshot["stocks"]:
        p = row["prediction"] or {}
        q = row["quote"] or {}
        content = {"symbol": row["symbol"], "name": row["name"], "captured_at": snapshot["captured_at"],
                   "gauge": row["gauge"], "quote_at": q.get("event_at"),
                   "prediction": {k: p.get(k) for k in ("target_date", "low", "median", "high", "method", "status", "model_version")}}
        state = row["gauge"]
        state_text = {"no_quote": "KRX 정규장 위험 판정용 시세가 없습니다. 화면의 통합 현재가 및 확정 일봉과 별도입니다.",
                      "stale": "자료의 최신성을 확인할 수 없어 현재 위험 판정을 제한합니다.",
                      "market_closed": "정규장 밖이므로 현재 장중 위험 판정을 하지 않습니다.",
                      "ready": "현재 자료로 발행된 예측 상한과 관측 변동폭을 비교할 수 있습니다."}.get(
                          state["status"], "데이터 또는 예측의 사용 조건을 충족하지 않아 위험 판정을 제한합니다.")
        narrative = f"{row['name']}: {state_text} 상태 코드: {state['status']}."
        if state["status"] == "ready":
            narrative += f" 관측 변동폭(%): {state.get('range_pct')}. 예측 상한 소진율(%): {state.get('consumption_pct')}."
        sources.append({"id": f"L{len(sources) + 1}", "kind": "local_calculation", "title": f"{row['name']} 위험 상태",
                        "text": narrative, "facts": content, "url": None,
                        "retrieved_at": snapshot["captured_at"], "content_hash": digest(content)})
    risk = snapshot.get("risk")
    if risk:
        content = {k: risk.get(k) for k in ("method", "confidence", "status", "n", "as_of", "exception_rate", "tests")}
        sources.append({"id": f"L{len(sources) + 1}", "kind": "local_calculation", "title": "포트폴리오 검증 상태",
                        "text": json.dumps(content, ensure_ascii=False), "url": None,
                        "retrieved_at": snapshot["captured_at"], "content_hash": digest(content)})
    if snapshot.get("market"):
        text = "코스피 시장 지수의 고저 변동폭 자료입니다. 개별 기업의 주가 자료가 아닙니다. " + "; ".join(
            f"{r['date'][:10]} 변동폭 {r.get('range_pct')}%" for r in snapshot["market"])
        sources.append({"id": f"L{len(sources) + 1}", "kind": "local_market", "title": "코스피 시장 지수 고저 변동폭 일봉 (개별 종목 자료 아님)",
                        "text": text, "facts": snapshot["market"], "url": None,
                        "retrieved_at": snapshot["captured_at"], "content_hash": digest(snapshot["market"])})
    return sources


def build_report(request, external, ai=None):
    snapshot = request["snapshot"]
    sources = local_evidence(snapshot) + external["sources"]
    findings, counter, uncertainties = [], [], list(external["limitations"])
    if snapshot.get("trigger"):
        trigger = snapshot["trigger"]
        findings.append({"text": f"{trigger['created_at']}에 {trigger['symbol']}의 "
                          f"{trigger['threshold']}% 임계값 알림이 기록됐습니다. 현재 상태와 구분합니다.",
                         "source_ids": ["A1"]})
    for row in snapshot["stocks"]:
        state = row["gauge"]
        ref = next(s["id"] for s in sources if s["title"] == f"{row['name']} 위험 상태")
        if state["status"] != "ready":
            uncertainties.append(f"{row['name']}: {state['status']} — 현재 예측 기반 위험 판정 제한")
        elif 100 in state["thresholds"]:
            findings.append({"text": f"{row['name']}: 관측 변동폭이 발행된 예측 상한을 초과했습니다.", "source_ids": [ref]})
        elif 80 in state["thresholds"]:
            findings.append({"text": f"{row['name']}: 예측 상한 소진율 80%에 도달했습니다.", "source_ids": [ref]})
        else:
            counter.append({"text": f"{row['name']}: 수집 시점에 80% 임계값 미도달. 향후 변동이 작다는 보장은 아닙니다.", "source_ids": [ref]})
    if snapshot["portfolio_state"] != "current":
        uncertainties.append("현재 데이터·보유 구성에 맞는 포트폴리오 계산 결과가 없습니다.")
    uncertainties += ["예측 상한 초과와 공시의 시간적 동반만으로 원인을 입증할 수 없습니다.",
                      "반대 근거가 없다는 것은 위험이 확실하다는 뜻이 아닙니다."]
    return {"schema_version": 1, "created_at": utcnow(), "snapshot": snapshot,
            "snapshot_hash": digest(snapshot), "sources": sources,
            "findings": findings, "counter_evidence": counter, "uncertainties": uncertainties,
            "collection": {k: v for k, v in external.items() if k != "sources"},
            "ai": ai or {"status": "not_requested"},
            "status": "partial" if external["status"] != "ok" or (ai and ai["status"] != "completed") else "completed",
            "scope": "근거 수집·위험 해석. 가격 방향·매매 추천·원인 확정 제외"}


def markdown_report(report):
    def plain(value):
        return re.sub(r"([\\`*_{}\[\]()<>!#|])", r"\\\1", str(value))

    snap = report["snapshot"]
    lines = ["# VolGuard KR 위험 근거 보고서", "", f"생성: {report['created_at']}",
             f"위험 상태 수집: {snap['captured_at']} · {snap['mode']} · {report['status']}", "",
             "## 관측 사실", ""]
    for row in snap["stocks"]:
        state = row["gauge"]
        lines.append(f"- {row['symbol']} {plain(row['name'])}: 상태 {state['status']}, "
                     f"관측 변동폭 {state['range_pct']}, 소진율 {state['consumption_pct']}")
    if snap.get("risk"):
        r = snap["risk"]
        lines += ["", "## 포트폴리오", "", f"VaR 검증: {r['status']} · 기준일 {r.get('as_of')}"]
        if r["status"] in {"passed", "cash_only"}:
            lines.append(f"1일 95% VaR: {r['var_krw']:,.0f}원")
        for scenario in r["stress"]:
            pnl = scenario["pnl_krw"]
            lines.append(f"- {scenario['scenario']}: {pnl:,.0f}원" if pnl is not None
                         else f"- {scenario['scenario']}: 자료 불완전")
    for title, key in (("위험 근거", "findings"), ("반대 근거·완화 관측", "counter_evidence")):
        lines += ["", f"## {title}", ""]
        for item in report[key]:
            lines.append(f"- {plain(item['text'])} [{', '.join(item['source_ids'])}]")
        if not report[key]:
            lines.append("확인된 항목 없음. 부재 자체로 결론을 내리지 않습니다.")
    lines += ["", "## 불확실성", ""] + [f"- {plain(v)}" for v in report["uncertainties"]]
    ai = report["ai"]
    lines += ["", "## AI 근거 선별" if ai.get("mode") == "extractive_selection" else "## AI 해석", "", f"상태: {ai['status']}"]
    if ai.get("provider"):
        lines.append(f"생성 모델: {plain(ai['provider'])} / {plain(ai['model'])}")
    if ai.get("input_scope"):
        scope = ai["input_scope"]
        lines.append(f"AI 입력 근거: {', '.join(scope['included_source_ids'])}. "
                     f"미포함: {', '.join(scope['omitted_source_ids']) or '없음'}. "
                     f"추가 발췌: {', '.join(scope['excerpted_source_ids']) or '없음'}.")
    for claim in ai.get("claims", []):
        lines.append(f"- ({claim['kind']}) {plain(claim['text'])} [{', '.join(claim['source_ids'])}]")
        if claim.get("supporting_quote"):
            lines.append(f"  - 원문 인용: {plain(claim['supporting_quote'])}")
    lines += ["", "## 출처", ""]
    for source in report["sources"]:
        title = plain(source["title"])
        label = f"[{title}]({source['url']})" if source.get("url") else title
        lines.append(f"- {source['id']}: {label} · 수집 {source['retrieved_at']} · {source['kind']}")
    lines += ["", report["scope"], ""]
    return "\n".join(lines)
