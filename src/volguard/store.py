import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .util import utcnow


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS watchlist(symbol TEXT PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS predictions(
                  symbol TEXT, target_date TEXT, payload TEXT NOT NULL,
                  PRIMARY KEY(symbol,target_date));
                CREATE TABLE IF NOT EXISTS quotes(symbol TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS alerts(
                  symbol TEXT, trade_date TEXT, threshold INTEGER, created_at TEXT, payload TEXT,
                  PRIMARY KEY(symbol,trade_date,threshold));
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS risk_cache(key TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS run_log(
                  id TEXT PRIMARY KEY, kind TEXT, started_at TEXT, finished_at TEXT,
                  status TEXT, rows INTEGER, elapsed REAL, error TEXT);
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

    def watchlist(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM watchlist ORDER BY symbol")]

    def set_watchlist(self, items, universe):
        symbols = [x["symbol"] for x in items]
        if len(symbols) > 30 or len(set(symbols)) != len(symbols):
            raise ValueError("관심종목은 중복 없이 최대 30개입니다")
        if not set(symbols).issubset(universe):
            raise ValueError("고정된 유니버스 안에서 선택하세요")
        with self.connect() as db:
            db.execute("DELETE FROM watchlist")
            db.executemany("INSERT INTO watchlist VALUES (?,?)", [(x["symbol"], x["name"]) for x in items])

    def publish(self, prediction):
        """First publication wins; intraday bounds can never be overwritten."""
        encoded = json.dumps(prediction, ensure_ascii=False, allow_nan=False)
        with self.connect() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO predictions VALUES (?,?,?)",
                (prediction["symbol"], prediction["target_date"], encoded),
            )
            return bool(cursor.rowcount)

    def prediction(self, symbol, day):
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM predictions WHERE symbol=? AND target_date=?", (symbol, day)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def put(self, table, key, payload):
        if table not in {"meta", "risk_cache", "quotes"}:
            raise ValueError("Unsupported table")
        with self.connect() as db:
            db.execute(
                f"INSERT OR REPLACE INTO {table} VALUES (?,?)",
                (key, json.dumps(payload, ensure_ascii=False, allow_nan=False)),
            )

    def get(self, table, key, default=None):
        if table not in {"meta", "risk_cache", "quotes"}:
            raise ValueError("Unsupported table")
        column = "symbol" if table == "quotes" else "key"
        with self.connect() as db:
            row = db.execute(f"SELECT payload FROM {table} WHERE {column}=?", (key,)).fetchone()
            return json.loads(row[0]) if row else default

    def alert(self, symbol, day, threshold, payload):
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO alerts VALUES (?,?,?,?,?)",
                (symbol, day, threshold, utcnow(), json.dumps(payload, allow_nan=False)),
            )

    def recent(self, table, limit=100):
        if table not in {"run_log", "alerts"}:
            raise ValueError("Unsupported table")
        with self.connect() as db:
            return [
                dict(r) for r in db.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT ?", (limit,))
            ]

    def batch_observations(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT payload FROM meta WHERE key LIKE 'batch:%'")]

    @contextmanager
    def run(self, kind):
        run_id, start = uuid.uuid4().hex, time.monotonic()
        summary = {"rows": 0, "status": "success"}
        with self.connect() as db:
            db.execute("INSERT INTO run_log VALUES (?,?,?,NULL,'running',0,0,NULL)", (run_id, kind, utcnow()))
        try:
            yield summary
        except Exception as exc:
            # Exception messages may contain URLs with credentials. Persist type only.
            summary.update(status="failed", error=type(exc).__name__)
            raise
        finally:
            with self.connect() as db:
                db.execute(
                    "UPDATE run_log SET finished_at=?,status=?,rows=?,elapsed=?,error=? WHERE id=?",
                    (
                        utcnow(),
                        summary["status"],
                        summary["rows"],
                        time.monotonic() - start,
                        summary.get("error"),
                        run_id,
                    ),
                )
