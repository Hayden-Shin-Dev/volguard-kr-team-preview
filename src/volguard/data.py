import json
import os
import uuid
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .util import atomic_json, digest, exclusive_run, utcnow

KEYS = ["symbol", "date", "exchange"]
NUMERIC = ["open", "high", "low", "close", "volume", "turnover", "market_cap", "adjustment_factor"]


def boolean(series):
    return series.astype(str).str.lower().isin(["true", "1"])


class Lake:
    def __init__(self, root: Path):
        self.root = root
        for directory in ("bronze", "silver", "reports", "config"):
            (root / directory).mkdir(parents=True, exist_ok=True)

    def bronze(self, content: bytes, source: str, metadata=None):
        identifier = digest(content)
        directory = self.root / "bronze" / source
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{identifier}.raw"
        if not target.exists():
            try:
                with target.open("xb") as f:
                    f.write(content)
            except FileExistsError:
                pass  # another download can return exactly the same immutable bytes
            else:
                atomic_json(
                    target.with_suffix(".json"),
                    {"hash": identifier, "source": source, "fetched_at": utcnow(), **(metadata or {})},
                )
        return identifier

    def ingest(self, frame: pd.DataFrame, source_hash: str):
        frame = normalize(frame)
        frame["source_hash"] = source_hash
        report = quality_report(frame)
        with exclusive_run(self.root / "silver.lock"):
            old = self.bars()
            merged = pd.concat([old, frame], ignore_index=True)
            merged = merged.drop_duplicates(KEYS, keep="last").sort_values(KEYS).reset_index(drop=True)
            temp = self.root / "silver" / f"{uuid.uuid4().hex}.parquet"
            merged.to_parquet(temp, index=False)
            os.replace(temp, self.root / "silver" / "bars.parquet")
            atomic_json(self.root / "reports" / "quality.json", {**report, "silver_rows": len(merged)})
        return report

    def bars(self):
        path = self.root / "silver" / "bars.parquet"
        if not path.exists():
            return pd.DataFrame()
        with duckdb.connect() as db:
            return db.execute("SELECT * FROM read_parquet(?) ORDER BY symbol, date", [str(path)]).df()

    def import_csv(self, path: Path):
        raw = path.read_bytes()
        version = self.bronze(raw, "normalized-csv", {"filename": path.name})
        return self.ingest(pd.read_csv(path, dtype={"symbol": str}), version)

    def set_calendar(self, path: Path):
        frame = pd.read_csv(path, dtype=str)
        required = {"date", "open_time", "close_time"}
        if not required.issubset(frame):
            raise ValueError("Calendar needs date, open_time, close_time; only actual sessions")
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        if frame["date"].duplicated().any():
            raise ValueError("Duplicate calendar date")
        for row in frame.to_dict("records"):
            start = pd.Timestamp(f"{row['date']} {row['open_time']}")
            end = pd.Timestamp(f"{row['date']} {row['close_time']}")
            if not start < end:
                raise ValueError("Invalid session hours")
        version = self.bronze(path.read_bytes(), "calendar")
        atomic_json(
            self.root / "config" / "calendar.json",
            {"source_hash": version, "sessions": frame.sort_values("date").to_dict("records")},
        )

    def calendar(self):
        path = self.root / "config" / "calendar.json"
        if not path.exists():
            raise ValueError("공식 거래일 달력을 먼저 import-calendar로 입력하세요")
        return json.loads(path.read_text(encoding="utf-8"))["sessions"]

    def universe(self):
        path = self.root / "config" / "universe.json"
        if not path.exists():
            raise ValueError("유니버스를 먼저 freeze-universe로 고정하세요")
        return json.loads(path.read_text(encoding="utf-8"))

    def freeze_universe(self, path: Path, as_of: str, source: str, demo=False, evidence=None):
        frame = pd.read_csv(path, dtype={"symbol": str})
        if not {"symbol", "name"}.issubset(frame) or frame.symbol.duplicated().any():
            raise ValueError("Need unique symbol,name columns")
        if not demo:
            if not frame.symbol.str.fullmatch(r"[0-9A-Z]{6}").all():
                raise ValueError("KRX 6자리 영숫자 종목코드 필요")
            if len(frame) != 200 and not (
                evidence
                and evidence.get("matched_symbols") == sorted(frame.symbol)
                and evidence.get("primary_snapshot_hash")
                and evidence.get("crosscheck_snapshot_hash")
            ):
                raise ValueError("200개가 아닌 구성은 두 출처의 일치 근거가 필요합니다")
        target = self.root / "config" / "universe.json"
        if target.exists():
            raise ValueError("이미 고정된 유니버스입니다. 새 연구는 별도 데이터 디렉터리를 사용하세요")
        atomic_json(
            target,
            {
                "as_of": as_of,
                "source": source,
                "demo": demo,
                "hash": digest(path.read_bytes()),
                "members": frame.to_dict("records"),
                "membership_evidence": evidence,
            },
        )


def normalize(frame):
    frame = frame.copy()
    required = {"symbol", "date", "open", "high", "low", "close", "volume", "available_at"}
    if not required.issubset(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    frame["symbol"] = frame.symbol.astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame.date).dt.normalize()
    frame["available_at"] = pd.to_datetime(frame.available_at, utc=True, errors="coerce")
    defaults = {
        "exchange": "KRX",
        "adjustment_factor": np.nan,
        "adjustment_verified": False,
        "pit_verified": False,
        "halted": False,
        "price_limit": False,
        "turnover": np.nan,
        "market_cap": np.nan,
    }
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("adjustment_verified", "pit_verified", "halted", "price_limit"):
        frame[column] = boolean(frame[column])
    prices = frame[["open", "high", "low", "close"]]
    good = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    good &= frame.high >= frame[["open", "close", "low"]].max(axis=1)
    good &= frame.low <= frame[["open", "close", "high"]].min(axis=1)
    good &= np.isfinite(frame.volume) & (frame.volume > 0)
    good &= (~frame.halted) & (frame.exchange == "KRX") & frame.available_at.notna()
    frame["valid_label"] = good
    frame["range_pct"] = np.where(good, 100 * (frame.high / frame.low - 1), np.nan)
    verified = frame.adjustment_verified & (frame.adjustment_factor > 0) & good
    frame["adjusted_close"] = (frame.close * frame.adjustment_factor).where(verified)
    return frame.drop_duplicates(KEYS, keep="last")


def quality_report(frame):
    return {
        "generated_at": utcnow(),
        "rows": len(frame),
        "valid_labels": int(frame.valid_label.sum()),
        "excluded_labels": int((~frame.valid_label).sum()),
        "zero_range_trading_days": int(((frame.range_pct == 0) & frame.valid_label).sum()),
        "adjustment_unverified": int((~frame.adjustment_verified).sum()),
        "pit_unverified": int((~frame.pit_verified).sum()),
        "start": str(frame.date.min().date()) if len(frame) else None,
        "end": str(frame.date.max().date()) if len(frame) else None,
    }


HAR_FEATURES = ["range_1", "range_5", "range_22"]
ML_FEATURES = HAR_FEATURES + ["return_1", "vol_22", "gap", "volume_ratio", "turnover_ratio", "market_range"]
ML_REQUIRED_FEATURES = HAR_FEATURES + ["market_range"]


def features(bars, sessions, crosschecked_research=False, issue_cutoffs=None):
    """Rows are issued after day t; labels strictly refer to the next actual session.

    Reindex BEFORE shifting: suspended/missing days must not turn a one-day target into a two-day target.
    Late-available inputs are masked before all rolling statistics.
    """
    bars = bars[bars.exchange.eq("KRX")]
    calendar = pd.DatetimeIndex(pd.to_datetime([s["date"] for s in sessions]))
    blocks = []
    for symbol, raw in bars.groupby("symbol", sort=True):
        part = raw.set_index("date").sort_index().reindex(calendar).copy()
        cutoff = (calendar + pd.Timedelta(hours=18, minutes=30)).tz_localize("Asia/Seoul").tz_convert("UTC")
        cutoff = pd.Series(cutoff, index=calendar)
        if issue_cutoffs is not None:
            cutoff = issue_cutoffs.reindex(calendar).fillna(cutoff)
        known = part.available_at.le(cutoff)
        valid = part.valid_label.eq(True)
        r = part.range_pct.where(known & valid)
        adj = part.adjusted_close.where(known & valid)
        adjustment_eligible = part.adjustment_verified.eq(True)
        if crosschecked_research:
            adj = part.candidate_adjusted_close.where(known & valid & part.adjustment_crosschecked.eq(True))
            adjustment_eligible = part.adjustment_crosschecked.eq(True)
        ret = adj.pct_change(fill_method=None)
        reference_gap = pd.Series(np.nan, index=calendar)
        if not crosschecked_research and "CMPPREVDD_PRC" in part:
            change = pd.to_numeric(part.CMPPREVDD_PRC.astype(str).str.replace(",", ""), errors="coerce")
            previous = part.close.shift(1)
            continuous = known & valid & known.shift(1, fill_value=False) & valid.shift(1, fill_value=False)
            continuous &= (part.close - change - previous).abs().le(0.5)
            ret = ret.fillna((part.close / previous - 1).where(continuous))
            reference_gap = (part.open / previous - 1).where(continuous)
        volume = part.volume.where(known & valid)
        turnover = part.turnover.where(known & valid)
        x = pd.DataFrame(index=calendar)
        x["symbol"] = symbol
        if issue_cutoffs is not None:
            x["issue_at"] = cutoff
        x["range_1"] = r
        x["range_5"] = r.rolling(5, min_periods=5).mean()
        x["range_22"] = r.rolling(22, min_periods=22).mean()
        x["body_pct"] = (100 * (part.close / part.open - 1)).where(known & valid)
        spread = (part.high - part.low).replace(0, np.nan)
        x["close_location"] = ((part.close - part.low) / spread).where(known & valid)
        x.loc[known & valid & r.eq(0), "close_location"] = 0.5
        x["range_max5"] = r.rolling(5).max()
        x["return_1"] = ret
        x["vol_22"] = ret.rolling(22, min_periods=22).std()
        adjusted_open = (part.open * part.adjustment_factor).where(known & adjustment_eligible)
        x["gap"] = (adjusted_open / adj.shift(1) - 1).fillna(reference_gap)
        x["volume_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)
        x["turnover_ratio"] = turnover / turnover.rolling(20).mean().replace(0, np.nan)
        x["emp_low"] = r.rolling(20).quantile(0.05)
        x["emp_mid"] = r.rolling(20).quantile(0.5)
        x["emp_high"] = r.rolling(20).quantile(0.95)
        x["target_date"] = pd.Series(calendar, index=calendar).shift(-1)
        x["target_available_at"] = part.available_at.shift(-1)
        x["y"] = part.range_pct.where(valid).shift(-1)
        x["pit_verified"] = part.pit_verified.eq(True) & part.pit_verified.shift(-1).eq(True)
        x.index.name = "date"
        blocks.append(x.reset_index())
    if not blocks:
        return pd.DataFrame()
    result = pd.concat(blocks, ignore_index=True)
    # Market proxy is explicitly an equal-weight universe range until a verified index adapter is supplied.
    market = result.groupby("date")["range_1"].mean()
    result["market_range"] = result.date.map(market)
    return result.replace([np.inf, -np.inf], np.nan).sort_values(["date", "symbol"]).reset_index(drop=True)


def adjusted_returns(bars, sessions, symbols):
    bars = bars[bars.exchange.eq("KRX")]
    idx = pd.DatetimeIndex(pd.to_datetime([s["date"] for s in sessions]))
    pivot = bars[bars.symbol.isin(symbols)].pivot(index="date", columns="symbol", values="adjusted_close")
    pivot = pivot.reindex(index=idx, columns=symbols)
    return pivot.pct_change(fill_method=None)
