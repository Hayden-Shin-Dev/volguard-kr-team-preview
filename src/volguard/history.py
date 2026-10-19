"""Build research-ready files from checkpointed official responses in one pass."""

import json
import os

import numpy as np
import pandas as pd

from .adjustments import auxiliary_prices, reconcile_adjustments
from .data import Lake, normalize, quality_report
from .providers import KRX
from .util import atomic_json, digest, exclusive_run, utcnow


def materialize(settings, require_complete=True):
    lake = Lake(settings.data_dir)
    checkpoints = lake.root / "config" / "downloads"
    symbols = {m["symbol"] for m in lake.universe()["members"]}
    candidate_dates = pd.bdate_range("2010-01-04", "2026-10-15").strftime("%Y-%m-%d")
    missing = [
        (day, kind)
        for day in candidate_dates
        for kind in ("stock", "index")
        if not (checkpoints / f"{day}-{kind}.json").exists()
    ]
    if missing and require_complete:
        raise ValueError(f"Backfill incomplete: {len(missing)} checkpoints missing")
    bars, market, sessions, hashes = [], [], [], []
    for day in candidate_dates:
        stock_path, index_path = (checkpoints / f"{day}-{kind}.json" for kind in ("stock", "index"))
        if not stock_path.exists() or not index_path.exists():
            continue
        info = [json.loads(p.read_text(encoding="utf8")) for p in (stock_path, index_path)]
        stock, index = [
            json.loads((lake.root / "bronze" / f"krx-{kind}-history" / f"{m['hash']}.raw").read_bytes())
            for kind, m in zip(("stock", "index"), info)
        ]
        entries = [r for r in index["OutBlock_1"] if r["IDX_NM"] == "코스피"]
        if not stock["OutBlock_1"] and not entries:
            continue  # confirmed empty responses, not an invented weekday session
        if len(entries) != 1 or not stock["OutBlock_1"]:
            raise ValueError(f"Stock/index session mismatch: {day}")
        item = entries[0]
        high, low = (float(item[k].replace(",", "")) for k in ("HGPRC_IDX", "LWPRC_IDX"))
        if not 0 < low <= high:
            raise ValueError(f"Invalid KOSPI OHLC: {day}")
        market.append(
            {
                "date": pd.Timestamp(day),
                "range_pct": 100 * (high / low - 1),
                "available_at": pd.Timestamp(info[1]["fetched_at"]),
                "pit_verified": False,
                "source_hash": info[1]["hash"],
            }
        )
        stock["OutBlock_1"] = [r for r in stock["OutBlock_1"] if r["ISU_CD"] in symbols]
        if stock["OutBlock_1"]:
            frame = KRX.normalize(stock, day)
            frame["available_at"] = info[0]["fetched_at"]
            frame["source_hash"] = info[0]["hash"]
            bars.append(frame)
        # These are session DATES established from actual observations. Session
        # hours are deliberately unverified; realtime must not use this calendar.
        sessions.append(
            {
                "date": day,
                "open_time": "09:00",
                "close_time": "15:30",
                "hours_verified": False,
                "source": "KRX stock and KOSPI daily observations",
            }
        )
        hashes.extend(m["hash"] for m in info)
    if not bars:
        raise ValueError("No complete stock/index history")
    merged = normalize(pd.concat(bars, ignore_index=True))
    merged = reconcile_adjustments(merged, auxiliary_prices(lake))
    report = {
        **quality_report(merged),
        "members": len(symbols),
        "sessions": len(sessions),
        "download_checkpoints_missing": len(missing),
        "adjustment_ohlc_crosschecked": int(merged.adjustment_direct_crosschecked.sum()),
        "adjustment_reference_continuity": int(merged.adjustment_basis.eq("krx_reference_continuity").sum()),
        "adjustment_unmatched_valid": int((merged.valid_label & ~merged.adjustment_crosschecked).sum()),
        "historical_publication_archive": False,
        "corporate_entitlements_verified": False,
        "timing_note": "As-retrieved history, not archived point-in-time data",
        "source_manifest_hash": digest(sorted(hashes)),
        "generated_at": utcnow(),
    }
    with exclusive_run(lake.root / "silver.lock"):
        for name, frame in (("bars", merged), ("market", pd.DataFrame(market))):
            temp = lake.root / "silver" / f"{name}.tmp.parquet"
            frame.to_parquet(temp, index=False)
            os.replace(temp, lake.root / "silver" / f"{name}.parquet")
        atomic_json(
            lake.root / "config" / "calendar.json",
            {"sessions": sessions, "basis": "observed_session_dates_only", "hours_verified": False},
        )
        atomic_json(lake.root / "reports" / "quality.json", report)
    return report


def retrospective_frame(lake, verified_price_returns=False):
    """Explicit EOD reconstruction used only by the retrospective audit.

    No production verification flags are upgraded. Prices are known by close
    as an ASSUMPTION; corrections and corporate entitlements are limitations.
    """
    from .data import features

    bars = lake.bars()
    bars["available_at"] = (
        (bars.date + pd.Timedelta(hours=18)).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
    )
    frame = features(bars, lake.calendar(), crosschecked_research=not verified_price_returns)
    market = pd.read_parquet(lake.root / "silver" / "market.parquet")
    frame["market_range"] = frame.date.map(market.set_index("date").range_pct)
    return frame.replace([np.inf, -np.inf], np.nan)
