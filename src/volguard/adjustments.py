"""Auxiliary adjusted-price crosschecks; never silently certify a vendor series."""

import ast
import json
import time

import httpx
import numpy as np
import pandas as pd

from .data import Lake
from .util import atomic_json, utcnow


def download_auxiliary(settings):
    lake = Lake(settings.data_dir)
    directory = lake.root / "config" / "adjustment-downloads"
    directory.mkdir(exist_ok=True)
    summaries = []
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        for member in lake.universe()["members"]:
            symbol = member["symbol"]
            checkpoint = directory / f"{symbol}.json"
            if checkpoint.exists():
                summaries.append(json.loads(checkpoint.read_text(encoding="utf8")))
                continue
            try:
                response = client.get(
                    "https://api.finance.naver.com/siseJson.naver",
                    params={
                        "symbol": symbol,
                        "requestType": "1",
                        "startTime": "20100101",
                        "endTime": "20261015",
                        "timeframe": "day",
                    },
                )
                response.raise_for_status()
                if len(response.content) > 2_000_000:
                    raise ValueError("UNEXPECTED_SIZE")
                rows = ast.literal_eval(response.text.strip())
                if not isinstance(rows, list) or not rows or len(rows[0]) != 7:
                    raise ValueError("UNEXPECTED_SCHEMA")
                version = lake.bronze(response.content, "naver-adjusted-crosscheck", {"symbol": symbol})
                result = {"symbol": symbol, "rows": len(rows) - 1, "hash": version, "fetched_at": utcnow()}
                atomic_json(checkpoint, result)
                summaries.append(result)
            except (httpx.HTTPError, ValueError, SyntaxError):
                summaries.append({"symbol": symbol, "error": "DOWNLOAD_OR_SCHEMA"})
            time.sleep(0.25)
            if len(summaries) % 25 == 0:
                print(f"adjustment candidates {len(summaries)}/{len(lake.universe()['members'])}", flush=True)
    atomic_json(
        lake.root / "reports" / "adjustment-access.json", {"items": summaries, "created_at": utcnow()}
    )
    return summaries


def auxiliary_prices(lake):
    frames = []
    for checkpoint in sorted((lake.root / "config" / "adjustment-downloads").glob("*.json")):
        info = json.loads(checkpoint.read_text(encoding="utf8"))
        raw = lake.root / "bronze" / "naver-adjusted-crosscheck" / f"{info['hash']}.raw"
        rows = ast.literal_eval(raw.read_text(encoding="utf8").strip())
        frame = pd.DataFrame(
            rows[1:], columns=["date", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume", "unused"]
        )
        frame["date"] = pd.to_datetime(frame.date, format="%Y%m%d")
        frame["symbol"] = info["symbol"]
        frame["adjustment_source_hash"] = info["hash"]
        frames.append(frame.drop(columns="unused"))
    if not frames:
        raise ValueError("No auxiliary adjustment candidates")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")


def reconcile_adjustments(bars, auxiliary):
    """A candidate multiplier must reproduce all four OHLC, not just close.

    Integer rounding in adjusted vendor OHLC is tolerated by 2 currency units.
    This verifies same-day multiplicative consistency, NOT corporate entitlements
    or a historical as-of archive. The distinction remains explicit in outputs.
    """
    result = bars.merge(auxiliary, on=["symbol", "date"], how="left", validate="one_to_one")
    factor = result.adj_close / result.close
    consistent = np.isfinite(factor) & factor.gt(0)
    for field in ("open", "high", "low", "close"):
        consistent &= (result[field] * factor - result[f"adj_{field}"]).abs().le(2.0)
    result["adjustment_direct_crosschecked"] = consistent
    result["adjustment_factor"] = factor.where(consistent)
    result["adjustment_basis"] = np.where(consistent, "auxiliary_ohlc_ratio", "unverified")
    # Recent auxiliary charts can have a different session basis. Never use their
    # close in that case. An existing multiplier can continue only when the
    # official KRX close-minus-change reference equals the prior raw close.
    if "CMPPREVDD_PRC" in result:
        change = pd.to_numeric(result.CMPPREVDD_PRC.astype(str).str.replace(",", ""), errors="coerce")
        result["reference_price"] = result.close - change
        for _, group in result.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
            previous_close, previous_factor = np.nan, np.nan
            for index, close, reference, direct, current_factor in zip(
                group.index,
                group.close,
                group.reference_price,
                group.adjustment_direct_crosschecked,
                group.adjustment_factor,
            ):
                if not direct and np.isfinite(previous_factor) and abs(reference - previous_close) < 0.5:
                    result.at[index, "adjustment_factor"] = previous_factor
                    result.at[index, "adjustment_basis"] = "krx_reference_continuity"
                    current_factor = previous_factor
                previous_close, previous_factor = close, current_factor
    result["adjustment_crosschecked"] = result.adjustment_factor.notna() & result.valid_label
    result["adjustment_verified"] = False  # entitlement/as-of evidence remains separate
    result["candidate_adjusted_close"] = (result.close * result.adjustment_factor).where(
        result.adjustment_crosschecked
    )
    return result


if __name__ == "__main__":
    from .config import Settings

    download_auxiliary(Settings.load().prepare())
