"""Bounded, cached historical external-feature feasibility study for three core stocks."""

import json
import time
from datetime import timedelta

import httpx
import pandas as pd

from .dart_evidence import api_key, download, xml, zip_members
from .platform_study import local_settings
from .providers import Kiwoom, request_json
from .util import atomic_json, digest, utcnow

SYMBOLS = ["005930", "000660", "005380"]


def run(settings):
    root = settings.data_dir / "research-inputs/direction-external"
    root.mkdir(parents=True, exist_ok=True)
    report = dict(
        symbols=SYMBOLS,
        start="2014-01-01",
        end="2023-12-31",
        errors=[],
        scope="Three-core-stock external-data ablation; not full-universe evidence",
        timing="Use only records dated strictly before the feature day; publication/revision archive unverified",
        source_contract="Historical Kiwoom KRX investor flow only; no tick/account/order data",
        source_hash=digest(__import__("pathlib").Path(__file__).read_bytes()),
    )
    broker = Kiwoom(settings.environment)
    broker.authenticate()
    for symbol in SYMBOLS:
        final = root / f"flow-{symbol}.parquet"
        if final.exists():
            continue
        cursor, rows = "20231228", []
        for page in range(35):
            cache = root / f"flow-{symbol}-{cursor}.json"
            if cache.exists():
                raw = json.loads(cache.read_text(encoding="utf8"))
            else:
                try:
                    raw = request_json(
                        "POST",
                        f"https://{broker.host}/api/dostk/stkinfo",
                        headers={"authorization": f"Bearer {broker.token}", "api-id": "ka10059"},
                        json={
                            "dt": cursor,
                            "stk_cd": symbol,
                            "amt_qty_tp": "1",
                            "trde_tp": "0",
                            "unit_tp": "1",
                        },
                    )
                    if str(raw.get("return_code")) != "0":
                        raise ValueError("BROKER_REJECTED")
                    atomic_json(cache, raw)
                    time.sleep(0.35)
                except Exception as e:
                    report["errors"].append(dict(source="flow", symbol=symbol, error=type(e).__name__))
                    break
            batch = raw.get("stk_invsr_orgn", [])
            if not batch:
                break
            dates = [r["dt"] for r in batch]
            if max(dates) > cursor:
                raise ValueError("Provider ignored historical date")
            rows.extend(batch)
            if min(dates) <= "20140101":
                break
            next_cursor = (pd.Timestamp(min(dates)) - timedelta(days=1)).strftime("%Y%m%d")
            if next_cursor >= cursor:
                raise ValueError("Nonadvancing historical cursor")
            cursor = next_cursor
        if rows:
            frame = pd.DataFrame(rows).drop_duplicates("dt").sort_values("dt")
            frame["symbol"] = symbol
            frame["date"] = pd.to_datetime(frame.dt)
            frame = frame[frame.date.between("2014-01-01", "2023-12-31")]
            frame.to_parquet(final, index=False)
            print(
                json.dumps(dict(source="flow", symbol=symbol, rows=len(frame), start=str(frame.date.min()))),
                flush=True,
            )
    key = api_key()
    dart_root = root / "dart"
    dart_root.mkdir(exist_ok=True)
    if key:
        with httpx.Client(timeout=20, follow_redirects=False) as client:
            try:
                raw, _ = download(client, dart_root, "corpCode.xml", {}, key, 86400, 10_000_000)
                companies = {}
                for _, data in zip_members(raw, 100_000_000):
                    for r in xml(data).findall("list"):
                        companies[(r.findtext("stock_code") or "").strip()] = r.findtext("corp_code")
                for symbol in SYMBOLS:
                    final = root / f"dart-{symbol}.parquet"
                    if final.exists():
                        continue
                    rows, complete = [], False
                    for page in range(1, 151):
                        params = dict(
                            corp_code=companies[symbol],
                            bgn_de="20140101",
                            end_de="20231231",
                            last_reprt_at="N",
                            page_count="100",
                            page_no=str(page),
                            sort="date",
                            sort_mth="asc",
                        )
                        data, _ = download(
                            client, dart_root, "list.json", params, key, 86400 * 365, 5_000_000
                        )
                        result = json.loads(data)
                        rows.extend(result.get("list", []))
                        if page >= int(result.get("total_page", 0)):
                            complete = True
                            break
                        time.sleep(0.2)
                    if not complete:
                        report["errors"].append(
                            dict(source="dart", symbol=symbol, error="PAGE_BUDGET_EXCEEDED")
                        )
                        continue
                    frame = pd.DataFrame(rows).drop_duplicates("rcept_no")
                    frame["symbol"] = symbol
                    frame["date"] = pd.to_datetime(frame.rcept_dt)
                    frame.to_parquet(final, index=False)
                    print(json.dumps(dict(source="dart", symbol=symbol, rows=len(frame))), flush=True)
            except Exception as e:
                report["errors"].append(dict(source="dart", error=type(e).__name__))
    else:
        report["errors"].append(dict(source="dart", error="KEY_MISSING"))
    report["files"] = {
        p.name: dict(hash=digest(p.read_bytes()), rows=len(pd.read_parquet(p)))
        for p in root.glob("*.parquet")
    }
    report["completed_at"] = utcnow()
    atomic_json(root / "manifest.json", report)
    atomic_json(settings.data_dir / "reports/direction-external-access.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    run(local_settings())
