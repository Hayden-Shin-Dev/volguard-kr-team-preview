"""Read-only, bounded external research inputs; isolated from live storage and models."""

import argparse
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import httpx
import pandas as pd

from .dart_evidence import api_key, download
from .platform_study import local_settings
from .providers import ECOS, Kiwoom
from .util import atomic_json, digest, utcnow

RUN = "direction-new-information-20261017"
CORE = ["005930", "000660", "005380"]
ALLOW = {"ka10051", "ka90010", "ka10047", "ka10080", "ka10079", "ka10004", "ka10059"}


def output():
    out = local_settings().data_dir / "runs" / RUN
    out.mkdir(exist_ok=True)
    return out


class Reader:
    def __init__(self, out):
        self.out = out
        self.spec = json.loads((out / "kiwoom-spec.json").read_text("utf8"))["apis"]
        self.broker = Kiwoom(local_settings().environment)
        self.broker.authenticate()
        self.client = httpx.Client(timeout=30, follow_redirects=False)

    def get(self, code, body, continuation="N", cursor=""):
        if code not in ALLOW:
            raise ValueError("Read-only research endpoint not allowed")
        key = digest([code, body, continuation, cursor])
        path = self.out / "broker-raw" / (key + ".json")
        if path.exists():
            return json.loads(path.read_text("utf8"))
        entry = next(v for v in self.spec.values() if v["meta"].get("API ID") == code)
        for attempt in range(3):
            r = self.client.post(
                "https://" + self.broker.host + entry["meta"]["URL"],
                headers={
                    "authorization": "Bearer " + self.broker.token,
                    "api-id": code,
                    "cont-yn": continuation,
                    "next-key": cursor,
                },
                json=body,
            )
            time.sleep(0.55)
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            if r.status_code != 200:
                raise ValueError("Broker HTTP " + str(r.status_code))
            data = r.json()
            if str(data.get("return_code")) != "0":
                raise ValueError("Broker code " + str(data.get("return_code")))
            result = dict(
                api=code,
                request=body,
                response=data,
                retrieved_at=utcnow(),
                continuation=r.headers.get("cont-yn"),
                next_key=r.headers.get("next-key"),
            )
            atomic_json(path, result)
            return result
        raise ValueError("Broker retry budget exhausted")


def save_frame(out, name, frame, metadata):
    path = out / (name + ".parquet")
    frame.to_parquet(path, index=False)
    atomic_json(
        out / (name + "-manifest.json"),
        dict(rows=len(frame), hash=digest(path.read_bytes()), collected_at=utcnow(), **metadata),
    )
    print(json.dumps(dict(source=name, rows=len(frame), **metadata), ensure_ascii=False), flush=True)


def sector():
    out = output()
    reader = Reader(out)
    context = pd.read_parquet(out.parent / "direction-math-20261017/context.parquet", columns=["date"])
    dates = sorted(context.loc[context.date.between("2014-01-01", "2023-12-31"), "date"].unique())
    rows, failed = [], []

    def fetch(day):
        day = pd.Timestamp(day).strftime("%Y%m%d")
        try:
            raw = reader.get("ka10051", dict(mrkt_tp="0", amt_qty_tp="0", base_dt=day, stex_tp="1"))
            batch = [dict(r, date=day) for r in raw["response"].get("inds_netprps", []) if r.get("cur_prc")]
            if not batch:
                raise ValueError("Empty sector day")
            return batch, None
        except Exception as e:
            return [], dict(date=day, error=type(e).__name__)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (batch, error) in enumerate(pool.map(fetch, dates)):
            rows.extend(batch)
            if error:
                failed.append(error)
            if i % 100 == 0:
                print(f"SECTOR {i + 1}/{len(dates)}", flush=True)
    save_frame(
        out,
        "sector",
        pd.DataFrame(rows),
        dict(
            requested_days=len(dates),
            failed=failed,
            timing="Lag at least one full KRX session; original revision vintages unavailable",
            membership="Use sector indices as common market inputs, never current stock-sector membership",
        ),
    )


def broker_history():
    out = output()
    reader = Reader(out)
    # Market program flow has historical-date input, unlike the order-book endpoint.
    rows = []
    cursor = "20231228"
    for page in range(30):
        raw = reader.get(
            "ka90010", dict(date=cursor, amt_qty_tp="1", mrkt_tp="P00101", min_tic_tp="0", stex_tp="1")
        )
        batch = [r for r in raw["response"].get("prm_trde_trnsn", []) if r.get("cntr_tm")]
        if not batch:
            break
        earliest = min(r["cntr_tm"][:8] for r in batch)
        if max(r["cntr_tm"][:8] for r in batch) > cursor:
            raise ValueError("Historical date ignored")
        rows.extend(batch)
        if earliest <= "20140101":
            break
        cursor = (pd.Timestamp(earliest) - timedelta(days=1)).strftime("%Y%m%d")
    save_frame(
        out,
        "program",
        pd.DataFrame(rows).drop_duplicates("cntr_tm"),
        dict(pages=page + 1, timing="Previous-session values; vintage unavailable"),
    )
    # Walk continuation to the actual boundary, with explicit page budget.
    probe_results = []
    for code, body, field, datefield, maxpages in [
        ("ka10047", dict(stk_cd="005930"), "cntr_str_daly", "dt", 80),
        (
            "ka10080",
            dict(stk_cd="005930", tic_scope="30", upd_stkpc_tp="0", base_dt="20261016"),
            "stk_min_pole_chart_qry",
            "cntr_tm",
            30,
        ),
        (
            "ka10079",
            dict(stk_cd="005930", tic_scope="1", upd_stkpc_tp="0"),
            "stk_tic_chart_qry",
            "cntr_tm",
            5,
        ),
    ]:
        rows = []
        cursor = ""
        cont = "N"
        seen = set()
        reason = "page_budget"
        for page in range(maxpages):
            raw = reader.get(code, body, cont, cursor)
            batch = [r for r in raw["response"].get(field, []) if r.get(datefield)]
            rows.extend(batch)
            if not batch or raw["continuation"] != "Y":
                reason = "provider_end"
                break
            if min(r[datefield][:8] for r in batch) <= "20140101":
                reason = "study_start_reached"
                break
            cursor = raw.get("next_key", "")
            cont = "Y"
            if not cursor or cursor in seen:
                reason = "nonadvancing_cursor"
                break
            seen.add(cursor)
        frame = pd.DataFrame(rows)
        meta = dict(
            api=code,
            pages=page + 1,
            stop_reason=reason,
            earliest=min((r[datefield] for r in rows), default=None),
            latest=max((r[datefield] for r in rows), default=None),
            session="Raw provider response; non-regular timestamps present, not admitted to historical model",
        )
        save_frame(out, code, frame, meta)
        probe_results.append(meta)
    atomic_json(out / "intraday-boundaries.json", probe_results)


def macro():
    out = output()
    client = ECOS()
    for name, table, item in [
        ("usdkrw", "731Y001", "0000001"),
        ("jpykrw", "731Y001", "0000002"),
        ("kr3y", "817Y002", "010200000"),
        ("kr10y", "817Y002", "010210000"),
        ("creditAA", "817Y002", "010300000"),
    ]:
        rows = []
        for year in range(2014, 2024):
            dest = out / f"ecos-{name}-{year}.json"
            if dest.exists():
                raw = json.loads(dest.read_text("utf8"))
            else:
                raw = client.query(table, "D", f"{year}0101", f"{year}1231", item)
                atomic_json(dest, raw)
                time.sleep(0.15)
            rows.extend(raw["StatisticSearch"].get("row", []))
        save_frame(
            out,
            "ecos-" + name,
            pd.DataFrame(rows),
            dict(
                table=table,
                item=item,
                timing="Conservative five-business-day release lag; current-vintage exploratory data, not proven PIT",
            ),
        )


def alfred():
    """Weekly historical vintages, never today's revised series substituted for history."""
    out = output()
    series = ["NASDAQCOM"]
    dates = pd.date_range("2014-01-03", "2023-12-29", freq="W-FRI")

    def fetch(day):
        date = day.strftime("%Y-%m-%d")
        dest = out / "alfred-raw" / f"{date}.zip"
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            with httpx.Client(timeout=40, follow_redirects=False) as c:
                r = c.get(
                    "https://alfred.stlouisfed.org/graph/alfredgraph.csv",
                    params={
                        "id": ",".join(series),
                        "vintage_date": date,
                        "cosd": (day - pd.Timedelta(days=45)).strftime("%Y-%m-%d"),
                        "coed": date,
                    },
                )
                if r.status_code != 200:
                    return [dict(error="HTTP_" + str(r.status_code), vintage_date=date)]
                dest.write_bytes(r.content)
        try:
            if zipfile.is_zipfile(dest):
                archive = zipfile.ZipFile(dest)
                if sum(i.file_size for i in archive.infolist()) > 20_000_000:
                    raise ValueError("Archive too large")
                frames = [
                    pd.read_csv(io.BytesIO(archive.read(i))) for i in archive.namelist() if i.endswith(".csv")
                ]
            else:
                frames = [pd.read_csv(dest, compression=None)]
            result = []
            for name in series:
                col = name + "_" + day.strftime("%Y%m%d")
                frame = next((f for f in frames if col in f), None)
                if frame is None:
                    raise ValueError("Vintage column missing")
                values = frame[["observation_date", col]].copy()
                values[col] = pd.to_numeric(values[col], errors="coerce")
                values = values.dropna()
                values = values[values.observation_date <= date].sort_values("observation_date")
                if len(values) < 2:
                    continue
                last = values.iloc[-1]
                prev = values.iloc[-2]
                # At least two calendar days after US vintage day, interpreted in KST.
                result.append(
                    dict(
                        series=name,
                        vintage_date=date,
                        observation_date=last.observation_date,
                        available_at=(day + pd.Timedelta(days=2)).tz_localize("Asia/Seoul").isoformat(),
                        value=float(last[col]),
                        previous=float(prev[col]),
                        raw_hash=digest(dest.read_bytes()),
                    )
                )
            return result
        except Exception as e:
            return [dict(error=type(e).__name__, vintage_date=date)]

    rows = []
    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, batch in enumerate(pool.map(fetch, dates)):
            rows.extend(r for r in batch if "error" not in r)
            errors.extend(r for r in batch if "error" in r)
            if i % 50 == 0:
                print(f"ALFRED {i + 1}/{len(dates)}", flush=True)
    save_frame(
        out,
        "alfred",
        pd.DataFrame(rows),
        dict(
            vintages=len(dates),
            errors=errors,
            timing="Weekly dated ALFRED snapshots; only available after US vintage day plus two calendar days; max observation age 14 days",
        ),
    )


def earnings():
    out = output()
    root = local_settings().data_dir
    key = api_key()
    if not key:
        raise ValueError("DART key missing")
    rows = []
    bad = []
    (out / "dart-raw").mkdir(exist_ok=True)
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        for symbol in CORE:
            events = pd.read_parquet(root / f"research-inputs/direction-external/dart-{symbol}.parquet")
            corp = events.corp_code.iloc[0]
            for year in range(2015, 2024):
                for code in ["11013", "11012", "11014", "11011"]:
                    data, _ = download(
                        client,
                        out / "dart-raw",
                        "fnlttSinglAcntAll.json",
                        dict(corp_code=corp, bsns_year=str(year), reprt_code=code, fs_div="CFS"),
                        key,
                        86400 * 365,
                        5_000_000,
                    )
                    result = json.loads(data)
                    if result.get("status") != "000":
                        bad.append(dict(symbol=symbol, year=year, code=code, status=result.get("status")))
                        continue
                    receipts = {r.get("rcept_no") for r in result.get("list", [])}
                    for receipt in receipts:
                        match = events[events.rcept_no.eq(receipt)]
                        if len(match) != 1:
                            bad.append(
                                dict(
                                    symbol=symbol,
                                    year=year,
                                    code=code,
                                    status="receipt_outside_verified_list",
                                )
                            )
                            continue
                        event = match.iloc[0]
                        # Corrections become known only on the correction receipt date.
                        for r in result["list"]:
                            if r.get("rcept_no") == receipt:
                                rows.append(
                                    dict(
                                        r,
                                        symbol=symbol,
                                        publication_date=event.rcept_dt,
                                        report_name=event.report_nm,
                                    )
                                )
                    time.sleep(0.2)
            print("EARNINGS " + symbol, flush=True)
    save_frame(
        out,
        "earnings",
        pd.DataFrame(rows),
        dict(
            failures=bad,
            timing="Join actual receipt to disclosure list, next-calendar-day cutoff; corrected filings not backdated; current API amount vintage still unverified",
        ),
    )


def strength():
    out = output()
    reader = Reader(out)
    all_rows = []
    for symbol in CORE:
        rows = []
        cursor = ""
        cont = "N"
        seen = set()
        for page in range(80):
            raw = reader.get("ka10047", dict(stk_cd=symbol), cont, cursor)
            batch = [dict(r, symbol=symbol) for r in raw["response"].get("cntr_str_daly", []) if r.get("dt")]
            rows.extend(batch)
            if not batch or min(r["dt"] for r in batch) <= "20140101" or raw["continuation"] != "Y":
                break
            cursor = raw.get("next_key", "")
            cont = "Y"
            if not cursor or cursor in seen:
                break
            seen.add(cursor)
        all_rows.extend(rows)
        print("STRENGTH " + symbol + " " + str(len(rows)), flush=True)
    save_frame(
        out,
        "strength",
        pd.DataFrame(all_rows).drop_duplicates(["symbol", "dt"]),
        dict(
            timing="Daily execution-strength statistic; lag one full KRX session; original revisions/session vintages unverified"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["sector", "broker_history", "macro", "alfred", "earnings", "strength"]
    )
    stage = parser.parse_args().stage
    try:
        globals()[stage]()
    except Exception as exc:
        # Provider exception messages can contain authenticated request URLs.
        print(json.dumps({"stage": stage, "status": "failed", "error": type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
