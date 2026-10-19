"""Read-only data providers. No account inquiry or order endpoints are exposed."""

import json
import os
import re
import time
from datetime import datetime

import httpx
import pandas as pd

from .config import KST, credential_status
from .util import atomic_json, utcnow


class ProviderError(RuntimeError):
    pass


def request_json(method, url, **kwargs):
    # No redirects: never forward an authentication header to an unexpected host.
    for attempt in range(3):
        try:
            response = httpx.request(method, url, timeout=25, follow_redirects=False, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
            if response.status_code != 200:
                raise ProviderError(f"HTTP_{response.status_code}")
            body = response.json()
            if not isinstance(body, dict):
                raise ProviderError("INVALID_RESPONSE")
            return body
        except (httpx.HTTPError, ValueError):
            if attempt == 2:
                raise ProviderError("TRANSPORT_OR_JSON_ERROR") from None
            time.sleep(2**attempt)
    raise ProviderError("REQUEST_FAILED")


class KRX:
    BASE = "https://data-dbg.krx.co.kr/svc/apis"

    def __init__(self):
        self.key = os.environ.get("KRX_API_KEY", "").strip()
        if not self.key:
            raise ProviderError("KRX_KEY_MISSING")

    def daily(self, day, index=False):
        date = pd.Timestamp(day).strftime("%Y%m%d")
        endpoint = "idx/kospi_dd_trd" if index else "sto/stk_bydd_trd"
        payload = request_json(
            "GET", f"{self.BASE}/{endpoint}", headers={"AUTH_KEY": self.key}, params={"basDd": date}
        )
        if not isinstance(payload.get("OutBlock_1"), list):
            raise ProviderError("KRX_NO_DATA_BLOCK_CHECK_SERVICE_APPROVAL")
        return payload

    @staticmethod
    def normalize(payload, day):
        mapping = {
            "ISU_CD": "symbol",
            "ISU_NM": "name",
            "TDD_OPNPRC": "open",
            "TDD_HGPRC": "high",
            "TDD_LWPRC": "low",
            "TDD_CLSPRC": "close",
            "ACC_TRDVOL": "volume",
            "ACC_TRDVAL": "turnover",
            "MKTCAP": "market_cap",
        }
        frame = pd.DataFrame(payload["OutBlock_1"]).rename(columns=mapping)
        if (
            "BAS_DD" in frame
            and not frame.BAS_DD.astype(str)
            .str.replace("/", "")
            .str.replace("-", "")
            .eq(pd.Timestamp(day).strftime("%Y%m%d"))
            .all()
        ):
            raise ProviderError("KRX_RESPONSE_DATE_MISMATCH")
        required = {"symbol", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame):
            raise ProviderError("KRX_SCHEMA_MISMATCH")
        for column in ("open", "high", "low", "close", "volume", "turnover", "market_cap"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(",", ""), errors="coerce")
        frame["date"] = pd.Timestamp(day).strftime("%Y-%m-%d")
        frame["available_at"] = utcnow()
        frame["exchange"] = "KRX"
        # Raw official data do not establish adjusted prices or historical publication timing.
        frame["adjustment_verified"] = False
        frame["pit_verified"] = False
        return frame


class Kiwoom:
    def __init__(self, environment="live"):
        if environment not in {"live", "mock"}:
            raise ValueError("Invalid environment")
        self.host = "api.kiwoom.com" if environment == "live" else "mockapi.kiwoom.com"
        self.token = None
        self.expiry = None

    def authenticate(self):
        key = os.environ.get("KIWOOM_APP_KEY", "").strip()
        secret = os.environ.get("KIWOOM_APP_SECRET", "").strip()
        if not key or not secret:
            raise ProviderError("KIWOOM_KEYS_MISSING")
        body = request_json(
            "POST",
            f"https://{self.host}/oauth2/token",
            json={"grant_type": "client_credentials", "appkey": key, "secretkey": secret},
        )
        if str(body.get("return_code", 0)) != "0" or not body.get("token"):
            raise ProviderError("KIWOOM_AUTH_REJECTED_CHECK_IP_AND_KEYS")
        self.token, self.expiry = body["token"], body.get("expires_dt")
        return {"authenticated": True, "expiry": self.expiry}

    def quote(self, symbol):
        if not re.fullmatch(r"[0-9A-Z]{6}", symbol):
            raise ValueError("KRX 6 character symbol required")
        if not self.token:
            self.authenticate()
        body = request_json(
            "POST",
            f"https://{self.host}/api/dostk/stkinfo",
            headers={"authorization": f"Bearer {self.token}", "api-id": "ka10001"},
            json={"stk_cd": symbol},
        )
        if str(body.get("return_code", 0)) != "0":
            raise ProviderError("KIWOOM_QUOTE_REJECTED")
        for key in ("high_pric", "low_pric", "cur_prc"):
            if key not in body:
                raise ProviderError("KIWOOM_QUOTE_SCHEMA_MISMATCH")
        return body

    def daily_snapshot(self, symbol):
        """Dated daily bar used for reconnect recovery; date must match current session."""
        if not re.fullmatch(r"[0-9A-Z]{6}", symbol):
            raise ValueError("KRX symbol required")
        if not self.token:
            self.authenticate()
        body = request_json(
            "POST",
            f"https://{self.host}/api/dostk/mrkcond",
            headers={"authorization": f"Bearer {self.token}", "api-id": "ka10005"},
            json={"stk_cd": symbol},
        )
        if str(body.get("return_code", 0)) != "0" or not body.get("stk_ddwkmm"):
            raise ProviderError("KIWOOM_DAILY_SNAPSHOT_UNAVAILABLE")
        return body["stk_ddwkmm"][0]


class ECOS:
    def query(self, table, cycle, start, end, item, first=1, last=1000):
        key = os.environ.get("ECOS_API_KEY", "").strip()
        if not key:
            raise ProviderError("ECOS_KEY_MISSING")
        for value in (table, cycle, start, end, item):
            if not re.fullmatch(r"[A-Za-z0-9]+", value):
                raise ValueError("Invalid ECOS query component")
        body = request_json(
            "GET",
            f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/"
            f"{first}/{last}/{table}/{cycle}/{start}/{end}/{item}",
        )
        if "StatisticSearch" not in body:
            raise ProviderError("ECOS_QUERY_REJECTED")
        return body


def probe(settings, day="2026-10-15"):
    """Persist only whitelisted diagnostics; never dump tokens, URLs, or response messages."""
    result = {"checked_at": utcnow(), "credentials_present": credential_status(), "checks": {}}
    checks = result["checks"]
    for label, date in (("krx_recent", day), ("krx_2010", "2010-01-04")):
        try:
            payload = KRX().daily(date)
            rows = payload["OutBlock_1"]
            checks[label] = {
                "status": "pass" if rows else "empty",
                "date": date,
                "rows": len(rows),
                "fields": sorted(rows[0]) if rows else [],
            }
        except ProviderError as exc:
            checks[label] = {"status": "unverified", "reason": str(exc)}
    try:
        client = Kiwoom(settings.environment)
        checks["kiwoom_auth"] = {"status": "pass", **client.authenticate()}
        quote = client.quote("005930")
        checks["kiwoom_rest_quote"] = {"status": "pass", "fields": sorted(quote)}
    except ProviderError as exc:
        checks["kiwoom"] = {"status": "unverified", "reason": str(exc)}
    if result["credentials_present"]["ECOS_API_KEY"]:
        try:
            body = ECOS().query("731Y001", "D", "20260102", "20260102", "0000001")
            checks["ecos"] = {"status": "pass", "rows": len(body["StatisticSearch"].get("row", []))}
        except ProviderError as exc:
            checks["ecos"] = {"status": "unverified", "reason": str(exc)}
    result["data_gate"] = "pending: universe/calendar/adjustments/publication-time/session evidence required"
    atomic_json(settings.data_dir / "reports" / "access.json", result)
    return result


def signed_price(value):
    number = abs(float(str(value).replace(",", "")))
    if not 0 < number < float("inf"):
        raise ValueError("Invalid price")
    return number


def parse_tick(item, received_at):
    """Verified official 0B fields: 20 time, 290 session, 9081 exchange."""
    values = item.get("values", {})
    if item.get("type") != "0B" or not re.fullmatch(r"[0-9A-Z]{6}", item.get("item", "")):
        return None
    if values.get("9081") != "KRX" or str(values.get("290")) != "2":
        return None
    stamp = str(values.get("20", ""))
    if not re.fullmatch(r"\d{6}", stamp):
        return None
    local = received_at.astimezone(KST)
    event = datetime.strptime(local.strftime("%Y%m%d") + stamp, "%Y%m%d%H%M%S").replace(tzinfo=KST)
    try:
        quote = {
            "symbol": item["item"],
            "trade_date": event.date().isoformat(),
            "exchange": "KRX",
            "session": "regular",
            "event_at": event.isoformat(),
            "received_at": received_at.isoformat(),
            "price": signed_price(values["10"]),
            "high": signed_price(values["17"]),
            "low": signed_price(values["18"]),
            "volume": abs(float(values["13"])),
        }
    except (ValueError, KeyError):
        return None
    if not quote["low"] <= quote["price"] <= quote["high"] or not 0 <= quote["volume"] < float("inf"):
        return None
    return quote


def ingest_krx(lake, day):
    client = KRX()
    payload = client.daily(day)
    version = lake.bronze(json.dumps(payload, ensure_ascii=False).encode(), "krx-daily", {"date": day})
    if not payload["OutBlock_1"]:
        raise ProviderError("KRX_EMPTY_DAY")
    frame = KRX.normalize(payload, day)
    members = {m["symbol"] for m in lake.universe()["members"]}
    frame = frame[frame.symbol.isin(members)].copy()
    frame["pit_verified"] = pd.Timestamp(day).date() == datetime.now(KST).date()
    result = lake.ingest(frame, version)
    # Index availability is independent: HAR can publish even if this optional
    # ML input is temporarily unavailable.
    try:
        index = client.daily(day, index=True)
        index_hash = lake.bronze(
            json.dumps(index, ensure_ascii=False).encode(), "krx-index-daily", {"date": day}
        )
        rows = [r for r in index["OutBlock_1"] if r["IDX_NM"] == "코스피"]
        if len(rows) == 1:
            high, low = (float(rows[0][k].replace(",", "")) for k in ("HGPRC_IDX", "LWPRC_IDX"))
            if not 0 < low <= high:
                raise ProviderError("KRX_INDEX_PRICE_INVALID")
            row = pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp(day),
                        "range_pct": 100 * (high / low - 1),
                        "available_at": pd.Timestamp(utcnow()),
                        "pit_verified": True,
                        "source_hash": index_hash,
                    }
                ]
            )
            path = lake.root / "silver" / "market.parquet"
            old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
            from .util import exclusive_run

            with exclusive_run(lake.root / "silver.lock"):
                merged = (
                    pd.concat([old, row], ignore_index=True)
                    .drop_duplicates("date", keep="last")
                    .sort_values("date")
                )
                temp = path.with_suffix(".tmp.parquet")
                merged.to_parquet(temp, index=False)
                os.replace(temp, path)
        result["index_status"] = "ready" if len(rows) == 1 else "unavailable"
    except ProviderError:
        result["index_status"] = "unavailable"
    return result
