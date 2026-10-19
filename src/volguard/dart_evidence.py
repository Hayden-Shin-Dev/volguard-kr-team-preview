"""Official OpenDART company search and bounded document excerpts, with raw provenance."""

import io
import json
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta
from html.parser import HTMLParser

import httpx
from dotenv import dotenv_values

from .config import KST, ROOT
from .util import atomic_json, digest, utcnow

BASE = "https://opendart.fss.or.kr/api"


def api_key():
    values = dotenv_values(ROOT / ".env")
    return os.getenv("DART_API_KEY") or values.get("DART_API_KEY") or os.getenv("OPENDART_API_KEY") or values.get("OPENDART_API_KEY")


class DartError(ValueError):
    def __init__(self, code):
        self.code = code if re.fullmatch(r"\d{3}", str(code)) else "invalid_response"
        super().__init__(self.code)


def xml(data):
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("Unsafe XML")
    return ET.fromstring(data)


def zip_members(data, limit):
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise DartError(xml(data).findtext("status"))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > 50 or sum(i.file_size for i in infos) > limit:
            raise ValueError("Document archive exceeds limit")
        # Never extract paths supplied by the archive onto disk.
        return [(i.filename, archive.read(i)) for i in infos if i.filename.lower().endswith((".xml", ".html", ".htm"))]


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, value):
        if not self.hidden and value.strip():
            self.parts.append(value.strip())


def document_excerpt(data, receipt, size=1800):
    members = zip_members(data, 20_000_000)
    if not members:
        raise ValueError("No readable document")
    members.sort(key=lambda item: (receipt not in item[0], item[0]))
    name, raw = members[0]
    if b"<!ENTITY" in raw.upper():
        raise ValueError("Unsafe document")
    for encoding in ("utf-8", "cp949"):
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Unsupported document encoding")
    parser = TextParser()
    parser.feed(decoded)
    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    if not text:
        raise ValueError("Empty document text")
    return {"text": text[:size], "document_member": name, "document_chars": len(text),
            "excerpt_start": 0, "excerpt_end": min(size, len(text)), "truncated": len(text) > size,
            "other_members": len(members) - 1}


def download(client, root, endpoint, params, key, ttl, max_bytes):
    cache = root / f"request-{digest([endpoint, params])}.json"
    if cache.exists():
        meta = json.loads(cache.read_text(encoding="utf8"))
        age = (datetime.now(KST) - datetime.fromisoformat(meta["retrieved_at"])).total_seconds()
        if 0 <= age <= ttl:
            data = (root / f"{meta['hash']}.bin").read_bytes()
            if digest(data) != meta["hash"]:
                raise ValueError("Evidence hash mismatch")
            return data, meta
    with client.stream("GET", f"{BASE}/{endpoint}", params={"crtfc_key": key, **params}) as response:
        response.raise_for_status()
        chunks, count = [], 0
        for chunk in response.iter_bytes():
            count += len(chunk)
            if count > max_bytes:
                raise ValueError("DART response exceeds limit")
            chunks.append(chunk)
    data = b"".join(chunks)
    # API errors are neither cached nor logged with their secret-bearing request URL.
    if data.lstrip().startswith(b"{"):
        status = json.loads(data).get("status")
        if status not in {"000", "013"}:
            raise DartError(status)
    elif not zipfile.is_zipfile(io.BytesIO(data)):
        raise DartError(xml(data).findtext("status"))
    meta = {"hash": digest(data), "retrieved_at": utcnow(), "endpoint": endpoint, "params": params}
    raw_path = root / f"{meta['hash']}.bin"
    if not raw_path.exists():
        raw_path.write_bytes(data)
    atomic_json(cache, meta)
    return data, meta


def collect_dart(settings, snapshot, key, client=None):
    root = settings.data_dir / "evidence" / "opendart"
    root.mkdir(parents=True, exist_ok=True)
    owned = client is None
    client = client or httpx.Client(timeout=15, follow_redirects=False)
    sources, searches, errors = [], [], []
    started, body_count = time.monotonic(), 0
    now = datetime.now(KST)
    start, end = (now.date() - timedelta(days=30)).strftime("%Y%m%d"), now.strftime("%Y%m%d")
    try:
        data, _ = download(client, root, "corpCode.xml", {}, key, 86400, 10_000_000)
        members = zip_members(data, 100_000_000)
        companies = {}
        for _, raw in members:
            for item in xml(raw).findall("list"):
                stock, corp = (item.findtext("stock_code") or "").strip(), (item.findtext("corp_code") or "").strip()
                if re.fullmatch(r"\d{6}", stock) and re.fullmatch(r"\d{8}", corp):
                    companies[stock] = corp
        for stock in snapshot["stocks"]:
            symbol = stock["symbol"]
            if symbol not in companies or time.monotonic() - started > 90:
                errors.append({"symbol": symbol, "reason": "company_mapping_missing_or_time_budget"})
                continue
            try:
                params = {"corp_code": companies[symbol], "bgn_de": start, "end_de": end,
                          "page_no": 1, "page_count": 100, "sort": "date", "sort_mth": "desc", "last_reprt_at": "N"}
                data, meta = download(client, root, "list.json", params, key, 900, 2_000_000)
                result = json.loads(data)
                items = result.get("list", [])
                searches.append({"symbol": symbol, "corp_code": companies[symbol], "start": start, "end": end,
                                 "total_count": int(result.get("total_count", 0)), "fetched_count": len(items),
                                 "selected_count": min(3, len(items)), "hash": meta["hash"], "retrieved_at": meta["retrieved_at"]})
                for index, item in enumerate(items[:3]):
                    receipt = item.get("rcept_no", "")
                    if not re.fullmatch(r"\d{14}", receipt) or item.get("corp_code") != companies[symbol]:
                        raise ValueError("Disclosure identity mismatch")
                    source = {"id": f"D{len(sources)+1}", "kind": "disclosure_title_only", "symbol": symbol,
                              "title": item["report_nm"], "text": item["report_nm"], "receipt": receipt,
                              "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}",
                              "receipt_date": item["rcept_dt"], "published_at": None,
                              "retrieved_at": meta["retrieved_at"], "content_hash": digest(item),
                              "remarks": item.get("rm", ""), "search_hash": meta["hash"]}
                    if index == 0 and body_count < 3 and time.monotonic() - started < 90:
                        body_count += 1
                        try:
                            raw, docmeta = download(client, root, "document.xml", {"rcept_no": receipt}, key,
                                                    86400, 8_000_000)
                            excerpt = document_excerpt(raw, receipt)
                            source.update(excerpt)
                            source.update(kind="disclosure_body_excerpt", document_hash=docmeta["hash"],
                                          document_retrieved_at=docmeta["retrieved_at"])
                        except (httpx.HTTPError, ValueError, ET.ParseError, OSError, KeyError, zipfile.BadZipFile) as exc:
                            source["body_status"] = type(exc).__name__
                    sources.append(source)
            except (httpx.HTTPError, ValueError, ET.ParseError, OSError, KeyError) as exc:
                errors.append({"symbol": symbol, "reason": type(exc).__name__, "code": getattr(exc, "code", None)})
    except (httpx.HTTPError, ValueError, ET.ParseError, OSError, KeyError, zipfile.BadZipFile) as exc:
        errors.append({"reason": type(exc).__name__, "code": getattr(exc, "code", None)})
    finally:
        if owned:
            client.close()
    return {"status": "partial" if errors else "ok", "provider": "opendart", "sources": sources,
            "searches": searches, "errors": errors, "retrieved_at": utcnow(),
            "limitations": ["최근 30일의 회사별 첫 100건 중 최신 3건을 선택했습니다. 전체 공시 조사 결과가 아닙니다.",
                             "보고서당 최대 3건의 본문 앞부분만 발췌했습니다. 전체 본문·첨부·정정 관계를 검증한 것은 아닙니다.",
                             "접수일자는 정확한 발표 시각이 아닙니다. 현재 조회 자료를 과거 알림 당시 정보로 소급하지 않습니다.",
                             "조회 실패나 일치 공시 없음은 관련 사건이 없다는 증거가 아닙니다."]}
