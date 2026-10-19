"""Bounded public DART evidence collection. Missing search hits are not absence proof."""

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, urlparse

import httpx

from .util import atomic_json, digest, utcnow

RSS_URL = "https://dart.fss.or.kr/api/todayRSS.xml"


def parse_rss(data, names, retrieved_at):
    if len(data) > 2_000_000 or b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("Unsafe or oversized RSS")
    root = ET.fromstring(data)
    if root.tag != "rss":
        raise ValueError("Not an RSS response")
    sources, seen = [], set()
    by_name = {re.sub(r"\s+", "", name): symbol for symbol, name in names.items()}
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        # Market notices can have the exchange as dc:creator; match exact company in title.
        match = re.match(r"^\([^)]*\)(.+?)\s+-\s+", title)
        name = re.sub(r"\s+", "", match.group(1)) if match else ""
        if name not in by_name:
            continue
        url = urlparse(item.findtext("link") or "")
        number = parse_qs(url.query).get("rcpNo", [""])[0]
        if url.scheme != "https" or url.hostname != "dart.fss.or.kr" or not re.fullmatch(r"\d{14}", number):
            continue
        if number in seen:
            continue
        stamp = item.findtext("pubDate")
        published = parsedate_to_datetime(stamp).astimezone(timezone.utc).isoformat() if stamp else None
        if published and datetime.fromisoformat(published) > datetime.fromisoformat(retrieved_at):
            continue
        seen.add(number)
        content = {"title": title, "published_at": published, "symbol": by_name[name], "receipt": number}
        sources.append({"id": f"D{len(sources)+1}", "kind": "disclosure_title_only", "title": title[:1000],
                        "text": title[:1000], "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={number}",
                        "symbol": by_name[name], "published_at": published, "retrieved_at": retrieved_at,
                        "content_hash": digest(content)})
    return sources


def collect(settings, snapshot, client=None):
    if snapshot["mode"] == "demo":
        return {"status": "demo", "sources": [], "limitations": ["합성 데모에서는 외부 공시를 조회하지 않습니다."]}
    from .dart_evidence import api_key, collect_dart

    key = api_key()
    if key:
        return collect_dart(settings, snapshot, key, client)
    root = settings.data_dir / "evidence"
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / "dart-rss-cache.json"
    cached = json.loads(metadata.read_text(encoding="utf8")) if metadata.exists() else None
    now = datetime.now(timezone.utc)
    owned = client is None
    client = client or httpx.Client(timeout=20, follow_redirects=False)
    try:
        if cached and 0 <= (now - datetime.fromisoformat(cached["retrieved_at"])).total_seconds() <= 900:
            data = (root / f"{cached['hash']}.xml").read_bytes()
            if digest(data) != cached["hash"]:
                raise ValueError("Evidence hash mismatch")
            retrieved = cached["retrieved_at"]
        else:
            with client.stream("GET", RSS_URL) as response:
                response.raise_for_status()
                chunks, count = [], 0
                for chunk in response.iter_bytes():
                    count += len(chunk)
                    if count > 2_000_000:
                        raise ValueError("RSS too large")
                    chunks.append(chunk)
                data = b"".join(chunks)
            retrieved = utcnow()
            parse_rss(data, {}, retrieved)  # Validate before caching.
            file_hash = digest(data)
            path = root / f"{file_hash}.xml"
            if not path.exists():
                path.write_bytes(data)
            atomic_json(metadata, {"hash": file_hash, "retrieved_at": retrieved, "url": RSS_URL})
        names = {r["symbol"]: r["name"] for r in snapshot["stocks"]}
        sources = parse_rss(data, names, retrieved)
        return {"status": "ok", "sources": sources, "source_hash": digest(data), "retrieved_at": retrieved,
                "limitations": ["DART 최근 공시 RSS의 최신 일부 목록만 조회했습니다. 관련 공시 전체를 검색한 결과가 아닙니다.",
                                 "공시 제목·접수시각만 확인했습니다. 본문 내용·정정 관계·위험의 인과관계는 미확인입니다.",
                                 "보고서 생성 시점에 조회한 공시는 과거 위험 신호 발생 당시 이미 알려졌던 정보가 아닐 수 있습니다."]}
    except (httpx.HTTPError, ValueError, ET.ParseError, OSError, KeyError) as exc:
        return {"status": "unavailable", "sources": [], "error_type": type(exc).__name__,
                "limitations": ["공시 조회 실패. 관련 공시가 없다고 판단하지 않습니다."]}
    finally:
        if owned:
            client.close()
