"""Evidence tools for analyst roles. All original provider responses remain local."""

import json
import re
from datetime import datetime

import httpx

from .config import KST
from .dart_evidence import api_key, download
from .providers import KRX
from .terminal_data import history, read_json
from .util import atomic_json, digest, utcnow


def financials(settings, searches, names):
    key = api_key()
    if not key:
        return [], ["DART 인증키 미설정으로 재무정보를 확인하지 못했습니다."]
    root = settings.data_dir / "evidence/opendart"
    root.mkdir(parents=True, exist_ok=True)
    year = datetime.now(KST).year
    month = datetime.now(KST).month
    periods = (
        ([(year, "11014", "3분기")] if month >= 11 else [])
        + ([(year, "11012", "반기")] if month >= 8 else [])
        + ([(year, "11013", "1분기")] if month >= 5 else [])
        + [(year - 1, "11011", "사업연도")]
    )
    sources, limitations = [], []
    with httpx.Client(timeout=20, follow_redirects=False) as client:
        for search in searches[:5]:
            symbol, corp = search["symbol"], search["corp_code"]
            found = False
            try:
                for y, report_code, period in periods:
                    for division in ("CFS", "OFS"):
                        raw, meta = download(
                            client,
                            root,
                            "fnlttSinglAcntAll.json",
                            {
                                "corp_code": corp,
                                "bsns_year": str(y),
                                "reprt_code": report_code,
                                "fs_div": division,
                            },
                            key,
                            86400,
                            4_000_000,
                        )
                        items = json.loads(raw).get("list", [])
                        selected = statement_accounts(items)
                        if not selected:
                            continue
                        rows = [
                            {
                                k: r.get(k)
                                for k in (
                                    "account_nm",
                                    "thstrm_nm",
                                    "thstrm_amount",
                                    "thstrm_add_amount",
                                    "currency",
                                    "sj_div",
                                )
                            }
                            for r in selected
                        ]
                        receipt = selected[0].get("rcept_no", "")
                        text = f"{names[symbol]} · {y}년 {period} · {'연결' if division == 'CFS' else '별도'} 재무제표. "
                        text += "손익의 당기금액은 분기값일 수 있으며 누적금액을 별도 표시합니다. "
                        for row in rows:
                            unit = "원" if row["currency"] == "KRW" else str(row["currency"] or "통화 미확인")
                            statement = {
                                "BS": "재무상태표",
                                "IS": "손익계산서",
                                "CIS": "포괄손익계산서",
                                "CF": "현금흐름표",
                            }.get(row["sj_div"], row["sj_div"])
                            text += f"{statement} · {row['account_nm']} · {row['thstrm_nm']} · 당기금액 {row['thstrm_amount']}{unit}. "
                            if row.get("thstrm_add_amount"):
                                text += f"{row['account_nm']} · 누적금액 {row['thstrm_add_amount']}{unit}. "
                        sources.append(
                            {
                                "id": f"F{len(sources) + 1}",
                                "symbol": symbol,
                                "kind": "financial_statement",
                                "title": f"{names[symbol]} {y} {period} 재무정보",
                                "text": text,
                                "facts": rows,
                                "period": f"{y} {period}",
                                "division": division,
                                "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}"
                                if re.fullmatch(r"\d{14}", receipt)
                                else None,
                                "retrieved_at": meta["retrieved_at"],
                                "content_hash": meta["hash"],
                            }
                        )
                        found = True
                        break
                    if found:
                        break
                if not found:
                    limitations.append(
                        f"{names[symbol]}: 조회한 최근 보고기간의 주요 재무계정을 확보하지 못했습니다."
                    )
            except (httpx.HTTPError, ValueError, KeyError, OSError):
                limitations.append(f"{names[symbol]}: 재무 조회 실패. 재무 위험이 없다는 의미가 아닙니다.")
    return sources, limitations


def statement_accounts(items):
    """Exclude equity-change components and CF profit adjustments from headline accounts."""
    allowed = {
        "ifrs-full_Revenue": ("IS", "CIS"),
        "dart_OperatingIncomeLoss": ("IS", "CIS"),
        "ifrs-full_ProfitLoss": ("IS", "CIS"),
        "ifrs-full_Assets": ("BS",),
        "ifrs-full_Liabilities": ("BS",),
        "ifrs-full_Equity": ("BS",),
        "ifrs-full_CashAndCashEquivalents": ("BS",),
        "ifrs-full_CashFlowsFromUsedInOperatingActivities": ("CF",),
    }
    selected = []
    for account, statements in allowed.items():
        for statement in statements:
            found = next(
                (r for r in items if r.get("account_id") == account and r.get("sj_div") == statement), None
            )
            if found:
                selected.append(found)
                break
    return selected


def market_evidence(settings, symbols):
    sources, comparisons = [], []
    from .data import Lake

    names = {r["symbol"]: r["name"] for r in Lake(settings.data_dir).universe()["members"]}
    for symbol in symbols:
        rows = history(settings, symbol, 22)
        if not rows:
            continue
        ranges = [100 * (r["high"] / r["low"] - 1) for r in rows]
        baseline = sum(r["volume"] for r in rows[:-1]) / max(1, len(rows) - 1)
        fact = {
            "symbol": symbol,
            "as_of": rows[-1]["time"],
            "range_pct": ranges[-1],
            "mean_range_pct": sum(ranges) / len(ranges),
            "volume_ratio": rows[-1]["volume"] / baseline if baseline else None,
        }
        comparisons.append(fact)
        text = f"{names[symbol]} · {fact['as_of']} · 고저 변동폭 {fact['range_pct']:.2f}%. 최근 {len(rows)}거래일 평균 고저 변동폭 {fact['mean_range_pct']:.2f}%. "
        text += (
            f"직전 {len(rows) - 1}거래일 평균 대비 거래량 {fact['volume_ratio']:.2f}배."
            if fact["volume_ratio"] is not None
            else "거래량 비교 자료 부족."
        )
        sources.append(
            {
                "id": f"P{len(sources) + 1}",
                "symbol": symbol,
                "kind": "price_volume",
                "title": f"{names[symbol]} 가격·거래량 관측",
                "text": text,
                "facts": fact,
                "retrieved_at": utcnow(),
                "url": None,
                "content_hash": digest(fact),
            }
        )
    day = max((r["as_of"] for r in comparisons), default=None)
    indices = []
    if day:
        cache = settings.data_dir / "evidence" / f"indices-{day}.json"
        try:
            raw = read_json(cache)
            if raw is None:
                raw = KRX().daily(day, index=True)
                atomic_json(cache, raw)
            for r in raw.get("OutBlock_1", []):
                if not r.get("LWPRC_IDX") or not r.get("HGPRC_IDX"):
                    continue
                low = float(r["LWPRC_IDX"].replace(",", ""))
                high = float(r["HGPRC_IDX"].replace(",", ""))
                if low > 0 and high >= low:
                    indices.append({"name": r["IDX_NM"], "range_pct": 100 * (high / low - 1), "as_of": day})
            # Every index retains its official name; no unsupported issuer-to-sector mapping.
            sources.append(
                {
                    "id": "M1",
                    "kind": "market_sector",
                    "title": "KRX 시장·업종 지수 당일 변동폭 비교",
                    "text": "시장·업종 지수 관측 (개별 종목의 업종 배속 미확정). "
                    + "; ".join(f"{r['name']} · {day} · 고저 변동폭 {r['range_pct']:.2f}%" for r in indices),
                    "facts": indices,
                    "url": None,
                    "retrieved_at": utcnow(),
                    "content_hash": digest(raw),
                }
            )
        except (RuntimeError, ValueError, KeyError, OSError, httpx.HTTPError):
            pass
    return sources, comparisons, indices
