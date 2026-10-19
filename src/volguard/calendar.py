"""Versioned auxiliary exchange schedule, checked against official observations."""

import exchange_calendars as xc
import pandas as pd

from .data import Lake
from .util import atomic_json, digest, utcnow


def build_schedule(settings, end="2026-11-18"):
    lake = Lake(settings.data_dir)
    observed = pd.read_parquet(lake.root / "silver" / "market.parquet").date.sort_values()
    start, last = observed.iloc[0], observed.iloc[-1]
    # This initial release is bounded. Later special-session notices must be
    # reviewed before extending it; a calendar library is not a live exchange feed.
    if pd.Timestamp(end) > pd.Timestamp("2026-11-18"):
        raise ValueError("2026-11-19 수능 특별장 시간의 KRX 공지 검토 후 달력을 연장하세요")
    schedule = xc.get_calendar("XKRX", start=str(start.date()), end=end).schedule
    overrides = {
        "2026-06-03": {"reason": "지방선거일", "source": "https://strn.krx.co.kr/corebbs5/BHPSTRN0401/list"},
        "2026-07-17": {
            "reason": "제헌절 공휴일 재지정",
            "source": "https://www.mofa.go.kr/us-newyork-ko/brd/m_4237/view.do?seq=1348072",
        },
    }
    schedule = schedule.drop(index=pd.to_datetime(list(overrides)), errors="ignore")
    expected = set(schedule.loc[:last].index)
    actual = set(observed)
    report = {
        "version": xc.__version__,
        "calendar": "XKRX",
        "source": "exchange_calendars",
        "observed_start": str(start.date()),
        "observed_end": str(last.date()),
        "planned_end": end,
        "official_observations_missing": sorted(str(d.date()) for d in expected - actual),
        "library_sessions_missing": sorted(str(d.date()) for d in actual - expected),
        "created_at": utcnow(),
        "reviewed_holiday_overrides": overrides,
        "note": "거래일은 KRX 지수 관측일과 전수 대조. 예정 개장시간은 버전 고정 보조 달력. WS 정규장 확인도 별도 필요.",
    }
    report["passed"] = actual == expected
    atomic_json(lake.root / "reports" / "calendar.json", report)
    if not report["passed"]:
        raise ValueError("KRX 실제 관측 거래일과 보조 달력이 다릅니다: calendar.json 확인")
    sessions = [
        {
            "date": str(day.date()),
            "open_time": row["open"].tz_convert("Asia/Seoul").strftime("%H:%M:%S"),
            "close_time": row["close"].tz_convert("Asia/Seoul").strftime("%H:%M:%S"),
            "hours_verified": True,
            "schedule_basis": f"exchange_calendars {xc.__version__} XKRX",
        }
        for day, row in schedule.iterrows()
    ]
    atomic_json(
        lake.root / "config" / "calendar.json",
        {
            "sessions": sessions,
            "schedule_report_hash": digest(
                {key: value for key, value in report.items() if key != "created_at"}
            ),
            "source": f"exchange_calendars {xc.__version__}; historical dates crosschecked with KRX",
        },
    )
    return report
