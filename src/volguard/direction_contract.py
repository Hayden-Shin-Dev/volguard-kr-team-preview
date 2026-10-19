"""Frozen close-to-next-close contract for prospective direction research.

This does not alter historical experiments or authorize a model release.
"""

from datetime import datetime
from decimal import Decimal

from .config import KST
from .util import atomic_json, digest, exclusive_run

CONTRACT = {
    "version": "direction-krx-close-v2",
    "exchange": "KRX",
    "session": "regular",
    "feature_cutoff": "anchor session official close",
    "target": "next KRX trading session close versus anchor close",
    "information_rule": "observed_at <= close; official regular-session daily aggregates may arrive after close, before issuance; other information must be available by close",
    "publication_rule": "same anchor day, after close; real issued_at, no backfill",
    "flat_rule": "separate flat outcome; binary probabilities conditional on nonflat",
    "price_rule": "comparable corporate-action-adjusted KRX regular closes",
    "research_only": True,
    "down_research_threshold": 0.60,
    "primary_evaluation": "two-sided probability reliability, Brier and Log Loss",
    "historical_results": "not relabelled as prospective or close-cutoff verified",
}


def aware(value):
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timezone-aware timestamp required")
    return dt.astimezone(KST)


def window(calendar, anchor):
    sessions = sorted(calendar["sessions"], key=lambda s: s["date"])
    for i, session in enumerate(sessions):
        if session["date"] == anchor:
            if i + 1 == len(sessions):
                raise ValueError("Next session not verified: extend reviewed calendar")
            target = sessions[i + 1]
            if not session.get("hours_verified") or not target.get("hours_verified"):
                raise ValueError("Exchange hours not verified")
            return {
                "anchor_date": anchor,
                "target_date": target["date"],
                "feature_cutoff": f"{anchor}T{session['close_time']}+09:00",
                "target_close": f"{target['date']}T{target['close_time']}+09:00",
            }
    raise ValueError("Anchor is not a reviewed KRX trading session")


def validate_inputs(evidence, cutoff, issued_at=None):
    if not evidence:
        raise ValueError("No point-in-time input evidence")
    for row in evidence:
        if not row.get("source_hash"):
            raise ValueError("Input source hash missing")
        finalized = row.get("kind") == "krx_regular_daily" and row.get("session") == "regular"
        limit = issued_at if finalized and issued_at is not None else cutoff
        if aware(row["observed_at"]) > aware(cutoff) or aware(row["available_at"]) > aware(limit):
            raise ValueError("Post-close input: exclude or use an earlier verified vintage")


def record(root, calendar, anchor, symbol, model_version, p_up, evidence, *, now=None, details=None):
    """Immutable research journal. Real clock by default; no inferred forecasts."""
    now = aware(now or datetime.now(KST))
    dates = window(calendar, anchor)
    if now.date().isoformat() != anchor or now < aware(dates["feature_cutoff"]):
        raise ValueError("Issue only after close on anchor day; no retroactive issuance")
    if not 0 <= p_up <= 1:
        raise ValueError("Probability must be finite and in [0, 1]")
    validate_inputs(evidence, dates["feature_cutoff"], now)
    entry = dict(
        contract=CONTRACT,
        contract_hash=digest(CONTRACT),
        **dates,
        issued_at=now.isoformat(),
        symbol=symbol,
        model_version=model_version,
        p_up=float(p_up),
        p_down=float(1 - p_up),
        input_evidence=evidence,
        details=details or {},
    )
    folder = root / "direction-forward" / dates["target_date"]
    key = digest({"symbol": symbol, "model": model_version, "contract": CONTRACT})
    path = folder / f"{key}.json"
    with exclusive_run(root / "direction-forward" / "journal.lock"):
        if path.exists():
            raise FileExistsError("Already issued: retain original probability")
        atomic_json(path, entry)
    return path


def outcome(entry, anchor_close, target_close, *, target_date, comparable, now=None):
    if target_date != entry["target_date"]:
        raise ValueError("Outcome must be the next trading session, not next available quote")
    if aware(now or datetime.now(KST)) < aware(entry["target_close"]):
        raise ValueError("Target session has not closed")
    if not comparable:
        raise ValueError("KRX session or corporate-action price basis unverified")
    a, b = Decimal(str(anchor_close)), Decimal(str(target_close))
    if not a.is_finite() or not b.is_finite() or a <= 0 or b <= 0:
        raise ValueError("Invalid close price")
    return {
        "direction": "up" if b > a else "down" if b < a else "flat",
        "label": int(b > a) if b != a else None,
        "return": float(b / a - 1),
        "flat": b == a,
    }
