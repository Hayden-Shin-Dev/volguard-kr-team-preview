"""Signed observations acknowledge actual visible browser paints during regular sessions."""

import hashlib
import hmac
import json
from datetime import datetime

from .realtime import within_session


def ticket(secret, quote, now, sessions):
    if not quote or not within_session(now, sessions):
        return None
    if not quote.get("complete") or quote.get("exchange") != "KRX" or quote.get("session") != "regular":
        return None
    if quote.get("trade_date") != now.date().isoformat() or not quote.get("verified_at"):
        return None
    body = json.dumps(
        {"symbol": quote["symbol"], "verified_at": quote["verified_at"], "sent_at": now.isoformat()},
        separators=(",", ":"),
    )
    return body + "|" + hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()


def acknowledge(store, secret, value, visible, now, sessions):
    if not visible or not within_session(now, sessions):
        return False
    try:
        body, signature = value.rsplit("|", 1)
        expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return False
        observation = json.loads(body)
        sent = datetime.fromisoformat(observation["sent_at"])
        verified = datetime.fromisoformat(observation["verified_at"])
        delay = (now - verified).total_seconds()
        if not 0 <= (now - sent).total_seconds() <= 300 or delay < 0:
            return False
        # One paint per confirmed quote state; retries and multiple tabs cannot bias the percentile.
        identity = observation["symbol"] + ":" + observation["verified_at"]
        with store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            last = db.execute("SELECT payload FROM meta WHERE key='terminal_render_seen'").fetchone()
            seen = json.loads(last[0]) if last else []
            if identity in seen:
                return False
            row = db.execute("SELECT payload FROM meta WHERE key='render_latencies'").fetchone()
            values = json.loads(row[0]) if row else []
            values.append(delay)
            for key, payload in [
                ("terminal_render_seen", (seen + [identity])[-60000:]),
                ("render_latencies", values[-60000:]),
            ]:
                db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(payload)))
        return True
    except (ValueError, KeyError, TypeError):
        return False
