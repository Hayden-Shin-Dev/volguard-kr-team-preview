"""Personal local service. Baseline release and capability-specific data checks."""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import KST
from .data import Lake, features
from .evaluation import coverage_health
from .models import sequential_predictions
from .store import Store
from .util import atomic_json, digest, exclusive_run, utcnow


def pipeline_hash():
    names = (
        "service.py",
        "models.py",
        "data.py",
        "evaluation.py",
        "providers.py",
        "store.py",
        "util.py",
        "research.py",
        "improvement.py",
        "config.py",
    )
    root = Path(__file__).parent
    return digest({name: digest((root / name).read_bytes()) for name in names})


def capabilities(lake):
    bars = lake.bars()
    reasons, warnings = [], []
    try:
        universe, sessions = lake.universe(), lake.calendar()
    except ValueError as exc:
        return {"passed": False, "reasons": [str(exc)], "capabilities": {}}
    if universe.get("demo"):
        reasons.append("합성 유니버스는 실운영에 사용할 수 없음")
    official = {
        p.stem
        for source in ("krx-stock-history", "krx-daily")
        for p in (lake.root / "bronze" / source).glob("*.raw")
    }
    members = {m["symbol"] for m in universe["members"]}
    if bars.empty or "source_hash" not in bars:
        reasons.append("공식 일봉 데이터 없음")
        eligible = []
    else:
        good = bars.valid_label & bars.exchange.eq("KRX") & bars.source_hash.isin(official)
        counts = bars[good].groupby("symbol").size()
        eligible = sorted(set(counts[counts.ge(252)].index) & members)
        if not eligible:
            reasons.append("KRX 원본으로 추적 가능한 학습 이력 부족")
        if not bars.pit_verified.all():
            warnings.append("과거 성능은 현재 조회본의 회고 평가이며 당시 정정 전 원본은 없음")
        if not bars.adjustment_verified.all():
            warnings.append("미확인 기업행동 경계의 수익률·VaR만 제한; 고저 비율은 별도 사용")
    if not sessions:
        reasons.append("거래일 달력 없음")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "eligible_symbols": eligible,
        "capabilities": {
            "range": not reasons,
            "returns": "per-row verified reference continuity or adjustment",
            "var": "per-portfolio sample and backtest gate",
            "external_features": "disabled",
        },
    }


def operational_frame(lake, as_of, historical_end):
    """Retrospective warm start is explicit; live dates use actual receipt times."""
    bars = lake.bars()
    day = pd.Timestamp(as_of).normalize()
    bars = bars[bars.date.le(day)].copy()
    historical = bars.date.le(historical_end)
    assumed = (bars.date + pd.Timedelta(hours=18)).dt.tz_localize(KST).dt.tz_convert("UTC")
    bars.loc[historical, "available_at"] = assumed[historical]
    cutoffs = bars.groupby("date").available_at.max()
    scheduled = (cutoffs.index + pd.Timedelta(hours=18, minutes=30)).tz_localize(KST).tz_convert("UTC")
    cutoffs = pd.Series([max(a, b) for a, b in zip(cutoffs, scheduled)], index=cutoffs.index)
    frame = features(bars, lake.calendar(), issue_cutoffs=cutoffs)
    market_path = lake.root / "silver" / "market.parquet"
    if market_path.exists():
        market = pd.read_parquet(market_path)
        available = market.date.le(historical_end) | market.available_at.le(market.date.map(cutoffs))
        frame["market_range"] = frame.date.map(market[available].set_index("date").range_pct)
    else:
        frame["market_range"] = np.nan
    return frame[frame.date.le(day)]


def activate_baseline(settings):
    from .research import history_fingerprint

    lake = Lake(settings.data_dir)
    gate = capabilities(lake)
    if not gate["passed"]:
        raise ValueError("; ".join(gate["reasons"]))
    report = json.loads((lake.root / "reports" / "test.json").read_text(encoding="utf8"))
    if report.get("demo") or not report.get("har_pass"):
        raise ValueError("실데이터 HAR 검증 관문 통과 결과 필요")
    path = Path(report["artifact_dir"]) / "models.joblib"
    if digest(path.read_bytes()) != report["model_hash"]:
        raise ValueError("검증 모델 무결성 불일치")
    existing = lake.root / "config" / "release.json"
    if existing.exists():
        old = json.loads(existing.read_text(encoding="utf8"))
        if old.get("kind") == "har_operational_v2":
            if old["pipeline_hash"] == pipeline_hash():
                return old
            if old["history_hash"] != history_fingerprint(lake):
                raise ValueError("운영 중 과거 데이터 정정: 자동 재승인 불가")
            atomic_json(lake.root / "runs" / "release-history" / f"{digest(old)}.json", old)
    # Reproduce archived HAR predictions exactly; this does not rescore or tune
    # a new model on the consumed holdout.
    from .history import retrospective_frame

    model = joblib.load(path)
    frame = retrospective_frame(lake)
    frame = frame[frame.target_date.between("2024-01-01", report["test_end"])]
    actual = sequential_predictions(model, frame, "har", report["test_start"])
    saved = pd.read_parquet(Path(report["artifact_dir"]) / "predictions.parquet")
    saved = saved[saved.method.eq("har")].sort_values(["symbol", "target_date"])
    actual = actual.sort_values(["symbol", "target_date"])
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), saved.reset_index(drop=True), check_exact=True
    )
    bars = lake.bars()
    release = {
        "kind": "har_operational_v2",
        "study_id": report["study_id"],
        "demo": False,
        "selected": "har",
        "har_pass": True,
        "activated_at": utcnow(),
        "artifact_dir": str(Path(report["artifact_dir"]).resolve()),
        "model_hash": report["model_hash"],
        "pipeline_hash": pipeline_hash(),
        "evaluation_report_hash": digest(report),
        "history_hash": history_fingerprint(lake),
        "eligible_symbols": sorted(set(report["eligible_symbols"]) & set(gate["eligible_symbols"])),
        "historical_end": old["historical_end"]
        if existing.exists() and old.get("kind") == "har_operational_v2"
        else str(bars.date.max().date()),
        "baseline_reproduction": "exact",
        "validation_basis": "retrospective holdout passed; personal monitored operation",
        "limitations": report.get("limitations", []),
        "operational_acceptance": "pending 20 actual sessions",
    }
    atomic_json(existing, release)
    atomic_json(lake.root / "reports" / "service.json", release)
    return release


def publish(settings, as_of, now=None):
    from .research import history_fingerprint

    now = now or datetime.now(KST)
    now = now.astimezone(KST)
    day = pd.Timestamp(as_of).normalize()
    if day.date() != now.astimezone(KST).date() or now.hour < 18 or (now.hour == 18 and now.minute < 30):
        raise ValueError("운영 예측은 당일 18:30 이후에만 발행; 소급 발행 금지")
    lake, store = Lake(settings.data_dir), Store(settings.db_path)
    release = json.loads((lake.root / "config" / "release.json").read_text(encoding="utf8"))
    if release["pipeline_hash"] != pipeline_hash():
        raise ValueError("운영 코드 무결성 불일치")
    if release["history_hash"] != history_fingerprint(lake):
        raise ValueError("검증된 과거 데이터 정정 감지: 재검토 필요")
    model_path = Path(release["artifact_dir"]) / "models.joblib"
    if digest(model_path.read_bytes()) != release["model_hash"]:
        raise ValueError("모델 무결성 불일치")
    sessions = lake.calendar()
    dates = [s["date"] for s in sessions]
    stamp = str(day.date())
    if stamp not in dates or dates.index(stamp) + 1 >= len(dates):
        raise ValueError("현재·다음 거래일 달력이 필요합니다")
    target = dates[dates.index(stamp) + 1]
    frame = operational_frame(lake, as_of, release["historical_end"])
    model = joblib.load(model_path)
    window = frame[frame.target_date.ge("2024-01-01")]
    output = sequential_predictions(model, window, "har", "2025-01-01")
    bars = lake.bars()
    current = bars[bars.date.eq(day)].set_index("symbol")
    expected = [m["symbol"] for m in store.watchlist()]
    missing, restricted, published, existing = [], [], 0, 0
    version = digest(pd.util.hash_pandas_object(bars, index=False).values.tobytes())
    with exclusive_run(lake.root / "publication.lock"):
        for symbol in expected:
            if store.prediction(symbol, target):
                existing += 1
                continue
            if symbol not in current.index or not bool(current.loc[symbol, "valid_label"]):
                missing.append(symbol)
                continue
            received = current.loc[symbol, "available_at"]
            if received > pd.Timestamp(now) or received.tz_convert(KST).date() != day.date():
                missing.append(symbol)
                continue
            source_hash = current.loc[symbol, "source_hash"]
            if not (lake.root / "bronze" / "krx-daily" / f"{source_hash}.raw").exists():
                missing.append(symbol)
                continue
            part = output[output.symbol.eq(symbol)]
            past = part[part.target_date.le(day) & part.target_available_at.le(pd.Timestamp(now))].dropna(
                subset=["y", "low", "high"]
            )
            # After activation use the saved issued bounds for operational
            # health, not a hindsight reconstruction under corrected inputs.
            with store.connect() as db:
                issued = [
                    json.loads(r[0])
                    for r in db.execute(
                        "SELECT payload FROM predictions WHERE symbol=? ORDER BY target_date", (symbol,)
                    )
                ]
            if issued:
                past = past[past.target_date.lt(min(p["target_date"] for p in issued))]
                realized = bars[bars.symbol.eq(symbol) & bars.valid_label & bars.date.le(day)].set_index(
                    "date"
                )
                actual = []
                for prediction in issued:
                    date = pd.Timestamp(prediction["target_date"])
                    if prediction.get("status") != "ready" or date not in realized.index:
                        continue
                    label = realized.loc[date]
                    if label.available_at <= pd.Timestamp(now):
                        actual.append(
                            {
                                "target_date": date,
                                "y": label.range_pct,
                                "low": prediction["low"],
                                "high": prediction["high"],
                            }
                        )
                past = pd.concat([past, pd.DataFrame(actual)], ignore_index=True).sort_values("target_date")
            health = coverage_health((past.y >= past.low) & (past.y <= past.high))
            health["basis"] = "historical warm-start plus immutable issued forecasts"
            latest = part[part.date.eq(day)].dropna(subset=["low", "median", "high"])
            ready = (
                symbol in release["eligible_symbols"] and health["status"] == "normal" and not latest.empty
            )
            if not ready:
                restricted.append(symbol)
            row = latest.iloc[-1] if not latest.empty else None
            payload = {
                "symbol": symbol,
                "target_date": target,
                "issued_for": stamp,
                "generated_at": now.isoformat(),
                "model_version": release["study_id"],
                "data_version": version,
                "method": "har" if ready else "observation_only",
                "status": "ready" if ready else "restricted",
                "demo": False,
                "low": float(row.low) if ready else None,
                "median": float(row["median"]) if ready else None,
                "high": float(row.high) if ready else None,
                "nominal_coverage": 0.9,
                "calibration_n": int(row.calibration_n) if ready else 0,
                "health": {"har": health},
                "validation_basis": release["validation_basis"],
            }
            published += store.publish(payload)
    try:
        _publish_shadow(settings, frame, target, stamp, now)
        store.put("meta", "shadow_error", None)
    except Exception as exc:
        store.put("meta", "shadow_error", {"at": utcnow(), "type": type(exc).__name__})
    return {
        "published": published,
        "already_published": existing,
        "missing_inputs": missing,
        "restricted_symbols": restricted,
        "target_date": target,
        "status": "partial" if missing else "success",
    }


def _publish_shadow(settings, frame, target, stamp, now):
    pointer = settings.data_dir / "config" / "shadow-release.json"
    if not pointer.exists():
        return
    from .improvement import enrich

    release = json.loads(pointer.read_text(encoding="utf8"))
    path = Path(release["artifact_dir"]) / "models.joblib"
    if digest(path.read_bytes()) != release["model_hash"]:
        raise ValueError("관찰 모델 무결성 불일치")
    frame = enrich(frame)
    baseline = json.loads((settings.data_dir / "config" / "release.json").read_text(encoding="utf8"))
    baseline_model = joblib.load(Path(baseline["artifact_dir"]) / "models.joblib")
    result = pd.concat(
        [
            sequential_predictions(
                joblib.load(path), frame[frame.target_date.ge(release["calibration_start"])], "ml", target
            ),
            sequential_predictions(baseline_model, frame[frame.target_date.ge("2024-01-01")], "har", target),
            sequential_predictions(baseline_model, frame, "empirical", target),
        ],
        ignore_index=True,
    )
    rows = result[result.date.eq(stamp) & result.symbol.isin(baseline["eligible_symbols"])].dropna(
        subset=["low", "high"]
    )
    directory = settings.data_dir / "forward"
    directory.mkdir(exist_ok=True)
    file = directory / f"{target}.json"
    if not file.exists():
        atomic_json(
            file,
            {
                "issued_at": now.isoformat(),
                "target_date": target,
                "model_hash": release["model_hash"],
                "role": "shadow_only",
                "baseline_model_hash": baseline["model_hash"],
                "predictions": rows[["symbol", "method", "low", "median", "high"]].to_dict("records"),
            },
        )


def run_daily(settings, day):
    from .operations import report
    from .providers import ingest_krx

    lake, store = Lake(settings.data_dir), Store(settings.db_path)
    now = datetime.now(KST)
    if day != now.date().isoformat() or now.strftime("%H:%M") < "18:30":
        raise ValueError("실운영 배치는 당일 18:30 이후 실행합니다")
    dates = [s["date"] for s in lake.calendar()]
    if day > max(dates):
        raise ValueError("운영 달력 만료: 거래소 일정 확인 후 연장 필요")
    if day not in dates:
        return {"status": "non_trading_day"}
    batch = {
        "input_date": day,
        "demo": False,
        "observed_at": now.isoformat(),
        "status": "running",
        "data_ready": False,
    }
    with exclusive_run(lake.root / "daily.lock"), store.run("daily") as log:
        store.put("meta", f"batch:{day}", batch)
        try:
            ingest_krx(lake, day)
            current = lake.bars()
            current = current[current.date.eq(pd.Timestamp(day))]
            expected = {m["symbol"] for m in store.watchlist()}
            batch["data_ready"] = bool(expected) and expected.issubset(
                set(current.loc[current.valid_label, "symbol"])
            )
            result = publish(settings, day)
            from .forward import evaluate

            evaluate(settings)
            batch.update(result)
            log.update(status=result["status"], rows=result["published"])
            return result
        except Exception as exc:
            batch.update(status="failed", error=type(exc).__name__)
            raise
        finally:
            store.put("meta", f"batch:{day}", batch)
            report(store, settings)
