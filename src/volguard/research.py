import json
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import KST
from .data import Lake, features
from .evaluation import coverage_health, metrics, promotion_gate
from .models import RangeModels, sequential_predictions
from .store import Store
from .util import atomic_json, digest, exclusive_run, utcnow

POLICY = {
    "version": "1.1",
    "train_start": "2010-01-01",
    "validation_years": [2022, 2023],
    "calibration_start": "2024-01-01",
    "test_start": "2025-01-01",
    "test_end": "2026-09-30",
    "coverage": [0.87, 0.93],
    "minimum_improvement": 0.05,
    "bootstrap_block": 20,
    "bootstrap_repeats": 2000,
    "seed": 731,
    "calibration_window": 252,
    "calibration_minimum": 120,
    "model_parameters": {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.035},
    "optional_features_missing": "LightGBM native NaN; require HAR features and observed market range",
    "comparison_scope": "symbols with converged HAR fitted on training data; cold starts restricted individually",
    "health": {"window": 60, "coverage_below": 0.8, "consecutive_days": 5},
}


def source_hash():
    files = sorted(Path(__file__).parent.glob("*.py"))
    content = {p.name: digest(p.read_bytes()) for p in files}
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    if lock.exists():
        content["uv.lock"] = digest(lock.read_bytes())
    return digest(content)


def data_gate(lake, demo=False):
    if not demo:
        from .service import capabilities
        return capabilities(lake)
    reasons = []
    try:
        universe = lake.universe()
        sessions = lake.calendar()
    except ValueError as exc:
        return {"passed": False, "reasons": [str(exc)], "demo": demo}
    bars = lake.bars()
    if bars.empty:
        reasons.append("일봉 데이터 없음")
    else:
        symbols = [m["symbol"] for m in universe["members"]]
        subset = bars[bars.symbol.isin(symbols)]
        if set(symbols) - set(subset.symbol):
            reasons.append("유니버스 종목 데이터 누락")
        if not demo:
            evidence = universe.get("membership_evidence") or {}
            if universe.get("demo") or (
                len(symbols) != 200 and evidence.get("matched_symbols") != sorted(symbols)
            ):
                reasons.append("공식 코스피200 고정 목록 필요")
            if not subset.pit_verified.all():
                reasons.append("과거 이용 가능 시점 검증 필요")
            valid = subset[subset.valid_label]
            if not valid.adjustment_verified.all() or valid.adjusted_close.isna().any():
                reasons.append("수정주가·기업행동 검증 필요")
            market_path = lake.root / "silver" / "market.parquet"
            if not market_path.exists():
                reasons.append("검증된 코스피 지수 피처 필요")
            else:
                market = pd.read_parquet(market_path)
                if not market.pit_verified.all() or not set(valid.date).issubset(set(market.date)):
                    reasons.append("코스피 지수 피처의 검증 또는 날짜 범위 불완전")
        if not sessions:
            reasons.append("거래일 달력 비어 있음")
    return {"passed": not reasons, "reasons": reasons, "demo": demo}


def research_frame(lake):
    universe = lake.universe()
    symbols = {m["symbol"] for m in universe["members"]}
    bars = lake.bars()
    bars = bars[bars.symbol.isin(symbols)]
    market_path = lake.root / "silver" / "market.parquet"
    frame = features(bars, lake.calendar())
    if market_path.exists():
        market = pd.read_parquet(market_path)
        cutoff = (
            (market.date + pd.Timedelta(hours=18, minutes=30))
            .dt.tz_localize("Asia/Seoul")
            .dt.tz_convert("UTC")
        )
        market["range_pct"] = market.range_pct.where(market.available_at.le(cutoff) & market.pit_verified)
        frame["market_range"] = frame.date.map(market.set_index("date").range_pct)
    return frame


def import_market(lake, path):
    frame = pd.read_csv(path)
    required = {"date", "range_pct", "available_at", "pit_verified"}
    if not required.issubset(frame):
        raise ValueError("시장 피처: date,range_pct,available_at,pit_verified 필요")
    frame["date"] = pd.to_datetime(frame.date)
    frame["available_at"] = pd.to_datetime(frame.available_at, utc=True)
    frame["pit_verified"] = frame.pit_verified.astype(str).str.lower().isin(["true", "1"])
    frame["range_pct"] = pd.to_numeric(frame.range_pct, errors="raise")
    if frame.date.duplicated().any() or not np.isfinite(frame.range_pct).all() or (frame.range_pct < 0).any():
        raise ValueError("시장 피처 중복·비유한 값·음수")
    lake.bronze(path.read_bytes(), "verified-market")
    frame.to_parquet(lake.root / "silver" / "market.parquet", index=False)


def data_fingerprint(lake):
    paths = [
        lake.root / "silver" / "bars.parquet",
        lake.root / "config" / "universe.json",
        lake.root / "config" / "calendar.json",
    ]
    market = lake.root / "silver" / "market.parquet"
    if market.exists():
        paths.append(market)
    return digest({str(p.relative_to(lake.root)): digest(p.read_bytes()) for p in paths})


def history_fingerprint(lake):
    """Detect revisions to evaluated history while permitting new operational dates."""
    bars = lake.bars()
    columns = [
        "symbol",
        "date",
        "exchange",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "adjustment_factor",
        "adjustment_verified",
        "available_at",
        "pit_verified",
    ]
    old = bars[bars.date.le(POLICY["test_end"])][columns].sort_values(["symbol", "date", "exchange"])
    return digest(pd.util.hash_pandas_object(old.reset_index(drop=True), index=False).values.tobytes())


def evaluate_window(frame, train_end, calibration_start, evaluation_start, evaluation_end):
    train = frame[
        (frame.target_date <= pd.Timestamp(train_end)) & (frame.target_date >= POLICY["train_start"])
    ]
    # Future targets with delayed publication must also be unavailable to model fitting.
    fit_cutoff = pd.Timestamp(calibration_start, tz="Asia/Seoul").tz_convert("UTC")
    train = train[train.target_available_at.lt(fit_cutoff)]
    model = RangeModels().fit(train)
    window = frame[(frame.target_date >= calibration_start) & (frame.target_date <= evaluation_end)]
    predictions = pd.concat(
        [
            sequential_predictions(model, window, method, evaluation_start)
            for method in ("har", "ml", "empirical")
        ],
        ignore_index=True,
    )
    # Determine supported symbols from TRAINING ONLY. New listings that could
    # not be fitted must be restricted individually, not turn a valid baseline
    # into a failed model for all established stocks. Never choose this set by
    # validation/test loss or coverage.
    eligible = sorted(model.har)
    report = promotion_gate(predictions[predictions.symbol.isin(eligible)])
    report.update(
        eligible_symbols=eligible,
        excluded_symbols=sorted(set(frame.symbol) - set(eligible)),
        eligibility_basis="HAR training sample/convergence only; fixed before this evaluation window",
        full_universe_metrics={method: metrics(group) for method, group in predictions.groupby("method")},
    )
    return model, predictions, report


def validate(settings, demo=False):
    lake = Lake(settings.data_dir)
    gate = data_gate(lake, demo)
    if not gate["passed"]:
        raise ValueError("데이터 관문 미통과: " + "; ".join(gate["reasons"]))
    frame = research_frame(lake)
    output = settings.data_dir / "runs" / f"validation-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    reports = []
    for year in POLICY["validation_years"]:
        _, prediction, report = evaluate_window(
            frame, f"{year - 2}-12-31", f"{year - 1}-01-01", f"{year}-01-01", f"{year}-12-31"
        )
        prediction.to_parquet(output / f"{year}.parquet", index=False)
        reports.append({"year": year, **report})
    result = {
        "kind": "development_validation",
        "demo": demo,
        "created_at": utcnow(),
        "policy": POLICY,
        "source_hash": source_hash(),
        "data_hash": data_fingerprint(lake),
        "folds": reports,
        "artifact_dir": str(output),
    }
    atomic_json(output / "report.json", result)
    atomic_json(settings.data_dir / "reports" / "validation.json", result)
    return result


def freeze(settings, demo=False):
    lake = Lake(settings.data_dir)
    path = settings.data_dir / "config" / "study.json"
    if path.exists():
        raise ValueError("연구 manifest가 이미 고정되었습니다")
    report_path = settings.data_dir / "reports" / "validation.json"
    if not report_path.exists():
        raise ValueError("먼저 validate를 실행하세요")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    current_hash, current_source = data_fingerprint(lake), source_hash()
    if report["data_hash"] != current_hash or report["source_hash"] != current_source:
        raise ValueError("검증 이후 데이터·코드 변경: 개발 검증을 다시 실행하세요")
    manifest = {
        "policy": POLICY,
        "data_hash": current_hash,
        "source_hash": current_source,
        "demo": demo,
        "frozen_at": utcnow(),
        "validation_report_hash": digest(report),
        "survivorship_bias": "current KOSPI200 membership applied retrospectively",
    }
    manifest["study_id"] = digest(manifest)[:16]
    atomic_json(path, manifest)
    return manifest


def final_test(settings, demo=False):
    lake = Lake(settings.data_dir)
    manifest_path = settings.data_dir / "config" / "study.json"
    if not manifest_path.exists():
        raise ValueError("freeze-study가 먼저 필요합니다")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["demo"] != demo or manifest["data_hash"] != data_fingerprint(lake):
        raise ValueError("고정 데이터·모드 불일치")
    if manifest["source_hash"] != source_hash() or manifest["policy"] != POLICY:
        raise ValueError("고정 코드·정책 불일치")
    gate = data_gate(lake, demo)
    if not gate["passed"]:
        raise ValueError("데이터 관문 미통과")
    marker = settings.data_dir / "config" / "test-consumed.json"
    # Atomic exclusive creation is a dataset-level seal, not a run-directory-level guard.
    try:
        with marker.open("x", encoding="utf-8") as f:
            json.dump({"study_id": manifest["study_id"], "started_at": utcnow()}, f)
    except FileExistsError:
        raise ValueError("잠금 테스트는 이미 소비되었습니다. 기존 결과를 사용하세요") from None
    output = settings.data_dir / "runs" / f"test-{manifest['study_id']}"
    output.mkdir(parents=True, exist_ok=True)
    frame = research_frame(lake)
    model, prediction, report = evaluate_window(
        frame, "2023-12-31", "2024-01-01", POLICY["test_start"], POLICY["test_end"]
    )
    prediction.to_parquet(output / "predictions.parquet", index=False)
    joblib.dump(model, output / "models.joblib")
    result = {
        **report,
        "study_id": manifest["study_id"],
        "demo": demo,
        "created_at": utcnow(),
        "test_start": POLICY["test_start"],
        "test_end": POLICY["test_end"],
        "data_hash": manifest["data_hash"],
        "source_hash": manifest["source_hash"],
        "artifact_dir": str(output),
        "model_hash": digest((output / "models.joblib").read_bytes()),
        "history_hash": history_fingerprint(lake),
    }
    atomic_json(output / "report.json", result)
    atomic_json(settings.data_dir / "reports" / "test.json", result)
    # Only this explicit release pointer is used by nightly publication.
    atomic_json(settings.data_dir / "config" / "release.json", result)
    return result


def publish_daily(settings, as_of, demo=False):
    pointer = settings.data_dir / "config" / "release.json"
    if not demo and pointer.exists():
        release = json.loads(pointer.read_text(encoding="utf8"))
        if release.get("kind") == "har_operational_v2":
            from .service import publish
            return publish(settings, as_of)
    lake, store = Lake(settings.data_dir), Store(settings.db_path)
    day = pd.Timestamp(as_of).normalize()
    if (
        not demo
        and datetime.now(KST) < (day + pd.Timedelta(hours=18, minutes=30)).tz_localize(KST).to_pydatetime()
    ):
        raise ValueError("예측은 해당일 18:30 KST 이후에만 발행합니다")
    sessions = lake.calendar()
    dates = [s["date"] for s in sessions]
    stamp = day.strftime("%Y-%m-%d")
    if stamp not in dates or dates.index(stamp) + 1 >= len(dates):
        raise ValueError("현재·다음 거래일 달력이 필요합니다")
    target = dates[dates.index(stamp) + 1]
    release_path = settings.data_dir / "config" / "release.json"
    if not release_path.exists():
        raise ValueError("검증된 release 없음: 관측 전용")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release["demo"] != demo:
        raise ValueError("합성 데이터 모델은 실데이터에 사용할 수 없습니다")
    if release["source_hash"] != source_hash():
        raise ValueError("검증 후 코드·의존성 변경: 새로운 검증 버전이 필요합니다")
    if release["history_hash"] != history_fingerprint(lake):
        raise ValueError("평가된 과거 데이터 수정 감지: 새로운 검증 버전이 필요합니다")
    model_path = Path(release["artifact_dir"]) / "models.joblib"
    if digest(model_path.read_bytes()) != release["model_hash"]:
        raise ValueError("모델 무결성 검사 실패")
    model = joblib.load(model_path)  # Local trusted artifacts only; never accept uploaded pickle files.
    # Cutting before feature creation prevents late/future bars from changing this issuance.
    frame = research_frame(lake)
    frame = frame[(frame.date <= day) & (frame.target_date >= "2024-01-01")]
    outputs = pd.concat(
        [sequential_predictions(model, frame, method, "2025-01-01") for method in ("har", "ml")],
        ignore_index=True,
    )
    published = 0
    missing = []
    current_data_version = data_fingerprint(lake)
    current = lake.bars()
    current = current[current.date.eq(day) & current.exchange.eq("KRX")]
    with exclusive_run(settings.data_dir / "publication.lock"):
        for symbol in sorted(model.symbols):
            input_row = current[current.symbol.eq(symbol)]
            if input_row.empty or not input_row.valid_label.all():
                missing.append(symbol)
                continue
            if not demo and not (input_row.pit_verified.all() and input_row.adjustment_verified.all()):
                missing.append(symbol)
                continue
            histories, health = {}, {}
            cutoff = (day + pd.Timedelta(hours=18, minutes=30)).tz_localize(KST).tz_convert("UTC")
            for method in ("har", "ml"):
                histories[method] = outputs[(outputs.symbol == symbol) & (outputs.method == method)]
                past = histories[method]
                past = past[(past.target_date <= day) & past.target_available_at.le(cutoff)]
                past = past.dropna(subset=["y", "low", "high"]).sort_values("target_date")
                health[method] = coverage_health((past.y >= past.low) & (past.y <= past.high))
            latch_key = f"restricted:{release['study_id']}:{symbol}"
            restricted = store.get("meta", latch_key, False)
            if health["ml"]["status"] == "restricted":
                restricted = True
                store.put("meta", latch_key, True)
            selected = release["selected"]
            if symbol not in release.get("eligible_symbols", model.symbols):
                selected = "observation_only"
            if selected == "ml" and (restricted or health["ml"]["status"] != "normal"):
                selected = "har"
            if selected == "har" and (not release["har_pass"] or health["har"]["status"] != "normal"):
                selected = "observation_only"
            payload = {
                "symbol": symbol,
                "target_date": target,
                "issued_for": stamp,
                "generated_at": utcnow(),
                "model_version": release["study_id"],
                "data_version": current_data_version,
                "method": selected,
                "status": "ready" if selected != "observation_only" else "restricted",
                "low": None,
                "median": None,
                "high": None,
                "nominal_coverage": 0.9,
                "health": health,
                "demo": demo,
            }
            if selected != "observation_only":
                last = histories[selected]
                last = last[last.date.eq(day)].dropna(subset=["low", "median", "high"])
                if last.empty:
                    # Missing input never reuses yesterday's forecast under a new target date.
                    missing.append(symbol)
                    continue
                row = last.iloc[-1]
                payload.update(
                    low=float(row.low),
                    median=float(row["median"]),
                    high=float(row.high),
                    calibration_n=int(row.calibration_n),
                )
            published += store.publish(payload)
    return {
        "published": published,
        "target_date": target,
        "missing_inputs": missing,
        "status": "partial" if missing else "success",
    }
