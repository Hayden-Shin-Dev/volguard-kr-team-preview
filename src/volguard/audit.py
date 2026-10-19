"""Honest retrospective performance audit, isolated from deployment approval.

An audit can measure real-market performance without pretending that downloaded
history is an archived as-of dataset. Reading the holdout still consumes it.
"""

import json
import uuid

import joblib
import pandas as pd

from .data import Lake
from .evaluation import metrics
from .history import retrospective_frame
from .research import POLICY, data_fingerprint, data_gate, evaluate_window, source_hash
from .risk import historical_var, stress_tests
from .util import atomic_json, digest, utcnow

LIMITATIONS = [
    "현재 코스피200 구성의 소급 적용에 따른 생존편향",
    "KRX 현재 조회본으로 과거를 복원: 당시 정정 전 원본 아카이브 없음",
    "당일 OHLCV는 18:00까지 이용 가능했다는 가정; 당시 API 공표시각 미확인",
    "보조 수정주가의 OHLC 배율 일치를 점검했으나 모든 기업행동 권리내역을 확정하지 못함",
    "실데이터 회고 평가이며 운영 배포 승인 또는 미래 성능 보장이 아님",
]


def breakdown(predictions):
    valid = predictions.dropna(subset=["y", "low", "high"])
    by_year = [
        {"year": int(year), "method": method, **metrics(group)}
        for (year, method), group in valid.groupby([valid.target_date.dt.year, "method"])
    ]
    by_symbol = [
        {"symbol": symbol, "method": method, **metrics(group)}
        for (symbol, method), group in valid.groupby(["symbol", "method"])
    ]
    # A predeclared realized stress subset is diagnostic, never a forecast input.
    regimes = []
    for label, subset in (
        ("realized_range_ge_5pct", valid[valid.y.ge(5)]),
        ("covid_2020_02_20_to_03_23", valid[valid.target_date.between("2020-02-20", "2020-03-23")]),
    ):
        for method, group in subset.groupby("method"):
            regimes.append({"regime": label, "method": method, **metrics(group)})
    return {"by_year": by_year, "by_symbol": by_symbol, "regimes": regimes}


def run_development(settings):
    lake = Lake(settings.data_dir)
    quality = json.loads((lake.root / "reports" / "quality.json").read_text(encoding="utf8"))
    if quality.get("download_checkpoints_missing"):
        raise ValueError("Complete the official history download first")
    frame = retrospective_frame(lake)
    output = lake.root / "runs" / f"market-audit-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    reports = []
    for year in (2020, 2022, 2023):
        print(f"Real-market development fold {year}: fitting", flush=True)
        model, prediction, report = evaluate_window(
            frame, f"{year - 2}-12-31", f"{year - 1}-01-01", f"{year}-01-01", f"{year}-12-31"
        )
        prediction.to_parquet(output / f"{year}.parquet", index=False)
        eligible_predictions = prediction[prediction.symbol.isin(report["eligible_symbols"])]
        reports.append(
            {"year": year, **report, **breakdown(eligible_predictions), "training": model.fit_summary}
        )
        print(json.dumps({"year": year, **report}, ensure_ascii=False), flush=True)
        atomic_json(output / "progress.json", {"completed_folds": reports})
    result = {
        "kind": "real_market_retrospective_development",
        "demo": False,
        "created_at": utcnow(),
        "limitations": LIMITATIONS,
        "policy": POLICY,
        "folds": reports,
        "source_hash": source_hash(),
        "data_hash": data_fingerprint(lake),
        "artifact_dir": str(output),
        "deployment_eligible": False,
    }
    atomic_json(output / "development.json", result)
    atomic_json(lake.root / "reports" / "market-development.json", result)
    return result


def run_holdout(settings):
    lake = Lake(settings.data_dir)
    development = json.loads((lake.root / "reports" / "market-development.json").read_text(encoding="utf8"))
    if development["source_hash"] != source_hash() or development["data_hash"] != data_fingerprint(lake):
        raise ValueError("Code/data changed after development; rerun development before sealing")
    manifest = {
        "policy": POLICY,
        "data_hash": data_fingerprint(lake),
        "source_hash": source_hash(),
        "limitations": LIMITATIONS,
        "frozen_at": utcnow(),
        "kind": "retrospective_audit",
        "development_hash": digest(development),
        "stress_range_threshold_pct": 5.0,
    }
    manifest["study_id"] = digest(manifest)[:16]
    # The SAME dataset-level seal as the production workflow: an audited holdout
    # cannot subsequently be reused as an unseen final test.
    marker = lake.root / "config" / "test-consumed.json"
    with marker.open("x", encoding="utf8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    atomic_json(lake.root / "config" / "audit-study.json", manifest)
    output = lake.root / "runs" / f"market-test-{manifest['study_id']}"
    output.mkdir(parents=True)
    print("Real-market frozen holdout 2025-01-01..2026-09-30: fitting", flush=True)
    model, prediction, statistical = evaluate_window(
        retrospective_frame(lake), "2023-12-31", "2024-01-01", POLICY["test_start"], POLICY["test_end"]
    )
    prediction.to_parquet(output / "predictions.parquet", index=False)
    joblib.dump(model, output / "models.joblib")
    result = {
        **statistical,
        **breakdown(prediction[prediction.symbol.isin(statistical["eligible_symbols"])]),
        **manifest,
        "demo": False,
        "statistical_selection": statistical["selected"],
        "selected": "observation_only",
        "deployment_eligible": False,
        "data_gate": data_gate(lake),
        "created_at": utcnow(),
        "test_start": POLICY["test_start"],
        "test_end": POLICY["test_end"],
        "artifact_dir": str(output),
        "model_hash": digest((output / "models.joblib").read_bytes()),
        "har_fitted_symbols": len(model.har),
        "ml_training_symbols": len(model.symbols),
        "training": model.fit_summary,
    }
    atomic_json(output / "report.json", result)
    atomic_json(lake.root / "reports" / "test.json", result)
    # No release pointer: deployment remains fail-closed.
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"by_symbol", "by_year"}}, ensure_ascii=False
        ),
        flush=True,
    )
    return result


def portfolio_audit(settings):
    lake = Lake(settings.data_dir)
    bars = lake.bars()
    dates = pd.to_datetime([s["date"] for s in lake.calendar()])
    # Fixed reference compositions, not selected for passing p-values.
    portfolios = {
        "samsung_10m": ({"005930": 10_000_000}, 0),
        "three_stocks_equal_30m": ({"005930": 10_000_000, "000660": 10_000_000, "005380": 10_000_000}, 0),
        "three_stocks_plus_cash": (
            {"005930": 10_000_000, "000660": 10_000_000, "005380": 10_000_000},
            10_000_000,
        ),
        "cash_only_10m": ({}, 10_000_000),
    }
    results = []
    for name, (holdings, cash) in portfolios.items():
        selected = bars[bars.symbol.isin(holdings)]
        prices = selected.pivot(index="date", columns="symbol", values="candidate_adjusted_close").reindex(
            dates
        )
        returns = prices.pct_change(fill_method=None)
        report = historical_var(returns, holdings, cash, as_of="2026-09-30")
        stress_bars = bars.copy()
        stress_bars["adjusted_close"] = stress_bars.candidate_adjusted_close
        results.append(
            {
                "name": name,
                "holdings": holdings,
                "cash": cash,
                **report,
                "stress": stress_tests(stress_bars, holdings, cash),
            }
        )
    result = {
        "kind": "retrospective_portfolio_audit",
        "created_at": utcnow(),
        "limitations": LIMITATIONS,
        "deployment_eligible": False,
        "portfolios": results,
    }
    atomic_json(lake.root / "reports" / "portfolio-audit.json", result)
    return result


if __name__ == "__main__":
    import argparse

    from .config import Settings

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["development", "holdout", "portfolio"])
    stage = parser.parse_args().stage
    {"development": run_development, "holdout": run_holdout, "portfolio": portfolio_audit}[stage](
        Settings.load().prepare()
    )
