"""Isolated, free-source acquisition and probability ablations. Never releases models."""
import argparse
import json
import shutil
import time
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .dart_evidence import api_key, download, xml, zip_members
from .direction import COLS
from .direction_calibration_study import fit_calibration, calibrated
from .direction_historical_reliability import compare, draws
from .event_direction import freeze
from .new_information_sources import Reader, ALLOW
from .platform_study import local_settings
from .probability_quality import evaluate
from .util import atomic_json, digest, utcnow

RUN = "free-information-probability-20261017"
CORE = ["005930", "000660", "005380"]
FLOW = [f"flow_{k}_{n}" for k in ["foreign", "institution", "individual"] for n in [1, 5, 20]]
EVENTS = {"earnings": "잠정|실적", "periodic": "사업보고서|분기보고서|반기보고서", "capital": "증자|전환사채|자기주식", "correction": "정정", "contract": "단일판매|공급계약"}
DART = [f"event_{k}_{n}" for k in EVENTS for n in [1, 5]]


def setup():
    root = local_settings().data_dir
    out = root / "runs" / RUN
    out.mkdir(exist_ok=True)
    protocol = out / "protocol.json"
    if not protocol.exists():
        bars = pd.read_parquet(root / "silver/bars.parquet")
        ranking = bars[bars.date.between("2021-01-01", "2021-12-31")].groupby("symbol").turnover.mean().sort_values(ascending=False)
        eligible = set(pd.read_parquet(root / "runs/direction-calibration-study-20261017/features.parquet", columns=["symbol"]).symbol)
        symbols = list(dict.fromkeys(CORE + [s for s in ranking.index if s in eligible]))[:30]
        protected = {str(p): digest(p.read_bytes()) for folder in ["config", "direction-forward", "forward"] for p in (root / folder).rglob("*") if p.is_file()}
        # Immutable registered model files as well as release configuration.
        for p in root.rglob("*.joblib"):
            if "free-information" not in str(p):
                protected[str(p)] = digest(p.read_bytes())
        freeze(protocol, dict(symbols=symbols, universe_rule="three core + top 2021 turnover; current surviving universe limitation", train="through 2021", calibration="2022", selection="2023 reused", evaluation="2024 reused retrospective, NOT independent", cutoff="regular session close; external receipt/flow data strictly before feature day", models=["initial", "flow", "dart", "flow_dart"], calibration_methods=["sigmoid", "beta", "isotonic"], selection_rule="minimum 2023 date-balanced Brier, log loss tie break; no final-period selection", protected=protected, frozen_at=utcnow(), code_hash=digest(Path(__file__).read_bytes())))
    p = json.loads(protocol.read_text("utf8"))
    return root, out, p.get("payload", p)


def number(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def collect(shard=None):
    root, out, protocol = setup()
    shutil.copyfile(root / "runs/direction-new-information-20261017/kiwoom-spec.json", out / "kiwoom-spec.json")
    reader = Reader(out)
    ALLOW.add("ka10081")  # Local process only; documented read-only daily endpoint.
    summary = []
    symbols = protocol["symbols"] if shard is None else protocol["symbols"][shard::2]
    for symbol in symbols:
        dest = out / f"flow-{symbol}.parquet"
        if dest.exists():
            continue
        cursor, rows, reason = "20241230", [], "page_budget"
        try:
            for page in range(100):
                raw = reader.get("ka10059", dict(dt=cursor, stk_cd=symbol, amt_qty_tp="1", trde_tp="0", unit_tp="1"))
                batch = [r for r in raw["response"].get("stk_invsr_orgn", []) if len(r.get("dt", "")) == 8]
                if not batch:
                    reason = "provider_end"
                    break
                first, last = min(r["dt"] for r in batch), max(r["dt"] for r in batch)
                if last > cursor:
                    raise ValueError("historical_date_ignored")
                rows.extend(batch)
                if first <= "20000101":
                    reason = "requested_start"
                    break
                new = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y%m%d")
                if new >= cursor:
                    raise ValueError("nonadvancing_date")
                cursor = new
            if rows:
                frame = pd.DataFrame(rows).drop_duplicates("dt").sort_values("dt")
                frame["symbol"] = symbol
                frame.to_parquet(dest, index=False)
                meta = dict(symbol=symbol, rows=len(frame), earliest=frame.dt.min(), latest=frame.dt.max(), stop_reason=reason, pages=page+1, hash=digest(dest.read_bytes()), timing="previous-date only; historical revisions unverified")
                atomic_json(out / f"flow-{symbol}-manifest.json", meta)
                print(json.dumps(meta), flush=True)
        except Exception as exc:
            summary.append(dict(source="flow", symbol=symbol, error=type(exc).__name__))
    reader.client.close()
    atomic_json(out / f"flow-errors-{shard}.json", summary)


def extras():
    root, out, protocol = setup()
    reader = Reader(out)
    ALLOW.add("ka10081")
    for symbol in CORE:
        for scope in ["1", "5", "daily"]:
            dest = out / f"bars-{symbol}-{scope}.parquet"
            if dest.exists():
                continue
            code = "ka10081" if scope == "daily" else "ka10080"
            field = "stk_dt_pole_chart_qry" if scope == "daily" else "stk_min_pole_chart_qry"
            datefield = "dt" if scope == "daily" else "cntr_tm"
            body = dict(stk_cd=symbol, base_dt="20241230", upd_stkpc_tp="1")
            if scope != "daily":
                body["tic_scope"] = scope
            rows, seen, cursor, cont, reason = [], set(), "", "N", "page_budget"
            try:
                for page in range(35):
                    raw = reader.get(code, body, cont, cursor)
                    batch = [r for r in raw["response"].get(field, []) if len(r.get(datefield, "")) >= 8]
                    rows.extend(batch)
                    if not batch or raw.get("continuation") != "Y":
                        reason = "provider_end"
                        break
                    if min(r[datefield][:8] for r in batch) <= "20000101":
                        reason = "requested_start"
                        break
                    cursor = raw.get("next_key", "")
                    if not cursor or cursor in seen:
                        reason = "nonadvancing_cursor"
                        break
                    seen.add(cursor)
                    cont = "Y"
                frame = pd.DataFrame(rows)
                frame.to_parquet(dest, index=False)
                meta = dict(symbol=symbol, scope=scope, rows=len(frame), earliest=min((r[datefield] for r in rows), default=None), latest=max((r[datefield] for r in rows), default=None), stop_reason=reason, request_cutoff="20241230", historical_request_honored=all(r[datefield][:8] <= "20241230" for r in rows), pages=page+1)
                atomic_json(out / f"bars-{symbol}-{scope}-manifest.json", meta)
                print(json.dumps(meta), flush=True)
            except Exception as exc:
                atomic_json(out / f"bars-{symbol}-{scope}-error.json", dict(error=type(exc).__name__))
    reader.client.close()
    key = api_key()
    if not key:
        atomic_json(out / "dart-error.json", dict(error="missing_key"))
        return
    cache = out / "dart-raw"
    cache.mkdir(exist_ok=True)
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        data, _ = download(client, cache, "corpCode.xml", {}, key, 86400, 10000000)
        companies = {}
        for _, raw in zip_members(data, 100000000):
            for r in xml(raw).findall("list"):
                companies[(r.findtext("stock_code") or "").strip()] = r.findtext("corp_code")
        for symbol in protocol["symbols"]:
            dest = out / f"dart-{symbol}.parquet"
            if dest.exists():
                continue
            rows = []
            try:
                for page in range(1, 301):
                    params = dict(corp_code=companies[symbol], bgn_de="20000101", end_de="20241231", last_reprt_at="N", page_count="100", page_no=str(page), sort="date", sort_mth="asc")
                    raw, _ = download(client, cache, "list.json", params, key, 31536000, 5000000)
                    data = json.loads(raw)
                    if data.get("status") not in ["000", "013"]:
                        raise ValueError("dart_rejected")
                    rows.extend(data.get("list", []))
                    if page >= int(data.get("total_page", 0)):
                        break
                    time.sleep(.25)
                else:
                    raise ValueError("dart_page_budget")
                frame = pd.DataFrame(rows)
                if len(frame):
                    frame = frame.drop_duplicates("rcept_no")
                frame.to_parquet(dest, index=False)
                print(json.dumps(dict(source="dart", symbol=symbol, rows=len(frame))), flush=True)
            except Exception as exc:
                atomic_json(out / f"dart-{symbol}-error.json", dict(error=type(exc).__name__))


def lag_join(left, source, cols):
    """Strictly prior observation day, maximum seven calendar days old."""
    left = left.sort_values("date")
    source = source.sort_values("observed_date")
    joined = pd.merge_asof(left, source[["observed_date"] + cols], left_on="date", right_on="observed_date", allow_exact_matches=False, direction="backward", tolerance=pd.Timedelta(days=7))
    if (joined.observed_date >= joined.date).any():
        raise ValueError("future feature")
    return joined.drop(columns="observed_date")


def build():
    root, out, protocol = setup()
    base = pd.read_parquet(root / "runs/direction-calibration-study-20261017/features.parquet")
    frames = []
    for symbol in protocol["symbols"]:
        path = out / f"flow-{symbol}.parquet"
        dart = out / f"dart-{symbol}.parquet"
        if not path.exists() or not dart.exists():
            continue
        f = base[base.symbol.eq(symbol)].copy()
        flow = pd.read_parquet(path).sort_values("dt")
        flow["observed_date"] = pd.to_datetime(flow.dt)
        active = pd.concat([number(flow[c]) for c in ["frgnr_invsr", "orgn", "ind_invsr"]], axis=1).abs().sum(axis=1).gt(0)
        # Provider returns long leading all-zero history for unavailable investor fields.
        flow = flow.loc[active.cummax()].copy()
        scale = number(flow.acc_trde_prica).where(lambda v: v > 0)
        for k, raw in [("foreign", "frgnr_invsr"), ("institution", "orgn"), ("individual", "ind_invsr")]:
            value = number(flow[raw]) / scale
            for n in [1, 5, 20]:
                flow[f"flow_{k}_{n}"] = value.rolling(n, min_periods=n).mean()
        f = lag_join(f, flow, FLOW)
        events = pd.read_parquet(dart)
        index = pd.date_range("2000-01-01", "2024-12-31")
        s = pd.DataFrame({"observed_date": index})
        for k, pattern in EVENTS.items():
            counts = pd.to_datetime(events.loc[events.report_nm.str.contains(pattern, na=False), "rcept_dt"]).value_counts().reindex(index, fill_value=0)
            for n in [1, 5]:
                s[f"event_{k}_{n}"] = counts.rolling(n, min_periods=1).sum().to_numpy()
        f = lag_join(f, s, DART)
        frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    # Same complete rows for baseline and every ablation; no missing-source-as-zero bias.
    f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=COLS + FLOW + DART)
    f.to_parquet(out / "features.parquet", index=False)
    atomic_json(out / "features-manifest.json", dict(rows=len(f), symbols=f.symbol.nunique(), first=str(f.date.min()), last=str(f.date.max()), hash=digest((out / "features.parquet").read_bytes()), timing="external information strictly previous calendar date; current-vintage flows remain exploratory; DART counts not consensus surprise"))
    return f


def train():
    root, out, protocol = setup()
    missing = [s for s in protocol["symbols"] if not (out / f"flow-{s}.parquet").exists() or not (out / f"dart-{s}.parquet").exists()]
    if missing:
        raise ValueError("Collection incomplete: " + ",".join(missing))
    f = build()
    train = f[f.target_date.lt("2022-01-01")]
    cal = f[f.target_date.dt.year.eq(2022)]
    dev = f[f.target_date.dt.year.eq(2023)]
    test = f[f.target_date.dt.year.eq(2024)]
    if min(map(len, [train, cal, dev, test])) < 500:
        raise ValueError("Insufficient split")
    prior = float(train.label.mean())
    atomic_json(out / "splits.json", {k: dict(rows=len(v), dates=v.target_date.nunique(), start=str(v.target_date.min()), end=str(v.target_date.max())) for k, v in [("train",train),("calibration",cal),("selection",dev),("evaluation",test)]})
    methods = protocol["calibration_methods"]
    objects, devresults = {}, {}
    with threadpool_limits(limits=3):
        for name, cols in {"initial": COLS, "flow": COLS+FLOW, "dart": COLS+DART, "flow_dart": COLS+FLOW+DART}.items():
            print("FIT", name, len(train), flush=True)
            model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=.1, max_iter=700, random_state=17))
            model.fit(train[cols], train.label)
            cp, dp = model.predict_proba(cal[cols])[:,1], model.predict_proba(dev[cols])[:,1]
            for method in methods:
                calibrator = fit_calibration(cal, cp, method)
                p = calibrated(calibrator, dev, dp, method)
                key = name + "__" + method
                objects[key] = (model, calibrator, cols, method)
                devresults[key] = evaluate(dev, p, prior)
                print("DEV", key, devresults[key]["date_balanced"]["brier"], flush=True)
    chosen = min(devresults, key=lambda k: (devresults[k]["date_balanced"]["brier"], devresults[k]["date_balanced"]["log_loss"]))
    selection_path = out / "selection.json"
    stamp = json.loads(selection_path.read_text("utf8"))["payload"]["frozen_at"] if selection_path.exists() else utcnow()
    freeze(selection_path, dict(selected=chosen, development=devresults, frozen_at=stamp))
    predictions = test[["symbol", "date", "target_date", "label"]].copy()
    results = {}
    for key, (model, calibrator, cols, method) in objects.items():
        with threadpool_limits(limits=3):
            p = calibrated(calibrator, test, model.predict_proba(test[cols])[:,1], method)
        predictions[key] = p
        results[key] = evaluate(test, p, prior)
        joblib.dump(dict(model=model, calibrator=calibrator, columns=cols, method=method), out / (key + ".joblib"))
    for label, filename in [("frozen_initial", "final-initial_refit__sigmoid.parquet"), ("frozen_recent", "final-technical_recent__market_sigmoid.parquet")]:
        saved = pd.read_parquet(root / "runs/direction-calibration-study-20261017" / filename)
        matched = test[["symbol", "target_date", "label"]].merge(saved[["symbol", "target_date", "label", "p_up"]], on=["symbol", "target_date"], how="left", validate="one_to_one", suffixes=("", "_saved"))
        if matched.p_up.isna().any() or not matched.label.eq(matched.label_saved).all():
            raise ValueError("Archived baseline does not match evaluation labels")
        predictions[label] = matched.p_up.to_numpy()
        results[label] = evaluate(test, matched.p_up.to_numpy(), prior)
    dates = sorted(test.target_date.unique())
    indices = draws(dates)
    reference = "initial__sigmoid"
    pairs = {k: compare(test, predictions[reference].to_numpy(), predictions[k].to_numpy(), dates, indices) for k in results if k != reference}
    from .probability_quality import losses
    rb, rl = losses(test.label.to_numpy(), predictions[reference].to_numpy())
    for key in pairs:
        b, l = losses(test.label.to_numpy(), predictions[key].to_numpy())
        daily = pd.DataFrame(dict(date=test.target_date.to_numpy(), brier=b-rb, log_loss=l-rl)).groupby("date").mean().reindex(dates)
        sampled = daily.to_numpy()[indices].mean(axis=1)
        pairs[key]["date_balanced"] = {c: dict(delta=float(daily[c].mean()), ci95=np.quantile(sampled[:,i], [.025,.975]).tolist()) for i,c in enumerate(daily.columns)}
    predictions.to_parquet(out / "predictions.parquet", index=False)
    unchanged = all(Path(p).exists() and digest(Path(p).read_bytes()) == h for p,h in protocol["protected"].items())
    report = dict(created_at=utcnow(), selected=chosen, reference=reference, results=results, comparison_new_minus_initial=pairs, production_unchanged=unchanged, independent_test=False, auto_promotion=False, source_hash=digest(Path(__file__).read_bytes()), symbols=int(f.symbol.nunique()), prior_up=prior, probability_contract="up/down conditional on non-flat next close; flat observations excluded equally", limitations=["2023/2024 reused retrospective", "flow historical revisions unverified; one-day delay does not prove original vintage", "30 current survivors selected on 2021 turnover", "DART event counts, no consensus surprise"], artifact_dir=str(out))
    atomic_json(out / "report.json", report)
    atomic_json(root / "reports/free-information-probability.json", report)
    print("DONE", chosen, "protected_unchanged", unchanged, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["collect", "extras", "train"])
    parser.add_argument("--shard", type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.stage == "collect":
        collect(args.shard)
    else:
        {"extras": extras, "train": train}[args.stage]()
