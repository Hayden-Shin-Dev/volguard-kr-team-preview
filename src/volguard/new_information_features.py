"""Availability joins for new external information, without changing price features."""

import json

import numpy as np
import pandas as pd

from .new_information_sources import CORE, output
from .platform_study import local_settings
from .util import atomic_json, digest, utcnow


def number(values):
    return pd.to_numeric(values.astype(str).str.replace(",", "", regex=False), errors="coerce")


def available(dates, business_lag=0):
    dates = pd.to_datetime(dates)
    # Dates without verified publication hours are never used on their observation day.
    result = dates + (pd.offsets.BDay(business_lag) if business_lag else pd.Timedelta(days=1))
    if business_lag:
        result = result + pd.Timedelta(hours=23, minutes=59)
    return result.dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")


def align(data, source, cols, name, max_age_days=10):
    """Only backward joins on explicit release availability, with stale-value rejection."""
    left = data[["date", "symbol"]].copy()
    left["_row"] = np.arange(len(left))
    left["issue_at"] = (
        (left.date + pd.Timedelta(hours=18, minutes=30)).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
    )
    source = source[
        cols + ["available_at", "observed_date"] + (["symbol"] if "symbol" in source else [])
    ].copy()
    source["available_at"] = pd.to_datetime(source.available_at, utc=True)
    source["observed_date"] = pd.to_datetime(source.observed_date)
    if source[cols].replace([np.inf, -np.inf], np.nan).isna().all().all():
        raise ValueError("No valid " + name + " observations")
    by = "symbol" if "symbol" in source else None
    source = source.drop_duplicates(([by] if by else []) + ["available_at"], keep="last")
    merged = pd.merge_asof(
        left.sort_values("issue_at"),
        source.sort_values("available_at"),
        left_on="issue_at",
        right_on="available_at",
        by=by,
        direction="backward",
    )
    future = merged.available_at.notna() & (merged.available_at > merged.issue_at)
    if future.any():
        raise ValueError("Future release joined")
    if (merged.observed_date > merged.date).any():
        raise ValueError("Future observation with invalid availability metadata")
    stale = (merged.date - merged.observed_date).dt.days > max_age_days
    merged.loc[stale, cols] = np.nan
    merged = merged.sort_values("_row")
    for col in cols:
        data[col] = merged[col].to_numpy()
    data[name + "__available_at"] = merged.available_at.to_numpy()
    data[name + "__observed_date"] = merged.observed_date.to_numpy()
    return dict(
        rows=len(source),
        first=str(source.observed_date.min()),
        last=str(source.observed_date.max()),
        complete_fraction=float(data[cols].notna().all(axis=1).mean()),
        stale_rows=int(stale.sum()),
        max_age_days=max_age_days,
    )


def build(stage="global"):
    out = output()
    root = local_settings().data_dir
    data = pd.read_parquet(out.parent / "direction-math-20261017/features.parquet")
    # Identical prior OOF-eligible universe. No new price transformation or target definition.
    data = data[data.range_relative.notna()].sort_values(["date", "symbol"]).reset_index(drop=True)
    if data.target_date.max() >= pd.Timestamp("2024-01-01"):
        raise ValueError("Protected period")
    if stage == "core":
        data = data[data.symbol.isin(CORE)].reset_index(drop=True)
    groups = {}
    quality = {}
    exclusions = {}

    def add(name, source, cols, ttl=10):
        groups[name] = cols
        quality[name] = align(data, source, cols, name, ttl)

    if stage == "global":
        if (out / "alfred.parquet").exists():
            s = pd.read_parquet(out / "alfred.parquet")
            s = s[s.series.eq("NASDAQCOM")].copy()
            s["nasdaq_vintage_return"] = s.value / s.previous - 1
            s["observed_date"] = pd.to_datetime(s.observation_date)
            add("nasdaq_vintage", s, ["nasdaq_vintage_return"], 14)
        for name, columns, ttl in [
            ("overseas_latest_delayed", ["NASDAQCOM", "VIXCLS", "DGS10", "DGS2"], 21),
            ("fx_rates_latest_delayed", ["usdkrw", "jpykrw", "kr3y", "kr10y", "creditAA"], 21),
        ]:
            cols = []
            for code in columns:
                if name.startswith("overseas"):
                    s = pd.read_csv(out / f"fred-{code}.csv")
                    s["date"] = pd.to_datetime(s.observation_date)
                    s["value"] = number(s[code])
                else:
                    s = pd.read_parquet(out / f"ecos-{code}.parquet")
                    s["date"] = pd.to_datetime(s.TIME)
                    s["value"] = number(s.DATA_VALUE)
                s = s.dropna(subset=["value"]).sort_values("date").drop_duplicates("date")
                # Simple changes of genuinely new observations, not extra transforms of KRX bars.
                change = code in ["NASDAQCOM", "VIXCLS", "usdkrw", "jpykrw"]
                s["x_" + code] = s.value.pct_change(fill_method=None) if change else s.value.diff()
                if code == "VIXCLS":
                    s["x_vix_level"] = s.value
                localcols = ["x_" + code] + (["x_vix_level"] if code == "VIXCLS" else [])
                s["available_at"] = available(s.date, 5)
                s["observed_date"] = s.date
                # Join each release separately: holidays never force equal observation dates.
                q = align(data, s, localcols, name + "_" + code, ttl)
                quality[name + "_" + code] = q
                cols += localcols
            groups[name] = cols
            quality[name] = dict(
                complete_fraction=float(data[cols].notna().all(axis=1).mean()),
                timing="Current vintage, five-business-day delayed exploratory; not strict PIT",
            )
        s = pd.read_parquet(out / "program.parquet")
        s["date"] = pd.to_datetime(s.cntr_tm.str[:8])
        s = s.sort_values("date").drop_duplicates("date")
        cols = []
        for prefix, net, buy, sell in [
            ("program", "all_netprps", "all_buy", "all_sel"),
            ("arbitrage", "dfrt_trde_netprps", "dfrt_trde_buy", "dfrt_trde_sel"),
            ("nonarbitrage", "ndiffpro_trde_netprps", "ndiffpro_trde_buy", "ndiffpro_trde_sel"),
        ]:
            denom = number(s[buy]) + number(s[sell])
            s[prefix + "_pressure"] = number(s[net]) / denom.replace(0, np.nan)
            s[prefix + "_pressure5"] = s[prefix + "_pressure"].rolling(5, min_periods=5).mean()
            cols += [prefix + "_pressure", prefix + "_pressure5"]
        s["program_basis"] = number(s.basis) / number(s.kospi200).abs().replace(0, np.nan)
        cols += ["program_basis"]
        s["available_at"] = available(s.date)
        s["observed_date"] = s.date
        add("market_program", s, cols)
        if (out / "sector.parquet").exists():
            source = pd.read_parquet(out / "sector.parquet")
            source["date"] = pd.to_datetime(source.date)
            for c in ["ind_netprps", "frgnr_netprps", "orgn_netprps"]:
                source[c] = number(source[c])
            denom = (
                source[["ind_netprps", "frgnr_netprps", "orgn_netprps"]].abs().sum(axis=1).replace(0, np.nan)
            )
            source["foreign_pressure"] = source.frgnr_netprps / denom
            source["institution_pressure"] = source.orgn_netprps / denom
            market = source[source.inds_cd.eq("001")].sort_values("date").copy()
            cols = []
            for prefix in ["foreign", "institution"]:
                c = prefix + "_pressure"
                market[c + "5"] = market[c].rolling(5, min_periods=5).mean()
                cols += [c, c + "5"]
            market["available_at"] = available(market.date)
            market["observed_date"] = market.date
            add("market_investor", market, cols)
            # Common cross-sector pressure only; no invented historical stock membership.
            sectors = source[~source.inds_cd.isin(["001", "002", "003", "004"])].copy()
            sectors = sectors[sectors[["foreign_pressure", "institution_pressure"]].notna().all(axis=1)]
            aggregates = (
                sectors.groupby("date")
                .agg(
                    sector_foreign_mean=("foreign_pressure", "mean"),
                    sector_foreign_dispersion=("foreign_pressure", "std"),
                    sector_institution_mean=("institution_pressure", "mean"),
                    sector_foreign_breadth=("foreign_pressure", lambda v: (v > 0).mean()),
                )
                .reset_index()
            )
            cols = [c for c in aggregates if c != "date"]
            aggregates["available_at"] = available(aggregates.date)
            aggregates["observed_date"] = aggregates.date
            add("sector_investor", aggregates, cols)
    else:
        s = pd.read_parquet(out / "strength.parquet")
        s["date"] = pd.to_datetime(s.dt)
        s = s[s.date < "2024-01-01"].sort_values(["symbol", "date"])
        strength = number(s.cntr_str)
        s["execution_pressure"] = np.log(strength.where(strength > 0) / 100).clip(-4, 4)
        s["execution_pressure5"] = s.groupby("symbol").execution_pressure.transform(
            lambda x: x.rolling(5, min_periods=5).mean()
        )
        s["available_at"] = available(s.date)
        s["observed_date"] = s.date
        add("execution_strength_core", s, ["execution_pressure", "execution_pressure5"])
        # Event categories use receipt dates, never fiscal-period end dates or future announcements.
        parts = []
        types = {
            "earnings_event": "잠정|실적",
            "periodic_event": "사업보고서|분기보고서|반기보고서",
            "capital_event": "증자|전환사채|자기주식",
            "correction_event": "정정",
            "contract_event": "단일판매|공급계약",
        }
        for symbol in CORE:
            raw = pd.read_parquet(root / f"research-inputs/direction-external/dart-{symbol}.parquet")
            index = pd.date_range("2014-01-01", "2023-12-31")
            s = pd.DataFrame({"date": index, "symbol": symbol})
            for name, pattern in types.items():
                count = (
                    raw[raw.report_nm.str.contains(pattern)]
                    .groupby("date")
                    .size()
                    .reindex(index, fill_value=0)
                )
                s[name + "1"] = count.to_numpy()
                s[name + "5"] = count.rolling(5, min_periods=1).sum().to_numpy()
            s["available_at"] = available(s.date)
            s["observed_date"] = s.date
            parts.append(s)
        cols = [k + suffix for k in types for suffix in ["1", "5"]]
        add("dart_events_core", pd.concat(parts), cols, 5)
        raw = pd.read_parquet(out / "earnings.parquet")
        records = []
        rejected = []
        for (symbol, receipt), g in raw.groupby(["symbol", "rcept_no"]):
            rec = dict(symbol=symbol, receipt=receipt, observed_date=pd.Timestamp(g.publication_date.iloc[0]))
            for name, ids in [
                ("revenue", ["ifrs_Revenue", "ifrs-full_Revenue"]),
                ("operating", ["dart_OperatingIncomeLoss"]),
            ]:
                choices = g[g.account_id.isin(ids) & g.currency.eq("KRW") & g.sj_div.isin(["IS", "CIS"])]
                vals = choices.drop_duplicates(["thstrm_amount", "frmtrm_amount", "frmtrm_q_amount"])
                if len(vals) != 1:
                    rejected.append(
                        dict(symbol=symbol, receipt=receipt, account=name, reason="missing_or_ambiguous")
                    )
                    continue
                row = vals.iloc[0]
                cur = pd.to_numeric(str(row.thstrm_amount).replace(",", ""), errors="coerce")
                prior = row.frmtrm_amount if row.reprt_code == "11011" else row.frmtrm_q_amount
                prev = pd.to_numeric(str(prior).replace(",", ""), errors="coerce")
                rec[name + "_current"] = cur
                rec[name + "_growth"] = (
                    (cur - prev) / max(abs(prev), 1) if pd.notna(prev) and prev != 0 else np.nan
                )
            if "revenue_current" not in rec or "operating_current" not in rec:
                continue
            rec["operating_margin"] = (
                rec["operating_current"] / rec["revenue_current"] if rec["revenue_current"] > 0 else np.nan
            )
            rec["annual_report"] = float(g.reprt_code.iloc[0] == "11011")
            records.append(rec)
        s = pd.DataFrame(records).sort_values(["observed_date", "receipt"])
        s["available_at"] = available(s.observed_date)
        cols = ["revenue_growth", "operating_growth", "operating_margin", "annual_report"]
        s[cols] = s[cols].replace([np.inf, -np.inf], np.nan).clip(-10, 10)
        add("earnings_core", s, cols, 150)
        atomic_json(
            out / "earnings-feature-audit.json",
            dict(
                rejected=rejected,
                receipts=len(s),
                note="Growth is reported accounting growth, not analyst-consensus earnings surprise; current API vintage unverified",
            ),
        )
        # Prior stock flow as a control: this source had already been tested, not claimed as new.
        parts = []
        for symbol in CORE:
            s = pd.read_parquet(
                root / f"research-inputs/direction-external/flow-{symbol}.parquet"
            ).sort_values("date")
            for prefix, field in [("stock_foreign", "frgnr_invsr"), ("stock_institution", "orgn")]:
                s[prefix] = number(s[field]) / number(s.acc_trde_prica).replace(0, np.nan)
            s["available_at"] = available(s.date)
            s["observed_date"] = s.date
            parts.append(s)
        add("previous_stock_flow_core", pd.concat(parts), ["stock_foreign", "stock_institution"])
    path = out / f"{stage}-features.parquet"
    data.to_parquet(path, index=False)
    manifest = dict(
        stage=stage,
        groups=groups,
        quality=quality,
        exclusions=exclusions,
        rows=len(data),
        symbols=data.symbol.nunique(),
        hash=digest(path.read_bytes()),
        created_at=utcnow(),
    )
    atomic_json(out / f"{stage}-features.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return data, manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["global", "core"])
    build(parser.parse_args().stage)
