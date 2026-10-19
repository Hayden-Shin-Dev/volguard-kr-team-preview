"""Causal financial-statistics features; research only, no production mutations.

All rolling features are built before future-label eligibility is applied.
Market means refer to the fixed current universe, not point-in-time sectors.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import Lake, features
from .direction import COLS, extend
from .direction_oof import RANGE_COLS, check_temporal
from .direction_study import EXTRA, enhance
from .util import atomic_json, digest

BASE = COLS + EXTRA
GROUPS = {
    "persistence": ["acf1", "acf2", "pacf2", "sign_persistence", "variance_ratio5"],
    "trend": ["efficiency20", "trend_t60", "reversal_z20", "momentum60", "drawdown60"],
    "conditional_vol": ["garch_sigma", "egarch_sigma", "garch_shock", "leverage20"],
    "residual": ["beta60", "residual1", "residual5", "residual20", "correlation60"],
    "relations": ["neighbor_return", "neighbor_momentum", "factor_return", "factor_residual", "common_share"],
    "tails": ["skew60", "kurtosis60", "tail05", "downside_ratio", "extreme_frequency"],
    "liquidity": ["amihud20", "money_flow20", "signed_volume20", "volume_return_corr", "liquidity_change"],
    "nonlinear": ["sign_entropy", "permutation_entropy", "change_z", "vol_change_log", "markov_high"],
    "qlib_subset": [f"q_{f}_{w}" for w in [5, 20, 60] for f in ["roc", "ma", "std", "rank", "cntp", "sump"]],
}


def rolling_features(frame):
    parts = []
    for _, g in frame.groupby("symbol", sort=True):
        g = g.sort_values("date").copy()
        r, m = g.return_1, g.market_return
        sd = r.rolling(60, min_periods=40).std().replace(0, np.nan)
        g["acf1"] = r.rolling(60, min_periods=40).corr(r.shift(1))
        g["acf2"] = r.rolling(60, min_periods=40).corr(r.shift(2))
        g["pacf2"] = (g.acf2 - g.acf1**2) / (1 - g.acf1**2).replace(0, np.nan)
        sign = np.sign(r)
        g["sign_persistence"] = (
            sign.eq(sign.shift(1)).where(r.notna() & r.shift(1).notna()).rolling(60).mean()
        )
        g["variance_ratio5"] = r.rolling(5).sum().rolling(60).var() / (5 * sd**2)
        g["efficiency20"] = r.rolling(20).sum() / r.abs().rolling(20).sum().replace(0, np.nan)
        g["trend_t60"] = r.rolling(60).mean() / sd * np.sqrt(60)
        g["reversal_z20"] = (r - r.rolling(20).mean()) / r.rolling(20).std().replace(0, np.nan)
        log = np.log1p(r)
        # Reset a synthetic adjusted-price chain at missing/invalid daily returns.
        chain = np.exp(log.groupby(r.isna().cumsum()).cumsum())
        valid60 = r.notna().rolling(60).sum().eq(60)
        g["momentum60"] = np.expm1(log.rolling(60).sum())
        g["drawdown60"] = (chain / chain.rolling(60).max() - 1).where(valid60)
        g["leverage20"] = r.rolling(20).corr(r.abs().shift(-0))
        g["beta60"] = r.rolling(60, min_periods=40).cov(m) / m.rolling(60, min_periods=40).var().replace(
            0, np.nan
        )
        g["correlation60"] = r.rolling(60, min_periods=40).corr(m)
        g["residual1"] = r - g.beta60.shift(1) * m
        g["residual5"] = g.residual1.rolling(5).sum()
        g["residual20"] = g.residual1.rolling(20).sum()
        g["skew60"], g["kurtosis60"] = r.rolling(60).skew(), r.rolling(60).kurt()
        g["tail05"] = r.rolling(60).quantile(0.05) / sd
        g["downside_ratio"] = np.sqrt((r.clip(upper=0) ** 2).rolling(60).mean()) / sd
        g["extreme_frequency"] = r.abs().gt(2 * sd.shift(1)).where(sd.shift(1).notna()).rolling(60).mean()
        amount = g.turnover.where(g.turnover > 0)
        g["amihud20"] = np.log1p((r.abs() / amount * 1e9).rolling(20).mean())
        g["money_flow20"] = ((2 * g.close_location - 1) * amount).rolling(20).sum() / amount.rolling(20).sum()
        g["signed_volume20"] = (sign * g.volume).rolling(20).sum() / g.volume.rolling(20).sum().replace(
            0, np.nan
        )
        g["volume_return_corr"] = r.rolling(20).corr(np.log1p(g.volume).diff())
        g["liquidity_change"] = amount.rolling(5).mean() / amount.rolling(60).mean() - 1
        p = r.gt(0).where(r.notna()).rolling(60).mean().clip(1e-8, 1 - 1e-8)
        g["sign_entropy"] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        a, b, c = r.shift(2), r.shift(1), r
        code = (a > b).astype(int) * 4 + (a > c).astype(int) * 2 + (b > c).astype(int)
        entropy = pd.Series(0.0, index=g.index)
        for k in [0, 1, 3, 4, 6, 7]:
            freq = code.eq(k).where(a.notna() & b.notna() & c.notna()).rolling(60).mean()
            entropy -= freq * np.log(freq.clip(lower=1e-12)) / np.log(6)
        g["permutation_entropy"] = entropy
        g["change_z"] = (r.rolling(5).mean() - r.shift(5).rolling(55).mean()) / sd
        g["vol_change_log"] = np.log(r.rolling(10).std() / sd)
        for w in [5, 20, 60]:
            valid = r.notna().rolling(w + 1).sum().eq(w + 1)
            g[f"q_roc_{w}"] = (chain.shift(w) / chain).where(valid)
            g[f"q_ma_{w}"] = (chain.rolling(w).mean() / chain).where(valid)
            g[f"q_std_{w}"] = (chain.rolling(w).std() / chain).where(valid)
            g[f"q_rank_{w}"] = chain.rolling(w).rank(pct=True).where(valid)
            g[f"q_cntp_{w}"] = r.gt(0).where(r.notna()).rolling(w).mean()
            delta = chain.diff()
            g[f"q_sump_{w}"] = (delta.clip(lower=0).rolling(w).sum() / delta.abs().rolling(w).sum()).where(
                valid
            )
        for lag in range(10):
            g[f"lag_{lag}"] = r.shift(lag)
        parts.append(g)
    return pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def relation_features(frame):
    """Monthly frozen 60-day correlations; today's peers are known at issue time."""
    panel = frame.pivot(index="date", columns="symbol", values="return_1").sort_index()
    out = {k: np.full(panel.shape, np.nan) for k in GROUPS["relations"]}
    arr = panel.to_numpy()
    mom = panel.rolling(5).sum().to_numpy()
    previous = None
    for t, date in enumerate(panel.index):
        period = date.to_period("M")
        if t < 60:
            continue
        if previous != period:
            history = arr[t - 60 : t]
            valid = np.isfinite(history).all(axis=0) & (np.nanstd(history, axis=0) > 1e-8)
            ids = np.flatnonzero(valid)
            weights = np.zeros((len(ids), len(ids)))
            if len(ids) < 3:
                continue
            corr = np.corrcoef(history[:, ids].T)
            vals, vecs = np.linalg.eigh(corr)
            loading = vecs[:, -1]
            if loading.sum() < 0:
                loading = -loading
            sd = np.std(history[:, ids], axis=0)
            share = vals[-1] / vals.sum()
            np.fill_diagonal(corr, -np.inf)
            for j in range(len(ids)):
                nn = np.argsort(corr[j])[-min(5, len(ids) - 1) :]
                weights[j, nn] = np.maximum(corr[j, nn], 0)
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
            previous = period
        cur = arr[t, ids]
        good = np.isfinite(cur)
        denom = weights @ good
        out["neighbor_return"][t, ids] = np.divide(
            weights @ np.nan_to_num(cur), denom, out=np.full(len(ids), np.nan), where=denom > 0.5
        )
        out["neighbor_momentum"][t, ids] = weights @ np.nan_to_num(mom[t, ids])
        if good.all():
            factor = float(loading @ (cur / sd))
            out["factor_return"][t, ids] = factor
            out["factor_residual"][t, ids] = cur / sd - loading * factor
        out["common_share"][t, ids] = share
    for col, values in out.items():
        series = pd.DataFrame(values, index=panel.index, columns=panel.columns).stack(future_stack=True)
        frame[col] = series.reindex(pd.MultiIndex.from_frame(frame[["date", "symbol"]])).to_numpy()
    return frame


def volatility_features(frame):
    from arch import arch_model

    audit = []
    for symbol, g in frame.groupby("symbol"):
        g = g.sort_values("date")
        for year in range(2013, 2024):
            train = (100 * g.loc[g.date.lt(f"{year}-01-01"), "return_1"]).dropna().tail(1000)
            test = g[g.date.dt.year.eq(year)]
            if len(train) < 500 or test.empty:
                continue
            for kind in ["GARCH", "EGARCH"]:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = arch_model(
                            train.to_numpy(),
                            mean="Zero",
                            vol=kind,
                            p=1,
                            o=int(kind == "EGARCH"),
                            q=1,
                            rescale=False,
                        ).fit(disp="off", show_warning=False, options={"maxiter": 100})
                    if fit.convergence_flag:
                        raise ValueError("optimizer did not converge")
                    params = fit.params
                    h = float(fit.conditional_volatility[-1] ** 2)
                    last = float(train.iloc[-1])

                    def step(h, e):
                        if kind == "GARCH":
                            return max(
                                1e-8,
                                float(params["omega"] + params["alpha[1]"] * e * e + params["beta[1]"] * h),
                            )
                        z = e / np.sqrt(h)
                        return float(
                            np.exp(
                                np.clip(
                                    params["omega"]
                                    + params["alpha[1]"] * (abs(z) - np.sqrt(2 / np.pi))
                                    + params["gamma[1]"] * z
                                    + params["beta[1]"] * np.log(h),
                                    -15,
                                    15,
                                )
                            )
                        )

                    h = step(h, last)
                    vals = []
                    for r in test.return_1:
                        if np.isfinite(r):
                            h = step(h, 100 * r)
                            vals.append(np.sqrt(h) / 100)
                        else:
                            vals.append(np.nan)
                    frame.loc[test.index, kind.lower() + "_sigma"] = vals
                    audit.append(
                        dict(
                            symbol=symbol,
                            year=year,
                            kind=kind,
                            n=len(train),
                            train_end=str(
                                g[g.date.lt(f"{year}-01-01") & g.return_1.notna()].date.max().date()
                            ),
                            status="ok",
                            params=params.to_dict(),
                        )
                    )
                except (ValueError, RuntimeError, OverflowError) as e:
                    audit.append(
                        dict(symbol=symbol, year=year, kind=kind, status="failed", reason=str(e)[:120])
                    )
    frame["garch_shock"] = frame.return_1 / frame.garch_sigma.replace(0, np.nan)
    return frame, audit


def regime_features(frame):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    market = frame.groupby("date").market_return.first().dropna().sort_index() * 100
    audit = []
    for year in range(2013, 2024):
        train = market[market.index < pd.Timestamp(f"{year}-01-01")].tail(1000)
        total = market[
            (market.index >= train.index.min()) & (market.index < pd.Timestamp(f"{year + 1}-01-01"))
        ]
        if len(train) < 500:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = MarkovRegression(train.to_numpy(), k_regimes=2, trend="c", switching_variance=True).fit(
                    disp=False, maxiter=100
                )
                if not fit.mle_retvals["converged"]:
                    raise ValueError("nonconvergence")
                filt = MarkovRegression(
                    total.to_numpy(), k_regimes=2, trend="c", switching_variance=True
                ).filter(fit.params)
            high = int(np.argmax(fit.params[-2:]))
            vals = pd.Series(filt.filtered_marginal_probabilities[:, high], index=total.index)
            mask = frame.date.dt.year.eq(year)
            frame.loc[mask, "markov_high"] = frame.loc[mask, "date"].map(vals)
            audit.append(dict(year=year, train_end=str(train.index.max().date()), status="ok"))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            audit.append(dict(year=year, status="failed", reason=str(e)[:120]))
    return frame, audit


def build(root, out):
    bars = pd.read_parquet(root / "silver/bars.parquet", filters=[("date", "<", pd.Timestamp("2024-01-01"))])
    input_hash = digest(pd.util.hash_pandas_object(bars, index=False).values.tobytes())
    bars["available_at"] = (
        (bars.date + pd.Timedelta(hours=18)).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
    )
    frame = enhance(extend(features(bars, [s for s in Lake(root).calendar() if s["date"] < "2024-01-01"])))
    market = pd.read_parquet(
        root / "silver/market.parquet", filters=[("date", "<", pd.Timestamp("2024-01-01"))]
    )
    frame["market_range"] = frame.date.map(market.drop_duplicates("date").set_index("date").range_pct)
    frame = frame.merge(
        bars[["date", "symbol", "volume", "turnover"]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    print("math rolling features", flush=True)
    frame = rolling_features(frame)
    with threadpool_limits(limits=3):
        frame = relation_features(frame)
        print("annual GARCH / EGARCH fits", flush=True)
        frame, vol_audit = volatility_features(frame)
        print("annual Markov filtering", flush=True)
        frame, regime_audit = regime_features(frame)
    pointer = json.loads((root / "reports/direction-range-oof.json").read_text(encoding="utf8"))
    oof_path = Path(pointer["artifact_dir"]) / "oof.parquet"
    if not oof_path.exists():
        oof_path = Path(pointer["artifact_dir"]) / "predictions.parquet"
    oof = pd.read_parquet(oof_path)
    check_temporal(oof)
    frame = frame.merge(
        oof[["symbol", "date", "target_date"] + RANGE_COLS],
        on=["symbol", "date", "target_date"],
        how="left",
        validate="one_to_one",
    )
    frame["range_relative"] = frame.range_forecast_median / frame.market_range.replace(0, np.nan)
    # Sequence context is saved BEFORE filtering target zero/missing observations.
    frame.to_parquet(out / "context.parquet", index=False)
    eligible = (
        frame.direction_return.notna()
        & frame.direction_return.ne(0)
        & frame.y.notna()
        & frame[BASE].notna().all(axis=1)
    )
    frame = frame[eligible & frame.target_date.lt("2024-01-01") & frame.date.ge("2015-01-01")].copy()
    quality = dict(
        input_hash=input_hash,
        oof_hash=digest(oof_path.read_bytes()),
        rows=len(frame),
        symbols=frame.symbol.nunique(),
        flat_excluded=int((~eligible).sum()),
        missing=frame[sum(GROUPS.values(), [])].isna().mean().to_dict(),
        volatility=vol_audit,
        regime=regime_audit,
    )
    frame.to_parquet(out / "features.parquet", index=False)
    atomic_json(out / "feature-quality.json", quality)
    return frame, quality


if __name__ == "__main__":
    import argparse

    from .platform_study import local_settings
    from .util import utcnow

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    atomic_json(
        args.out / "pre-registration.json",
        dict(
            created_at=utcnow(),
            development="2021-2022",
            confirmation="2023 REUSED development, not independent",
            calibration="prior year rolling OOF",
            threshold_grid=[0.5, 0.55, 0.6, 0.65, 0.7],
            selection="development accuracy, Brier no worse than baseline; ties by Brier",
            threshold_selection="development accuracy with coverage >=5% and n>=1000; otherwise .5",
            production_mutations=False,
        ),
    )
    (args.out / "feature-source.py").write_bytes(Path(__file__).read_bytes())
    build(local_settings().data_dir, args.out)
