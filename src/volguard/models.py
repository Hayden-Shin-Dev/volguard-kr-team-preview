import heapq
import math
import warnings
from collections import deque

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMRegressor

from .data import HAR_FEATURES, ML_FEATURES, ML_REQUIRED_FEATURES

QUANTILES = (0.05, 0.50, 0.95)


class RangeModels:
    """HAR-shaped per-stock quantile regression and a pooled quantile LightGBM."""

    def __init__(self, config=None):
        self.har = {}
        self.ml = []
        self.symbols = []
        self.config = config or {}

    def fit(self, frame, baseline=None):
        train = frame.dropna(subset=HAR_FEATURES + ["y"])
        if len(train) < 500:
            raise ValueError("학습 표본이 500개 미만입니다")
        self.symbols = sorted(train.symbol.unique())
        self.fit_summary = {
            "har_training_rows": len(train),
            "candidate_symbols": len(self.symbols),
            "target_start": str(train.target_date.min().date()),
            "target_end": str(train.target_date.max().date()),
        }
        if baseline is not None:
            self.har = baseline.har.copy()
        for symbol, group in [] if baseline is not None else train.groupby("symbol"):
            if len(group) < 252:
                continue
            x = group[HAR_FEATURES].to_numpy(float)
            center, scale = x.mean(axis=0), x.std(axis=0)
            scale[scale == 0] = 1
            x = sm.add_constant((x - center) / scale, has_constant="add")
            y = np.log1p(group.y.to_numpy())
            coefficients = []
            with warnings.catch_warnings(record=True) as caught:
                for quantile in QUANTILES:
                    fit = sm.QuantReg(y, x).fit(q=quantile, max_iter=5000, p_tol=1e-6)
                    coefficients.append(fit.params)
            if any("maximum number" in str(w.message).lower() for w in caught):
                continue  # Non-converged baseline must not silently become eligible.
            self.har[symbol] = (center, scale, np.array(coefficients))
        # LightGBM natively handles missing optional features. A failed price
        # adjustment crosscheck removes returns/gaps, not otherwise valid range
        # observations. Never impute unverifiable returns to zero.
        ml_train = train.dropna(subset=ML_REQUIRED_FEATURES)
        years = self.config.get("recent_years")
        if years:
            ml_train = ml_train[
                ml_train.target_date.gt(ml_train.target_date.max() - pd.DateOffset(years=years))
            ]
        self.fit_summary.update(
            har_fitted_symbols=len(self.har),
            ml_training_rows=len(ml_train),
            ml_training_symbols=int(ml_train.symbol.nunique()),
        )
        if len(ml_train) < 500 or not self.har:
            raise ValueError("필수 피처 표본 부족 또는 HAR 적합 실패")
        x = self._ml_x(ml_train)
        for quantile in QUANTILES:
            model = LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=self.config.get("n_estimators", 200),
                num_leaves=self.config.get("num_leaves", 15),
                learning_rate=0.035,
                min_child_samples=40,
                reg_lambda=1.0,
                verbosity=-1,
                random_state=731,
                deterministic=True,
                force_col_wise=True,
                n_jobs=2,
            )
            target = ml_train.y
            if self.config.get("target_scale"):
                target = target / ml_train[self.config["target_scale"]].clip(lower=0.1)
            model.fit(x, np.log1p(target), categorical_feature=["symbol_code"])
            self.ml.append(model)
        return self

    def _ml_x(self, frame):
        x = frame[getattr(self, "config", {}).get("features", ML_FEATURES)].copy()
        codes = {symbol: i for i, symbol in enumerate(self.symbols)}
        x["symbol_code"] = frame.symbol.map(codes).fillna(-1).astype(int)
        return x

    def predict(self, frame, method):
        result = np.full((len(frame), 3), np.nan)
        if method == "empirical":
            return frame[["emp_low", "emp_mid", "emp_high"]].to_numpy(float)
        if method == "ml":
            valid = frame[ML_REQUIRED_FEATURES].notna().all(axis=1).to_numpy() & frame.symbol.isin(
                self.symbols
            )
            if valid.any():
                x = self._ml_x(frame.loc[valid])
                result[valid] = np.column_stack([model.predict(x) for model in self.ml])
        elif method == "har":
            for symbol, (center, scale, coefficients) in self.har.items():
                valid = frame.symbol.eq(symbol) & frame[HAR_FEATURES].notna().all(axis=1)
                if valid.any():
                    x = sm.add_constant(
                        (frame.loc[valid, HAR_FEATURES].to_numpy() - center) / scale, has_constant="add"
                    )
                    result[valid] = x @ coefficients.T
        else:
            raise ValueError("Unknown model")
        # Numerical explosions are invalid predictions, not clipped into apparently valid outputs.
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.maximum(0, np.expm1(result))
        scale_feature = getattr(self, "config", {}).get("target_scale") if method == "ml" else None
        if scale_feature:
            result *= frame[scale_feature].clip(lower=0.1).to_numpy()[:, None]
        result[~np.isfinite(result)] = np.nan
        return np.sort(result, axis=1)


def conformal_bounds(raw, scores, minimum=120):
    if len(scores) < minimum or not np.isfinite(raw).all():
        return None
    ordered = np.sort(np.asarray(scores))
    rank = math.ceil((len(ordered) + 1) * 0.9)
    if rank > len(ordered):
        return None
    # Nonnegative correction is conservative and prevents inverted intervals under overcoverage.
    adjustment = max(0.0, float(ordered[rank - 1]))
    low = max(0.0, float(raw[0]) - adjustment)
    high = max(low, float(raw[2]) + adjustment)
    return low, float(np.clip(raw[1], low, high)), high


def sequential_predictions(model, frame, method, start_date):
    raw = model.predict(frame, method)
    work = frame.copy()
    work[["raw_low", "raw_mid", "raw_high"]] = raw
    records = []
    config = getattr(model, "config", {}) if method == "ml" else {}
    for symbol, group in work.groupby("symbol", sort=True):
        scores, pending = deque(maxlen=config.get("calibration_window", 252)), []
        for counter, row in enumerate(group.sort_values("date").itertuples()):
            cutoff = (
                (row.date + pd.Timedelta(hours=18, minutes=30)).tz_localize("Asia/Seoul").tz_convert("UTC")
            )
            if hasattr(row, "issue_at") and pd.notna(row.issue_at):
                cutoff = row.issue_at
            while pending and pending[0][0] <= cutoff:
                _, _, score = heapq.heappop(pending)
                scores.append(score)
            raw_row = np.array([row.raw_low, row.raw_mid, row.raw_high])
            scale = max(0.1, raw_row[2] - raw_row[0]) if config.get("scaled_conformal") else 1.0
            bound = tuple(raw_row) if method == "empirical" and np.isfinite(raw_row).all() else None
            if method != "empirical":
                bound = conformal_bounds(raw_row / scale, scores)
                if bound is not None:
                    bound = tuple(value * scale for value in bound)
            if pd.notna(row.target_date) and row.target_date >= pd.Timestamp(start_date):
                low, middle, high = bound if bound is not None else (np.nan, np.nan, np.nan)
                records.append(
                    {
                        "symbol": symbol,
                        "date": row.date,
                        "target_date": row.target_date,
                        "y": row.y,
                        "low": low,
                        "median": middle,
                        "high": high,
                        "method": method,
                        "calibration_n": len(scores),
                        "target_available_at": row.target_available_at,
                    }
                )
            if np.isfinite(raw_row).all() and pd.notna(row.y) and pd.notna(row.target_available_at):
                # Never release a target before that trading session has finished.
                close_cutoff = (
                    (row.target_date + pd.Timedelta(hours=18, minutes=30))
                    .tz_localize("Asia/Seoul")
                    .tz_convert("UTC")
                )
                release = max(row.target_available_at, close_cutoff)
                score = max(float(raw_row[0] - row.y), float(row.y - raw_row[2])) / scale
                heapq.heappush(pending, (release, counter, score))
    return pd.DataFrame(records)
