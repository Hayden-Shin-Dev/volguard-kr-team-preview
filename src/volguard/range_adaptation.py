"""Past-only tail correction experiments on archived DEVELOPMENT predictions."""

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .direction import block_ci
from .platform_study import local_settings, measures
from .terminal_data import read_json
from .util import atomic_json, digest, utcnow


def adapt(frame, strength, window=252, minimum=120):
    result = frame.copy()
    result["calibration_n"] = 0
    for _, g in frame.groupby(["fold", "symbol"]):
        g = g.sort_values("target_date")
        if g.target_date.duplicated().any():
            raise ValueError("Duplicate symbol/target observations")
        width = (g.high - g.low).replace(0, np.nan)
        # Shift excludes the currently evaluated outcome from its own correction.
        lower_error = ((g.y - g.low) / width).shift(1)
        upper_error = ((g.y - g.high) / width).shift(1)
        lower_q = lower_error.rolling(window, min_periods=minimum).quantile(0.05).fillna(0)
        upper_q = upper_error.rolling(window, min_periods=minimum).quantile(0.95).fillna(0)
        lower = np.maximum(0, g.low + strength * width.fillna(0) * lower_q)
        upper = g.high + strength * width.fillna(0) * upper_q
        # Preserve a valid quantile ordering without moving the archived median.
        result.loc[g.index, "low"] = np.minimum(lower, g["median"])
        result.loc[g.index, "high"] = np.maximum(upper, g["median"])
        result.loc[g.index, "calibration_n"] = lower_error.rolling(window, min_periods=1).count().fillna(0)
    return result


def interval_scores(frame):
    return frame.high - frame.low + 20 * np.maximum(np.maximum(frame.low - frame.y, frame.y - frame.high), 0)


def run(settings):
    root = settings.data_dir
    source = read_json(root / "reports/ensemble-study.json")
    out = root / "runs" / ("range-adaptation-" + uuid.uuid4().hex[:10])
    out.mkdir()
    manifest = dict(
        created_at=utcnow(),
        source_report_hash=digest(source),
        folds=source["folds"],
        strengths=[0.25, 0.5, 1.0],
        windows=[120, 252],
        minimum=120,
        source_hash=digest(Path(__file__).read_bytes()),
        evaluation="reused development only; never final test; no promotion",
        rule="each stock/fold uses only previously settled errors; initial 120 predictions unchanged",
    )
    atomic_json(out / "manifest.json", manifest)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    frames = []
    for year in source["folds"]:
        if int(year) >= 2024:
            raise ValueError("Only development years allowed")
        f = pd.read_parquet(Path(source["artifact_dir"]) / f"pred-nested7_tail50-{year}.parquet")
        frames.append(f.assign(fold=year))
    df = pd.concat(frames, ignore_index=True)
    df = df[df.method.eq("ml") & np.isfinite(df[["low", "median", "high", "y"]]).all(axis=1)].copy()
    if not (df.target_date < pd.Timestamp("2024-01-01")).all():
        raise ValueError("Development boundary violated")
    base = measures(df)
    candidates = []
    for window in manifest["windows"]:
        for strength in manifest["strengths"]:
            candidate = adapt(df, strength, window)
            score_delta = interval_scores(candidate) - interval_scores(df)
            ci = block_ci(score_delta.groupby(df.target_date).mean())
            m = measures(candidate)
            row = dict(
                window=window,
                strength=strength,
                **m,
                score_delta_ci=ci,
                width_improvement_pct=100 * (1 - m["width"] / base["width"]),
                score_improvement_pct=100 * (1 - m["score"] / base["score"]),
                adjusted_n=int(candidate.calibration_n.ge(120).sum()),
                by_fold={str(y): measures(g) for y, g in candidate.groupby("fold")},
            )
            row["development_pass"] = bool(
                0.87 <= m["coverage"] <= 0.93
                and m["score"] < base["score"]
                and m["width"] < base["width"]
                and ci[1] < 0
            )
            candidates.append(row)
            candidate.to_parquet(out / f"w{window}-s{strength}.parquet", index=False)
    report = dict(
        **manifest,
        baseline=base,
        candidates=candidates,
        artifact_dir=str(out),
        improving_candidates=[c for c in candidates if c["development_pass"]],
        completed_at=utcnow(),
        limitations="Adaptive tail quantiles have no unconditional coverage guarantee. Multiple comparisons on reused development. Existing release unchanged.",
    )
    atomic_json(out / "report.json", report)
    atomic_json(root / "reports/platform-range-adaptation.json", report)
    print(__import__("json").dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    run(local_settings())
