"""Fixed-model paired ablations for actual new external information only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .direction_math_features import BASE
from .direction_math_study import config, measure, paired_ci, run_raw, safe_json, subset
from .new_information_sources import output
from .util import atomic_json, digest, utcnow


def run(stage):
    out = output()
    folder = out / stage
    folder.mkdir(exist_ok=True)
    featurefile = out / f"{stage}-features.parquet"
    metadata = json.loads((out / f"{stage}-features.json").read_text("utf8"))
    data = pd.read_parquet(featurefile)
    if digest(featurefile.read_bytes()) != metadata["hash"]:
        raise ValueError("Feature identity changed")
    groups = metadata["groups"]
    manifest = dict(
        features_hash=metadata["hash"],
        config=config(BASE),
        groups=groups,
        source_hash=digest(Path(__file__).read_bytes()),
        fitting_hash=digest(Path(__file__).with_name("direction_math_study.py").read_bytes()),
    )
    identity = folder / "identity.json"
    if identity.exists() and json.loads(identity.read_text("utf8")) != manifest:
        raise ValueError("Changed experiment: do not reuse stale predictions")
    if not identity.exists():
        atomic_json(identity, manifest)
    issue = (data.date + pd.Timedelta(hours=18, minutes=30)).dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
    for c in [c for c in data if c.endswith("__available_at")]:
        stamp = pd.to_datetime(data[c], utc=True)
        if (stamp.notna() & (stamp > issue)).any():
            raise ValueError("Feature availability after issue")
    candidates = {
        "baseline": BASE,
        **{k: BASE + v for k, v in groups.items()},
        "all_new_information": BASE + sum(groups.values(), []),
    }
    dev = {}
    metrics = []
    for name, cols in candidates.items():
        frame = run_raw(data, config(cols), folder, name, [2021, 2022])
        frame["p_up"] = frame.raw_p
        dev[name] = frame
        metrics.append(dict(name=name, **measure(frame)))
        print("DEVELOP " + stage + " " + name, flush=True)
    # Select only on development; no architecture/calibration/threshold search.
    baseline = measure(dev["baseline"])
    valid = [r for r in metrics if r["accuracy"] >= baseline["accuracy"] and r["brier"] < baseline["brier"]]
    selected = min(valid, key=lambda x: x["brier"])["name"] if valid else "baseline"
    selection = dict(selected=selected, development=metrics, threshold=0.60)
    selectionpath = folder / "selection.json"
    if selectionpath.exists():
        existing = json.loads(selectionpath.read_text("utf8"))
        if any(existing[k] != v for k, v in safe_json(selection).items()):
            raise ValueError("Selection changed")
    else:
        atomic_json(selectionpath, safe_json(dict(selection, frozen_at=utcnow())))
    predictions = {}
    rows = []
    for name, cols in candidates.items():
        frame = run_raw(data, config(cols), folder, name, [2023])
        frame["p_up"] = frame.raw_p
        frame["candidate"] = name
        predictions[name] = frame
        row = dict(
            name=name,
            **measure(frame),
            by_symbol={str(k): measure(g) for k, g in frame.groupby("symbol")},
            by_regime={str(k): measure(g) for k, g in frame.groupby(frame.markov_high.gt(0.5))},
        )
        if name != "baseline":
            row["comparison"] = paired_ci(frame, predictions["baseline"])
            extra = [c for c in cols if c not in BASE]
            validkeys = data.loc[data[extra].notna().all(axis=1), ["symbol", "target_date"]]
            observed = frame.merge(validkeys, on=["symbol", "target_date"], validate="one_to_one")
            reference = predictions["baseline"].merge(
                validkeys, on=["symbol", "target_date"], validate="one_to_one"
            )
            row["observed_coverage"] = len(observed) / len(frame)
            row["observed_only"] = measure(observed) if len(observed) else None
            row["observed_only_baseline"] = measure(reference) if len(reference) else None
            row["observed_comparison"] = paired_ci(observed, reference) if len(observed) > 100 else None
        rows.append(row)
    chosen = predictions[selected]
    high = subset(chosen, 0.60)
    keys = chosen.loc[np.maximum(chosen.p_up, 1 - chosen.p_up) >= 0.60, ["symbol", "target_date"]]
    ref = predictions["baseline"].merge(keys, on=["symbol", "target_date"], validate="one_to_one")
    report = safe_json(
        dict(
            stage=stage,
            created_at=utcnow(),
            selected=selected,
            development=metrics,
            rows=len(chosen),
            symbols=chosen.symbol.nunique(),
            results=rows,
            confidence=high,
            confidence_matched_baseline=measure(ref) if len(ref) else None,
            feature_quality=metadata["quality"],
            independent_evaluation=False,
            operating_changes=False,
            conclusion="Reused development research; vintage assumptions prevent promotion; unchanged production",
            timing_classes={
                g: (
                    "historical-vintage"
                    if g == "nasdaq_vintage"
                    else "receipt-date-event"
                    if g == "dart_events_core"
                    else "current-vintage-delayed-exploratory"
                )
                for g in groups
            },
        )
    )
    pd.concat(predictions.values(), ignore_index=True).to_parquet(folder / "predictions.parquet", index=False)
    atomic_json(folder / "report.json", report)
    print("STUDY_DONE " + stage + " " + selected, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["global", "core"])
    run(parser.parse_args().stage)
