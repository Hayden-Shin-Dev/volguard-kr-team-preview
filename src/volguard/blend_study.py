"""Two fixed equal-weight blends of already calibrated development intervals.

Motivated by the opposing width/score results of signed and tail calibration.
This is additional development selection, not independent confirmation or CQR.
"""

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from .calibration_study import paired_comparison
from .config import Settings
from .evaluation import promotion_gate
from .util import atomic_json, digest, utcnow


def blend(left, right):
    keys = ["method", "symbol", "target_date"]
    left = left.set_index(keys).sort_index()
    right = right.set_index(keys).sort_index()
    pd.testing.assert_index_equal(left.index, right.index)
    pd.testing.assert_series_equal(left.y, right.y)
    out = left.copy()
    for column in ("low", "median", "high"):
        # Missing in either component stays missing; never drop the harder cases.
        out[column] = (left[column] + right[column]) / 2
    out["empty_expanded"] = left.empty_expanded.eq(True) | right.empty_expanded.eq(True)
    return out.reset_index()


def run(settings):
    control = json.loads((settings.data_dir / "reports/calibration-study.json").read_text(encoding="utf8"))
    root = Path(control["artifact_dir"])
    output = settings.data_dir / "runs" / f"blend-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    manifest = {"created_at": utcnow(), "folds": control["folds"],
                "models": ["original", "official_normalized"], "components": ["signed252", "tails252"],
                "weights": [0.5, 0.5], "control_artifact": str(root), "control_hash": digest(control),
                "source_hash": digest(Path(__file__).read_bytes()), "selection": control["selection"],
                "evaluation_role": "additional development selection; no asserted coverage theorem"}
    atomic_json(output / "manifest.json", manifest)
    shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    original = pd.concat([pd.read_parquet(root / f"pred-original__current-{y}.parquet")
                          for y in control["folds"]], ignore_index=True)
    results = []
    for name in manifest["models"]:
        key, predictions = f"{name}__blend50", []
        for year in control["folds"]:
            pred = blend(*(pd.read_parquet(root / f"pred-{name}__{c}-{year}.parquet")
                           for c in manifest["components"]))
            pred.to_parquet(output / f"pred-{key}-{year}.parquet", index=False)
            predictions.append(pred)
        pred = pd.concat(predictions, ignore_index=True)
        ml = pred.query("method == 'ml'")
        frozen = promotion_gate(pd.concat([ml, original.query("method != 'ml'")], ignore_index=True))
        same, compare = promotion_gate(pred), paired_comparison(ml, original.query("method == 'ml'"))
        folds = [{"year": y, **promotion_gate(p)} for y, p in zip(control["folds"], predictions)]
        passed = (frozen["ml_pass"] and same["ml_pass"] and 0.89 <= frozen["paired_ml_coverage"] <= 0.91
                  and compare["width_reduction"] > 0 and compare["score_improvement"] > 0
                  and all(0.87 <= f["paired_ml_coverage"] <= 0.93 for f in folds))
        results.append({"candidate": key, "versus_frozen_har": frozen, "versus_same_calibration_har": same,
                        "versus_original_ml": compare, "folds": folds, "development_pass": bool(passed),
                        "empty_expanded": int(ml.empty_expanded.sum())})
    passing = [r for r in results if r["development_pass"]]
    best = min(passing or results, key=lambda r: r["versus_frozen_har"]["daily_scores"]["ml"])
    report = {**manifest, "results": results, "champion": best["candidate"],
              "development_pass": bool(passing), "deployment_eligible": False,
              "artifact_dir": str(output.resolve()), "completed_at": utcnow()}
    atomic_json(output / "report.json", report)
    atomic_json(settings.data_dir / "reports/blend-study.json", report)
    print(json.dumps({"champion": best["candidate"], "result": best}), flush=True)


if __name__ == "__main__":
    run(Settings.load())
