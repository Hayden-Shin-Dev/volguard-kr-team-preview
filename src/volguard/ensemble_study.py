"""Two fixed half-weight forecast ensembles, additional development selection."""

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from .blend_study import blend
from .config import Settings
from .learning_study import evaluate_candidate
from .util import atomic_json, digest, utcnow

CANDIDATES = {f"nested{leaves}_tail50": [f"leaves{leaves}_nested", "tail_features_s100"]
              for leaves in (7, 15)}


def run(settings):
    studies = {name: json.loads((settings.data_dir / f"reports/{name}-study.json").read_text(encoding="utf8"))
               for name in ("learning", "boosting")}
    previous = json.loads((settings.data_dir / "reports/forecast-improvement.json").read_text(encoding="utf8"))
    if any(s["previous_report_hash"] != digest(previous) for s in studies.values()):
        raise ValueError("Ensemble components have different controls")
    roots = {name: Path(s["artifact_dir"]) for name, s in studies.items()}
    control, prior_root = Path(studies["learning"]["control_artifact"]), Path(previous["artifact_dir"])
    output = settings.data_dir / "runs" / f"ensemble-{uuid.uuid4().hex[:10]}"
    output.mkdir(parents=True)
    manifest = {"created_at": utcnow(), "candidates": CANDIDATES, "weights": [.5, .5],
                "folds": studies["learning"]["folds"], "previous_artifact": str(prior_root),
                "previous_champion": previous["champion"], "previous_report_hash": digest(previous),
                "source_hash": digest(Path(__file__).read_bytes()),
                "components": {name: digest(s) for name, s in studies.items()},
                "component_hashes": {f"{name}/{p.name}": digest(p.read_bytes())
                                     for name, root in roots.items() for p in root.glob("pred-*.parquet")},
                "evaluation_role": "additional reused development after component diagnostics; not independent",
                "selection": "same full gates; both width and score must improve over previous champion"}
    atomic_json(output / "manifest.json", manifest)
    shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    originals = pd.concat([pd.read_parquet(control / f"pred-original__current-{y}.parquet")
                           for y in manifest["folds"]], ignore_index=True)
    prior = pd.concat([pd.read_parquet(prior_root / f"pred-{previous['champion']}-{y}.parquet")
                       for y in manifest["folds"]], ignore_index=True)
    results = []
    for key, (boosted, tail) in CANDIDATES.items():
        parts = []
        for year in manifest["folds"]:
            pred = blend(pd.read_parquet(roots["boosting"] / f"pred-{boosted}-{year}.parquet"),
                         pd.read_parquet(roots["learning"] / f"pred-{tail}-{year}.parquet"))
            pred.to_parquet(output / f"pred-{key}-{year}.parquet", index=False)
            parts.append(pred)
        results.append({"candidate": key, **evaluate_candidate(parts, originals, prior, manifest["folds"])})
    passing = [r for r in results if r["development_pass"]]
    best = min(passing or results, key=lambda r: r["versus_frozen_har"]["daily_scores"]["ml"])
    report = {**manifest, "results": results, "champion": best["candidate"],
              "development_pass": bool(passing), "deployment_eligible": False,
              "artifact_dir": str(output.resolve()), "completed_at": utcnow()}
    atomic_json(output / "report.json", report)
    atomic_json(settings.data_dir / "reports/ensemble-study.json", report)
    print(json.dumps({"champion": best["candidate"], "result": best}), flush=True)


if __name__ == "__main__":
    run(Settings.load())
