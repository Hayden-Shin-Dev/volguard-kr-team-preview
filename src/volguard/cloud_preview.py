"""Single-process Render entry point. Seed contains market data only."""
import json
import os
import shutil
from pathlib import Path

from .config import Settings
from .public_preview import create_preview


def bootstrap(seed: Path, base: Path):
    live = base / "live"
    if not (seed / "config/universe.json").is_file():
        raise RuntimeError("Market seed missing; run scripts/export-render-seed.py before deployment")
    for source in seed.rglob("*"):
        if source.is_file():
            target = live / source.relative_to(seed)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    manifest_path = live / "config/forecast-release.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    for component in manifest["components"]:
        component["path"] = str(live / "models" / Path(component["name"]).with_suffix(".joblib"))
    manifest["baseline_path"] = str(live / "models/baseline.joblib")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf8")
    return Settings(live, os.environ.get("KIWOOM_ENV", "live")).prepare()


def main():
    import uvicorn
    base = Path(os.environ.get("VOLGUARD_DATA_DIR", "/var/data/volguard"))
    source = bootstrap(Path(os.environ.get("VOLGUARD_SEED_DIR", "/app/deploy/seed")), base)
    app = create_preview(source, base / "public-preview", broker_refresh=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "10000")),
                access_log=False, workers=1)


if __name__ == "__main__":
    main()
