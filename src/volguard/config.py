import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

KST = ZoneInfo("Asia/Seoul")
ROOT = Path(os.environ.get("VOLGUARD_PROJECT_DIR", Path.cwd()))


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    environment: str = "live"

    @classmethod
    def load(cls, demo: bool = False):
        load_dotenv(ROOT / ".env", override=False)
        base = os.environ.get("VOLGUARD_DATA_DIR") or str(
            Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local/share")) / "VolGuardKR"
        )
        environment = os.environ.get("KIWOOM_ENV", "live")
        if environment not in {"live", "mock"}:
            raise ValueError("KIWOOM_ENV must be live or mock")
        directory = Path(base) / ("demo" if demo else "live")
        sentinel = directory / "silver" / "bars.parquet"
        if sentinel.exists():
            directory = sentinel.resolve().parents[1]
        return cls(directory, environment)

    def prepare(self):
        for directory in ("bronze", "silver", "runs", "reports", "config"):
            (self.data_dir / directory).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def db_path(self):
        return self.data_dir / "state.sqlite"


def credential_status():
    """Return booleans only; never expose credential values."""
    Settings.load()
    return {
        key: bool(os.environ.get(key, "").strip())
        for key in ("KRX_API_KEY", "KIWOOM_APP_KEY", "KIWOOM_APP_SECRET", "ECOS_API_KEY")
    }
