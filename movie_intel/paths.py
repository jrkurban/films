from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STAGING_DIR = DATA_DIR / "staging"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

DEFAULT_MERGED = PROCESSED_DIR / "movies.parquet"
DEFAULT_MODEL = MODELS_DIR / "rating_model.joblib"


def ensure_dirs() -> None:
    for path in (RAW_DIR, STAGING_DIR, PROCESSED_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
