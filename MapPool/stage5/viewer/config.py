"""Viewer configuration constants."""

from pathlib import Path

STAGE5_DIR = Path(__file__).parent.parent
DB_PATH = STAGE5_DIR / "pipeline.db"
IMAGE_DIR = Path("/Volumes/PHD/phd/mappool_data/stage5_downloaded")
THUMB_DIR = STAGE5_DIR / "thumbnails"
THUMB_MAX = 400
PER_PAGE = 30
