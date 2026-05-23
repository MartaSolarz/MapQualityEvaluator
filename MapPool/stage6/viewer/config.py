"""Viewer configuration for stage 6."""
from pathlib import Path

STAGE6_DIR = Path(__file__).parent.parent
DB_PATH = STAGE6_DIR / "pipeline.db"
IMAGE_DIR = Path("/Volumes/PHD/phd/mappool_data/stage5_downloaded")
THUMB_DIR = STAGE6_DIR / "thumbnails"
THUMB_MAX = 400
PER_PAGE = 30
