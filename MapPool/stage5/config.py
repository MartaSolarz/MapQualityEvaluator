"""
Stage 5 pipeline configuration.

All tuneable parameters in one place.
Criteria definitions are in criteria.py (separate from pipeline config).
"""

from pathlib import Path
import os

# Load .env if present (API key etc.)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# --- Paths ---
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent
PREDICTIONS_FILE = BASE_DIR.parent / "stage4" / "results" / "predictions_50M.parquet"
DB_PATH = BASE_DIR / "pipeline.db"
IMAGE_CACHE_DIR = BASE_DIR / "images"
PROMPT_TEMPLATE_PATH = BASE_DIR / "criteria_prompt.md"

# --- Sampling ---
PRED_PROBA_MIN = 0.70
DOMAIN_CAP = 50
SAMPLE_TARGET = 25_000
STRATA = [
    # (low, high, fraction_of_target)
    (0.70, 0.80, 0.20),   # lower confidence — more diverse/lower quality
    (0.80, 0.90, 0.35),   # medium confidence
    (0.90, 1.01, 0.45),   # high confidence (1.01 to include 1.0)
]

# --- URL Validation ---
URL_TIMEOUT = 8          # seconds per HEAD request
URL_WORKERS = 30         # concurrent HEAD requests
URL_BATCH_SIZE = 500     # commit to DB every N items

# --- Image Download ---
DOWNLOAD_TIMEOUT = 15    # seconds per GET request
DOWNLOAD_WORKERS = 15    # concurrent downloads
MAX_IMAGE_SIZE_MB = 20   # skip very large files
ACCEPTED_FORMATS = {"JPEG", "PNG", "GIF", "WEBP", "BMP", "TIFF"}

# --- Claude Vision API ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL_PILOT = "claude-sonnet-4-20250514"
CLAUDE_MODEL_BULK = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 600
CLAUDE_WORKERS = 5       # concurrent API calls
CLAUDE_MAX_IMAGE_DIM = 1024  # resize before sending (reduces cost)
CLAUDE_RETRY_ATTEMPTS = 3
CLAUDE_RETRY_DELAY = 5   # seconds between retries

# --- Export ---
EXPORT_DIR = BASE_DIR / "results"
