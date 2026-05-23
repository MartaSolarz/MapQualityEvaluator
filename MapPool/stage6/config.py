"""
Stage 6 pipeline configuration: disqualifying criteria check.
"""

from pathlib import Path
import os

# Load .env if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# --- Paths ---
BASE_DIR = Path(__file__).parent
STAGE5_DIR = BASE_DIR.parent / "stage5"
STAGE5_DB = STAGE5_DIR / "pipeline.db"
DB_PATH = BASE_DIR / "pipeline.db"
IMAGE_DIR = Path("/Volumes/PHD/phd/mappool_data/stage5_downloaded")
PROMPT_TEMPLATE_PATH = BASE_DIR / "criteria_prompt.md"

# --- API ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL_PILOT = "claude-sonnet-4-20250514"
CLAUDE_MODEL_BULK = "claude-sonnet-4-20250514"  # Sonnet for both (needs deeper understanding)
CLAUDE_MAX_TOKENS = 1200  # longer responses (10 criteria + descriptions)
CLAUDE_MAX_IMAGE_DIM = 1024
CLAUDE_WORKERS = 3  # fewer workers (Sonnet is slower + more expensive)
CLAUDE_RETRY_ATTEMPTS = 3
CLAUDE_RETRY_DELAY = 5

# --- Caching ---
CLAUDE_USE_CACHE = True  # set True for bulk (saves ~90%), False for testing (fresh prompt each time)

# --- Quality Score ---
Q_MAX = 60  # max score for stage 1 (to be calibrated empirically)
