"""Application configuration."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
MAX_PAGES_PER_PDF = int(os.getenv("MAX_PAGES_PER_PDF", "50"))

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def path_for_api(path: str | Path) -> str:
    """Relative path from project root (posix), safe for JSON/GitHub examples."""
    p = Path(path).resolve()
    try:
        return p.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return Path(path).name


def pdf_filename(path: str | Path) -> str:
    """PDF basename only — no machine-specific directories."""
    return Path(path).name
