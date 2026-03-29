"""Repository root and standard directories for all scripts."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
IMG_DIR = REPO_ROOT / "images"
