from pathlib import Path

def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

REPO_ROOT = get_repo_root()

