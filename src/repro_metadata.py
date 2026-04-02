"""Reproducibility helpers: git commit, raw data DVC pointer snippet, MLflow run id file."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def project_root_from_here(here: Path | None = None) -> Path:
    """Repository root (parent of src/)."""
    if here is None:
        here = Path(__file__).resolve().parent
    return here.parent


def git_commit_short(project_root: Path) -> str:
    env = os.environ.get("GIT_COMMIT_SHORT") or os.environ.get("GIT_COMMIT")
    if env:
        return env[:7] if len(env) > 40 else env
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def dvc_raw_data_snippet(project_root: Path) -> str:
    dvc_file = (
        project_root / "data" / "raw" / "House_Rent_10M_balanced_40cities.csv.dvc"
    )
    if dvc_file.is_file():
        try:
            return dvc_file.read_text(encoding="utf-8")[:500]
        except OSError:
            pass
    return "n/a"


def write_mlflow_run_id(output_dir: str | Path, run_id: str) -> Path:
    """Persist active MLflow run id for register_model.py (Docker / Airflow)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "mlflow_run_id.txt"
    path.write_text(run_id.strip(), encoding="utf-8")
    return path
