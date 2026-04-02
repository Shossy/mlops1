"""
Register the sklearn model from the latest training MLflow run into the Model Registry (Staging).

Requires a tracking URI with registry support (e.g. MLflow server + SQL backend).
Run id: MLFLOW_RUN_ID env or data/models/mlflow_run_id.txt (after train).

Usage:
    python src/register_model.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    model_name = os.environ.get("MLFLOW_MODEL_NAME", "house_rent_rf")
    stage = os.environ.get("MLFLOW_MODEL_STAGE", "Staging")

    run_id = os.environ.get("MLFLOW_RUN_ID", "").strip()
    if not run_id:
        rid_file = Path(
            os.environ.get(
                "MLFLOW_RUN_ID_FILE",
                str(project_root / "data/models/mlflow_run_id.txt"),
            )
        )
        if rid_file.is_file():
            run_id = rid_file.read_text(encoding="utf-8").strip()
    if not run_id:
        logger.error("No MLflow run id: set MLFLOW_RUN_ID or run train first")
        return 1

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"runs:/{run_id}/random_forest_model"
    try:
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info("Registered %s version %s", model_name, mv.version)
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage=stage,
            archive_existing_versions=False,
        )
        logger.info("Model version %s -> %s", mv.version, stage)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Registry unavailable or failed (use MLflow server + DB for full support): %s",
            e,
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
