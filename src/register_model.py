"""
src/register_model.py
Register the latest trained model in MLflow Model Registry with Staging stage.

Reads metrics.json for context, finds the most recent MLflow run in the
experiment, and registers its logged sklearn model artifact.

Usage:
    python src/register_model.py
    python src/register_model.py --metrics-path metrics.json --model-name HouseRentRF
"""

import argparse
import json
import logging
import os

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register model in MLflow Registry")
    parser.add_argument(
        "--metrics-path",
        default=os.environ.get("METRICS_PATH", "metrics.json"),
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "HouseRentRF"),
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "House_Rent_Prediction"),
    )
    parser.add_argument(
        "--stage",
        default=os.environ.get("MODEL_STAGE", "Staging"),
    )
    parser.add_argument(
        "--artifact-path",
        default="random_forest_model",
        help="Artifact path used in mlflow.sklearn.log_model during training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    metrics = {}
    if os.path.isfile(args.metrics_path):
        with open(args.metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        logger.info("Loaded metrics from %s: test_r2=%.4f", args.metrics_path,
                     metrics.get("test_r2", -1))

    client = MlflowClient()

    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        logger.error("Experiment '%s' not found", args.experiment_name)
        raise SystemExit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        logger.error("No runs found in experiment '%s'", args.experiment_name)
        raise SystemExit(1)

    latest_run = runs[0]
    run_id = latest_run.info.run_id
    model_uri = f"runs:/{run_id}/{args.artifact_path}"
    logger.info("Registering model from run %s, URI: %s", run_id, model_uri)

    mv = mlflow.register_model(model_uri, args.model_name)
    logger.info("Registered %s version %s", args.model_name, mv.version)

    client.transition_model_version_stage(
        name=args.model_name,
        version=mv.version,
        stage=args.stage,
    )
    logger.info("Transitioned to stage: %s", args.stage)

    client.set_model_version_tag(args.model_name, mv.version,
                                 "registered_by", "airflow_dag")
    if "test_r2" in metrics:
        client.set_model_version_tag(args.model_name, mv.version,
                                     "test_r2", str(metrics["test_r2"]))
    if "git_commit" in metrics:
        client.set_model_version_tag(args.model_name, mv.version,
                                     "git_commit", metrics["git_commit"])

    logger.info("Model %s v%s -> %s (done)", args.model_name, mv.version, args.stage)


if __name__ == "__main__":
    main()
