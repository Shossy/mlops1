"""
src/evaluate.py
Standalone evaluation stage: load a trained model, evaluate on test data,
produce metrics.json and confusion_matrix.png, log results to MLflow.

The last stdout line is a JSON string of key metrics — used by Airflow XCom.

Usage:
    python src/evaluate.py <model_path> <test_data_path>
    python src/evaluate.py data/models/model.pkl data/prepared/test.csv
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_commit_short() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _dvc_raw_data_snippet() -> str:
    dvc_file = (
        PROJECT_ROOT / "data" / "raw" / "House_Rent_10M_balanced_40cities.csv.dvc"
    )
    if dvc_file.is_file():
        try:
            return dvc_file.read_text(encoding="utf-8")[:500]
        except OSError:
            pass
    return "n/a"


def compute_metrics(y_true, y_pred, prefix: str = "") -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        f"{prefix}rmse": round(rmse, 5),
        f"{prefix}mae": round(mae, 5),
        f"{prefix}r2": round(r2, 5),
    }


def plot_actual_vs_predicted(y_true, y_pred, r2: float, save_path: str) -> None:
    """Actual vs Predicted scatter — regression equivalent of a confusion matrix."""
    plt.figure(figsize=(8, 6))
    sample_n = min(3000, len(y_true))
    plt.scatter(
        y_true[:sample_n],
        y_pred[:sample_n],
        alpha=0.3,
        color="darkorange",
        s=10,
    )
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "k--", linewidth=1.5, label="Ideal")
    plt.xlabel("Actual Rent (INR)")
    plt.ylabel("Predicted Rent (INR)")
    plt.title(f"Actual vs Predicted — Test set (R²={r2:.4f})")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Evaluation plot saved: %s", save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("model_path", help="Path to model.pkl")
    parser.add_argument("test_data_path", help="Path to test CSV")
    parser.add_argument(
        "--metrics-path",
        default=os.environ.get("METRICS_PATH", "metrics.json"),
    )
    parser.add_argument(
        "--plot-path",
        default=os.environ.get("PLOT_PATH", "confusion_matrix.png"),
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "House_Rent_Prediction"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading model from %s", args.model_path)
    model = joblib.load(args.model_path)

    logger.info("Loading test data from %s", args.test_data_path)
    test_df = pd.read_csv(args.test_data_path)
    X_test = test_df.drop(columns=["Rent"])
    y_test = test_df["Rent"]

    logger.info("Test samples: %d | Features: %d", len(X_test), X_test.shape[1])

    test_pred = model.predict(X_test)

    # Log-space metrics (target is log1p(Rent))
    test_metrics = compute_metrics(y_test, test_pred, prefix="test_")

    # INR-space metrics (expm1 to restore original scale)
    y_test_inr = np.expm1(y_test)
    test_pred_inr = np.expm1(test_pred)
    inr_metrics = {
        "test_mae_inr": round(float(mean_absolute_error(y_test_inr, test_pred_inr)), 2),
        "test_rmse_inr": round(
            float(np.sqrt(mean_squared_error(y_test_inr, test_pred_inr))), 2
        ),
    }

    # Read random_state from params.yaml if available
    random_state = 42
    params_file = PROJECT_ROOT / "params.yaml"
    if params_file.is_file():
        with open(params_file, "r", encoding="utf-8") as fh:
            params = yaml.safe_load(fh)
        random_state = params.get("train", {}).get("random_state", 42)

    all_metrics = {**test_metrics, **inr_metrics}

    payload = {
        **all_metrics,
        "random_state": random_state,
        "git_commit": _git_commit_short(),
        "dvc_raw_dvc_head": _dvc_raw_data_snippet(),
    }

    # Save metrics.json
    metrics_dir = os.path.dirname(args.metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Metrics written to %s", args.metrics_path)

    # Save confusion_matrix.png (actual vs predicted scatter for regression)
    plot_actual_vs_predicted(
        y_test_inr.values,
        test_pred_inr,
        r2=test_metrics["test_r2"],
        save_path=args.plot_path,
    )

    # Log to MLflow
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.set_tag("stage", "evaluate")
        mlflow.set_tag("git_commit", payload["git_commit"])
        mlflow.log_metrics(all_metrics)
        mlflow.log_artifact(args.metrics_path)
        mlflow.log_artifact(args.plot_path)
        logger.info(
            "Test R²: %.4f | Test RMSE (INR): %.0f",
            test_metrics["test_r2"],
            inr_metrics["test_rmse_inr"],
        )

    # Last stdout line: JSON for Airflow XCom capture
    print(json.dumps(all_metrics))


if __name__ == "__main__":
    main()
