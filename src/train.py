"""
src/train.py
DVC pipeline stage: train RandomForest on prepared data and log results to MLflow.

Usage (via DVC):
    python src/train.py data/prepared data/models
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _git_commit_short(project_root: Path) -> str:
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


def _dvc_raw_data_snippet(project_root: Path) -> str:
    dvc_file = (
        project_root / "data" / "raw" / "House_Rent_10M_balanced_40cities.csv.dvc"
    )
    if dvc_file.is_file():
        try:
            return dvc_file.read_text(encoding="utf-8")[:500]
        except OSError:
            pass
    return "n/a"


def _write_metrics_json(
    path: str,
    all_metrics: dict,
    random_state: int,
    git_commit: str,
    dvc_snippet: str,
) -> None:
    payload = {
        **all_metrics,
        "random_state": random_state,
        "git_commit": git_commit,
        "dvc_raw_dvc_head": dvc_snippet,
    }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics written to {path}")


def compute_metrics(y_true, y_pred, prefix: str = "") -> dict:
    """Compute RMSE, MAE, R²."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        f"{prefix}rmse": round(rmse, 5),
        f"{prefix}mae": round(mae, 5),
        f"{prefix}r2": round(r2, 5),
    }


def plot_feature_importance(model, feature_names: list, save_path: str):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(len(importances)),
        importances[indices],
        color="steelblue",
        edgecolor="white",
    )
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right",
    )
    plt.title("Feature Importance (RandomForest)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Feature importance plot saved: {save_path}")


def plot_predictions(
    y_true, y_pred, save_path: str, title: str = "Actual vs Predicted Rent (INR)"
):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true[:2000], y_pred[:2000], alpha=0.3, color="darkorange", s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "k--", linewidth=1.5, label="Ideal line")
    plt.xlabel("Actual Rent (INR)")
    plt.ylabel("Predicted Rent (INR)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Predictions plot saved: {save_path}")


def main():
    if len(sys.argv) != 3:
        print(f"Using: python {sys.argv[0]} <prepared_dir> <output_dir>")
        sys.exit(1)

    prepared_dir = sys.argv[1]
    output_dir = sys.argv[2]

    with open("params.yaml", "r", encoding="utf-8") as fh:
        params = yaml.safe_load(fh)["train"]

    n_estimators = params["n_estimators"]
    max_depth = params["max_depth"]
    min_samples_split = params["min_samples_split"]
    min_samples_leaf = params["min_samples_leaf"]
    max_features = params["max_features"]
    random_state = params.get("random_state", 42)
    experiment_name = params.get("experiment_name", "House_Rent_Prediction")

    project_root = Path(__file__).resolve().parents[1]
    if os.environ.get("CI", "").lower() == "true":
        ne = os.environ.get("TRAIN_N_ESTIMATORS")
        if ne:
            n_estimators = int(ne)
            logger.info(
                f"CI: TRAIN_N_ESTIMATORS override -> n_estimators={n_estimators}"
            )

    logger.info(f"Reading prepared data from: {prepared_dir}")
    train_df = pd.read_csv(os.path.join(prepared_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(prepared_dir, "test.csv"))

    X_train = train_df.drop(columns=["Rent"])
    y_train = train_df["Rent"]
    X_test = test_df.drop(columns=["Rent"])
    y_test = test_df["Rent"]

    logger.info(
        f"Train: {len(X_train):,} | Test: {len(X_test):,} | Features: {X_train.shape[1]}"
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)

    mlflow.set_experiment(experiment_name)
    run_name = f"RF_depth{max_depth}_trees{n_estimators}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("author", "student")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset", "House_Rent_10M_India")
        mlflow.set_tag("target", "log_rent")

        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "random_state": random_state,
            }
        )
        logger.info(
            "Params: n_estimators=%s max_depth=%s min_samples_split=%s min_samples_leaf=%s max_features=%s random_state=%s",
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            random_state,
        )

        logger.info("Training model...")
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_metrics = compute_metrics(y_train, train_pred, prefix="train_")
        test_metrics = compute_metrics(y_test, test_pred, prefix="test_")

        train_pred_inr = np.expm1(train_pred)
        test_pred_inr = np.expm1(test_pred)
        y_train_inr = np.expm1(y_train)
        y_test_inr = np.expm1(y_test)

        inr_metrics = {
            "train_mae_inr": round(mean_absolute_error(y_train_inr, train_pred_inr), 2),
            "test_mae_inr": round(mean_absolute_error(y_test_inr, test_pred_inr), 2),
            "train_rmse_inr": round(
                np.sqrt(mean_squared_error(y_train_inr, train_pred_inr)), 2
            ),
            "test_rmse_inr": round(
                np.sqrt(mean_squared_error(y_test_inr, test_pred_inr)), 2
            ),
        }

        all_metrics = {**train_metrics, **test_metrics, **inr_metrics}
        all_metrics["overfit_r2_gap"] = round(
            train_metrics["train_r2"] - test_metrics["test_r2"], 5
        )

        mlflow.log_metrics(all_metrics)
        logger.info(
            f"Train R²: {train_metrics['train_r2']:.4f} | Test R²: {test_metrics['test_r2']:.4f}"
        )
        logger.info(
            f"Test MAE: \u20b9{inr_metrics['test_mae_inr']:,.0f} | Test RMSE: \u20b9{inr_metrics['test_rmse_inr']:,.0f}"
        )
        logger.info(f"Overfitting gap: {all_metrics['overfit_r2_gap']:.4f}")

        fi_path = "notebooks/feature_importance.png"
        pred_test_path = "notebooks/predictions_test.png"
        pred_train_path = "notebooks/predictions_train.png"

        plot_feature_importance(model, list(X_train.columns), fi_path)
        plot_predictions(
            y_test_inr.values,
            test_pred_inr,
            pred_test_path,
            title="Actual vs Predicted — Test set (INR)",
        )
        plot_predictions(
            y_train_inr.values,
            train_pred_inr,
            pred_train_path,
            title="Actual vs Predicted — Train set (INR)",
        )

        mlflow.log_artifact(fi_path)
        mlflow.log_artifact(pred_test_path)
        mlflow.log_artifact(pred_train_path)

        model_path = os.path.join(output_dir, "model.pkl")
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path="random_forest_model")

        logger.info(f"Model saved to {model_path}")
        metrics_path = os.environ.get("METRICS_PATH", "metrics.json")
        _write_metrics_json(
            metrics_path,
            all_metrics,
            random_state=random_state,
            git_commit=_git_commit_short(project_root),
            dvc_snippet=_dvc_raw_data_snippet(project_root),
        )
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run complete. Run ID: {run_id}")
        logger.info("Open MLflow UI: mlflow ui  ->  http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
