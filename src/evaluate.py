"""
Evaluate trained regressor on the test split: regression metrics, lab-style derived-binary
metrics + confusion matrix (auxiliary reporting only; model is trained for regression).

Usage:
    python src/evaluate.py <model_path> <prepared_dir>

Writes metrics.json (path from METRICS_PATH) and confusion_matrix.png (CONFUSION_MATRIX_PATH).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.repro_metadata import (
    dvc_raw_data_snippet,
    git_commit_short,
    project_root_from_here,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _compute_regression_metrics(y_true_log, y_pred_log, prefix: str = "test_") -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae = float(mean_absolute_error(y_true_log, y_pred_log))
    r2 = float(r2_score(y_true_log, y_pred_log))
    return {
        f"{prefix}rmse": round(rmse, 5),
        f"{prefix}mae": round(mae, 5),
        f"{prefix}r2": round(r2, 5),
    }


def _derived_high_rent_binary(
    y_inr: np.ndarray, pred_inr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Auxiliary binary labels for confusion matrix / accuracy (lab artifact).
    "High rent" = above test-set median actual rent (INR). Not the training objective.
    """
    med = np.median(y_inr)
    y_bin = (y_inr > med).astype(int)
    pred_bin = (pred_inr > med).astype(int)
    return y_bin, pred_bin


def run_evaluate(
    model_path: str,
    prepared_dir: str,
    *,
    metrics_path: str | None = None,
    cm_path: str | None = None,
    project_root: Path | None = None,
) -> dict:
    metrics_path = metrics_path or os.environ.get("METRICS_PATH", "metrics.json")
    cm_path = cm_path or os.environ.get(
        "CONFUSION_MATRIX_PATH", "artifacts/confusion_matrix.png"
    )
    project_root = project_root or project_root_from_here()
    params_all = yaml.safe_load(
        (project_root / "params.yaml").read_text(encoding="utf-8")
    )
    train_params = params_all.get("train", {})
    eval_params = params_all.get("evaluate", {})
    random_state = int(train_params.get("random_state", 42))

    model = joblib.load(model_path)
    test_df = pd.read_csv(os.path.join(prepared_dir, "test.csv"))
    X_test = test_df.drop(columns=["Rent"])
    y_test = test_df["Rent"]

    pred_log = model.predict(X_test)
    reg = _compute_regression_metrics(y_test, pred_log, prefix="test_")

    y_inr = np.expm1(y_test.values)
    pred_inr = np.expm1(pred_log)
    reg["test_mae_inr"] = round(float(mean_absolute_error(y_inr, pred_inr)), 2)
    reg["test_rmse_inr"] = round(float(np.sqrt(mean_squared_error(y_inr, pred_inr))), 2)

    y_bin, pred_bin = _derived_high_rent_binary(y_inr, pred_inr)
    accuracy = float(accuracy_score(y_bin, pred_bin))
    f1 = float(f1_score(y_bin, pred_bin, zero_division=0))

    cm = confusion_matrix(y_bin, pred_bin)
    out_cm = Path(cm_path)
    out_cm.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["low", "high"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (derived high/low rent vs predictions)")
    fig.tight_layout()
    fig.savefig(out_cm, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_cm)

    payload = {
        **reg,
        "accuracy": round(accuracy, 5),
        "f1": round(f1, 5),
        "random_state": random_state,
        "git_commit": git_commit_short(project_root),
        "dvc_raw_dvc_head": dvc_raw_data_snippet(project_root),
        "confusion_matrix_path": str(out_cm),
        "evaluate_note": (
            "accuracy/f1/confusion_matrix use derived binary labels (median rent); "
            "primary task is regression (test_r2)."
        ),
    }

    mdir = os.path.dirname(metrics_path) or "."
    os.makedirs(mdir, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", metrics_path)

    return {
        "test_r2": payload["test_r2"],
        "accuracy": payload["accuracy"],
        "metrics_path": metrics_path,
        "r2_threshold": float(eval_params.get("r2_threshold", 0.15)),
    }


def main() -> dict:
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <model_path> <prepared_dir>")
        sys.exit(1)
    model_path = os.environ.get("MODEL_PATH", sys.argv[1])
    prepared_dir = os.environ.get("PREPARED_DIR", sys.argv[2])
    return run_evaluate(model_path, prepared_dir)


if __name__ == "__main__":
    main()
