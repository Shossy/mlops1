"""Post-train: artifacts exist and Quality Gate on regression metrics."""

import json
import os

import joblib
import pytest


@pytest.mark.post_train
def test_model_artifact_exists(model_path):
    assert os.path.isfile(model_path), f"Model not found: {model_path}"


@pytest.mark.post_train
def test_model_loadable(model_path):
    m = joblib.load(model_path)
    assert hasattr(m, "predict"), "Loaded object has no predict()"


@pytest.mark.post_train
def test_metrics_json_exists(metrics_path):
    assert os.path.isfile(metrics_path), f"metrics.json not found: {metrics_path}"


@pytest.mark.post_train
def test_metrics_json_schema(metrics_path):
    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "test_r2" in data, "metrics.json must contain test_r2 for Quality Gate"
    assert "git_commit" in data
    assert "random_state" in data


@pytest.mark.post_train
def test_report_visualization_exists(report_image_path):
    assert os.path.isfile(
        report_image_path
    ), f"Report image missing: {report_image_path}"
    assert os.path.getsize(report_image_path) > 100, "Report image file too small"


@pytest.mark.post_train
def test_quality_gate_r2(metrics_path):
    threshold = float(os.environ.get("R2_THRESHOLD", "0.15"))
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    r2 = float(metrics["test_r2"])
    assert r2 >= threshold, f"Quality Gate failed: test_r2={r2:.4f} < {threshold:.4f}"
