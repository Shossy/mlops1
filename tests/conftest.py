"""Shared paths for Lab 4 tests (overridable via environment variables)."""

import os

import pytest


@pytest.fixture
def train_csv_path() -> str:
    return os.environ.get("DATA_TRAIN_PATH", "data/prepared/train.csv")


@pytest.fixture
def test_csv_path() -> str:
    return os.environ.get("DATA_TEST_PATH", "data/prepared/test.csv")


@pytest.fixture
def model_path() -> str:
    return os.environ.get("MODEL_PATH", "data/models/model.pkl")


@pytest.fixture
def metrics_path() -> str:
    return os.environ.get("METRICS_PATH", "metrics.json")


@pytest.fixture
def report_image_path() -> str:
    return os.environ.get("REPORT_IMAGE_PATH", "notebooks/predictions_test.png")
