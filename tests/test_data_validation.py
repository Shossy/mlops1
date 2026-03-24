"""Pre-train checks: prepared data schema and basic quality."""

import os

import pandas as pd
import pytest

REQUIRED_COLS = {"Rent"}
MIN_ROWS = 50


@pytest.mark.pre_train
def test_prepared_train_exists(train_csv_path):
    assert os.path.isfile(train_csv_path), f"Missing train data: {train_csv_path}"


@pytest.mark.pre_train
def test_prepared_test_exists(test_csv_path):
    assert os.path.isfile(test_csv_path), f"Missing test data: {test_csv_path}"


@pytest.mark.pre_train
def test_train_schema_and_quality(train_csv_path):
    df = pd.read_csv(train_csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"
    assert df["Rent"].notna().all(), "Rent contains nulls"
    assert (
        df.shape[0] >= MIN_ROWS
    ), f"Too few rows for training smoke test: {df.shape[0]}"
    # Prepared target is log1p(Rent); expect finite reasonable range
    assert df["Rent"].between(-0.5, 20).all(), "Rent (log1p) out of expected range"


@pytest.mark.pre_train
def test_test_schema_and_quality(test_csv_path):
    df = pd.read_csv(test_csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"
    assert df["Rent"].notna().all(), "Rent contains nulls"
    assert df.shape[0] >= max(10, MIN_ROWS // 5), "Too few test rows"


@pytest.mark.pre_train
def test_feature_columns_numeric_except_none(train_csv_path):
    df = pd.read_csv(train_csv_path)
    X = df.drop(columns=["Rent"])
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(
            X[col]
        ), f"Non-numeric feature column: {col}"
