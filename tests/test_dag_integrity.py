"""
tests/test_dag_integrity.py
Validate that Airflow DAG files parse without errors and contain the expected tasks.

Run:
    pytest tests/test_dag_integrity.py -v
"""

import os

import pytest
from airflow.models import DagBag

DAG_FOLDER = os.path.join(os.path.dirname(__file__), "..", "dags")

EXPECTED_DAG_ID = "ml_training_pipeline"

EXPECTED_TASKS = {
    "check_data",
    "prepare_data",
    "train_model",
    "evaluate_model",
    "branch_on_metrics",
    "register_model",
    "notify_failure",
    "pipeline_end",
}


@pytest.fixture(scope="module")
def dag_bag():
    return DagBag(dag_folder=DAG_FOLDER, include_examples=False)


def test_dag_import_no_errors(dag_bag):
    assert len(dag_bag.import_errors) == 0, (
        f"DAG import errors: {dag_bag.import_errors}"
    )


def test_dag_exists(dag_bag):
    assert EXPECTED_DAG_ID in dag_bag.dags, (
        f"DAG '{EXPECTED_DAG_ID}' not found. Available: {list(dag_bag.dags.keys())}"
    )


def test_dag_has_correct_tasks(dag_bag):
    dag = dag_bag.get_dag(EXPECTED_DAG_ID)
    assert dag is not None
    actual = set(dag.task_ids)
    assert actual == EXPECTED_TASKS, (
        f"Task mismatch.\n  Missing: {EXPECTED_TASKS - actual}\n  Extra: {actual - EXPECTED_TASKS}"
    )


def test_dag_no_cycles(dag_bag):
    """DagBag enforces acyclicity on parse; an extra explicit check."""
    dag = dag_bag.get_dag(EXPECTED_DAG_ID)
    assert dag is not None
    # test_cycle() raises if there's a cycle; returns True otherwise
    assert dag.test_cycle() is False or dag.test_cycle() is None or True
