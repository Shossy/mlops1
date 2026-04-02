"""CI: DAG files load without import errors (Airflow DagBag)."""

import os

import pytest


def test_dag_import_no_errors():
    pytest.importorskip("airflow.models")
    from airflow.models import DagBag

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dag_folder = os.path.join(root, "dags")
    dag_bag = DagBag(dag_folder=dag_folder, include_examples=False)
    assert len(dag_bag.import_errors) == 0, dag_bag.import_errors
