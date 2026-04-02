"""
Lab 5: ML training DAG — data check, prepare, train, evaluate, branch on R², MLflow registration.

Paths use PROJECT_ROOT and RAW_DATA_PATH (set in docker-compose / Airflow env).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.python import PythonSensor

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/airflow/project")
RAW_DATA_PATH = os.environ.get(
    "RAW_DATA_PATH", os.path.join(PROJECT_ROOT, "data/raw/house_rent_sample.csv")
)


def _raw_data_ready() -> bool:
    """Sensor callable: raw CSV present (equivalent to FileSensor for bind-mounted repo)."""
    return os.path.isfile(os.environ.get("RAW_DATA_PATH", RAW_DATA_PATH))


def _evaluate_task(**_context):
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    from src.evaluate import run_evaluate

    model_path = os.path.join(PROJECT_ROOT, "data/models/model.pkl")
    prepared_dir = os.path.join(PROJECT_ROOT, "data/prepared")
    return run_evaluate(model_path, prepared_dir)


def _branch_on_r2(**context):
    ti = context["ti"]
    payload = ti.xcom_pull(task_ids="evaluate_model")
    if not payload:
        return "stop_pipeline"
    threshold = float(payload.get("r2_threshold", 0.15))
    test_r2 = float(payload["test_r2"])
    if test_r2 >= threshold:
        return "register_model"
    return "stop_pipeline"


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="DVC-style prepare/train + evaluate + conditional MLflow registry",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "lab5"],
) as dag:
    wait_raw_data = PythonSensor(
        task_id="wait_raw_data",
        python_callable=_raw_data_ready,
        poke_interval=30,
        timeout=3600,
        mode="poke",
    )

    data_prep = BashOperator(
        task_id="data_prep",
        bash_command=(
            'cd "${PROJECT_ROOT}" && python src/prepare.py "${RAW_DATA_PATH}" data/prepared'
        ),
        env={"RAW_DATA_PATH": RAW_DATA_PATH, "PROJECT_ROOT": PROJECT_ROOT},
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            'cd "${PROJECT_ROOT}" && python src/train.py data/prepared data/models'
        ),
        env={
            "PROJECT_ROOT": PROJECT_ROOT,
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
        },
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_task,
    )

    branch_quality = BranchPythonOperator(
        task_id="branch_quality",
        python_callable=_branch_on_r2,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command='cd "${PROJECT_ROOT}" && python src/register_model.py',
        env={
            "PROJECT_ROOT": PROJECT_ROOT,
            "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", ""),
        },
    )

    stop_pipeline = EmptyOperator(task_id="stop_pipeline")

    wait_raw_data >> data_prep >> train_model >> evaluate_model >> branch_quality
    branch_quality >> [register_model, stop_pipeline]
