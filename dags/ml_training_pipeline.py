"""
dags/ml_training_pipeline.py
Airflow DAG: end-to-end ML training pipeline with conditional model registration.

Flow:
  check_data (FileSensor)
    -> prepare_data (BashOperator)
    -> train_model (BashOperator)
    -> evaluate_model (BashOperator)
    -> branch_on_metrics (BranchPythonOperator)
        -> register_model (BashOperator)   [if test_r2 >= threshold]
        -> notify_failure (BashOperator)   [if test_r2 < threshold]
    -> pipeline_end (EmptyOperator)
"""

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.sensors.filesystem import FileSensor

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/airflow/project")
R2_THRESHOLD = float(os.environ.get("R2_THRESHOLD", "0.5"))
METRICS_PATH = os.environ.get("METRICS_PATH", "metrics.json")

# Resolve raw data path: prefer sample CSV that is committed to Git
RAW_DATA_FILE = "data/raw/house_rent_sample.csv"

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def _check_model_quality(**kwargs):
    """Read metrics.json and decide whether to register or stop."""
    metrics_file = os.path.join(PROJECT_DIR, METRICS_PATH)

    # Fallback: try XCom value pushed by evaluate_model (last stdout line)
    ti = kwargs["ti"]
    xcom_value = ti.xcom_pull(task_ids="evaluate_model")
    metrics = None
    if xcom_value:
        try:
            metrics = json.loads(xcom_value) if isinstance(xcom_value, str) else xcom_value
        except (json.JSONDecodeError, TypeError):
            pass

    # Primary source: metrics.json file on disk
    if metrics is None or "test_r2" not in metrics:
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    test_r2 = float(metrics["test_r2"])
    print(f"test_r2={test_r2:.4f}, threshold={R2_THRESHOLD}")

    if test_r2 >= R2_THRESHOLD:
        return "register_model"
    return "notify_failure"


with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="House Rent ML: prepare -> train -> evaluate -> conditional register",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "training"],
) as dag:

    # 1. Sensor: wait for raw data to be available
    check_data = FileSensor(
        task_id="check_data",
        filepath=os.path.join(PROJECT_DIR, RAW_DATA_FILE),
        poke_interval=30,
        timeout=120,
        mode="poke",
    )

    # 2. Data preparation (DVC prepare stage)
    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f"python src/prepare.py {RAW_DATA_FILE} data/prepared"
        ),
    )

    # 3. Model training
    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/train.py data/prepared data/models"
        ),
    )

    # 4. Evaluation: produces metrics.json + confusion_matrix.png
    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/evaluate.py data/models/model.pkl data/prepared/test.csv"
        ),
    )

    # 5. Branching decision based on R² threshold
    branch_on_metrics = BranchPythonOperator(
        task_id="branch_on_metrics",
        python_callable=_check_model_quality,
    )

    # 6a. Register model in MLflow (quality gate passed)
    register_model = BashOperator(
        task_id="register_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/register_model.py"
        ),
    )

    # 6b. Notify that model did not pass threshold
    notify_failure = BashOperator(
        task_id="notify_failure",
        bash_command=(
            'echo "Model did NOT meet quality threshold '
            f'(R² < {R2_THRESHOLD}). Skipping registration."'
        ),
    )

    # 7. Join node — succeeds if either branch completes
    pipeline_end = EmptyOperator(
        task_id="pipeline_end",
        trigger_rule="none_failed_min_one_success",
    )

    # DAG wiring
    (
        check_data
        >> prepare_data
        >> train_model
        >> evaluate_model
        >> branch_on_metrics
        >> [register_model, notify_failure]
    )
    register_model >> pipeline_end
    notify_failure >> pipeline_end
