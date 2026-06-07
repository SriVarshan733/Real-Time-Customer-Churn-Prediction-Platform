"""
Airflow DAG — Automated Churn Model Retraining
Runs nightly: feature engineering → model training → evaluation → deployment gate.
If new model beats production AUC by 0.5%, auto-promotes to Production in MLflow.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule

# ─── Default Args ─────────────────────────────────────────────────────────────

default_args = {
    "owner": "data-science-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email": ["alerts@company.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=3),
}


# ─── Task Functions ───────────────────────────────────────────────────────────

def run_feature_engineering(**context):
    """Execute PySpark feature engineering pipeline."""
    from loguru import logger
    import subprocess

    logger.info("Starting feature engineering pipeline...")
    result = subprocess.run(
        ["python", "/app/src/features/feature_pipeline.py"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Feature pipeline failed:\n{result.stderr}")

    logger.info("Feature engineering complete")
    context["ti"].xcom_push(key="feature_count", value=result.stdout.split("records")[-2].strip().split()[-1])


def run_model_training(**context):
    """Train new candidate model and return MLflow run ID."""
    import sys
    sys.path.insert(0, "/app")
    from src.training.train import train

    run_id = train()
    context["ti"].xcom_push(key="new_run_id", value=run_id)
    return run_id


def evaluate_and_gate(**context):
    """
    Compare new model AUC vs current Production model.
    Returns 'promote_model' if new model is better, else 'skip_promotion'.
    """
    import mlflow
    from loguru import logger

    ti = context["ti"]
    new_run_id = ti.xcom_pull(task_ids="train_model", key="new_run_id")

    mlflow.set_tracking_uri("http://mlflow:5000")
    client = mlflow.MlflowClient()

    # Get new model's AUC
    new_run = client.get_run(new_run_id)
    new_auc = new_run.data.metrics.get("test_auc_roc", 0)

    # Get production model's AUC (stored as tag)
    try:
        prod_versions = client.get_latest_versions("churn_predictor", stages=["Production"])
        if prod_versions:
            prod_run_id = prod_versions[0].run_id
            prod_run = client.get_run(prod_run_id)
            prod_auc = prod_run.data.metrics.get("test_auc_roc", 0)
        else:
            prod_auc = 0.0
    except Exception:
        prod_auc = 0.0

    logger.info(f"New model AUC: {new_auc:.4f} | Production AUC: {prod_auc:.4f}")
    context["ti"].xcom_push(key="new_auc", value=new_auc)
    context["ti"].xcom_push(key="prod_auc", value=prod_auc)

    # Gate: promote only if improvement >= 0.5%
    if new_auc >= prod_auc + 0.005:
        logger.info("New model is better — promoting to Production")
        return "promote_model"
    else:
        logger.info("New model not significantly better — keeping current Production")
        return "skip_promotion"


def promote_model(**context):
    """Transition new model version to Production in MLflow registry."""
    import mlflow
    from loguru import logger

    ti = context["ti"]
    new_run_id = ti.xcom_pull(task_ids="train_model", key="new_run_id")

    client = mlflow.MlflowClient()

    # Find the model version for this run
    versions = client.search_model_versions(f"run_id='{new_run_id}'")
    if not versions:
        raise RuntimeError(f"No model version found for run {new_run_id}")

    new_version = versions[0].version

    # Archive current Production
    prod_versions = client.get_latest_versions("churn_predictor", stages=["Production"])
    for v in prod_versions:
        client.transition_model_version_stage(
            name="churn_predictor",
            version=v.version,
            stage="Archived",
        )

    # Promote new version
    client.transition_model_version_stage(
        name="churn_predictor",
        version=new_version,
        stage="Production",
        archive_existing_versions=False,
    )

    logger.info(f"Model v{new_version} promoted to Production")
    context["ti"].xcom_push(key="promoted_version", value=new_version)


def check_data_drift(**context):
    """
    Check for feature distribution drift using PSI (Population Stability Index).
    Alerts if PSI > 0.2 for any feature.
    """
    import pandas as pd
    import numpy as np
    import sqlalchemy
    from loguru import logger

    engine = sqlalchemy.create_engine("postgresql://churn_user:churn_pass@postgres:5432/churn_db")

    # Compare last 7 days vs previous 30 days
    recent = pd.read_sql("""
        SELECT * FROM customer_features
        WHERE event_timestamp >= NOW() - INTERVAL '7 days'
    """, engine)

    baseline = pd.read_sql("""
        SELECT * FROM customer_features
        WHERE event_timestamp BETWEEN NOW() - INTERVAL '37 days' AND NOW() - INTERVAL '7 days'
    """, engine)

    numeric_cols = ["monthly_charges", "tenure_months", "num_support_calls_3m",
                    "nps_score", "email_open_rate", "engagement_score"]

    drift_results = {}
    for col in numeric_cols:
        if col in recent.columns and col in baseline.columns:
            # PSI calculation
            expected = np.histogram(baseline[col].dropna(), bins=10)[0] / len(baseline)
            actual = np.histogram(recent[col].dropna(), bins=10)[0] / len(recent)
            expected = np.where(expected == 0, 0.0001, expected)
            actual = np.where(actual == 0, 0.0001, actual)
            psi = np.sum((actual - expected) * np.log(actual / expected))
            drift_results[col] = round(psi, 4)

    high_drift = {k: v for k, v in drift_results.items() if v > 0.2}

    if high_drift:
        logger.warning(f"HIGH DRIFT DETECTED: {high_drift}")
        # In production: send Slack/email alert
    else:
        logger.info(f"No significant drift detected. PSI values: {drift_results}")

    context["ti"].xcom_push(key="drift_results", value=drift_results)


def send_slack_report(**context):
    """Send daily retraining summary to Slack."""
    ti = context["ti"]
    new_auc = ti.xcom_pull(task_ids="evaluate_model", key="new_auc")
    prod_auc = ti.xcom_pull(task_ids="evaluate_model", key="prod_auc")
    drift = ti.xcom_pull(task_ids="check_data_drift", key="drift_results")

    message = f"""
    *Churn Model — Daily Retraining Report* ✅
    - New model AUC: `{new_auc:.4f}`
    - Production AUC: `{prod_auc:.4f}`
    - Drift PSI: `{drift}`
    - Date: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}`
    """
    # Use requests.post to hit your Slack webhook URL
    import os, requests
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if webhook:
        requests.post(webhook, json={"text": message})


# ─── DAG Definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="churn_model_retraining",
    default_args=default_args,
    description="Nightly churn model retraining and promotion pipeline",
    schedule_interval="0 2 * * *",         # 2 AM UTC every night
    catchup=False,
    max_active_runs=1,
    tags=["ml", "churn", "production"],
) as dag:

    start = DummyOperator(task_id="start")

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_feature_engineering,
    )

    drift_check = PythonOperator(
        task_id="check_data_drift",
        python_callable=check_data_drift,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_model_training,
    )

    evaluate_model = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_and_gate,
    )

    promote_model_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    skip_promotion = DummyOperator(task_id="skip_promotion")

    send_report = PythonOperator(
        task_id="send_report",
        python_callable=send_slack_report,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    end = DummyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # ─── DAG Dependencies ──────────────────────────────────────────────────────
    (
        start
        >> [feature_engineering, drift_check]
        >> train_model
        >> evaluate_model
        >> [promote_model_task, skip_promotion]
        >> send_report
        >> end
    )