"""
Model Training Pipeline
Full ML training workflow with:
- Class imbalance handling (SMOTE)
- Hyperparameter tuning (Optuna)
- Cross-validation
- MLflow experiment tracking
- SHAP explainability
- Model registration
"""

import os
import warnings
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import shap
import sqlalchemy
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Config ───────────────────────────────────────────────────────────────────

DB_URL = "postgresql://churn_user:churn_pass@localhost:5432/churn_db"
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "churn-prediction"
MODEL_NAME = "churn_predictor"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "tenure_months", "monthly_charges", "total_charges",
    "avg_monthly_usage_gb", "num_logins_last_30d", "num_features_used",
    "last_login_days_ago", "num_support_calls_3m", "num_complaints_6m",
    "avg_support_resolution_hours", "num_late_payments", "nps_score",
    "email_open_rate", "referrals_made",
    "plan_encoded", "contract_encoded", "payment_encoded",
    "charges_per_tenure", "total_vs_expected_ratio",
    "logins_per_feature", "is_dormant", "engagement_score",
    "support_distress_score", "calls_per_tenure",
    "is_promoter", "is_detractor", "is_long_tenure",
    "risk_composite", "rolling_avg_logins_30d", "rolling_avg_support_calls",
    "prev_monthly_charges", "charges_change_pct",
]
TARGET_COL = "churned"


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """Load feature store data from PostgreSQL."""
    engine = sqlalchemy.create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM customer_features", engine)
    logger.info(f"Loaded {len(df):,} rows from feature store")
    logger.info(f"Churn rate: {df[TARGET_COL].mean():.2%}")
    return df


# ─── Preprocessing ────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """Split into train/validation/test and handle class imbalance."""
    X = df[FEATURE_COLS].fillna(0)
    y = df[TARGET_COL]

    # 60% train | 20% val | 20% test — stratified
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    logger.info(f"Train churn rate: {y_train.mean():.2%}")

    # Apply SMOTE to training set only to handle class imbalance
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {len(X_train_resampled):,} training samples (balanced)")

    return X_train_resampled, X_val, X_test, y_train_resampled, y_val, y_test


# ─── Hyperparameter Tuning ────────────────────────────────────────────────────

def objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    """Optuna objective — maximise AUC-ROC on validation set."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": 1,
        "use_label_encoder": False,
        "eval_metric": "auc",
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred_proba)


def tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
    """Run Optuna hyperparameter optimisation."""
    logger.info(f"Running Optuna search with {n_trials} trials...")
    study = optuna.create_study(direction="maximize", study_name="xgb_churn")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    logger.info(f"Best AUC-ROC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params


# ─── Cross Validation ─────────────────────────────────────────────────────────

def cross_validate_model(X_train, y_train, params: dict, n_folds: int = 5) -> dict:
    """5-fold stratified cross-validation for robust performance estimates."""
    logger.info(f"Running {n_folds}-fold cross-validation...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_aucs, cv_f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)

        y_proba = model.predict_proba(X_fold_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        cv_aucs.append(roc_auc_score(y_fold_val, y_proba))
        cv_f1s.append(f1_score(y_fold_val, y_pred))

        logger.debug(f"Fold {fold+1}: AUC={cv_aucs[-1]:.4f}, F1={cv_f1s[-1]:.4f}")

    results = {
        "cv_auc_mean": np.mean(cv_aucs),
        "cv_auc_std": np.std(cv_aucs),
        "cv_f1_mean": np.mean(cv_f1s),
        "cv_f1_std": np.std(cv_f1s),
    }
    logger.info(f"CV AUC: {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
    logger.info(f"CV F1:  {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")
    return results


# ─── Model Evaluation ─────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Comprehensive model evaluation on held-out test set."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "test_auc_roc": roc_auc_score(y_test, y_proba),
        "test_avg_precision": average_precision_score(y_test, y_proba),
        "test_f1": f1_score(y_test, y_pred),
        "test_threshold": threshold,
    }

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["test_precision_churn"] = report["1"]["precision"]
    metrics["test_recall_churn"] = report["1"]["recall"]

    logger.info("=== Test Set Evaluation ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    return metrics


# ─── SHAP Explainability ──────────────────────────────────────────────────────

def compute_shap_values(model, X_val: pd.DataFrame, n_samples: int = 500) -> np.ndarray:
    """Compute SHAP values for feature importance and explainability."""
    logger.info("Computing SHAP values...")
    X_sample = X_val.sample(min(n_samples, len(X_val)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Log top 10 most important features
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    logger.info("Top 10 features by SHAP importance:")
    for _, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

    return shap_values, feature_importance


# ─── Baseline Comparison ──────────────────────────────────────────────────────

def train_baselines(X_train, y_train, X_val, y_val) -> dict:
    """Train baseline models for comparison (shows rigour to interviewers)."""
    baselines = {}

    # Logistic Regression
    lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])
    baselines["logistic_regression_auc"] = lr_auc

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    baselines["random_forest_auc"] = rf_auc

    logger.info(f"Baseline LR AUC: {lr_auc:.4f} | RF AUC: {rf_auc:.4f}")
    return baselines


# ─── Main Training Run ────────────────────────────────────────────────────────

def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_features()
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)

    # Train baselines first
    baseline_metrics = train_baselines(X_train, y_train, X_val, y_val)

    # Hyperparameter search
    best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50)

    # Cross-validation
    cv_results = cross_validate_model(X_train, y_train, best_params)

    # Final model training with MLflow tracking
    with mlflow.start_run(run_name="xgboost-optimised") as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Log params
        mlflow.log_params(best_params)
        mlflow.log_params({"smote_enabled": True, "n_features": len(FEATURE_COLS)})

        # Log CV metrics
        mlflow.log_metrics(cv_results)
        mlflow.log_metrics(baseline_metrics)

        # Train final model
        final_params = {**best_params, "random_state": 42, "n_jobs": -1}
        model = xgb.XGBClassifier(**final_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=100,
        )

        # Evaluate
        test_metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(test_metrics)

        # SHAP values
        shap_values, feature_importance = compute_shap_values(model, X_val)

        # Save feature importance CSV
        fi_path = MODELS_DIR / "feature_importance.csv"
        feature_importance.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        # Log model with input example and signature
        input_example = X_val.head(5)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=input_example,
        )

        # Save model locally
        model_path = MODELS_DIR / "churn_model.json"
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

        logger.info(f"=== Training Complete ===")
        logger.info(f"Test AUC-ROC: {test_metrics['test_auc_roc']:.4f}")
        logger.info(f"Test F1:      {test_metrics['test_f1']:.4f}")
        logger.info(f"MLflow UI:    {MLFLOW_TRACKING_URI}")

    return run.info.run_id


if __name__ == "__main__":
    run_id = train()
    logger.info(f"Training complete. Run ID: {run_id}")