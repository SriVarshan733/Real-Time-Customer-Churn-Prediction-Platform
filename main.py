"""
FastAPI Prediction Service
Production-grade REST API for real-time churn prediction.
Includes: input validation, Prometheus metrics, health checks,
model loading from MLflow, request logging, SHAP explanations.
"""

import time
import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, validator
from starlette.responses import Response

# ─── Config ───────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "churn_predictor")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/churn_model.json")

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


# ─── Prometheus Metrics ───────────────────────────────────────────────────────

PREDICT_REQUESTS = Counter(
    "churn_prediction_requests_total",
    "Total prediction requests",
    ["status"]
)
PREDICT_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Prediction latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
CHURN_PROBABILITY = Histogram(
    "churn_probability_distribution",
    "Distribution of predicted churn probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
HIGH_RISK_COUNTER = Counter(
    "high_risk_customers_total",
    "Customers predicted with >70% churn probability"
)
MODEL_VERSION = Gauge(
    "model_version_loaded",
    "Version of the model currently loaded"
)


# ─── Request/Response Schemas ─────────────────────────────────────────────────

class ChurnPredictionRequest(BaseModel):
    """Input schema with validation. All fields map to model features."""
    customer_id: str = Field(..., description="Unique customer identifier")

    # Contract features
    tenure_months: int = Field(..., ge=0, le=120, description="Months as a customer")
    monthly_charges: float = Field(..., ge=0, le=1000, description="Monthly bill amount (£)")
    total_charges: float = Field(..., ge=0, description="Total spend to date (£)")
    plan_encoded: int = Field(..., ge=0, le=3, description="Plan tier: 0=Basic, 1=Standard, 2=Premium, 3=Enterprise")
    contract_encoded: int = Field(..., ge=0, le=2, description="Contract: 0=monthly, 1=annual, 2=biennial")
    payment_encoded: int = Field(..., ge=0, le=3, description="Payment: 0=credit, 1=bank, 2=paypal, 3=direct_debit")

    # Usage features
    avg_monthly_usage_gb: float = Field(default=50.0, ge=0)
    num_logins_last_30d: int = Field(default=10, ge=0)
    num_features_used: int = Field(default=5, ge=0, le=50)
    last_login_days_ago: int = Field(default=3, ge=0, le=365)

    # Support features
    num_support_calls_3m: int = Field(default=0, ge=0, le=100)
    num_complaints_6m: int = Field(default=0, ge=0, le=50)
    avg_support_resolution_hours: float = Field(default=24.0, ge=0)
    num_late_payments: int = Field(default=0, ge=0, le=50)

    # Satisfaction
    nps_score: int = Field(default=7, ge=0, le=10)
    email_open_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    referrals_made: int = Field(default=0, ge=0)

    # Billing
    billing_issues_flag: int = Field(default=0, ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "C_42091",
                "tenure_months": 8,
                "monthly_charges": 75.50,
                "total_charges": 604.0,
                "plan_encoded": 1,
                "contract_encoded": 0,
                "payment_encoded": 2,
                "num_support_calls_3m": 4,
                "num_complaints_6m": 2,
                "nps_score": 5,
                "last_login_days_ago": 21,
                "num_logins_last_30d": 2,
                "email_open_rate": 0.08,
            }
        }


class ChurnPredictionResponse(BaseModel):
    """Prediction output with probability, risk tier, and SHAP explanation."""
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_tier: str                          # low | medium | high | critical
    top_risk_factors: list[dict]            # SHAP-based explanations
    model_version: str
    latency_ms: float


# ─── Model Loader ─────────────────────────────────────────────────────────────

class ModelManager:
    """Handles model loading and inference. Singleton for app lifetime."""

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.model_version: str = "unknown"

    def load(self):
        """Load model from MLflow registry or fall back to local file."""
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            self.model = mlflow.xgboost.load_model(model_uri)
            self.model_version = MODEL_STAGE
            logger.info(f"Model loaded from MLflow: {model_uri}")
        except Exception as e:
            logger.warning(f"MLflow load failed ({e}), loading local model...")
            self.model = xgb.XGBClassifier()
            self.model.load_model(LOCAL_MODEL_PATH)
            self.model_version = "local"
            logger.info(f"Model loaded from {LOCAL_MODEL_PATH}")

        self.explainer = shap.TreeExplainer(self.model)
        MODEL_VERSION.set(1 if self.model_version == MODEL_STAGE else 0)
        logger.info("Model and SHAP explainer ready")

    def is_loaded(self) -> bool:
        return self.model is not None


model_manager = ModelManager()


# ─── Feature Computation ──────────────────────────────────────────────────────

def compute_derived_features(req: ChurnPredictionRequest) -> pd.DataFrame:
    """Compute all derived features from raw API inputs."""
    raw = req.dict()

    # Derived features (same logic as feature_pipeline.py)
    row = {
        # Raw features
        "tenure_months": raw["tenure_months"],
        "monthly_charges": raw["monthly_charges"],
        "total_charges": raw["total_charges"],
        "avg_monthly_usage_gb": raw["avg_monthly_usage_gb"],
        "num_logins_last_30d": raw["num_logins_last_30d"],
        "num_features_used": raw["num_features_used"],
        "last_login_days_ago": raw["last_login_days_ago"],
        "num_support_calls_3m": raw["num_support_calls_3m"],
        "num_complaints_6m": raw["num_complaints_6m"],
        "avg_support_resolution_hours": raw["avg_support_resolution_hours"],
        "num_late_payments": raw["num_late_payments"],
        "nps_score": raw["nps_score"],
        "email_open_rate": raw["email_open_rate"],
        "referrals_made": raw["referrals_made"],
        "plan_encoded": raw["plan_encoded"],
        "contract_encoded": raw["contract_encoded"],
        "payment_encoded": raw["payment_encoded"],

        # Derived
        "charges_per_tenure": raw["monthly_charges"] / (raw["tenure_months"] + 1),
        "total_vs_expected_ratio": raw["total_charges"] / (raw["monthly_charges"] * raw["tenure_months"] + 1),
        "logins_per_feature": raw["num_logins_last_30d"] / (raw["num_features_used"] + 1),
        "is_dormant": int(raw["last_login_days_ago"] > 30),
        "engagement_score": (
            raw["num_logins_last_30d"] * 0.4 +
            raw["num_features_used"] * 0.3 +
            raw["email_open_rate"] * 10 * 0.3
        ),
        "support_distress_score": (
            raw["num_support_calls_3m"] * 2 +
            raw["num_complaints_6m"] * 3 +
            raw["billing_issues_flag"] * 2 +
            raw["num_late_payments"]
        ),
        "calls_per_tenure": raw["num_support_calls_3m"] / (raw["tenure_months"] + 1),
        "is_promoter": int(raw["nps_score"] >= 9),
        "is_detractor": int(raw["nps_score"] <= 6),
        "is_long_tenure": int(raw["tenure_months"] >= 24),
        "risk_composite": (
            raw["num_support_calls_3m"] * 0.25 +
            raw["num_complaints_6m"] * 0.20 +
            (1 - raw["email_open_rate"]) * 0.15 +
            (raw["last_login_days_ago"] / 90) * 0.20 +
            raw["num_late_payments"] * 0.20
        ),
        # Window features — use raw as approximation for real-time
        "rolling_avg_logins_30d": raw["num_logins_last_30d"],
        "rolling_avg_support_calls": raw["num_support_calls_3m"],
        "prev_monthly_charges": raw["monthly_charges"],
        "charges_change_pct": 0.0,
    }

    return pd.DataFrame([row])[FEATURE_COLS]


def get_risk_tier(probability: float) -> str:
    if probability < 0.25:
        return "low"
    elif probability < 0.50:
        return "medium"
    elif probability < 0.75:
        return "high"
    else:
        return "critical"


def get_top_risk_factors(features_df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Return top-N SHAP-based risk factors in human-readable form."""
    shap_vals = model_manager.explainer.shap_values(features_df)[0]
    feature_impacts = sorted(
        zip(FEATURE_COLS, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:n]

    return [
        {
            "feature": feat,
            "impact": round(float(impact), 4),
            "direction": "increases_churn_risk" if impact > 0 else "decreases_churn_risk",
            "value": round(float(features_df[feat].iloc[0]), 4),
        }
        for feat, impact in feature_impacts
    ]


# ─── App Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    logger.info("Starting Churn Prediction API...")
    model_manager.load()
    yield
    logger.info("Shutting down API...")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time ML-powered churn prediction with SHAP explanations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware ───────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({duration*1000:.1f}ms)"
    )
    return response


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Operations"])
async def health_check():
    """Kubernetes/ECS liveness probe."""
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_version": model_manager.model_version,
        "model_name": MODEL_NAME,
    }


@app.get("/ready", tags=["Operations"])
async def readiness_check():
    """Kubernetes/ECS readiness probe."""
    return {"status": "ready"}


@app.get("/metrics", tags=["Operations"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    tags=["Prediction"],
    summary="Predict customer churn probability",
)
async def predict_churn(request: ChurnPredictionRequest):
    """
    Predict churn probability for a single customer.
    Returns probability, risk tier, and top SHAP-based risk factors.
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not ready")

    start_time = time.time()

    try:
        # Feature computation
        features_df = compute_derived_features(request)

        # Inference
        churn_proba = float(model_manager.model.predict_proba(features_df)[0][1])
        churn_prediction = churn_proba >= 0.5
        risk_tier = get_risk_tier(churn_proba)

        # SHAP explanations
        top_factors = get_top_risk_factors(features_df)

        latency_ms = (time.time() - start_time) * 1000

        # Prometheus metrics
        PREDICT_REQUESTS.labels(status="success").inc()
        PREDICT_LATENCY.observe(latency_ms / 1000)
        CHURN_PROBABILITY.observe(churn_proba)
        if churn_proba > 0.7:
            HIGH_RISK_COUNTER.inc()

        logger.info(
            f"customer={request.customer_id} | "
            f"prob={churn_proba:.3f} | "
            f"tier={risk_tier} | "
            f"latency={latency_ms:.1f}ms"
        )

        return ChurnPredictionResponse(
            customer_id=request.customer_id,
            churn_probability=round(churn_proba, 4),
            churn_prediction=churn_prediction,
            risk_tier=risk_tier,
            top_risk_factors=top_factors,
            model_version=model_manager.model_version,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        PREDICT_REQUESTS.labels(status="error").inc()
        logger.error(f"Prediction error for {request.customer_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(requests: list[ChurnPredictionRequest]):
    """Batch prediction endpoint — up to 1000 customers per request."""
    if len(requests) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 requests per batch")

    results = []
    for req in requests:
        try:
            result = await predict_churn(req)
            results.append(result)
        except Exception as e:
            results.append({"customer_id": req.customer_id, "error": str(e)})

    return {"predictions": results, "total": len(results)}


@app.get("/model/info", tags=["Operations"])
async def model_info():
    """Return metadata about the currently loaded model."""
    return {
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_version": model_manager.model_version,
        "n_features": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
    }