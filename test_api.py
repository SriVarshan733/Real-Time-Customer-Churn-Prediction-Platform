"""
Unit Tests — Churn Prediction API
Tests: input validation, feature computation, prediction endpoint, health checks.
Run with: pytest tests/unit/ -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

# ─── Fixtures ─────────────────────────────────────────────────────────────────

VALID_REQUEST = {
    "customer_id": "TEST_001",
    "tenure_months": 12,
    "monthly_charges": 75.50,
    "total_charges": 906.0,
    "plan_encoded": 1,
    "contract_encoded": 0,
    "payment_encoded": 0,
    "avg_monthly_usage_gb": 50.0,
    "num_logins_last_30d": 10,
    "num_features_used": 5,
    "last_login_days_ago": 3,
    "num_support_calls_3m": 2,
    "num_complaints_6m": 0,
    "avg_support_resolution_hours": 24.0,
    "num_late_payments": 0,
    "nps_score": 7,
    "email_open_rate": 0.45,
    "referrals_made": 1,
    "billing_issues_flag": 0,
}

HIGH_RISK_REQUEST = {
    **VALID_REQUEST,
    "customer_id": "HIGH_RISK_001",
    "num_support_calls_3m": 10,
    "num_complaints_6m": 5,
    "nps_score": 2,
    "last_login_days_ago": 45,
    "num_logins_last_30d": 1,
    "email_open_rate": 0.02,
    "num_late_payments": 4,
}


@pytest.fixture
def mock_model():
    """Create a mock XGBoost model."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.35, 0.65]])
    return model


@pytest.fixture
def mock_explainer():
    """Create a mock SHAP explainer."""
    explainer = MagicMock()
    explainer.shap_values.return_value = np.array([[0.1, -0.2, 0.3, 0.05, -0.1,
                                                    0.08, -0.15, 0.12, 0.04, 0.09,
                                                    -0.06, 0.11, -0.03, 0.07,
                                                    0.02, -0.08, 0.14, 0.01, -0.05,
                                                    0.06, 0.13, -0.04, 0.09, 0.03,
                                                    -0.07, 0.02, 0.08, -0.01, 0.05,
                                                    0.04, 0.01, -0.02]])
    return explainer


@pytest.fixture
def client(mock_model, mock_explainer):
    """Create test client with mocked model."""
    with patch("src.api.main.model_manager") as mock_manager:
        mock_manager.model = mock_model
        mock_manager.explainer = mock_explainer
        mock_manager.model_version = "test-v1"
        mock_manager.is_loaded.return_value = True

        from src.api.main import app
        with TestClient(app) as c:
            yield c


# ─── Health Check Tests ───────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_health_returns_200_when_model_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_version" in data

    def test_ready_returns_200(self, client):
        response = client.get("/ready")
        assert response.status_code == 200

    def test_health_returns_503_when_model_not_loaded(self):
        with patch("src.api.main.model_manager") as mock_manager:
            mock_manager.is_loaded.return_value = False
            from src.api.main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                response = c.get("/health")
                assert response.status_code == 503


# ─── Prediction Endpoint Tests ────────────────────────────────────────────────

class TestPredictionEndpoint:
    def test_valid_prediction_returns_200(self, client):
        response = client.post("/predict", json=VALID_REQUEST)
        assert response.status_code == 200

    def test_response_schema(self, client):
        response = client.post("/predict", json=VALID_REQUEST)
        data = response.json()
        assert "customer_id" in data
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_tier" in data
        assert "top_risk_factors" in data
        assert "latency_ms" in data
        assert "model_version" in data

    def test_customer_id_preserved(self, client):
        response = client.post("/predict", json=VALID_REQUEST)
        assert response.json()["customer_id"] == "TEST_001"

    def test_probability_in_valid_range(self, client):
        response = client.post("/predict", json=VALID_REQUEST)
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_risk_tier_valid_values(self, client):
        response = client.post("/predict", json=VALID_REQUEST)
        tier = response.json()["risk_tier"]
        assert tier in ["low", "medium", "high", "critical"]

    def test_top_risk_factors_structure(self, client):
        response = client.post("/predict", json=VALID_REQUEST)
        factors = response.json()["top_risk_factors"]
        assert len(factors) > 0
        for factor in factors:
            assert "feature" in factor
            assert "impact" in factor
            assert "direction" in factor
            assert factor["direction"] in ["increases_churn_risk", "decreases_churn_risk"]

    def test_high_probability_yields_high_tier(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        response = client.post("/predict", json=HIGH_RISK_REQUEST)
        assert response.json()["risk_tier"] in ["high", "critical"]

    def test_low_probability_yields_low_tier(self, client, mock_model):
        mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])
        response = client.post("/predict", json=VALID_REQUEST)
        assert response.json()["risk_tier"] == "low"


# ─── Input Validation Tests ───────────────────────────────────────────────────

class TestInputValidation:
    def test_missing_required_field(self, client):
        bad_request = {k: v for k, v in VALID_REQUEST.items() if k != "tenure_months"}
        response = client.post("/predict", json=bad_request)
        assert response.status_code == 422

    def test_negative_tenure_rejected(self, client):
        bad_request = {**VALID_REQUEST, "tenure_months": -1}
        response = client.post("/predict", json=bad_request)
        assert response.status_code == 422

    def test_nps_out_of_range_rejected(self, client):
        bad_request = {**VALID_REQUEST, "nps_score": 11}
        response = client.post("/predict", json=bad_request)
        assert response.status_code == 422

    def test_invalid_email_open_rate_rejected(self, client):
        bad_request = {**VALID_REQUEST, "email_open_rate": 1.5}
        response = client.post("/predict", json=bad_request)
        assert response.status_code == 422

    def test_empty_customer_id_rejected(self, client):
        bad_request = {**VALID_REQUEST, "customer_id": ""}
        response = client.post("/predict", json=bad_request)
        assert response.status_code == 422


# ─── Batch Prediction Tests ───────────────────────────────────────────────────

class TestBatchPrediction:
    def test_batch_of_two(self, client):
        batch = [VALID_REQUEST, {**VALID_REQUEST, "customer_id": "TEST_002"}]
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200
        assert response.json()["total"] == 2

    def test_batch_exceeds_limit(self, client):
        batch = [VALID_REQUEST] * 1001
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 400


# ─── Feature Engineering Tests ────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_derived_features_computed(self):
        from src.api.main import compute_derived_features, ChurnPredictionRequest
        req = ChurnPredictionRequest(**VALID_REQUEST)
        features_df = compute_derived_features(req)
        assert "engagement_score" in features_df.columns
        assert "support_distress_score" in features_df.columns
        assert "risk_composite" in features_df.columns
        assert "charges_per_tenure" in features_df.columns

    def test_dormant_flag_set_correctly(self):
        from src.api.main import compute_derived_features, ChurnPredictionRequest
        dormant_req = ChurnPredictionRequest(**{**VALID_REQUEST, "last_login_days_ago": 45})
        features_df = compute_derived_features(dormant_req)
        assert features_df["is_dormant"].iloc[0] == 1

    def test_promoter_flag_set_correctly(self):
        from src.api.main import compute_derived_features, ChurnPredictionRequest
        req = ChurnPredictionRequest(**{**VALID_REQUEST, "nps_score": 9})
        features_df = compute_derived_features(req)
        assert features_df["is_promoter"].iloc[0] == 1

    def test_no_null_features(self):
        from src.api.main import compute_derived_features, ChurnPredictionRequest
        req = ChurnPredictionRequest(**VALID_REQUEST)
        features_df = compute_derived_features(req)
        assert features_df.isnull().sum().sum() == 0
