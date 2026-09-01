"""
tests/test_api.py

Integration tests for the ReturnShield AI FastAPI backend.
Verifies health check status, prediction endpoints, strict schema validation rules
(blocking target leakage), and defense-only decision rekomendations.
"""

import pytest
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient

from src.api.main import app


# ---------------------------------------------------------------------------
# Test fixtures and clients
# ---------------------------------------------------------------------------
@pytest.fixture
def api_client():
    """
    FastAPI TestClient helper.
    Using app lifespan context manager ensures ML artifacts load properly on startup.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture
def valid_request_payload():
    """
    A valid transaction feature payload based on returns.csv schema.
    Contains only features available at inference time (excludes target).
    """
    return {
        "order_id": "ORD_123456_0",
        "customer_id": "CUST_000001",
        "timestamp": "2025-06-15 14:30:00",
        "order_amount": 249.99,
        "product_category": "Electronics",
        "payment_method": "COD",
        "customer_age_days": 45,
        "previous_orders": 2,
        "previous_returns": 2,
        "customer_return_rate": 1.0,
        "orders_last_7_days": 3,
        "orders_last_30_days": 5,
        "returns_last_7_days": 2,
        "returns_last_30_days": 3,
        "average_order_value": 120.0,
        "discount_percentage": 10.5,
        "delivery_days": 3,
        "return_days_after_delivery": 1,
        "address_change_count": 2,
        "payment_failures": 1,
        "previous_chargebacks": 1,
        "is_first_order": 0,
        "is_high_value_order": 1,
        "review_threshold": 0.30
    }


# ---------------------------------------------------------------------------
# Health endpoint checks
# ---------------------------------------------------------------------------
def test_health_check_status(api_client):
    """
    Checks GET /health status 200, checks model loaded is True, status is 'ok'.
    """
    response = api_client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model" in data["artifacts"]
    assert "feature_engineer" in data["artifacts"]
    assert "explainer" in data["artifacts"]


# ---------------------------------------------------------------------------
# Prediction endpoint checks
# ---------------------------------------------------------------------------
def test_prediction_endpoint_valid_run(api_client, valid_request_payload):
    """
    Verifies that POST /predict with a valid payload returns 200 and matches
    the structured PredictionResponse schema parameters.
    """
    response = api_client.post("/predict", json=valid_request_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction_id" in data
    assert data["order_id"] == valid_request_payload["order_id"]
    assert data["customer_id"] == valid_request_payload["customer_id"]
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert 0 <= data["risk_score"] <= 100
    assert data["risk_band"] in ["Low", "Medium", "High", "Very High"]
    assert data["decision"] in ["ALLOW", "REVIEW"]
    assert isinstance(data["summary"], str)
    assert isinstance(data["positive_factors"], list)
    assert isinstance(data["negative_factors"], list)
    assert isinstance(data["base_value_log_odds"], float)
    
    # Check that OHE mappings are clean
    for factor in data["positive_factors"] + data["negative_factors"]:
        assert "feature" in factor
        assert "display_name" in factor
        assert "value" in factor
        assert "contribution" in factor
        assert "direction" in factor
        # Confirm raw OHE variable names are not leaked to customer response
        assert "product_category_" not in factor["feature"]
        assert "payment_method_" not in factor["feature"]


def test_prediction_validation_missing_fields(api_client, valid_request_payload):
    """
    Checks that POST /predict with missing required fields raises a validation
    error and returns HTTP 422.
    """
    payload = valid_request_payload.copy()
    del payload["order_amount"]  # missing required numeric
    del payload["product_category"]  # missing required categorical
    
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_validation_invalid_types(api_client, valid_request_payload):
    """
    Checks that POST /predict with incorrect data types returns HTTP 422.
    """
    payload = valid_request_payload.copy()
    payload["order_amount"] = "one hundred rupees"  # string instead of float
    payload["previous_orders"] = -5  # invalid negative range
    
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_validation_invalid_categories(api_client, valid_request_payload):
    """
    Checks that POST /predict with unexpected string categories returns HTTP 422.
    """
    payload = valid_request_payload.copy()
    payload["product_category"] = "Automotive"  # not in Literals list
    payload["payment_method"] = "Bitcoin"  # not in Literals list
    
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_blocks_target_leakage(api_client, valid_request_payload):
    """
    Checks that sending the target column 'is_abusive_return' or any other
    extra field in the request body is rejected with a 422 Unprocessable Entity,
    enforcing target isolation.
    """
    payload = valid_request_payload.copy()
    payload["is_abusive_return"] = 1  # Target leakage field
    
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422
    
    data = response.json()
    assert "extra_forbidden" in str(data) or "is_abusive_return" in str(data)


def test_prediction_blocks_unauthorized_extra_fields(api_client, valid_request_payload):
    """
    Checks that sending arbitrary extra fields in the request body is rejected
    with 422 validation error.
    """
    payload = valid_request_payload.copy()
    payload["fraudulent_chargebacks_flag"] = True  # Extra field
    
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_defense_only_recommendation(api_client, valid_request_payload):
    """
    Verifies that the API remains defense-only: decisions must recommend REVIEW
    or ALLOW outcomes, never automatically deny checkout or reject returns.
    """
    response = api_client.post("/predict", json=valid_request_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["decision"] in ["ALLOW", "REVIEW"]
    # Ensure no automated enforcement fields (like block_customer, reject_transaction) are returned
    assert "block_customer" not in data
    assert "deny_return" not in data
