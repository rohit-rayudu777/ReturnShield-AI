"""
tests/test_model.py

Unit tests for ReturnShield AI — Phase 2C model module.

All tests use small synthetic mock data (not the full 100k dataset) to keep
the test suite fast and isolated.

Tests verify:
  - Training completes successfully
  - predict() returns correct shape and valid classes
  - predict_proba() values are in [0, 1]
  - predict_proba() output shape is (n_samples, 2)
  - Saved artifact can be reloaded from disk
  - Loaded model produces identical predictions to in-memory model
  - No NaN or Inf in model input or output
  - Target column is NOT present in feature matrix (leakage check)
  - predict_risk_score() returns a float in [0, 1]
"""

import os
import tempfile
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import pytest

from src.features.feature_engineering import FeatureEngineer
from src.model.model import (
    train_model,
    evaluate_model,
    save_artifacts,
    load_artifacts,
    predict_risk_score,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_raw_df():
    """
    A small (30-row) mock DataFrame with the same schema as data/returns.csv.
    Includes is_abusive_return so it can be split for training.
    """
    np.random.seed(42)
    n = 30
    return pd.DataFrame({
        "order_id":                  [f"ORD_{i}" for i in range(n)],
        "customer_id":               [f"CUST_{i % 5}" for i in range(n)],
        "timestamp":                 pd.date_range("2025-01-01", periods=n, freq="D")
                                     .strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "order_amount":              np.random.uniform(20, 300, n).round(2),
        "product_category":          np.random.choice(
                                         ["Electronics", "Clothing", "Beauty", "Home", "Books"], n),
        "payment_method":            np.random.choice(
                                         ["Credit Card", "Debit Card", "UPI", "Netbanking", "COD"], n),
        "customer_age_days":         np.random.randint(30, 600, n),
        "previous_orders":           np.random.randint(0, 20, n),
        "previous_returns":          np.random.randint(0, 10, n),
        "customer_return_rate":      np.random.uniform(0, 1, n).round(4),
        "orders_last_7_days":        np.random.randint(0, 5, n),
        "orders_last_30_days":       np.random.randint(0, 15, n),
        "returns_last_7_days":       np.random.randint(0, 4, n),
        "returns_last_30_days":      np.random.randint(0, 10, n),
        "average_order_value":       np.random.uniform(0, 200, n).round(2),
        "discount_percentage":       np.random.uniform(0, 70, n).round(1),
        "delivery_days":             np.random.randint(1, 10, n),
        "return_days_after_delivery":np.random.randint(0, 30, n),
        "address_change_count":      np.random.randint(0, 5, n),
        "payment_failures":          np.random.randint(0, 5, n),
        "previous_chargebacks":      np.random.randint(0, 3, n),
        "is_first_order":            np.random.randint(0, 2, n),
        "is_high_value_order":       np.random.randint(0, 2, n),
        "is_abusive_return":         np.random.randint(0, 2, n),
    })


@pytest.fixture
def prepared_data(mock_raw_df):
    """
    Fits a FeatureEngineer on the mock data and returns X, y, and the fitted fe.
    Uses the full mock set for both train and test (small data — unit test only).
    """
    fe = FeatureEngineer()
    X = fe.fit_transform(mock_raw_df)
    y = mock_raw_df["is_abusive_return"].copy()
    return X, y, fe


@pytest.fixture
def trained_model_and_data(prepared_data):
    """Returns a trained model pipeline alongside X, y, and fe."""
    X, y, fe = prepared_data
    model = train_model(X, y)
    return model, X, y, fe


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_training_completes(prepared_data):
    """Training must complete without raising an exception."""
    X, y, _ = prepared_data
    model = train_model(X, y)
    assert model is not None


def test_predict_shape_and_classes(trained_model_and_data):
    """predict() must return an array of shape (n_samples,) with values in {0, 1}."""
    model, X, y, _ = trained_model_and_data
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_predict_proba_shape(trained_model_and_data):
    """predict_proba() must return shape (n_samples, 2)."""
    model, X, y, _ = trained_model_and_data
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_predict_proba_values_in_range(trained_model_and_data):
    """All probability values must be in [0, 1]."""
    model, X, y, _ = trained_model_and_data
    proba = model.predict_proba(X)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_predict_proba_rows_sum_to_one(trained_model_and_data):
    """Each row of predict_proba() must sum to 1.0 (within floating-point tolerance)."""
    model, X, y, _ = trained_model_and_data
    proba = model.predict_proba(X)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_no_nan_or_inf_in_features(prepared_data):
    """Feature matrix must not contain NaN or Inf values."""
    X, _, _ = prepared_data
    assert not X.isnull().any().any(), "NaN found in feature matrix"
    assert not np.isinf(X.values).any(), "Inf found in feature matrix"


def test_no_nan_or_inf_in_predictions(trained_model_and_data):
    """Predictions and probabilities must not contain NaN or Inf values."""
    model, X, y, _ = trained_model_and_data
    preds = model.predict(X)
    proba = model.predict_proba(X)
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()
    assert not np.isnan(proba).any()
    assert not np.isinf(proba).any()


def test_target_not_in_feature_matrix(mock_raw_df):
    """
    Target column 'is_abusive_return' must never appear inside the
    transformed feature matrix X — leakage check.
    """
    fe = FeatureEngineer()
    X = fe.fit_transform(mock_raw_df)
    assert "is_abusive_return" not in X.columns


def test_save_and_load_artifacts(trained_model_and_data):
    """Saved artifacts must be loadable and produce identical results."""
    model, X, y, fe = trained_model_and_data

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        fe_path = os.path.join(tmpdir, "feature_engineer.joblib")

        save_artifacts(model, fe, model_path=model_path, fe_path=fe_path)

        assert os.path.exists(model_path)
        assert os.path.exists(fe_path)

        loaded_model, loaded_fe = load_artifacts(model_path=model_path, fe_path=fe_path)

    # Loaded model must produce identical predictions
    preds_original = model.predict(X)
    preds_loaded = loaded_model.predict(X)
    np.testing.assert_array_equal(preds_original, preds_loaded)

    # Loaded model must produce identical probabilities
    proba_original = model.predict_proba(X)
    proba_loaded = loaded_model.predict_proba(X)
    np.testing.assert_allclose(proba_original, proba_loaded, atol=1e-9)


def test_predict_risk_score_range(trained_model_and_data, mock_raw_df):
    """
    predict_risk_score() must return values in [0, 1].
    Raw DF must not include the target column during inference.
    """
    model, X, y, fe = trained_model_and_data

    # Remove target so it mimics a real inference request
    inference_df = mock_raw_df.drop(columns=["is_abusive_return"])

    scores = predict_risk_score(model, fe, inference_df)
    assert scores.shape == (len(inference_df),)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_predict_risk_score_rejects_target_column(trained_model_and_data, mock_raw_df):
    """
    predict_risk_score() must raise ValueError if is_abusive_return is in the
    input DataFrame (prevents accidental target leakage at inference time).
    """
    model, X, y, fe = trained_model_and_data

    with pytest.raises(ValueError, match="is_abusive_return"):
        predict_risk_score(model, fe, mock_raw_df)  # target column still present


def test_evaluate_model_returns_expected_keys(trained_model_and_data):
    """evaluate_model() must return all required metric keys."""
    model, X, y, _ = trained_model_and_data
    results = evaluate_model(model, X, y)
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"}
    assert expected_keys.issubset(results.keys())


def test_evaluate_model_metric_ranges(trained_model_and_data):
    """All scalar evaluation metrics must be in [0, 1]."""
    model, X, y, _ = trained_model_and_data
    results = evaluate_model(model, X, y)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= results[key] <= 1.0, f"{key} out of range: {results[key]}"
