"""
tests/test_explainability.py

Unit tests for Phase 2E - Explainability module.

Tests only essential functionality:
- Explanation can be generated for a valid raw inference record.
- Risk probability is between 0 and 1.
- Risk score is between 0 and 100.
- Risk band is valid.
- Positive/negative contributors are returned.
- Categorical features map back to raw columns and active categories.
- Explanation does not require target column is_abusive_return.
- Target column present in inference raw data is rejected.
"""

import numpy as np
import pandas as pd
import pytest
# pyrefly: ignore [missing-import]
import shap

from src.features.feature_engineering import FeatureEngineer
from src.model.model import train_model
from src.model.explainability import explain_prediction, get_explainer


@pytest.fixture
def mock_pipeline_and_data():
    """
    Sets up a small mock training dataset and model pipeline.
    """
    np.random.seed(42)
    n = 20
    
    raw_df = pd.DataFrame({
        "order_amount":              np.random.uniform(20, 200, n).round(2),
        "product_category":          np.random.choice(["Electronics", "Clothing", "Beauty"], n),
        "payment_method":            np.random.choice(["Credit Card", "UPI", "COD"], n),
        "customer_age_days":         np.random.randint(30, 300, n),
        "previous_orders":           np.random.randint(0, 10, n),
        "previous_returns":          np.random.randint(0, 5, n),
        "customer_return_rate":      np.random.uniform(0, 1, n).round(4),
        "orders_last_7_days":        np.random.randint(0, 3, n),
        "orders_last_30_days":       np.random.randint(0, 8, n),
        "returns_last_7_days":       np.random.randint(0, 2, n),
        "returns_last_30_days":      np.random.randint(0, 5, n),
        "average_order_value":       np.random.uniform(0, 100, n).round(2),
        "discount_percentage":       np.random.uniform(0, 50, n).round(1),
        "delivery_days":             np.random.randint(1, 5, n),
        "return_days_after_delivery":np.random.randint(0, 15, n),
        "address_change_count":      np.random.randint(0, 3, n),
        "payment_failures":          np.random.randint(0, 3, n),
        "previous_chargebacks":      np.random.randint(0, 2, n),
        "is_first_order":            np.random.randint(0, 2, n),
        "is_high_value_order":       np.random.randint(0, 2, n),
        "is_abusive_return":         np.random.randint(0, 2, n),
    })
    
    fe = FeatureEngineer()
    X = fe.fit_transform(raw_df)
    y = raw_df["is_abusive_return"].copy()
    model = train_model(X, y)
    
    return model, fe, raw_df


def test_explain_prediction_valid_run(mock_pipeline_and_data):
    """
    Verifies that explain_prediction executes without error and outputs
    all required scoring and explanation dictionary keys.
    """
    model, fe, raw_df = mock_pipeline_and_data
    explainer = get_explainer(model, X_train_sample=fe.transform(raw_df))
    
    # Take a single raw inference row (must not contain the target column)
    single_raw = raw_df.head(1).drop(columns=["is_abusive_return"]).copy()
    
    exp = explain_prediction(single_raw, model, fe, explainer)
    
    assert isinstance(exp, dict)
    assert "probability" in exp
    assert "risk_score" in exp
    assert "risk_band" in exp
    assert "recommendation" in exp
    assert "summary" in exp
    assert "positive_factors" in exp
    assert "negative_factors" in exp
    assert "base_value_log_odds" in exp


def test_explanation_metric_boundaries(mock_pipeline_and_data):
    """
    Verifies metric output formats and risk boundaries.
    """
    model, fe, raw_df = mock_pipeline_and_data
    explainer = get_explainer(model, X_train_sample=fe.transform(raw_df))
    single_raw = raw_df.head(1).drop(columns=["is_abusive_return"]).copy()
    
    exp = explain_prediction(single_raw, model, fe, explainer)
    
    assert 0.0 <= exp["probability"] <= 1.0
    assert 0 <= exp["risk_score"] <= 100
    assert exp["risk_band"] in ["Low", "Medium", "High", "Very High"]
    assert exp["recommendation"] in ["ALLOW", "REVIEW"]


def test_categorical_feature_mapping(mock_pipeline_and_data):
    """
    Checks that categorical variables like product_category map back from their
    one-hot representation to the raw feature and string value.
    """
    model, fe, raw_df = mock_pipeline_and_data
    explainer = get_explainer(model, X_train_sample=fe.transform(raw_df))
    
    single_raw = raw_df.head(1).drop(columns=["is_abusive_return"]).copy()
    raw_category = single_raw.iloc[0]["product_category"]
    
    exp = explain_prediction(single_raw, model, fe, explainer)
    
    # Scan factors to find the mapped product_category factor
    all_factors = exp["positive_factors"] + exp["negative_factors"]
    prod_cat_factors = [f for f in all_factors if f["feature"] == "product_category"]
    
    # We should have exactly 1 mapped factor matching the active category
    assert len(prod_cat_factors) == 1
    assert prod_cat_factors[0]["value"] == raw_category
    assert "product_category_" not in prod_cat_factors[0]["feature"]
    assert isinstance(prod_cat_factors[0]["contribution"], float)


def test_rejects_target_column(mock_pipeline_and_data):
    """
    Checks that explain_prediction raises ValueError if target column
    is_abusive_return is present in raw input data.
    """
    model, fe, raw_df = mock_pipeline_and_data
    explainer = get_explainer(model, X_train_sample=fe.transform(raw_df))
    
    with pytest.raises(ValueError, match="is_abusive_return"):
        explain_prediction(raw_df.head(1), model, fe, explainer)
