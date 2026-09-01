"""
src/model/explainability.py

Phase 2E - SHAP Explainability for ReturnShield AI.

Provides reusable functionality for generating SHAP explanations for individual
return predictions, resolving one-hot encoded features back to their original
categorical representations, and generating natural, grounded explanations.

IMPORTANT - DEFENSE ONLY:
  This system is a decision-support tool only.
  It recommends REVIEW or ALLOW outcomes.
  It does NOT automatically deny returns, block customers, or suspend accounts.

REUSE GUIDELINE:
  This module loads and reuse the pre-trained Logistic Regression baseline
  model and pre-fitted FeatureEngineer. It does NOT retrain the model.
"""

import os
# pyrefly: ignore [missing-import]
import joblib
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import shap
from typing import Dict, List, Optional, Tuple, Any

from src.model.model import (
    load_training_data,
    MODEL_ARTIFACT_PATH,
    FEATURE_ENGINEER_ARTIFACT_PATH,
)
from src.model.risk_scoring import (
    prob_to_score,
    score_to_band,
    classify_return,
    DEFAULT_REVIEW_THRESHOLD,
    DEFAULT_RISK_BANDS,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
EXPLAINER_ARTIFACT_PATH = os.path.join(
    os.path.dirname(MODEL_ARTIFACT_PATH), "explainer.joblib"
)

# ---------------------------------------------------------------------------
# Friendly feature name mapping
# ---------------------------------------------------------------------------
FRIENDLY_NAMES: Dict[str, str] = {
    # Raw numerical features
    "order_amount": "Order Amount",
    "customer_age_days": "Account Age (Days)",
    "previous_orders": "Previous Orders",
    "previous_returns": "Previous Returns",
    "customer_return_rate": "Customer Return Rate",
    "orders_last_7_days": "Orders in Last 7 Days",
    "orders_last_30_days": "Orders in Last 30 Days",
    "returns_last_7_days": "Returns in Last 7 Days",
    "returns_last_30_days": "Returns in Last 30 Days",
    "average_order_value": "Average Order Value",
    "discount_percentage": "Discount Percentage",
    "delivery_days": "Delivery Days",
    "return_days_after_delivery": "Return Days After Delivery",
    "address_change_count": "Address Change Count",
    "payment_failures": "Payment Failures",
    "previous_chargebacks": "Previous Chargebacks",
    "is_first_order": "Is First Order",
    "is_high_value_order": "Is High Value Order",
    
    # Engineered features
    "previous_return_frequency": "Previous Return Frequency",
    "return_to_delivery_days_ratio": "Return to Delivery Days Ratio",
    "total_return_delay_days": "Total Return Delay Days",
    "refund_to_order_value_ratio": "Refund to Order Value Ratio",
    "returns_velocity_ratio_7d_30d": "Returns Velocity (7d/30d)",
    "recent_returns_to_orders_ratio_7d": "Recent Returns/Orders Ratio (7d)",
    "recent_returns_to_orders_ratio_30d": "Recent Returns/Orders Ratio (30d)",
    "payment_failure_rate": "Payment Failure Rate",
    "address_change_rate": "Address Change Rate",
    "chargeback_rate": "Chargeback Rate",
    "orders_velocity_ratio_7d_30d": "Orders Velocity (7d/30d)",
    "order_frequency_days": "Order Frequency (Days)",
    "high_value_return_interaction": "High-Value Return Interaction",
    
    # Categorical variables (prefixes)
    "product_category": "Product Category",
    "payment_method": "Payment Method",
    "item_condition": "Item Condition",
    "return_reason": "Return Reason",
}


# ---------------------------------------------------------------------------
# Explainer retrieval
# ---------------------------------------------------------------------------
def get_explainer(
    model,
    X_train_sample: Optional[pd.DataFrame] = None,
    explainer_path: str = EXPLAINER_ARTIFACT_PATH,
) -> shap.LinearExplainer:
    """
    Retrieves the SHAP LinearExplainer. Loads from disk if saved, otherwise
    constructs it using the model's scaler and classifier, and saves it.

    Parameters
    ----------
    model          : fitted sklearn Pipeline
    X_train_sample : optional training sample to construct background distribution
                     if the explainer is not already saved.
    explainer_path : destination path to load/save the explainer artifact

    Returns
    -------
    shap.LinearExplainer
    """
    if os.path.exists(explainer_path):
        return joblib.load(explainer_path)

    # If background data not passed, attempt to load training data to sample
    if X_train_sample is None:
        try:
            X_train, _, _, _, _ = load_training_data("data/returns.csv")
            # Use 100 representative training samples as background reference
            X_train_sample = X_train.head(100)
        except Exception as e:
            raise RuntimeError(
                f"Explainer artifact not found and training data could not be loaded: {e}. "
                "Provide X_train_sample explicitly to fit the explainer."
            )

    scaler = model.named_steps["scaler"]
    clf = model.named_steps["classifier"]
    
    bg_scaled = scaler.transform(X_train_sample)
    
    # LogisticRegression is a linear classifier, so LinearExplainer is exact and fast
    explainer = shap.LinearExplainer(clf, bg_scaled)
    
    # Save the explainer to artifacts directory
    os.makedirs(os.path.dirname(os.path.abspath(explainer_path)), exist_ok=True)
    joblib.dump(explainer, explainer_path)
    print(f"Fitted and saved SHAP LinearExplainer -> {explainer_path}")
    
    return explainer


# ---------------------------------------------------------------------------
# Prediction explanation
# ---------------------------------------------------------------------------
def explain_prediction(
    raw_df: pd.DataFrame,
    model,
    feature_engineer,
    explainer: shap.LinearExplainer,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    risk_bands: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Generates a complete explanation for an individual prediction request.

    Parameters
    ----------
    raw_df           : pd.DataFrame containing exactly 1 row matching returns.csv
                       schema (must NOT contain 'is_abusive_return')
    model            : fitted sklearn Pipeline
    feature_engineer : fitted FeatureEngineer
    explainer        : fitted shap.LinearExplainer
    review_threshold : probability threshold for REVIEW outcome
    risk_bands       : optional custom risk bands

    Returns
    -------
    dict containing score prediction details and feature contributions
    """
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("raw_df must be a pandas DataFrame")
    if len(raw_df) != 1:
        raise ValueError("explain_prediction() supports explaining exactly 1 record at a time")
    if "is_abusive_return" in raw_df.columns:
        raise ValueError(
            "raw_df must not contain 'is_abusive_return' during inference. "
            "Remove target column before calling explain_prediction()."
        )

    # 1. Transform raw record
    X_trans = feature_engineer.transform(raw_df)
    
    # 2. Get scaled representation
    scaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(X_trans)
    
    # 3. Generate probabilities and scores
    prob = float(model.predict_proba(X_trans)[0, 1])
    score = prob_to_score(prob)
    band = score_to_band(score, risk_bands)
    recommendation = classify_return(prob, review_threshold)
    
    # 4. Generate SHAP values (in log-odds space)
    shap_out = explainer(X_scaled)
    shap_vals = shap_out.values[0]
    
    # 5. Reverse-map categorical columns and build list of contributions
    contributions = []
    
    categorical_prefixes = [f"{col}_" for col in feature_engineer.categorical_cols]
    
    for idx, feature_name in enumerate(feature_engineer.feature_names_):
        shap_val = float(shap_vals[idx])
        
        # Check if feature is an encoded categorical
        is_categorical = False
        for prefix in categorical_prefixes:
            if feature_name.startswith(prefix):
                is_categorical = True
                raw_col = prefix[:-1]
                category_val = feature_name[len(prefix):]
                
                # Retrieve engineered binary value
                eng_val = X_trans.iloc[0][feature_name]
                
                # Only include the active category (e.g. value is 1.0)
                # Discard inactive categories (0.0) as they are uninformative offsets
                if eng_val > 0.5:
                    contributions.append({
                        "feature": raw_col,
                        "display_name": FRIENDLY_NAMES.get(raw_col, raw_col),
                        "value": category_val,
                        "contribution": shap_val,
                        "direction": "increases risk" if shap_val > 0 else "decreases risk",
                    })
                break
                
        if not is_categorical:
            # Numerical feature (raw or engineered)
            val = float(X_trans.iloc[0][feature_name])
            contributions.append({
                "feature": feature_name,
                "display_name": FRIENDLY_NAMES.get(feature_name, feature_name),
                "value": val,
                "contribution": shap_val,
                "direction": "increases risk" if shap_val > 0 else "decreases risk",
            })
            
    # 6. Separate and sort contributions
    positive_factors = [c for c in contributions if c["contribution"] > 0]
    negative_factors = [c for c in contributions if c["contribution"] <= 0]
    
    # Sort: positive descending, negative ascending (most risk-reducing first)
    positive_factors.sort(key=lambda x: x["contribution"], reverse=True)
    negative_factors.sort(key=lambda x: x["contribution"])
    
    # 7. Generate grounded summary text
    summary = generate_summary_text(positive_factors, negative_factors)
    
    return {
        "probability":          round(prob, 4),
        "risk_score":           score,
        "risk_band":            band,
        "recommendation":       recommendation,
        "summary":              summary,
        "positive_factors":     positive_factors,
        "negative_factors":     negative_factors,
        "base_value_log_odds":  float(explainer.expected_value),
    }


# ---------------------------------------------------------------------------
# Human-readable explanation text generator
# ---------------------------------------------------------------------------
def generate_summary_text(
    pos_contribs: List[Dict[str, Any]],
    neg_contribs: List[Dict[str, Any]],
) -> str:
    """
    Constructs a descriptive human-readable summary grounded in the actual
    top SHAP values.

    Parameters
    ----------
    pos_contribs : positive contributors sorted descending
    neg_contribs : negative contributors sorted ascending

    Returns
    -------
    str summary text
    """
    parts = []
    
    if pos_contribs:
        top_pos = pos_contribs[0]
        feat = top_pos["feature"]
        val = top_pos["value"]
        disp = top_pos["display_name"]
        
        # Format the value to a clean string
        if isinstance(val, float):
            val_str = f"{val:.1%}" if feat in ["customer_return_rate", "previous_return_frequency"] else f"{val:.2f}"
        else:
            val_str = str(val)
            
        # Specific natural phrasing for the most critical features
        if feat == "customer_return_rate" or feat == "previous_return_frequency":
            parts.append(
                f"High customer return history ({val_str}) was the primary driver "
                f"increasing the predicted abuse risk."
            )
        elif feat == "previous_chargebacks" or feat == "chargeback_rate":
            parts.append(
                f"A history of previous chargebacks ({val_str}) "
                f"significantly elevated the risk classification."
            )
        elif feat == "payment_failures" or feat == "payment_failure_rate":
            parts.append(
                f"Frequent payment failures ({val_str}) "
                f"contributed to the high risk score."
            )
        elif feat == "refund_to_order_value_ratio":
            parts.append(
                f"An unusually high refund amount relative to average order value ({val_str}) "
                f"increased the predicted risk."
            )
        else:
            parts.append(
                f"The feature '{disp}' (value: {val_str}) "
                f"was the primary driver increasing the predicted risk (+{top_pos['contribution']:.2f} log-odds)."
            )
            
    if neg_contribs:
        top_neg = neg_contribs[0]
        feat = top_neg["feature"]
        val = top_neg["value"]
        disp = top_neg["display_name"]
        
        # Format the value to a clean string
        val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
        
        parts.append(
            f"This risk was partially mitigated by '{disp}' (value: {val_str}), "
            f"which reduced the overall score by {-top_neg['contribution']:.2f} log-odds."
        )
        
    if not parts:
        return "No single feature had a dominant impact on the risk score prediction."
        
    return " ".join(parts)
