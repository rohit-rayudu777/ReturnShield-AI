"""
src/api/schemas.py

Pydantic schemas for the ReturnShield AI FastAPI backend.
Defines strict validation rules for prediction requests, factor explanations,
and health check status.

Target Leakage Prevention:
  - ReturnRecordRequest does NOT define 'is_abusive_return'.
  - extra = 'forbid' config forces Pydantic to raise a 422 validation error
    if any extra field (such as the target label) is present in the payload.
"""

from typing import List, Literal, Dict, Union
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, field_validator


class ReturnRecordRequest(BaseModel):
    # Enforce strict validation: forbid extra fields to block target leakage (like is_abusive_return)
    model_config = {
        "extra": "forbid"
    }

    # Identifiers
    order_id: str = Field(..., description="Unique transaction/order identifier.")
    customer_id: str = Field(..., description="Unique customer identifier.")
    timestamp: str = Field(..., description="Timestamp of the return request (format: YYYY-MM-DD HH:MM:SS).")

    # Numeric raw features (validated to be non-negative)
    order_amount: float = Field(..., ge=0.0, description="Total amount of the order.")
    customer_age_days: int = Field(..., ge=0, description="Age of customer account in days.")
    previous_orders: int = Field(..., ge=0, description="Total previous orders placed by the customer.")
    previous_returns: int = Field(..., ge=0, description="Total previous returns made by the customer.")
    customer_return_rate: float = Field(..., ge=0.0, le=1.0, description="Historical customer return rate.")
    orders_last_7_days: int = Field(..., ge=0, description="Number of orders placed in the last 7 days.")
    orders_last_30_days: int = Field(..., ge=0, description="Number of orders placed in the last 30 days.")
    returns_last_7_days: int = Field(..., ge=0, description="Number of returns made in the last 7 days.")
    returns_last_30_days: int = Field(..., ge=0, description="Number of returns made in the last 30 days.")
    average_order_value: float = Field(..., ge=0.0, description="Average order value of customer history.")
    discount_percentage: float = Field(..., ge=0.0, le=100.0, description="Discount percentage applied.")
    delivery_days: int = Field(..., ge=0, description="Days taken to deliver the order.")
    return_days_after_delivery: int = Field(..., ge=0, description="Days after delivery return was initiated.")
    address_change_count: int = Field(..., ge=0, description="Number of address changes on account.")
    payment_failures: int = Field(..., ge=0, description="Number of payment failures on account.")
    previous_chargebacks: int = Field(..., ge=0, description="Number of previous chargebacks on account.")
    
    # Binary variables (Literal 0 or 1)
    is_first_order: Literal[0, 1] = Field(..., description="1 if this is the customer's first order, else 0.")
    is_high_value_order: Literal[0, 1] = Field(..., description="1 if order value is high, else 0.")

    # Categorical raw features
    product_category: Literal["Electronics", "Clothing", "Beauty", "Home", "Books"] = Field(
        ..., description="Category of the returned product."
    )
    payment_method: Literal["Credit Card", "Debit Card", "UPI", "Netbanking", "COD"] = Field(
        ..., description="Payment method used for the order."
    )

    # NOTE: review_threshold is intentionally NOT a client-facing field.
    # The operational decision threshold is controlled server-side only
    # (see REVIEW_THRESHOLD env var / main.py) to prevent threshold manipulation.


class FactorResponse(BaseModel):
    feature: str = Field(..., description="Original raw feature name.")
    display_name: str = Field(..., description="Human-readable feature name.")
    value: Union[str, float, int] = Field(..., description="Value of the feature in the transaction.")
    contribution: float = Field(..., description="SHAP contribution score in log-odds space.")
    direction: Literal["increases risk", "decreases risk"] = Field(..., description="Direction of risk impact.")


class PredictionResponse(BaseModel):
    prediction_id: str = Field(..., description="Unique UUID assigned to this prediction request.")
    order_id: str = Field(..., description="The ID of the order evaluated.")
    customer_id: str = Field(..., description="The ID of the customer evaluated.")
    risk_probability: float = Field(..., description="Model predicted probability of abusive return (0.0 to 1.0).")
    risk_score: int = Field(..., description="Risk score scaled to 0-100.")
    risk_band: Literal["Low", "Medium", "High", "Very High"] = Field(..., description="Risk band classification.")
    decision: Literal["ALLOW", "REVIEW"] = Field(..., description="Recommended action: human review vs auto-allow.")
    summary: str = Field(..., description="Human-readable explanation of risk drivers.")
    positive_factors: List[FactorResponse] = Field(..., description="Top risk-increasing factors.")
    negative_factors: List[FactorResponse] = Field(..., description="Top risk-mitigating factors.")
    base_value_log_odds: float = Field(..., description="Model baseline log-odds average.")


class HealthResponse(BaseModel):
    status: str = Field(..., description="API operational status ('ok').")
    model_loaded: bool = Field(..., description="True if trained ML artifacts are loaded and ready.")
    artifacts: Dict[str, bool] = Field(..., description="Availability status of individual model components.")
