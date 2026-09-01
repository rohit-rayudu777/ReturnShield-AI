# FastAPI Backend API — ReturnShield AI

This document describes the FastAPI backend API for ReturnShield AI implemented in Phase 3.

---

## 1. API Purpose & Architecture

The API exposes endpoints to query health status and evaluate transaction risk scores with human-readable SHAP explanations.

```
                  ┌───────────────────────────────────────────────┐
                  │               FastAPI Backend                 │
                  │              (src/api/main.py)                │
                  └───────────────┬───────────────┬───────────────┘
                                  │               │
                                  ▼               ▼
                      ┌───────────────────────┐ ┌───────────────────────┐
                      │  Risk Scoring module  │ │ Explainability module │
                      │ (model/risk_scoring.py)│ │(model/explainability) │
                      └───────────────────────┘ └───────────────────────┘
                                  │               │
                                  └───────┬───────┘
                                          ▼
                      ┌───────────────────────────────────────────────┐
                      │                 Model Pipeline                │
                      │          StandardScaler -> LogisticRegression  │
                      └───────────────────────────────────────────────┘
```

- **Lifespan Caching**: Model objects, FeatureEngineer, and SHAP explainer are loaded once during application startup and cached in the app state.
- **Strict Validation**: All input schemas are validated with Pydantic. Extra fields (including target variables) are explicitly forbidden to prevent target leakage.
- **Local Dev CORS**: CORS is configured to allow `localhost` origins only.
- **Defense-Only recommendations**: The API returns ALLOW or REVIEW decisions for human decision support; no automatic blocking or return denial features exist.

---

## 2. Local Startup Instructions

To start the API server locally:

1. **Activate the Virtual Environment**:
   ```powershell
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```
2. **Launch the Uvicorn Server**:
   ```powershell
   python -m uvicorn src.api.main:app --reload
   ```
3. **Interactive Documentation**:
   The interactive Swagger documentation is available in your browser at:
   - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - ReDoc UI: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 3. Endpoints

### GET `/health`

Verifies the system operational status and checks whether trained ML models and feature engineering artifacts are loaded and ready.

#### Example Response (200 OK)
```json
{
  "status": "ok",
  "model_loaded": true,
  "artifacts": {
    "model": true,
    "feature_engineer": true,
    "explainer": true
  }
}
```

---

### POST `/predict`

Analyzes transaction details and behavior history of a return request to generate risk probability, score, band, decision recommendation, and friendly SHAP explanation drivers.

#### Request Schema (Pydantic: `ReturnRecordRequest`)

- **Strict Validation**: Set to `extra = 'forbid'`. Extra parameters (like `is_abusive_return`) are rejected with `422 Unprocessable Entity`.
- **Fields**:

| Field | Type | Description | Validation |
|---|---|---|---|
| `order_id` | `str` | Unique transaction/order identifier | Required |
| `customer_id` | `str` | Unique customer identifier | Required |
| `timestamp` | `str` | Format: `YYYY-MM-DD HH:MM:SS` | Required |
| `order_amount` | `float` | Value of order in INR | Required, `>= 0.0` |
| `customer_age_days` | `int` | Account age in days | Required, `>= 0` |
| `previous_orders` | `int` | Count of prior orders | Required, `>= 0` |
| `previous_returns` | `int` | Count of prior returns | Required, `>= 0` |
| `customer_return_rate` | `float` | Prior return rate | Required, `0.0` to `1.0` |
| `orders_last_7_days` | `int` | Prior 7 days order count | Required, `>= 0` |
| `orders_last_30_days` | `int` | Prior 30 days order count | Required, `>= 0` |
| `returns_last_7_days` | `int` | Prior 7 days return count | Required, `>= 0` |
| `returns_last_30_days` | `int` | Prior 30 days return count | Required, `>= 0` |
| `average_order_value` | `float` | Prior average order value | Required, `>= 0.0` |
| `discount_percentage` | `float` | Applied discount percentage | Required, `0.0` to `100.0` |
| `delivery_days` | `int` | Delivery days | Required, `>= 0` |
| `return_days_after_delivery`| `int` | Days elapsed before return | Required, `>= 0` |
| `address_change_count` | `int` | Count of account address changes | Required, `>= 0` |
| `payment_failures` | `int` | Count of payment failures | Required, `>= 0` |
| `previous_chargebacks` | `int` | Count of previous chargebacks | Required, `>= 0` |
| `is_first_order` | `int` | 1 if first order, else 0 | Required, `0` or `1` |
| `is_high_value_order` | `int` | 1 if high value order, else 0 | Required, `0` or `1` |
| `product_category` | `str` | Product category of returned item | Required, one of: `Electronics`, `Clothing`, `Beauty`, `Home`, `Books` |
| `payment_method` | `str` | Payment method used | Required, one of: `Credit Card`, `Debit Card`, `UPI`, `Netbanking`, `COD` |
| `review_threshold` | `float` | Option custom decision threshold | Optional, `0.0` to `1.0` (default: `0.30` per Phase 2D optimization) |

#### Example Request Body
```json
{
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
```

#### Example Response (200 OK)
```json
{
  "prediction_id": "bfd5d796-035f-4a0b-9dfd-b4b126d400be",
  "order_id": "ORD_123456_0",
  "customer_id": "CUST_000001",
  "risk_probability": 0.9999,
  "risk_score": 100,
  "risk_band": "Very High",
  "decision": "REVIEW",
  "summary": "High customer return history (100.0%) was the primary driver increasing the predicted abuse risk. This risk was partially mitigated by 'Address Change Count' (value: 2.00), which reduced the overall score by 0.17 log-odds.",
  "positive_factors": [
    {
      "feature": "customer_return_rate",
      "display_name": "Customer Return Rate",
      "value": "1.0",
      "contribution": 1.9958,
      "direction": "increases risk"
    },
    {
      "feature": "payment_failures",
      "display_name": "Payment Failures",
      "value": "1.0",
      "contribution": 0.0462,
      "direction": "increases risk"
    }
  ],
  "negative_factors": [
    {
      "feature": "address_change_count",
      "display_name": "Address Change Count",
      "value": "2.0",
      "contribution": -0.174,
      "direction": "decreases risk"
    }
  ],
  "base_value_log_odds": 0.7026
}
```

---

## 4. Error Responses

- **400 Bad Request**: Input parameter error (e.g. invalid date formats, logic validation errors in scoring).
- **422 Unprocessable Entity**: Schema error (e.g. missing required field, negative value for a positive field, invalid category string, or target leakage column presence).
- **503 Service Unavailable**: ML Model pipeline is not loaded/initialized on the startup of the API.
- **500 Internal Server Error**: Inference processing error. Stack traces and file paths are stripped from detail payloads and logged locally on the server for safety.

#### Example Target Leakage Rejection Response (422)
```json
{
  "detail": [
    {
      "type": "extra_forbidden",
      "loc": ["body", "is_abusive_return"],
      "msg": "Extra inputs are not permitted",
      "input": 1
    }
  ]
}
```

---

## 5. Structured Audit Logging

Every transaction evaluated in `/predict` writes a prediction record to the local JSON lines file:

```
data/audit_log.jsonl
```

- **Format**: JSON Lines (`.jsonl`), one JSON dictionary per line.
- **Fields**:
  - `timestamp`: UTC ISO 8601 string
  - `prediction_id`: Unique UUIDv4 assigned to the transaction
  - `order_id`: Plain order string
  - `customer_id`: Plain customer string
  - `risk_probability`: Raw float (`0.0` to `1.0`)
  - `risk_score`: Scaled integer (`0` to `100`)
  - `risk_band`: Risk classification string
  - `decision`: REVIEW or ALLOW outcome
- **Data Privacy**: No sensitive customer details, tokens, passwords, or credit card numbers are ever written to the audit log.

#### Example Audit Log Entry
```json
{"timestamp": "2026-08-29T11:15:32.408102+00:00", "prediction_id": "bfd5d796-035f-4a0b-9dfd-b4b126d400be", "order_id": "ORD_123456_0", "customer_id": "CUST_000001", "risk_probability": 0.9999, "risk_score": 100, "risk_band": "Very High", "decision": "REVIEW"}
```

---

## 6. Defense-Only Policy

The ReturnShield AI API is strictly a **decision-support tool** for risk analyst teams. It provides recommendations (`decision: "ALLOW"` or `decision: "REVIEW"`) to assist operations. The API does not contain any routes or triggers for automatically blocking customer accounts, suspending checkouts, or rejecting payments.
