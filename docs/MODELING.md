# Modeling — ReturnShield AI

This document describes the complete model development and evaluation pipeline for Phase 2C of ReturnShield AI.

---

## 1. Target Definition

| Column | Type | Values | Meaning |
|---|---|---|---|
| `is_abusive_return` | `int` | 0 / 1 | **1** = abusive return (wardrobing, empty box, friendly fraud). **0** = legitimate return. |

The model outputs a **probability score in [0.0, 1.0]** for class 1. This is a risk-ranking/decision-support score — not an automated enforcement action.

---

## 2. Train / Test Split

| Parameter | Value |
|---|---|
| Method | Stratified random split |
| Test fraction | 20% |
| Random seed | 42 |
| Stratified on | `is_abusive_return` |
| Training samples | 80,000 |
| Test samples | 20,000 |

**Why stratified random (not chronological)?**

For the baseline model evaluation, stratified random split gives a representative class distribution in both splits and a stable estimate of model performance. Chronological splitting (used in Phase 2B for leakage analysis) is appropriate for temporal leakage studies but can produce artificially skewed evaluation sets when records are not perfectly uniformly distributed across time.

> **Note:** All scaler and model parameters are fitted **exclusively on the training split**. The test set is never used during fitting.

---

## 3. Preprocessing

The Phase 2B `FeatureEngineer` is reused without modification:

1. **Median imputation** of missing numerical values (medians computed on training data only).
2. **Clipping** of numerical features to `>= 0.0`.
3. **Inf/NaN replacement** with `0.0` for engineered features.
4. **OneHotEncoding** of `product_category` and `payment_method` with predefined categories and `handle_unknown='ignore'`.
5. **13 engineered risk features** (see `FEATURE_ENGINEERING.md`).

Output: a dense float matrix of shape `(n_samples, 41)`.

---

## 4. Model Selected

**Logistic Regression** (scikit-learn `LogisticRegression`) wrapped in a `sklearn.pipeline.Pipeline`:

```
Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42,
    ))
])
```

**Why Logistic Regression?**

- Fully interpretable: coefficients directly indicate feature importance and direction of risk.
- Fast to train even on 80,000 samples.
- Outputs calibrated probabilities via `predict_proba()`.
- Suitable baseline before considering tree-based models in later phases.

**Why StandardScaler?**

Logistic Regression is sensitive to feature scale. Raw features span very different numeric ranges (e.g., `customer_age_days`: 30–1365 vs. `chargeback_rate`: 0–1). `StandardScaler` is fitted on training data only and applied to test data, preventing any leakage.

---

## 5. Class Imbalance Handling

| Class | Count | Percentage |
|---|---|---|
| 0 (normal) | 36,385 | 36.38% |
| 1 (abusive) | 63,615 | 63.62% |

**Strategy: `class_weight='balanced'`**

This automatically adjusts the per-class loss weight inversely proportional to frequency:
- Normal weight ≈ 1.38
- Abusive weight ≈ 0.79

This is the minimum necessary intervention. It avoids the complexity and reproducibility concerns of SMOTE/oversampling, and is appropriate for this mild-to-moderate imbalance.

---

## 6. Evaluation Metrics

All metrics are computed on the **held-out test set (20,000 samples)**.

| Metric | Value |
|---|---|
| Accuracy | **0.7736** |
| Precision | **0.8838** |
| Recall | **0.7416** |
| F1-Score | **0.8065** |
| ROC-AUC | **0.8615** |

### Confusion Matrix

|  | Predicted Normal | Predicted Abusive |
|---|---|---|
| **Actual Normal** | 6,037 (TN) | 1,240 (FP) |
| **Actual Abusive** | 3,288 (FN) | 9,435 (TP) |

### Precision / Recall Trade-Off

In this deployment context:

- **False Positive (FP)**: A legitimate return is flagged as abusive. Cost = customer friction / lost goodwill.
- **False Negative (FN)**: An abusive return is missed. Cost = financial loss (item + shipping).

The current model achieves **Precision 0.88 > Recall 0.74**, meaning:
- Of the returns flagged as abusive, 88% are genuinely abusive — low false-alarm rate.
- 26% of abusive returns are missed (FN = 3,288).

For a **decision-support / operator-review** system this is a reasonable baseline. Operators are not overwhelmed with false alarms, and the ROC-AUC of 0.86 means the risk score is a strong ranking signal even if the 0.5 threshold is later adjusted.

> **Threshold tuning** (optimising the decision threshold based on FP/FN cost ratio) is deferred to a later phase per the cost-benefit framework described in `ARCHITECTURE.md`.

---

## 7. Model Artifact Location

| Artifact | Path |
|---|---|
| Trained model | `src/model/artifacts/model.joblib` |
| Fitted FeatureEngineer | `src/model/artifacts/feature_engineer.joblib` |

Both are serialised with `joblib` for reliable sklearn object persistence.

---

## 8. How to Load the Model

```python
from src.model.model import load_artifacts

model, feature_engineer = load_artifacts()
# model            : fitted sklearn Pipeline (StandardScaler -> LogisticRegression)
# feature_engineer : fitted FeatureEngineer with medians and encoder categories
```

Custom paths:
```python
model, fe = load_artifacts(
    model_path="src/model/artifacts/model.joblib",
    fe_path="src/model/artifacts/feature_engineer.joblib",
)
```

---

## 9. How to Generate a Risk Probability Score

```python
import pandas as pd
from src.model.model import load_artifacts, predict_risk_score

# Load pre-fitted artifacts
model, feature_engineer = load_artifacts()

# Prepare a raw return request (same schema as data/returns.csv)
# Do NOT include 'is_abusive_return' in inference data
raw_request = pd.DataFrame([{
    "order_id": "ORD_999999_0",
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
    "discount_percentage": 0.0,
    "delivery_days": 3,
    "return_days_after_delivery": 1,
    "address_change_count": 2,
    "payment_failures": 1,
    "previous_chargebacks": 1,
    "is_first_order": 0,
    "is_high_value_order": 1,
}])

# Returns array of shape (n_samples,) with values in [0.0, 1.0]
scores = predict_risk_score(model, feature_engineer, raw_request)
risk_score = float(scores[0])
print(f"Abusive-return risk score: {risk_score:.4f}")
```

A score close to **1.0** indicates high abuse risk; close to **0.0** indicates low risk.

---

## 10. How to Re-train the Model

```powershell
# From project root with .venv active:
python src/model/train.py

# Custom data path:
python src/model/train.py --data data/returns.csv
```

This will:
1. Re-split the data (same random seed = deterministic).
2. Re-fit the FeatureEngineer and scaler on training data.
3. Re-train Logistic Regression.
4. Print all evaluation metrics.
5. Overwrite artifacts in `src/model/artifacts/`.

---

## 11. Limitations

| Limitation | Details |
|---|---|
| **Synthetic data only** | The model is trained on synthetic data generated by `src/data/generator.py`. Real-world performance will differ and must be re-evaluated on genuine transaction data. |
| **Baseline model** | Logistic Regression is the Phase 2C baseline. More powerful models (e.g., XGBoost, LightGBM) may yield significantly better recall without sacrificing precision. |
| **Fixed threshold** | Default sklearn threshold of 0.5 is used. Threshold optimisation based on the FP/FN cost ratio is deferred to a later phase. |
| **No temporal validation** | A simple random split is used. Walk-forward or rolling-window cross-validation may give more realistic estimates for deployment. |
| **No feature selection** | All 41 features are used. Regularisation (L2, C=1.0) mitigates overfitting but no formal feature selection was performed. |
| **No calibration check** | predict_proba outputs are used directly as risk scores without isotonic/Platt calibration verification. |
| **Class imbalance** | class_weight='balanced' is a simple correction. More sophisticated techniques (ADASYN, cost-sensitive learning) may further improve recall of the minority class (normal returns). |
