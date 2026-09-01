# SHAP Explainability — ReturnShield AI

This document describes the explainability architecture of ReturnShield AI implemented in Phase 2E. It explains the importances of explainability, the choice of SHAP model explainer, features mapping back to original categorical variables, and lists system limitations.

---

## 1. Rationale for Explainability in Risk Management

ReturnShield AI is a **defense-only decision-support system**. It flags suspicious return requests for REVIEW or ALLOW recommendations but does not make automated denial decisions.

To support human review teams, the system must explain **why** a transaction was flagged. Providing clear, human-readable explanations:
- Speeds up manual review times by highlighting key risk factors.
- Promotes transparency and consistency in operations.
- Helps developers debug model edge-cases or identify potential feature leakage.

---

## 2. SHAP Approach & Explainer Type

We use **SHAP (SHapley Additive exPlanations)** to provide mathematically robust, additive local explanations for individual predictions.

### Explainer Choice: `shap.LinearExplainer`

The Phase 2C baseline model is a Logistic Regression classifier wrapped in a Pipeline with a `StandardScaler`.
- Because the classifier is a linear model, the most appropriate SHAP explainer is `shap.LinearExplainer`.
- Linear SHAP values are computed in **log-odds space** (the margin output of the classifier before the sigmoid function).
- **Log-odds space is additive**:
  $$log\_odds(p) = base\_value + \sum_{i} SHAP_i$$
- This mathematical form is exact, computed directly from the model weights and training background distributions. It does not suffer from sampling noise or convergence errors associated with kernel-based approximation explainers (e.g. `KernelExplainer`).

---

## 3. Background Reference Dataset

To represent the expected risk baseline, the explainer is initialized with a **100-sample representative background dataset** extracted from the training split.
- The `expected_value` of the explainer (approx. `0.70` log-odds) represents the average prediction log-odds score over this reference dataset.
- The SHAP values represent how much a specific transaction's features push the prediction above or below this average.

---

## 4. Preprocessing & Categorical Feature Mapping

One-hot encoding transforms a single categorical variable (e.g. `product_category`) into multiple binary columns (e.g. `product_category_Electronics`, `product_category_Clothing`).
- Displaying multiple binary columns with uninformative zero values (e.g. `product_category_Clothing = 0.0`) confuses review teams.
- **Reverse-Mapping Strategy**:
  - The explanation pipeline maps one-hot variables back to their original variables.
  - If a record has `product_category = 'Electronics'`, the column `product_category_Electronics` is active (`1.0`). We capture its SHAP contribution and assign it to the display variable `product_category = 'Electronics'`.
  - All inactive category columns (where value is `0.0`) represent baseline offsets and are filtered out of the active contributors list.

---

## 5. Structured Explanation Output

The explanation function returns a structured dictionary containing:

1. **Probability**: the raw classifier probability in `[0.0, 1.0]`
2. **Risk Score**: the scaled risk score in `[0, 100]`
3. **Risk Band**: "Low", "Medium", "High", or "Very High"
4. **Recommendation**: "REVIEW" or "ALLOW"
5. **Positive Factors**: list of factors increasing risk sorted descending
6. **Negative Factors**: list of factors reducing risk sorted ascending
7. **Summary**: A natural-language sentence summarizing the top contributors

### Example Structured Output
```json
{
  "probability": 0.8251,
  "risk_score": 83,
  "risk_band": "Very High",
  "recommendation": "REVIEW",
  "summary": "High customer return history (100.0%) was the primary driver increasing the predicted abuse risk. This risk was partially mitigated by 'Payment Failure Rate' (0.0%), which reduced the overall score by 0.06 log-odds.",
  "positive_factors": [
    {
      "feature": "customer_return_rate",
      "display_name": "Customer Return Rate",
      "value": 1.0,
      "contribution": 1.9958,
      "direction": "increases risk"
    },
    {
      "feature": "payment_failures",
      "display_name": "Payment Failures",
      "value": 1.0,
      "contribution": 0.0462,
      "direction": "increases risk"
    }
  ],
  "negative_factors": [
    {
      "feature": "payment_failure_rate",
      "display_name": "Payment Failure Rate",
      "value": 0.0,
      "contribution": -0.0647,
      "direction": "decreases risk"
    }
  ],
  "base_value_log_odds": 0.7026
}
```

---

## 6. Interpretation of SHAP Values

- **Positive SHAP Value (> 0)**: The feature pushes the transaction prediction towards higher risk (increases the probability of return abuse).
- **Negative SHAP Value (< 0)**: The feature pulls the prediction towards lower risk (mitigates the risk score).
- **Log-odds vs. Probability**: Because SHAP calculations for Logistic Regression occur in log-odds space, contributions cannot be simply added directly to the final probability. Instead, they are added to the `base_value_log_odds`, and the sum is converted to probability using the sigmoid function:
  $$probability = \frac{1}{1 + e^{-(base\_value + \sum SHAP)}}$$

---

## 7. Key Warnings & Limitations

> [!WARNING]
> **Correlation vs. Causation**
> SHAP values explain what features the *model* associated with risk, based on statistical correlations found in the training dataset. They **do not prove physical causation**. For example, if a high order amount contributes positively to risk, it does not mean placing high-value orders causes a customer to commit fraud, only that the model has learned an association.

> [!IMPORTANT]
> **Defense-Only System**
> These explanations are intended solely for human operators review. They must never be used to automate punitive customer actions, automatically block user accounts, or reject checkout attempts.

Other limitations of SHAP:
- **Baseline Dependency**: SHAP values are relative to the selected 100-sample background reference dataset. Changing the reference dataset changes the individual SHAP values and base value.
- **Linear Explainer Constraints**: `LinearExplainer` assumes linear relationship and additive contribution in log-odds space. It cannot capture non-linear feature interactions that a tree-based model might have.
- **Preprocessing Dependency**: If the feature pipeline transforms values (e.g. logs or scaling), SHAP values must be computed using the preprocessed feature matrix, requiring careful mapping to correspond to the original raw inputs.
