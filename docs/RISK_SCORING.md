# Risk Scoring & False-Positive Cost Analysis — ReturnShield AI

> **DEFENSE-ONLY SYSTEM**
> This module converts model probabilities into decision-support recommendations (REVIEW / ALLOW).
> It does NOT automatically deny returns, block customers, or suspend accounts.
> All cost figures in this document are ILLUSTRATIVE EXAMPLES ONLY and do not represent
> actual Razorpay or business figures.

---

## 1. Probability to Risk Score Conversion

The Phase 2C Logistic Regression model outputs a probability `p` in [0.0, 1.0]
for class 1 (abusive return).

This is converted to a **risk score** in [0, 100] by linear scaling:

```
risk_score = round(p * 100)
```

| Probability | Risk Score |
|---|---|
| 0.00 | 0 |
| 0.25 | 25 |
| 0.50 | 50 |
| 0.75 | 75 |
| 1.00 | 100 |

---

## 2. Risk Bands

Risk scores are grouped into four named bands:

| Band | Score Range | Interpretation |
|---|---|---|
| **Low** | 0 – 29 | Return is likely legitimate. Allow automatically. |
| **Medium** | 30 – 59 | Moderate risk. May warrant spot-check review. |
| **High** | 60 – 79 | Elevated risk. Recommend operator review before processing. |
| **Very High** | 80 – 100 | Strong abuse signal. Flag for priority review. |

> These band boundaries are **configurable**. Operators can pass a custom
> `risk_bands` dictionary to all scoring functions.

---

## 3. Review Threshold

The **review threshold** is the probability above which a return is flagged
for human operator review:

```
recommendation = "REVIEW" if probability >= review_threshold else "ALLOW"
```

Default: `review_threshold = 0.50`

This threshold is **configurable per deployment**. The threshold analysis below
helps operators select an appropriate value based on their cost assumptions.

---

## 4. Threshold Analysis (Held-Out Test Set — 20,000 samples)

The analysis below was computed on the Phase 2C held-out test set using the
**already-trained** model. No retraining was performed.

**Illustrative costs used:**
- False Positive cost: **100 units** (cost of wrongly flagging a legitimate return)
- False Negative cost: **500 units** (cost of missing a genuinely abusive return)

> These are example values only. See Section 7 for the cost framework.

| Threshold | Precision | Recall | F1 | TP | TN | FP | FN | FP Rate | Total Cost |
|---|---|---|---|---|---|---|---|---|---|
| 0.30 | 0.8007 | 0.8771 | 0.8372 | 11,159 | 4,500 | 2,777 | 1,564 | 0.3816 | 1,059,700 |
| 0.40 | 0.8454 | 0.8131 | 0.8289 | 10,345 | 5,385 | 1,892 | 2,378 | 0.2600 | 1,378,200 |
| **0.50** | **0.8838** | **0.7416** | **0.8065** | **9,435** | **6,037** | **1,240** | **3,288** | **0.1704** | **1,768,000** |
| 0.60 | 0.9130 | 0.6620 | 0.7675 | 8,423 | 6,474 | 803 | 4,300 | 0.1103 | 2,230,300 |
| 0.70 | 0.9367 | 0.5665 | 0.7060 | 7,207 | 6,790 | 487 | 5,516 | 0.0669 | 2,806,700 |
| 0.80 | 0.9591 | 0.4666 | 0.6277 | 5,936 | 7,024 | 253 | 6,787 | 0.0348 | 3,418,800 |

---

## 5. Precision/Recall Trade-Off

Increasing the threshold:
- **Raises precision** (fewer false alarms for operators)
- **Lowers recall** (more abusive returns are missed)
- **Reduces false-positive rate** (fewer legitimate customers are flagged)
- **Increases total cost** under our cost assumptions (FN cost >> FP cost)

Decreasing the threshold:
- **Raises recall** (catches more abusive returns)
- **Lowers precision** (more legitimate returns are flagged for review)
- **Increases operator workload**

This is the fundamental precision/recall trade-off for any risk-scoring system.
The right threshold depends entirely on the relative business impact of FP vs FN.

---

## 6. False-Positive Cost Analysis

### What is a False Positive here?

A **False Positive (FP)** is a **legitimate return that is flagged for review**.
It is not automatically rejected — it requires human review before processing.

Potential costs of a FP:
- Operator time / review cost
- Customer frustration and friction
- Risk of losing a loyal customer
- Reputational damage if customers feel treated unfairly

### What is a False Negative here?

A **False Negative (FN)** is an **abusive return that is not flagged**.
The abusive return is allowed through without additional review.

Potential costs of a FN:
- Direct financial loss (item cost + shipping)
- Replacement or refund processing cost
- Inventory shrinkage

---

## 7. Total Cost Calculation

```
total_cost = (FP * fp_cost) + (FN * fn_cost)

avg_cost_per_return = total_cost / n_test_samples
```

**Illustrative example values (NOT real business figures):**

| Parameter | Value | Meaning |
|---|---|---|
| `fp_cost` | 100 units | Estimated cost per wrongly flagged legitimate return |
| `fn_cost` | 500 units | Estimated cost per missed abusive return |

These values reflect a **5:1 FN:FP cost ratio**, meaning missing an abusive return
is assumed to be 5x more costly than unnecessarily reviewing a legitimate one.
This is a reasonable starting assumption for return-fraud defense systems,
but must be calibrated against actual business data before deployment.

---

## 8. Selected Threshold and Rationale

**Selected threshold: 0.30**

Under the illustrative cost assumptions (FP=100, FN=500):

| Threshold | Total Cost | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **0.30** | **1,059,700** | **2,777** | **1,564** | **0.8007** | **0.8771** | **0.8372** |
| 0.50 | 1,768,000 | 1,240 | 3,288 | 0.8838 | 0.7416 | 0.8065 |

Threshold 0.30 minimises illustrative total cost because:
- FN cost (500 units each) dominates the total cost calculation
- 0.30 catches significantly more abusive returns (recall 0.88 vs 0.74 at 0.50)
- The additional FP burden (2,777 vs 1,240) costs only 153,700 extra units
- The FN reduction (3,288 → 1,564) saves 862,000 units

> **IMPORTANT**: This selection is cost-assumption-dependent.
> If the FP:FN cost ratio were reversed (FP more costly than FN),
> a higher threshold (e.g. 0.70) would be appropriate.
> Operators must determine actual costs before setting the production threshold.

---

## 9. How to Use the Risk Scoring Module

### Batch Scoring (Inference)

```python
import joblib, pandas as pd
from src.model.risk_scoring import score_returns

model = joblib.load("src/model/artifacts/model.joblib")
fe    = joblib.load("src/model/artifacts/feature_engineer.joblib")

# raw_df must NOT include is_abusive_return
scored = score_returns(
    model, fe, raw_df,
    review_threshold=0.30,   # configurable
)
# Returns DataFrame: order_id, probability, risk_score, risk_band, recommendation
```

### Threshold Analysis (New Data)

```python
from src.model.risk_scoring import run_threshold_analysis, select_best_threshold

analysis_df = run_threshold_analysis(
    y_true=y_test.values,
    y_prob=y_prob,
    thresholds=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    fp_cost=100.0,
    fn_cost=500.0,
)
best = select_best_threshold(analysis_df)
print(best["explanation"])
```

---

## 10. Output File

The threshold analysis from the Phase 2C held-out test set is saved at:

```
data/threshold_analysis.csv
```

Columns: `threshold, precision, recall, f1, true_positives, true_negatives,
false_positives, false_negatives, fp_rate, fp_cost_unit, fn_cost_unit,
total_cost, avg_cost_per_return`

---

## 11. Limitations

| Limitation | Details |
|---|---|
| **Illustrative costs only** | FP/FN cost values (100 / 500 units) are examples. Real deployment requires actual business cost data. |
| **Synthetic data** | Analysis is performed on data generated by `src/data/generator.py`. Real-world performance will differ. |
| **Binary outcomes only** | The system flags for REVIEW or ALLOW. More nuanced multi-level outcomes (e.g., auto-approve, light review, full investigation) are not implemented. |
| **Single threshold** | A single global threshold is applied. In practice, category-specific or customer-segment-specific thresholds may perform better. |
| **No calibration** | Probability outputs are used directly. Isotonic regression or Platt scaling calibration is not applied. |
| **No feedback loop** | The system does not learn from operator review outcomes. A retraining pipeline with human-in-the-loop feedback is deferred to a later phase. |

---

## 12. System Classification

| Property | Value |
|---|---|
| **System type** | Decision-support / risk-ranking |
| **Automated enforcement** | NONE — human review required for all REVIEW recommendations |
| **Customer actions** | None — no blocking, suspension, or denial |
| **Data used** | Synthetic only (`data/returns.csv`) |
| **Real customer data** | NOT used |
