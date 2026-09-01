# Buildathon Submission Summary — ReturnShield AI

> **Track 2: AI Risk Manager — Razorpay Buildathon**

---

## 1. Project Information

- **Project Name**: ReturnShield AI
- **Track**: Razorpay Buildathon (Track 2 — AI Risk Manager)
- **Repository**: ReturnShield_ai
- **System Posture**: Defense-Only Decision Support System

---

## 2. Problem & Core Solution

### Problem
E-commerce merchants lose tens of billions of dollars annually to abusive returns (wardrobing, empty box returns, counterfeit swaps, and friendly fraud). Traditional rigid fraud rules cause massive false-positive customer friction, while lax policies lead to crippling inventory loss.

### Core Solution
ReturnShield AI provides an **explainable, cost-aware return-risk intelligence platform**. It predicts the likelihood of return abuse at transaction/return time, converts probabilities into calibrated 0–100 risk scores, optimizes decision thresholds against business cost trade-offs, and decomposes every decision into plain-English SHAP feature attributions to empower human fraud analysts.

---

## 3. Key Technical Components

1. **Synthetic Dataset**: 100,000 synthetic transaction records with realistic customer age, transaction velocity, payment failures, chargeback flags, and return patterns.
2. **Feature Engineering Pipeline**: 41 engineered features calculating historical return rates, payment failure rates, average order values, velocity ratios (7-day vs 30-day), and delivery delay interaction metrics.
3. **Supervised ML Model**: Scaled Logistic Regression classifier trained with class balancing (`class_weight='balanced'`) and fixed random seed (42) for 100% reproducible training.
4. **Held-Out Evaluation**: Evaluated on a 20,000-sample held-out test set (no data leakage).
5. **Calibrated Risk Scoring**: Probability-to-score linear scaling (0–100) mapped into 4 configurable risk bands (`Low`, `Medium`, `High`, `Very High`).
6. **False-Positive Cost Analysis**: Threshold optimization model balancing false-positive customer friction costs (100 units) against false-negative fraud loss costs (500 units), establishing `0.30` as the optimal business threshold.
7. **SHAP Explainability**: Local feature attributions using `shap.LinearExplainer`, reverse-mapping one-hot encoded variables back to original column names, raw values, and natural language summary reports.
8. **FastAPI REST API**: Async API endpoints (`GET /health`, `POST /predict`) with strict Pydantic v2 validation and audit logging.
9. **React Dashboard**: High-contrast obsidian black & warm gold editorial interface featuring preset demo controls, interactive evaluation form, SVG ring risk gauge, and SHAP factor attribution bars.

---

## 4. Empirical Evaluation Results

Evaluated on held-out synthetic test set (20,000 records, stratified split):

| Metric | Measured Value |
|---|---|
| **Accuracy** | **77.36%** |
| **Precision** | **88.38%** |
| **Recall** | **74.16%** |
| **F1 Score** | **80.65%** |
| **ROC-AUC** | **86.15%** |

---

## 5. Defense-Only & Safety Safeguard Statement

- **Recommendation Only**: Recommends **`ALLOW`** or **`REVIEW`** outcomes.
- **No Automatic Enforcement**: Never automatically blocks customer accounts, cancels transactions, or rejects returns.
- **Human-in-the-Loop**: Designed strictly to prioritize review queues for human fraud analysts.
- **Synthetic Data**: Trained and evaluated exclusively on synthetic datasets.

---

## 6. Quick Demo Instructions for Judges

### 1. Launch Backend API
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn src.api.main:app --reload
```
API running at `http://127.0.0.1:8000` | Docs at `http://127.0.0.1:8000/docs`

### 2. Launch React Frontend
```powershell
cd frontend
npm install
npm run dev
```
Dashboard running at `http://localhost:3000`

### 3. Verification Sequence
1. Open `http://localhost:3000` in browser.
2. Click **"Explore Risk Console →"**.
3. Click **Low Risk**, **Medium Risk**, or **High Risk** preset buttons:
   - **Low Risk**: Score 5 | `Low` Band | Recommendation `ALLOW`
   - **Medium Risk**: Score 52 | `Medium` Band | Recommendation `REVIEW`
   - **High Risk**: Score 84 | `Very High` Band | Recommendation `REVIEW`
4. Click **"Analyze Return Risk"** to execute real-time `POST /predict`.
5. Review score ring gauge, decision callout, natural language summary report, and SHAP factor bars.
