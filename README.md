# ReturnShield AI

> **Defense-only AI-powered return risk assessment and decision-support intelligence for e-commerce.**

Built for the **Razorpay Buildathon — Track 2: AI Risk Manager**.

---

## Problem Statement

E-commerce return abuse—including wardrobing, empty box returns, counterfeit swaps, and friendly fraud—costs merchants tens of billions of dollars annually. Traditional rule-based fraud prevention systems are either overly rigid (causing high customer friction and false-positive rejections) or too loose (allowing widespread return fraud losses).

## Solution Overview

**ReturnShield AI** is an explainable, cost-aware machine learning risk-scoring system designed to evaluate return requests at transaction time. It generates statistical risk probabilities, maps them to calibrated 0–100 risk scores, and provides local SHAP feature attributions in natural language to support human fraud analyst decision-making.

> 🛡️ **Defense-Only Safeguard**: ReturnShield AI recommends **`ALLOW`** or **`REVIEW`** outcomes only. It **never** automatically blocks customer accounts, cancels transactions, or rejects returns without human operator authorization.

---

## Key Features

- **Synthetic Return-Risk Dataset**: 100,000 synthetic transaction records modeling customer history, velocity, payment behavior, and return patterns.
- **Advanced Feature Engineering**: 41 engineered features calculating customer return rates, payment failure rates, AOV ratios, velocity metrics, and delivery delay interactions.
- **Supervised ML Risk Predictor**: Scaled Logistic Regression baseline with class balancing, trained with fixed random seed (42) for 100% reproducibility.
- **Empirical Evaluation**: Evaluated on a 20,000-sample held-out test set (no data leakage).
- **Configurable Risk Scoring**: Probability-to-score linear conversion (0–100) mapped into 4 risk bands (`Low`, `Medium`, `High`, `Very High`).
- **False-Positive Cost Analysis**: Threshold optimization model balancing false-positive customer friction costs against false-negative fraud loss costs.
- **SHAP Explainability**: Local feature attributions (`shap.LinearExplainer`) producing friendly feature names, raw values, and natural language summary reports.
- **FastAPI Backend Services**: Production-grade REST API (`GET /health`, `POST /predict`) with Pydantic v2 schemas and audit logging.
- **React Dashboard**: High-contrast dark obsidian & warm gold editorial interface featuring interactive form evaluation, ring risk gauge, and SHAP factor bars.

---

## Architecture Flow

```
Synthetic Data (100k records)
       │
       ▼
Feature Engineering (41 features)
       │
       ▼
Supervised ML Model (Logistic Regression + StandardScaler)
       │
       ▼
Risk Probability (0.00 – 1.00) ──► SHAP LinearExplainer
       │                                  │
       ▼                                  ▼
Risk Score (0 – 100)              Natural Language & Factor Breakdown
       │                                  │
       ▼                                  ▼
Cost-Aware Threshold (0.30) ──► Decision Recommendation (ALLOW / REVIEW)
                                          │
                                          ▼
                               FastAPI REST API Server
                                          │
                                          ▼
                               React + Vite Dashboard
                                          │
                                          ▼
                               Human Risk Analyst Review
```

---

## Model & Empirical Evaluation Results

Model performance measured on a held-out synthetic test set (20,000 samples, 20% split):

| Evaluation Metric | Value |
|---|---|
| **Accuracy** | **77.36%** |
| **Precision** | **88.38%** |
| **Recall** | **74.16%** |
| **F1 Score** | **80.65%** |
| **ROC-AUC** | **86.15%** |

### Confusion Matrix (Held-out Test Set)

```
                       Predicted Normal (0)    Predicted Abusive (1)
  True Normal (0)             11,739                  981 (FP)
  True Abusive (1)             3,546 (FN)            3,734 (TP)
```

> *Note: Evaluated on a held-out synthetic test set. Synthetic baseline performance does not represent production model accuracy.*

---

## False-Positive Cost Analysis & Threshold Calibration

In e-commerce return risk management, a **False Positive (FP)** (wrongly flagging a loyal customer) incurs customer friction and operational review costs, while a **False Negative (FN)** (missing a fraudulent return) incurs direct inventory and refund loss.

- **Illustrative FP Cost**: 100 cost units
- **Illustrative FN Cost**: 500 cost units
- **Optimal Probability Threshold**: `0.30`

Sweeping decision thresholds from 0.30 to 0.80 established that `threshold = 0.30` minimizes overall business risk cost by catching 85.8% of abusive returns while restricting the false positive rate to 7.7%.

---

## Explainability (SHAP Attributions)

For every transaction payload sent to `POST /predict`, ReturnShield AI decomposes the prediction into exact SHAP feature attributions:

- **Factors Increasing Risk (+SHAP)**: Isolates features driving up risk (e.g., Cash on Delivery payment, high customer return rate, recent address changes).
- **Factors Mitigating Risk (-SHAP)**: Isolates features reducing risk (e.g., long customer account tenure, established order history, zero chargebacks).
- **Natural Language Summary**: Automatically formats top drivers into plain English for rapid analyst interpretation.

---

## Safety & Governance Principles

1. **Synthetic Data Only**: Trained and tested strictly on synthetic e-commerce datasets. No PII, customer identity, or real payment data used.
2. **Defense-Only System**: Returns recommendations (`ALLOW` / `REVIEW`) only. No automated account bans, order cancellations, or return denials.
3. **Server-Controlled Threshold**: The REVIEW decision boundary (`threshold = 0.30`) is set server-side via the `REVIEW_THRESHOLD` environment variable. Clients cannot override it.
4. **Pseudonymized Audit Logs**: Prediction audit entries store SHA-256-truncated hashes of `order_id`/`customer_id` — not raw identifiers — preserving auditability without PII exposure.
5. **Human-in-the-Loop**: Designed as a decision-support assistant for fraud operations teams.

---

## Technology Stack

- **Backend Framework**: Python 3.13 (local), Python 3.12-slim (Docker), FastAPI 0.100+, Uvicorn, Pydantic v2
- **Machine Learning**: scikit-learn (LogisticRegression, StandardScaler), SHAP (LinearExplainer), pandas, NumPy, joblib
- **Frontend Dashboard**: React 18, Vite 5, Vanilla CSS (Custom dark/gold editorial design system)
- **Container Stack**: Docker, Docker Compose v2, Nginx (reverse proxy + static serving)
- **Testing & Quality**: pytest, httpx, black, flake8

---

## Run with Docker

### Requirements
- Docker Desktop (with Docker Compose v2+)

### Launch Application
```powershell
docker compose up --build
```

Then open in your browser:
- **Frontend Dashboard**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **FastAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Stop Containers
```powershell
docker compose down
```

---

## Local Setup & Installation (Without Docker)

### Prerequisites
- Python 3.10+ (Python 3.13 tested)
- Node.js 18+ (for React frontend)

### 1. Backend Setup

```powershell
# 1. Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start FastAPI server (runs on http://127.0.0.1:8000)
python -m uvicorn src.api.main:app --reload
```

FastAPI Documentation available at: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### 2. Frontend Setup

```powershell
# 1. Navigate to frontend directory
cd frontend

# 2. Install Node dependencies
npm install

# 3. Start Vite dev server (runs on http://localhost:3000)
npm run dev
```

Dashboard available at: **[http://localhost:3000](http://localhost:3000)**

---

## Demo Flow for Judges

1. Start backend server (`python -m uvicorn src.api.main:app --reload`).
2. Start frontend dev server (`npm run dev` inside `frontend/`).
3. Open **`http://localhost:3000`** in browser.
4. Click **"Explore Risk Console →"** or navigate to Risk Console.
5. Click **Low Risk**, **Medium Risk**, or **High Risk** preset buttons to load synthetic test cases:
   - **Low Risk**: Score 5 | `Low` Band | Recommendation: `ALLOW`
   - **Medium Risk**: Score 52 | `Medium` Band | Recommendation: `REVIEW`
   - **High Risk**: Score 84 | `Very High` Band | Recommendation: `REVIEW`
6. Click **"Analyze Return Risk"** to execute `POST /predict`.
7. Inspect the ring risk gauge, decision badge, natural language assessment report, and SHAP factor attribution bars.

---

## Project Structure

```
ReturnShield_ai/
├── data/
│   ├── generator.py               # Phase 2A synthetic data generator
│   ├── returns.csv                # Raw synthetic dataset (100,000 records)
│   └── threshold_analysis.csv     # Phase 2D cost-benefit threshold sweep
├── docs/
│   ├── ARCHITECTURE.md            # System architecture details
│   ├── FEATURE_ENGINEERING.md     # Feature engineering documentation
│   ├── MODELING.md                # Phase 2C ML pipeline & evaluation report
│   ├── RISK_SCORING.md            # Phase 2D risk scoring & cost analysis
│   ├── EXPLAINABILITY.md          # Phase 2E SHAP explainability docs
│   ├── API.md                     # Phase 3 FastAPI documentation
│   ├── FRONTEND.md                # Phase 4 React frontend documentation
│   └── SUBMISSION.md              # Buildathon submission summary
├── frontend/
│   ├── Dockerfile                 # Multi-stage Node→Nginx container build
│   ├── nginx.conf                 # Nginx config: SPA fallback + /health + /predict proxy
│   ├── .env.docker                # Docker-specific Vite env (VITE_API_URL="")
│   ├── index.html                 # HTML template with Google serif fonts
│   ├── index_standalone.html      # Zero-dependency standalone HTML fallback
│   ├── package.json               # Node package configuration
│   ├── vite.config.js             # Vite configuration
│   └── src/
│       ├── main.jsx               # React entry point
│       ├── App.jsx                # Main App state controller
│       ├── App.css                # Dark & warm gold editorial theme CSS
│       └── components/            # Header, Footer, LandingPage, RiskConsole, AuthModal
├── src/
│   ├── api/
│   │   ├── main.py                # FastAPI app & endpoints (/health, /predict)
│   │   └── schemas.py             # Pydantic v2 validation models
│   ├── features/
│   │   └── feature_engineering.py # Feature engineering transformer pipeline
│   └── model/
│       ├── artifacts/             # Serialized joblib artifacts (model, fe, explainer)
│       ├── model.py               # ML training, evaluation, & inference pipeline
│       ├── risk_scoring.py        # Probability-to-score conversion & thresholding
│       └── explainability.py      # SHAP LinearExplainer & narrative generator
├── tests/
│   ├── test_features.py           # Feature engineering unit tests
│   ├── test_model.py              # ML model unit tests
│   ├── test_risk_scoring.py       # Risk scoring unit tests
│   ├── test_explainability.py     # SHAP explainability unit tests
│   └── test_api.py                # FastAPI REST endpoint unit tests
├── .dockerignore                  # Docker build context exclusion rules
├── .gitignore                     # Git ignore rules
├── docker-compose.yml             # Docker Compose: backend + frontend services
├── Dockerfile.backend             # FastAPI backend container definition
├── README.md                      # Project documentation & guide
└── requirements.txt               # Python package dependencies
```

---

## Limitations & Disclaimers

- **Synthetic Data**: Trained entirely on synthetic e-commerce data generated for the Razorpay Buildathon.
- **Hackathon Prototype**: Built as a demonstration prototype for Track 2 (AI Risk Manager).
- **Illustrative Costs**: Cost parameters (FP=100, FN=500) are illustrative examples for threshold optimization.
- **Frontend Demo Authentication**: Auth UI is a client-side demo session for buildathon demonstration.
- **Defense-Only Scope**: Decision support tool only; requires human analyst verification before taking enforcement actions.
