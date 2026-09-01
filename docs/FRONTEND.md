# ReturnShield AI — React Frontend / Risk Dashboard

## Overview

The Phase 4 frontend is a **single-page React application** (built with Vite) that
provides a premium, dark-mode fintech risk dashboard for the ReturnShield AI system.
It communicates directly with the Phase 3 FastAPI backend (`/health`, `/predict`) and
renders risk scores, decisions, natural-language summaries, and SHAP feature attributions
in real time.

> **Defense-Only**: The dashboard displays ALLOW / REVIEW recommendations for human
> operator use only. It does **not** automatically block accounts, reject returns, or
> take any automated enforcement action.

---

## Technology Stack

| Component | Technology |
|---|---|
| Framework | React 18 + Vite 5 |
| Styling | Vanilla CSS (custom dark-mode fintech theme) |
| Font | Google Fonts — Inter & Outfit |
| HTTP | Native `fetch` API |
| Backend API | FastAPI on `http://localhost:8000` |

---

## Directory Structure

```
frontend/
├── index.html              # HTML template with Google Fonts preconnect
├── index_standalone.html   # Zero-dependency standalone version (no Node required)
├── package.json            # npm project config (React + Vite)
├── vite.config.js          # Vite config — serves on port 3000, CORS enabled
└── src/
    ├── main.jsx            # React entry point
    ├── App.jsx             # Core application — form, API integration, all components
    ├── App.css             # High-fidelity dark-theme stylesheet
    └── index.css           # Global body/reset styles
```

---

## Dashboard Features

### Left Panel — Form Input
- All **20 transaction features** required by `POST /predict` grouped into:
  - **Identifiers** — Order ID, Customer ID, Timestamp
  - **Transaction Basics** — Amount, Discount %, Category, Payment Method
  - **Customer History** — Account Age, Previous Orders/Returns, Return Rate, AOV
  - **Velocity Metrics** — Orders/Returns in last 7 and 30 days
  - **Risk Flags & Timing** — Delivery days, Return delay, Address changes, Payment failures, Chargebacks
  - **Order Flags** — Is First Order, Is High Value Order
- **Configurable `review_threshold`** (default `0.30`) — changing this value adjusts
  the ALLOW / REVIEW decision boundary in real time on the next submission.
- **Load Demo Case** bar with three presets: **Low Risk**, **Medium Risk**, **High Risk**.

### Right Panel — Results
| State | Display |
|---|---|
| Inactive | Instructions callout + empty state illustration |
| Loading | Animated circular spinner |
| Error | Error banner with API re-check button |
| Success | Full risk assessment output (see below) |

#### Success Output
1. **SVG Risk Score Gauge** — animated circular gauge (0–100), colored by risk band.
2. **Decision Recommendation Badge** — `ALLOW` (green) or `REVIEW` (red).
3. **Risk Band + Abuse Probability** metric boxes.
4. **Natural Language Summary** — the `summary` string from the API response.
5. **SHAP Factor Cards**:
   - *Factors Increasing Risk* (red bars) — top positive SHAP contributors.
   - *Factors Mitigating Risk* (green bars) — top negative SHAP contributors.
   - Each factor shows its display name, raw value, and SHAP contribution magnitude.

### Header
- **API Health badge** — live green/offline indicator that polls `GET /health`.
- **Refresh button** — manually re-checks API health.

---

## Quick Start (React + Vite — requires Node.js ≥ 18)

```powershell
# 1. Install Node.js from https://nodejs.org if not already installed

# 2. Install frontend dependencies
cd frontend
npm install

# 3. Start the Vite dev server (port 3000)
npm run dev

# 4. In a separate terminal, start the FastAPI backend
cd ..
.venv\Scripts\Activate.ps1
python -m uvicorn src.api.main:app --reload
```

Then open **http://localhost:3000** in your browser.

---

## Standalone Mode (No Node.js Required)

If Node.js is not available, open `frontend/index_standalone.html` directly in a browser.
This file is a fully self-contained version of the dashboard built with vanilla JS and
the same CSS design system. It connects to the same `http://localhost:8000` FastAPI backend.

---

## Configuration

The Vite app reads the API base URL from the `VITE_API_URL` environment variable:

```
VITE_API_URL=http://localhost:8000   # default
```

To point to a different backend, create `frontend/.env.local`:
```
VITE_API_URL=http://your-api-host:8000
```

---

## Design System

| Token | Value | Purpose |
|---|---|---|
| `--bg-primary` | `#0a0d12` | Page background |
| `--bg-panel` | `#161c28` | Card/panel backgrounds |
| `--accent-color` | `#3b82f6` | Blue — interactive highlights |
| `--color-low` | `#10b981` | Low risk (green) |
| `--color-med` | `#eab308` | Medium risk (gold) |
| `--color-high` | `#f97316` | High risk (orange) |
| `--color-veryhigh` | `#ef4444` | Very High risk (red) |

---

## Defense-Only Constraints

The dashboard enforces the following constraints by design:

- Returns only `ALLOW` or `REVIEW` recommendations — no enforcement actions.
- Does not call any account-blocking, transaction-cancellation, or return-rejection APIs.
- The `review_threshold` is a **display parameter** that changes the recommendation label
  in the response; it does not modify any backend model or enforcement logic.
- Audit logging is handled server-side by the FastAPI backend.
