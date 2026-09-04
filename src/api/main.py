"""
src/api/main.py

FastAPI backend application for ReturnShield AI.
Exposes a health check endpoint and a risk prediction scoring endpoint with SHAP
explanations. Implements prediction audit logging to data/audit_log.jsonl.

IMPORTANT - DEFENSE ONLY:
  - Recommends ALLOW or REVIEW outcomes.
  - Never performs automated customer blocking or return rejections.
  - Strict CORS configured to local development origins only.
"""

import datetime
import hashlib
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, status
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.api.schemas import ReturnRecordRequest, PredictionResponse, HealthResponse
from src.model.model import load_artifacts, MODEL_ARTIFACT_PATH, FEATURE_ENGINEER_ARTIFACT_PATH
from src.model.explainability import get_explainer, explain_prediction, EXPLAINER_ARTIFACT_PATH

# ---------------------------------------------------------------------------
# Global ML Resources State
# ---------------------------------------------------------------------------
ml_resources: Dict[str, Any] = {
    "model": None,
    "feature_engineer": None,
    "explainer": None,
    "loaded": False,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan handler. Loads the pre-trained Logistic Regression baseline
    pipeline, fitted FeatureEngineer, and pre-saved SHAP explainer on startup.
    Cleans up cached objects on shutdown.
    """
    try:
        # Re-use already saved artifacts
        model, fe = load_artifacts(
            model_path=MODEL_ARTIFACT_PATH,
            fe_path=FEATURE_ENGINEER_ARTIFACT_PATH
        )
        explainer = get_explainer(model, explainer_path=EXPLAINER_ARTIFACT_PATH)
        
        ml_resources["model"] = model
        ml_resources["feature_engineer"] = fe
        ml_resources["explainer"] = explainer
        ml_resources["loaded"] = True
        print("  [Startup] All model artifacts loaded successfully.")
    except Exception as e:
        print(f"  [Startup] ERROR: Failed to load model artifacts: {e}")
        ml_resources["loaded"] = False
        
    yield
    
    # Cleanup resources on shutdown
    ml_resources.clear()
    ml_resources["loaded"] = False
    print("  [Shutdown] Cleaned up cached model resources.")


# ---------------------------------------------------------------------------
# FastAPI Application Config
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ReturnShield AI API",
    description="REST API for ecommerce return abuse risk assessment and explainability. Decision-support only.",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------------------------
# Operational Threshold — server-side only, not client-controlled
# ---------------------------------------------------------------------------
# Read from environment variable REVIEW_THRESHOLD; fall back to Phase 2D
# calibrated value of 0.30. Clients cannot override this.
_REVIEW_THRESHOLD: float = float(os.environ.get("REVIEW_THRESHOLD", "0.30"))

# Configure CORS - local development origins only (no unrestricted '*')
origins = [
    "http://localhost",
    "http://localhost:3000",  # React
    "http://localhost:5173",  # Vite
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Logging Helper
# ---------------------------------------------------------------------------
def _pseudonymize(value: str) -> str:
    """Return the first 12 hex characters of the SHA-256 hash of value.
    Preserves auditability (uniqueness) without storing raw identifiers."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def write_audit_log(
    order_id: str,
    customer_id: str,
    probability: float,
    score: int,
    band: str,
    decision: str
) -> str:
    """
    Logs non-sensitive prediction events in JSON Lines format for auditability.
    order_id and customer_id are SHA-256 pseudonymized before storage.
    Saves to data/audit_log.jsonl.
    """
    prediction_id = str(uuid.uuid4())
    log_entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prediction_id": prediction_id,
        "order_id_hash": _pseudonymize(order_id),
        "customer_id_hash": _pseudonymize(customer_id),
        "risk_probability": round(probability, 4),
        "risk_score": score,
        "risk_band": band,
        "decision": decision
    }
    
    log_dir = "data"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "audit_log.jsonl")
    
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"ERROR: Failed to write to audit log: {e}")
        
    return prediction_id


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Get API health status.",
    description="Returns the status of the API and shows whether all required ML artifacts are loaded and ready."
)
async def health_check():
    # Verify presence of the joblib files on disk as well as cached state
    artifacts_ready = {
        "model": os.path.exists(MODEL_ARTIFACT_PATH),
        "feature_engineer": os.path.exists(FEATURE_ENGINEER_ARTIFACT_PATH),
        "explainer": os.path.exists(EXPLAINER_ARTIFACT_PATH),
    }
    
    is_healthy = ml_resources["loaded"] and all(artifacts_ready.values())
    
    return HealthResponse(
        status="ok" if is_healthy else "degraded",
        model_loaded=is_healthy,
        artifacts=artifacts_ready
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Evaluate return abuse risk.",
    description="Analyzes return request transaction and behavior details to return probability, risk score, band, recommendation, and SHAP explainability reasons."
)
async def predict_risk(request: ReturnRecordRequest):
    # Guard against missing artifacts
    if not ml_resources["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded on the server. Please check the /health status."
        )
        
    try:
        # Convert Pydantic request to dict (no review_threshold field in schema)
        request_data = request.model_dump()
        
        # Use server-controlled threshold — not client-provided
        review_threshold = _REVIEW_THRESHOLD
        
        # Prepare 1-row DataFrame for inference pipeline
        df_row = pd.DataFrame([request_data])
        
        # Resolve models from cache
        model = ml_resources["model"]
        fe = ml_resources["feature_engineer"]
        explainer = ml_resources["explainer"]
        
        # Execute explain prediction flow (which computes shap + scoring)
        explanation = explain_prediction(
            raw_df=df_row,
            model=model,
            feature_engineer=fe,
            explainer=explainer,
            review_threshold=review_threshold
        )
        
        # Record structured audit log (defense-only)
        prediction_id = write_audit_log(
            order_id=request.order_id,
            customer_id=request.customer_id,
            probability=explanation["probability"],
            score=explanation["risk_score"],
            band=explanation["risk_band"],
            decision=explanation["recommendation"]
        )
        
        # Construct final response payload
        return PredictionResponse(
            prediction_id=prediction_id,
            order_id=request.order_id,
            customer_id=request.customer_id,
            risk_probability=explanation["probability"],
            risk_score=explanation["risk_score"],
            risk_band=explanation["risk_band"],
            decision=explanation["recommendation"],
            summary=explanation["summary"],
            positive_factors=explanation["positive_factors"],
            negative_factors=explanation["negative_factors"],
            base_value_log_odds=explanation["base_value_log_odds"]
        )
        
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as e:
        # Log error locally, return generic exception detail to client to avoid path/trace leakage
        print(f"Prediction Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during prediction scoring."
        )
