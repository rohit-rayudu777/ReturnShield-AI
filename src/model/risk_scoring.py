"""
src/model/risk_scoring.py

Phase 2D - Risk Scoring and False-Positive Cost Analysis for ReturnShield AI.

This module converts the model's abusive-return probability into a structured
risk score (0-100), applies configurable decision thresholds, performs threshold
analysis on the held-out test set, and calculates illustrative false-positive
and false-negative business costs.

IMPORTANT - DEFENSE ONLY:
  This system is a decision-support tool only.
  It recommends REVIEW or ALLOW outcomes.
  It does NOT automatically deny returns, block customers, or suspend accounts.

COST VALUES:
  All cost figures used in this module are ILLUSTRATIVE EXAMPLES ONLY.
  They do not represent actual Razorpay or business figures.
  Operators must calibrate these values against real loss data before deployment.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------------------------
# Risk band configuration (configurable, not hard-coded in logic)
# ---------------------------------------------------------------------------

DEFAULT_RISK_BANDS: Dict[str, Tuple[int, int]] = {
    "Low":       (0,  29),
    "Medium":    (30, 59),
    "High":      (60, 79),
    "Very High": (80, 100),
}

# Default probability threshold for flagging a return for review
DEFAULT_REVIEW_THRESHOLD: float = 0.50

# Thresholds to sweep during analysis
DEFAULT_THRESHOLDS: List[float] = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# Illustrative business costs (units are arbitrary; see RISK_SCORING.md)
DEFAULT_FP_COST: float = 100.0   # cost of wrongly flagging a legitimate return
DEFAULT_FN_COST: float = 500.0   # cost of missing a genuinely abusive return


# ---------------------------------------------------------------------------
# Core conversion: probability -> risk score (0-100) and risk band
# ---------------------------------------------------------------------------

def prob_to_score(probability: float) -> int:
    """
    Converts an abusive-return probability in [0.0, 1.0] to a risk score
    in [0, 100] by linear scaling.

    Parameters
    ----------
    probability : float
        Model's predicted probability for class 1 (abusive return).

    Returns
    -------
    int
        Risk score in [0, 100].
    """
    prob = float(np.clip(probability, 0.0, 1.0))
    return int(round(prob * 100))


def prob_to_score_array(probabilities: np.ndarray) -> np.ndarray:
    """
    Vectorised version of prob_to_score for arrays.

    Parameters
    ----------
    probabilities : np.ndarray of shape (n_samples,)

    Returns
    -------
    np.ndarray of int, shape (n_samples,)
    """
    clipped = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    return np.round(clipped * 100).astype(int)


def score_to_band(
    score: int,
    risk_bands: Optional[Dict[str, Tuple[int, int]]] = None,
) -> str:
    """
    Maps a risk score (0-100) to a named risk band.

    Parameters
    ----------
    score      : int in [0, 100]
    risk_bands : optional custom band config; defaults to DEFAULT_RISK_BANDS

    Returns
    -------
    str : band name e.g. 'Low', 'Medium', 'High', 'Very High'
    """
    bands = risk_bands if risk_bands is not None else DEFAULT_RISK_BANDS
    for band_name, (lo, hi) in bands.items():
        if lo <= score <= hi:
            return band_name
    return "Unknown"


def classify_return(
    probability: float,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> str:
    """
    Converts a probability into a decision recommendation.

    Outcomes:
      "REVIEW" - probability >= review_threshold: flag for human operator review
      "ALLOW"  - probability <  review_threshold: allow the return automatically

    IMPORTANT: This function returns a recommendation only.
    No automatic blocking, denial, or account action is taken.

    Parameters
    ----------
    probability       : float in [0, 1]
    review_threshold  : float, configurable decision boundary

    Returns
    -------
    str : 'REVIEW' or 'ALLOW'
    """
    return "REVIEW" if probability >= review_threshold else "ALLOW"


# ---------------------------------------------------------------------------
# Full scoring for a batch of raw return requests
# ---------------------------------------------------------------------------

def score_returns(
    model,
    feature_engineer,
    raw_df: pd.DataFrame,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    risk_bands: Optional[Dict[str, Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """
    Produces a full risk-scoring report for a batch of return records.

    Parameters
    ----------
    model             : fitted sklearn Pipeline (loaded from artifacts)
    feature_engineer  : fitted FeatureEngineer
    raw_df            : DataFrame with same schema as data/returns.csv
                        (must NOT contain 'is_abusive_return')
    review_threshold  : probability above which a return is flagged for REVIEW
    risk_bands        : optional custom band configuration

    Returns
    -------
    pd.DataFrame with columns:
      order_id (if present), probability, risk_score, risk_band, recommendation
    """
    if "is_abusive_return" in raw_df.columns:
        raise ValueError(
            "raw_df must not contain 'is_abusive_return' during inference. "
            "Remove the target column before calling score_returns()."
        )

    X = feature_engineer.transform(raw_df)
    probabilities = model.predict_proba(X)[:, 1]
    scores = prob_to_score_array(probabilities)
    bands = [score_to_band(s, risk_bands) for s in scores]
    recommendations = [classify_return(p, review_threshold) for p in probabilities]

    result = pd.DataFrame({
        "probability":    probabilities.round(4),
        "risk_score":     scores,
        "risk_band":      bands,
        "recommendation": recommendations,
    }, index=raw_df.index)

    if "order_id" in raw_df.columns:
        result.insert(0, "order_id", raw_df["order_id"].values)

    return result


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------

def analyse_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> dict:
    """
    Evaluates a single probability threshold and calculates all key metrics
    including illustrative business costs.

    Parameters
    ----------
    y_true    : true binary labels (0/1)
    y_prob    : predicted probabilities for class 1
    threshold : decision threshold in [0, 1]
    fp_cost   : illustrative cost per false positive (default 100 units)
    fn_cost   : illustrative cost per false negative (default 500 units)

    Returns
    -------
    dict with threshold metrics and costs
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = len(y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # false positive rate

    total_cost   = fp * fp_cost + fn * fn_cost
    avg_cost     = total_cost / total if total > 0 else 0.0

    return {
        "threshold":       round(threshold, 2),
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
        "true_positives":  int(tp),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "fp_rate":         round(fpr, 4),
        "fp_cost_unit":    fp_cost,
        "fn_cost_unit":    fn_cost,
        "total_cost":      round(total_cost, 2),
        "avg_cost_per_return": round(avg_cost, 4),
    }


def run_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> pd.DataFrame:
    """
    Sweeps multiple thresholds and returns a DataFrame of results.

    Parameters
    ----------
    y_true     : true binary labels (0/1)
    y_prob     : predicted probabilities for class 1
    thresholds : list of thresholds to evaluate (default: DEFAULT_THRESHOLDS)
    fp_cost    : illustrative cost per false positive
    fn_cost    : illustrative cost per false negative

    Returns
    -------
    pd.DataFrame with one row per threshold, sorted by threshold ascending
    """
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    rows = [
        analyse_threshold(y_true, y_prob, t, fp_cost, fn_cost)
        for t in thresholds
    ]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_best_threshold(
    analysis_df: pd.DataFrame,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> dict:
    """
    Identifies the threshold with the lowest total illustrative cost
    from a threshold analysis DataFrame.

    If two thresholds tie on total cost, the higher threshold (fewer flags)
    is preferred to reduce operator workload.

    Parameters
    ----------
    analysis_df : output of run_threshold_analysis()
    fp_cost     : illustrative FP cost used (for documentation in output)
    fn_cost     : illustrative FN cost used (for documentation in output)

    Returns
    -------
    dict with the selected threshold row and a human-readable explanation
    """
    best_idx = analysis_df["total_cost"].idxmin()
    best_row = analysis_df.loc[best_idx].to_dict()
    explanation = (
        f"Threshold {best_row['threshold']:.2f} produces the lowest illustrative "
        f"total cost ({best_row['total_cost']:,.0f} units) under the assumption that "
        f"each false positive costs {fp_cost:.0f} units and each false negative costs "
        f"{fn_cost:.0f} units. "
        f"At this threshold: precision={best_row['precision']:.4f}, "
        f"recall={best_row['recall']:.4f}, "
        f"FP={best_row['false_positives']}, FN={best_row['false_negatives']}. "
        f"NOTE: The optimal threshold is a policy decision. Operators should adjust "
        f"FP/FN cost assumptions to reflect actual business impact before deployment."
    )
    return {"best_row": best_row, "explanation": explanation}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_threshold_analysis(
    analysis_df: pd.DataFrame,
    output_path: str = "data/threshold_analysis.csv",
) -> None:
    """
    Saves the threshold analysis DataFrame to CSV.

    Parameters
    ----------
    analysis_df : output of run_threshold_analysis()
    output_path : destination CSV path
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    analysis_df.to_csv(output_path, index=False)
    print(f"Threshold analysis saved -> {output_path}")
