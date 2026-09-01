"""
tests/test_risk_scoring.py

Targeted unit tests for Phase 2D - Risk Scoring module.

Tests only essential functionality:
- probability -> 0-100 score conversion
- risk band assignment
- configurable threshold / recommendation
- cost calculation correctness
- threshold analysis produces expected columns
"""

import numpy as np
import pandas as pd
import pytest

from src.model.risk_scoring import (
    prob_to_score,
    prob_to_score_array,
    score_to_band,
    classify_return,
    analyse_threshold,
    run_threshold_analysis,
    select_best_threshold,
    DEFAULT_RISK_BANDS,
    DEFAULT_FP_COST,
    DEFAULT_FN_COST,
)


# ---------------------------------------------------------------------------
# prob_to_score
# ---------------------------------------------------------------------------

def test_prob_to_score_boundaries():
    assert prob_to_score(0.0) == 0
    assert prob_to_score(1.0) == 100


def test_prob_to_score_midpoint():
    assert prob_to_score(0.5) == 50


def test_prob_to_score_clamps_above_one():
    assert prob_to_score(1.5) == 100


def test_prob_to_score_clamps_below_zero():
    assert prob_to_score(-0.3) == 0


def test_prob_to_score_rounding():
    # 0.635 -> 63.5 -> rounds to 64
    assert prob_to_score(0.635) == 64


def test_prob_to_score_array():
    probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    scores = prob_to_score_array(probs)
    np.testing.assert_array_equal(scores, [0, 25, 50, 75, 100])


# ---------------------------------------------------------------------------
# score_to_band
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected_band", [
    (0,   "Low"),
    (15,  "Low"),
    (29,  "Low"),
    (30,  "Medium"),
    (45,  "Medium"),
    (59,  "Medium"),
    (60,  "High"),
    (70,  "High"),
    (79,  "High"),
    (80,  "Very High"),
    (95,  "Very High"),
    (100, "Very High"),
])
def test_score_to_band_default(score, expected_band):
    assert score_to_band(score) == expected_band


def test_score_to_band_custom_config():
    custom_bands = {"Safe": (0, 49), "Risky": (50, 100)}
    assert score_to_band(30, custom_bands) == "Safe"
    assert score_to_band(75, custom_bands) == "Risky"


# ---------------------------------------------------------------------------
# classify_return
# ---------------------------------------------------------------------------

def test_classify_return_above_threshold():
    assert classify_return(0.7, review_threshold=0.5) == "REVIEW"


def test_classify_return_below_threshold():
    assert classify_return(0.3, review_threshold=0.5) == "ALLOW"


def test_classify_return_at_threshold():
    # At exactly the threshold it should be REVIEW (>=)
    assert classify_return(0.5, review_threshold=0.5) == "REVIEW"


def test_classify_return_custom_threshold():
    # With a strict threshold of 0.8
    assert classify_return(0.75, review_threshold=0.8) == "ALLOW"
    assert classify_return(0.80, review_threshold=0.8) == "REVIEW"


# ---------------------------------------------------------------------------
# analyse_threshold - cost calculation
# ---------------------------------------------------------------------------

def test_analyse_threshold_cost_calculation():
    """
    With known y_true / y_prob, verify that FP/FN counts and costs are correct.
    """
    # Construct a scenario with known outcomes at threshold=0.5:
    # y_true: [0, 0, 1, 1]
    # y_prob: [0.2, 0.8, 0.3, 0.9]
    # Predictions at 0.5: [0, 1, 0, 1]
    # TN=1, FP=1, FN=1, TP=1
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.9])

    result = analyse_threshold(y_true, y_prob, threshold=0.5, fp_cost=100.0, fn_cost=500.0)

    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_positives"]  == 1
    assert result["true_negatives"]  == 1
    assert result["total_cost"]      == pytest.approx(600.0)   # 1*100 + 1*500
    assert result["avg_cost_per_return"] == pytest.approx(150.0)  # 600 / 4


def test_analyse_threshold_returns_expected_keys():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.6])
    result = analyse_threshold(y_true, y_prob, threshold=0.5)
    expected_keys = {
        "threshold", "precision", "recall", "f1",
        "true_positives", "true_negatives", "false_positives", "false_negatives",
        "fp_rate", "total_cost", "avg_cost_per_return",
    }
    assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# run_threshold_analysis
# ---------------------------------------------------------------------------

def test_run_threshold_analysis_columns():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9, 0.3, 0.7])
    df = run_threshold_analysis(y_true, y_prob, thresholds=[0.3, 0.5, 0.7])

    assert len(df) == 3
    assert "threshold"       in df.columns
    assert "precision"       in df.columns
    assert "recall"          in df.columns
    assert "f1"              in df.columns
    assert "false_positives" in df.columns
    assert "false_negatives" in df.columns
    assert "total_cost"      in df.columns


def test_run_threshold_analysis_sorted_ascending():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.4, 0.6])
    df = run_threshold_analysis(y_true, y_prob, thresholds=[0.7, 0.3, 0.5])
    thresholds = df["threshold"].tolist()
    assert thresholds == sorted(thresholds)


# ---------------------------------------------------------------------------
# select_best_threshold
# ---------------------------------------------------------------------------

def test_select_best_threshold_picks_lowest_cost():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7, 0.45, 0.6])
    df = run_threshold_analysis(y_true, y_prob, thresholds=[0.3, 0.5, 0.7])
    result = select_best_threshold(df)

    assert "best_row" in result
    assert "explanation" in result
    # The selected threshold must actually be in the analysis
    assert result["best_row"]["threshold"] in [0.3, 0.5, 0.7]
    # Its total cost must be the minimum
    assert result["best_row"]["total_cost"] == df["total_cost"].min()
