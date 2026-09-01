"""
src/model/model.py

Core model module for ReturnShield AI — Phase 2C.

Provides reusable functions for:
  - Loading and preparing data for training
  - Training the baseline Logistic Regression classifier
  - Evaluating the trained model on a held-out test set
  - Saving and loading model + feature-engineering artifacts
  - Generating abusive-return probability (risk score) for inference

Design notes:
  - All feature engineering is delegated to FeatureEngineer (Phase 2B).
    No feature logic is duplicated here.
  - The model is a sklearn Pipeline:  StandardScaler -> LogisticRegression.
    StandardScaler is required because Logistic Regression is sensitive to feature
    scale and the 41 features span very different numeric ranges.
  - class_weight='balanced' handles the 63.6% / 36.4% class imbalance without
    oversampling.
  - All randomness is controlled via RANDOM_STATE = 42.
  - Artifacts are saved with joblib for reliable sklearn object serialisation.
"""

import os
# pyrefly: ignore [missing-import]
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from src.features.feature_engineering import (
    FeatureEngineer,
    load_and_preprocess_data,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_ARTIFACT_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
FEATURE_ENGINEER_ARTIFACT_PATH = os.path.join(ARTIFACTS_DIR, "feature_engineer.joblib")


# ---------------------------------------------------------------------------
# Data loading and split
# ---------------------------------------------------------------------------
def load_training_data(csv_path: str = "data/returns.csv"):
    """
    Loads the raw dataset, performs a stratified random 80/20 train/test split,
    applies the FeatureEngineer (fit on train only), and returns ready-to-use
    feature matrices and target vectors.

    Parameters
    ----------
    csv_path : str
        Path to the raw returns CSV file.

    Returns
    -------
    X_train : pd.DataFrame  — engineered training features (n_train × 41)
    X_test  : pd.DataFrame  — engineered test features    (n_test  × 41)
    y_train : pd.Series     — training labels
    y_test  : pd.Series     — test labels
    fe      : FeatureEngineer — fitted feature engineering pipeline
    """
    # 1. Load raw data + target
    df, y = load_and_preprocess_data(csv_path)
    if y is None:
        raise ValueError("Target column 'is_abusive_return' not found in dataset.")

    # 2. Stratified random split (preserves class ratio in both splits)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    y_train = train_df["is_abusive_return"].copy()
    y_test = test_df["is_abusive_return"].copy()

    # 3. Fit FeatureEngineer on TRAIN only, then transform both splits
    fe = FeatureEngineer()
    X_train = fe.fit_transform(train_df)
    X_test = fe.transform(test_df)

    # Safety: confirm target is not in feature matrix
    assert "is_abusive_return" not in X_train.columns, "Target leakage detected in X_train!"
    assert "is_abusive_return" not in X_test.columns, "Target leakage detected in X_test!"

    return X_train, X_test, y_train, y_test, fe


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Builds and fits a sklearn Pipeline:
        StandardScaler -> LogisticRegression(class_weight='balanced')

    StandardScaler is necessary because LR is sensitive to feature scale and
    the 41 features span very different numeric ranges (e.g. customer_age_days
    can be 30–1365 while chargeback_rate is 0–1).

    class_weight='balanced' automatically compensates for the 63.6% / 36.4%
    class imbalance without oversampling.

    Parameters
    ----------
    X_train : pd.DataFrame  — engineered training features
    y_train : pd.Series     — training labels (0 = normal, 1 = abusive)

    Returns
    -------
    model : fitted sklearn Pipeline
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,          # default regularisation strength — no aggressive tuning
        )),
    ])
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluates the fitted model on the held-out test set.

    Metrics returned:
      accuracy, precision, recall, f1, roc_auc, confusion_matrix

    Note on threshold: sklearn's default decision threshold (0.5) is used here.
    In production a different threshold may be preferred based on the
    cost-benefit analysis (FP cost vs FN cost). That is a Phase 2D concern.

    Parameters
    ----------
    model   : fitted sklearn Pipeline
    X_test  : engineered test features
    y_test  : true test labels

    Returns
    -------
    results : dict with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1 (abusive)

    results = {
        "accuracy":         round(float(accuracy_score(y_test, y_pred)), 4),
        "precision":        round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":           round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":               round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":          round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test":           int(len(y_test)),
        "n_positives":      int(y_test.sum()),
        "n_negatives":      int((y_test == 0).sum()),
    }
    return results


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------
def save_artifacts(
    model: Pipeline,
    feature_engineer: FeatureEngineer,
    model_path: str = MODEL_ARTIFACT_PATH,
    fe_path: str = FEATURE_ENGINEER_ARTIFACT_PATH,
) -> None:
    """
    Saves the trained model pipeline and fitted FeatureEngineer to disk using
    joblib.

    Parameters
    ----------
    model            : fitted sklearn Pipeline
    feature_engineer : fitted FeatureEngineer
    model_path       : destination path for model artifact
    fe_path          : destination path for feature-engineer artifact
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(fe_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(feature_engineer, fe_path)

    print(f"  Model saved      -> {model_path}")
    print(f"  FeatureEngineer  -> {fe_path}")


def load_artifacts(
    model_path: str = MODEL_ARTIFACT_PATH,
    fe_path: str = FEATURE_ENGINEER_ARTIFACT_PATH,
):
    """
    Loads the trained model pipeline and fitted FeatureEngineer from disk.

    Parameters
    ----------
    model_path : path to saved model artifact
    fe_path    : path to saved feature-engineer artifact

    Returns
    -------
    model            : loaded sklearn Pipeline
    feature_engineer : loaded FeatureEngineer
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not os.path.exists(fe_path):
        raise FileNotFoundError(f"FeatureEngineer artifact not found: {fe_path}")

    model = joblib.load(model_path)
    feature_engineer = joblib.load(fe_path)
    return model, feature_engineer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_risk_score(
    model: Pipeline,
    feature_engineer: FeatureEngineer,
    raw_df: pd.DataFrame,
) -> np.ndarray:
    """
    Generates the abusive-return probability (risk score in [0, 1]) for one or
    more raw return records.

    Parameters
    ----------
    model            : fitted sklearn Pipeline (loaded via load_artifacts)
    feature_engineer : fitted FeatureEngineer  (loaded via load_artifacts)
    raw_df           : pd.DataFrame with the same raw columns as data/returns.csv
                       (is_abusive_return must NOT be present)

    Returns
    -------
    scores : np.ndarray of shape (n_samples,) — probability of class 1 (abusive)
             Values are in [0.0, 1.0].
    """
    if "is_abusive_return" in raw_df.columns:
        raise ValueError(
            "raw_df must not contain 'is_abusive_return' during inference. "
            "Remove the target column before calling predict_risk_score()."
        )

    X = feature_engineer.transform(raw_df)
    scores = model.predict_proba(X)[:, 1]
    return scores
