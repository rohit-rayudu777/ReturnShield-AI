"""
src/model/train.py

Runnable end-to-end training script for ReturnShield AI - Phase 2C.

Usage (from project root, with .venv active):
    python src/model/train.py
    python src/model/train.py --data data/returns.csv

This script:
  1. Loads data/returns.csv
  2. Performs a stratified 80/20 train/test split (random_state=42)
  3. Fits the FeatureEngineer on training data only
  4. Trains a Logistic Regression baseline model
  5. Evaluates on the held-out test set
  6. Prints all evaluation metrics
  7. Saves model + FeatureEngineer artifacts to src/model/artifacts/

No API, deployment, or frontend logic is included.
"""

import argparse
import sys
import os

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.model import (
    load_training_data,
    train_model,
    evaluate_model,
    save_artifacts,
    MODEL_ARTIFACT_PATH,
    FEATURE_ENGINEER_ARTIFACT_PATH,
)


def print_section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_metrics(results: dict) -> None:
    cm = results["confusion_matrix"]
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print(f"  ROC-AUC   : {results['roc_auc']:.4f}")
    print()
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print("                  Pred Normal  Pred Abusive")
    print(f"  Actual Normal :  {cm[0][0]:>9}    {cm[0][1]:>9}")
    print(f"  Actual Abusive:  {cm[1][0]:>9}    {cm[1][1]:>9}")
    print()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"  True Negatives  (TN): {tn}")
    print(f"  False Positives (FP): {fp}  (legitimate returns flagged as abusive)")
    print(f"  False Negatives (FN): {fn}  (abusive returns missed)")
    print(f"  True Positives  (TP): {tp}")


def main(csv_path: str = "data/returns.csv") -> None:

    # ------------------------------------------------------------------
    print_section("ReturnShield AI - Phase 2C Model Training")
    # ------------------------------------------------------------------

    print_section("1. Loading & Splitting Data")
    print(f"  Dataset  : {csv_path}")
    print("  Split    : stratified random 80/20  (random_state=42)")
    print("  Fitting FeatureEngineer on training data only...")

    X_train, X_test, y_train, y_test, fe = load_training_data(csv_path)

    print(f"  Training samples : {len(X_train):>7,}")
    print(f"  Test samples     : {len(X_test):>7,}")
    print(f"  Features         : {X_train.shape[1]}")

    train_abuse_rate = y_train.mean()
    test_abuse_rate = y_test.mean()
    print(f"\n  Class distribution (train) - normal: {1 - train_abuse_rate:.2%}  abusive: {train_abuse_rate:.2%}")
    print(f"  Class distribution (test)  - normal: {1 - test_abuse_rate:.2%}  abusive: {test_abuse_rate:.2%}")

    # ------------------------------------------------------------------
    print_section("2. Training Model")
    # ------------------------------------------------------------------
    print("  Model    : sklearn Pipeline(StandardScaler -> LogisticRegression)")
    print("  Scaler   : StandardScaler  (fit on train only)")
    print("  Solver   : lbfgs  |  C=1.0  |  max_iter=1000")
    print("  Imbalance: class_weight='balanced'")
    print("  Training...")

    model = train_model(X_train, y_train)
    print("  Training complete.")

    # ------------------------------------------------------------------
    print_section("3. Evaluation on Held-Out Test Set")
    # ------------------------------------------------------------------
    results = evaluate_model(model, X_test, y_test)
    print_metrics(results)

    # Precision/recall trade-off commentary
    precision = results["precision"]
    recall = results["recall"]
    print("  Precision/Recall Trade-Off Note:")
    if recall > precision:
        print(f"  Recall ({recall:.4f}) > Precision ({precision:.4f}).")
        print("  The model catches more abusive returns but may flag some legitimate ones.")
        print("  This is appropriate for a risk-ranking system: operators review flagged cases.")
    else:
        print(f"  Precision ({precision:.4f}) >= Recall ({recall:.4f}).")
        print("  Fewer false positives but may miss some abusive returns.")

    # ------------------------------------------------------------------
    print_section("4. Saving Artifacts")
    # ------------------------------------------------------------------
    save_artifacts(model, fe)

    # ------------------------------------------------------------------
    print_section("5. Summary")
    # ------------------------------------------------------------------
    print(f"  Model artifact       : {MODEL_ARTIFACT_PATH}")
    print(f"  FeatureEngineer art. : {FEATURE_ENGINEER_ARTIFACT_PATH}")
    print()
    print("  Phase 2C complete.")
    print("  No API / deployment / frontend was built.")
    print("  data/returns.csv was not modified.")
    print("  No real-world data was used.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReturnShield AI - Phase 2C Training Script")
    parser.add_argument(
        "--data",
        type=str,
        default="data/returns.csv",
        help="Path to the raw returns CSV (default: data/returns.csv)",
    )
    args = parser.parse_args()
    main(csv_path=args.data)
