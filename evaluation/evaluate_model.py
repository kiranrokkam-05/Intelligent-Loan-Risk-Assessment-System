"""Evaluate the persisted RiskLens pipelines on the deterministic hold-out split."""

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "preprocessing" / "loan_dataset.csv"
MODEL_PATH = PROJECT_DIR / "models" / "model.joblib"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models.Model_Logic import add_engineered_features


def evaluate_model() -> int:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {DATASET_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    raw_df = pd.read_csv(DATASET_PATH)
    if "application_id" in raw_df.columns:
        raw_df = raw_df.drop(columns=["application_id"])

    required_target = "defaulted_within_12_months"
    if required_target not in raw_df.columns:
        raise ValueError(f"Dataset is missing required target column: {required_target}")

    df = add_engineered_features(raw_df)
    X = df.drop(columns=[required_target])
    y = df[required_target]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    saved_data = joblib.load(MODEL_PATH)
    pipelines = saved_data.get("pipelines")
    if not pipelines:
        raise ValueError("Model artifact does not contain persisted pipelines")

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Evaluation dataset: {DATASET_PATH}")
    print(f"Hold-out samples: {len(y_test)}")

    for model_key, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"\nModel: {saved_data.get('metrics', {}).get(model_key, {}).get('name', model_key)} ({model_key})")
        print(f"Accuracy : {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall   : {recall * 100:.2f}%")
        print(f"F1 Score : {f1 * 100:.2f}%")
        print(f"ROC-AUC  : {roc_auc * 100:.2f}%")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

    print("=" * 60)
    print(f"Selected model: {saved_data.get('best_key', 'unknown')}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(evaluate_model())
    except Exception as exc:
        print(f"ERROR: Model evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
