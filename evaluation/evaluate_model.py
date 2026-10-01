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
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "preprocessing" / "loan_dataset.csv"
MODEL_PATH = PROJECT_DIR / "models" / "model.joblib"
REPORT_PATH = PROJECT_DIR / "evaluation" / "model_risk_report.json"

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

    report = {
        "report_version": "1.0",
        "evaluation": {"split": "stratified hold-out", "test_size": 0.2, "random_state": 42,
                       "samples": int(len(y_test)), "labels_available": True},
        "fairness": {"status": "unavailable",
                     "reason": "No protected-attribute columns were present in the evaluation dataset.",
                     "protected_attributes": []},
        "models": {},
        "governance_note": "This report is evidence for human review, not a legal or regulatory compliance claim.",
    }
    candidates = [c for c in ("gender", "sex", "race", "ethnicity", "age_group") if c in raw_df.columns]
    if candidates:
        report["fairness"] = {"status": "available", "protected_attributes": candidates,
                              "note": "Review group metrics with an accountable human before deployment."}

    for model_key, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        fraction, mean_predicted = calibration_curve(y_test, y_proba, n_bins=10, strategy="uniform")
        model_report = {
            "name": saved_data.get("metrics", {}).get(model_key, {}).get("name", model_key),
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "roc_auc": roc_auc, "brier_score": brier,
            "calibration": {"bins": len(fraction), "mean_predicted": mean_predicted.tolist(),
                            "fraction_positive": fraction.tolist()},
            "decision_threshold": 0.5,
            "threshold_note": "0.5 is the classifier default; threshold selection requires documented human-approved costs.",
        }
        if candidates:
            fairness_metrics = {}
            protected_test = raw_df.loc[X_test.index, candidates]
            for attribute in candidates:
                groups = {}
                for group, mask in protected_test[attribute].fillna("missing").items():
                    group_mask = protected_test[attribute].fillna("missing") == group
                    actual = y_test[group_mask]
                    predicted = pd.Series(y_pred, index=y_test.index)[group_mask]
                    groups[str(group)] = {
                        "n": int(group_mask.sum()),
                        "selection_rate": float(predicted.mean()) if len(predicted) else None,
                        "true_positive_rate": float(((predicted == 1) & (actual == 1)).sum() / (actual == 1).sum())
                        if (actual == 1).sum() else None,
                    }
                fairness_metrics[attribute] = groups
            model_report["fairness"] = fairness_metrics
        report["models"][model_key] = model_report

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
    REPORT_PATH.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
    print(f"Risk governance report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(evaluate_model())
    except Exception as exc:
        print(f"ERROR: Model evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
