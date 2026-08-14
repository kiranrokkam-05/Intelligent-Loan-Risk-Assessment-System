import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score


# ============================================================
# 1. FEATURE ENGINEERING FUNCTION
# ============================================================

def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Adds domain-specific financial features for loan default risk prediction."""
    df = data.copy()
    
    # Total annual income
    df['total_income'] = df['annual_income'] + df.get('additional_income', 0)
    
    # Debt-to-Income ratio (DTI) - annual debt obligations vs total income
    annual_debt_obligation = (df['monthly_emi'] * 12) + df['existing_debt']
    df['dti'] = np.where(df['total_income'] > 0, (annual_debt_obligation / df['total_income']) * 100, 100.0)
    
    # Loan-to-Income ratio (LTI)
    df['lti'] = np.where(df['total_income'] > 0, df['loan_amount'] / df['total_income'], 10.0)
    
    # Monthly EMI to Monthly Income ratio (%)
    monthly_income = df['total_income'] / 12.0
    df['emi_to_income'] = np.where(monthly_income > 0, (df['monthly_emi'] / monthly_income) * 100, 100.0)
    
    return df


# ============================================================
# 2. MAIN TRAINING AND EVALUATION PIPELINE
# ============================================================

def train_and_evaluate():
    base_dir = Path(__file__).parent
    root_dir = base_dir.parent
    dataset_path = root_dir / "preprocessing" / "loan_dataset.csv"
    if not dataset_path.exists():
        dataset_path = root_dir / "loan_dataset.csv"
        
    print(f"Loading raw dataset from: {dataset_path}")
    raw_df = pd.read_csv(dataset_path)
    
    if "application_id" in raw_df.columns:
        raw_df = raw_df.drop(columns=["application_id"])
        
    print("Applying Feature Engineering...")
    df = add_engineered_features(raw_df)
    
    X = df.drop(columns=["defaulted_within_12_months"])
    y = df["defaulted_within_12_months"]
    
    categorical_columns = ["emp_type", "loan_purpose"]
    ordinal_columns = ["repayment_status"]
    repayment_order = ["poor", "fair", "good", "excellent"]
    
    numerical_columns = [
        "age",
        "emp_exp_years",
        "annual_income",
        "additional_income",
        "loan_amount",
        "loan_term_months",
        "existing_debt",
        "monthly_emi",
        "credit_score",
        "previous_defaults",
        "total_income",
        "dti",
        "lti",
        "emi_to_income"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
            ("ord", OrdinalEncoder(categories=[repayment_order]), ordinal_columns)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nFitting Preprocessor...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # SMOTE Class Imbalance Handling
    print("\n------------------------------------------------------------")
    print("        SMOTE CLASS IMBALANCE RESAMPLING                    ")
    print("------------------------------------------------------------")
    orig_counts = {int(k): int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()}
    print(f"Original Class Counts (Training Set): {orig_counts}")
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_trans, y_train)
    res_counts = {int(k): int(v) for k, v in pd.Series(y_train_res).value_counts().to_dict().items()}
    print(f"SMOTE Resampled Class Counts:         {res_counts}")
    print("------------------------------------------------------------")
    
    # Define Member 2 Models
    model_defs = {
        "lr": ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        "rf": ("Random Forest", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)),
        "xgboost": ("XGBoost", XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42, eval_metric="logloss"))
    }
    
    pipelines = {}
    metrics_summary = {}
    best_key = None
    best_f1 = -1.0
    
    print("\n============================================================")
    print("        MODEL TRAINING & COMPARATIVE EVALUATION             ")
    print("============================================================")
    
    for key, (name, clf) in model_defs.items():
        clf.fit(X_train_res, y_train_res)
        y_pred = clf.predict(X_test_trans)
        y_proba = clf.predict_proba(X_test_trans)[:, 1]
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_proba))
        
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipelines[key] = pipe
        
        metrics_summary[key] = {
            "key": key,
            "name": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4)
        }
        
        print(f"\nModel: {name} ({key})")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        if f1 > best_f1:
            best_f1 = f1
            best_key = key
            
    print("============================================================")
    print(f"BEST PERFORMING MODEL: {metrics_summary[best_key]['name']} (F1-Score: {best_f1:.4f})")
    print("============================================================")
    
    smote_info = {
        "method": "SMOTE (Synthetic Minority Over-sampling Technique)",
        "original_counts": orig_counts,
        "resampled_counts": res_counts
    }
    
    model_output_path = base_dir / "model.joblib"
    joblib.dump({
        "pipelines": pipelines,
        "metrics": metrics_summary,
        "smote_info": smote_info,
        "best_key": best_key,
        # backward compatibility fallback
        "pipeline": pipelines[best_key],
        "model_name": metrics_summary[best_key]["name"]
    }, model_output_path)
    
    print(f"\nTrained models and SMOTE metadata saved to: {model_output_path}")
    return pipelines, metrics_summary, smote_info


if __name__ == "__main__":
    train_and_evaluate()
