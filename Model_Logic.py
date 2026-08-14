import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
# 2. MAIN TRAINING, CROSS-VALIDATION, TUNING & EVALUATION PIPELINE
# ============================================================

def train_and_evaluate():
    """Member 3 Workflow: Model Evaluation, Cross-Validation, Hyperparameter Tuning & Selection."""
    base_dir = Path(__file__).parent
    dataset_path = base_dir / "preprocessing" / "loan_dataset.csv"
    if not dataset_path.exists():
        dataset_path = base_dir / "loan_dataset.csv"
        
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
    
    # SMOTE Class Imbalance Handling
    print("\n------------------------------------------------------------")
    print("        SMOTE CLASS IMBALANCE RESAMPLING                    ")
    print("------------------------------------------------------------")
    orig_counts = {int(k): int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()}
    print(f"Original Class Counts (Training Set): {orig_counts}")
    
    X_train_trans = preprocessor.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_trans, y_train)
    res_counts = {int(k): int(v) for k, v in pd.Series(y_train_res).value_counts().to_dict().items()}
    print(f"SMOTE Resampled Class Counts:         {res_counts}")
    print("------------------------------------------------------------")
    
    # Define Hyperparameter Search Grids for Member 3
    candidate_configs = {
        "lr": {
            "name": "Logistic Regression",
            "base_clf": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs", "liblinear"]
            }
        },
        "rf": {
            "name": "Random Forest",
            "base_clf": RandomForestClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [100, 150, 200],
                "classifier__max_depth": [5, 10, 15],
                "classifier__min_samples_split": [2, 5]
            }
        },
        "xgboost": {
            "name": "XGBoost",
            "base_clf": XGBClassifier(random_state=42, eval_metric="logloss"),
            "param_grid": {
                "classifier__n_estimators": [100, 150, 200],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.03, 0.05, 0.1]
            }
        },
        "knn": {
            "name": "K-Nearest Neighbors",
            "base_clf": KNeighborsClassifier(),
            "param_grid": {
                "classifier__n_neighbors": [3, 5, 7, 9],
                "classifier__weights": ["uniform", "distance"]
            }
        }
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = {}
    metrics_summary = {}
    tuning_results = {}
    best_key = None
    best_f1 = -1.0
    best_auc = -1.0
    
    print("\n============================================================")
    print("  MEMBER 3: MODEL EVALUATION, CV & HYPERPARAMETER TUNING     ")
    print("============================================================")
    
    for key, config in candidate_configs.items():
        name = config["name"]
        print(f"\n---> Optimizing & Cross-Validating: {name} ({key})")
        
        # Build ImbPipeline with SMOTE inside each CV fold
        imb_pipe = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("classifier", config["base_clf"])
        ])
        
        # Hyperparameter Tuning using GridSearchCV
        grid_search = GridSearchCV(
            estimator=imb_pipe,
            param_grid=config["param_grid"],
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )
        grid_search.fit(X_train_trans, y_train)
        
        best_params_clean = {k.replace("classifier__", ""): v for k, v in grid_search.best_params_.items()}
        print(f"  Best Hyperparameters: {best_params_clean}")
        print(f"  Best 5-Fold CV F1-Score: {grid_search.best_score_:.4f}")
        
        # Perform multi-metric cross validation on training data
        cv_scores = cross_validate(
            grid_search.best_estimator_,
            X_train_trans,
            y_train,
            cv=cv,
            scoring=["f1", "roc_auc", "accuracy", "precision", "recall"]
        )
        
        # Train final best estimator on entire resampled training set
        best_clf = grid_search.best_estimator_["classifier"]
        best_clf.fit(X_train_res, y_train_res)
        
        # Evaluate on Test Set
        X_test_trans = preprocessor.transform(X_test)
        y_pred = best_clf.predict(X_test_trans)
        y_proba = best_clf.predict_proba(X_test_trans)[:, 1]
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_proba))
        
        # Create full production inference pipeline (Preprocessor + Classifier)
        prod_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", best_clf)
        ])
        pipelines[key] = prod_pipeline
        
        cv_summary = {
            "f1_mean": round(float(cv_scores["test_f1"].mean()), 4),
            "f1_std": round(float(cv_scores["test_f1"].std()), 4),
            "auc_mean": round(float(cv_scores["test_roc_auc"].mean()), 4),
            "auc_std": round(float(cv_scores["test_roc_auc"].std()), 4),
            "acc_mean": round(float(cv_scores["test_accuracy"].mean()), 4),
            "acc_std": round(float(cv_scores["test_accuracy"].std()), 4)
        }
        
        metrics_summary[key] = {
            "key": key,
            "name": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auc": round(auc, 4),
            "cv_scores": cv_summary,
            "best_params": best_params_clean
        }
        
        print(f"  Test Evaluation Results -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print("  Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Model Selection Rule: Primary Target F1-Score, Secondary ROC-AUC
        if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-6 and auc > best_auc):
            best_f1 = f1
            best_auc = auc
            best_key = key

    print("============================================================")
    print("                 MODEL COMPARISON LEADERBOARD               ")
    print("============================================================")
    leaderboard = []
    for k, m in metrics_summary.items():
        leaderboard.append({
            "Model": m["name"],
            "Key": k,
            "Test F1": m["f1"],
            "Test ROC-AUC": m["auc"],
            "Test Accuracy": m["accuracy"],
            "5-Fold CV F1 (Mean ± Std)": f"{m['cv_scores']['f1_mean']} ± {m['cv_scores']['f1_std']}",
            "Best Parameters": m["best_params"]
        })
    df_leaderboard = pd.DataFrame(leaderboard)
    print(df_leaderboard.to_string(index=False))
    
    print("\n============================================================")
    print(f"AUTOMATED SELECTION -> BEST MODEL: {metrics_summary[best_key]['name']} (F1: {best_f1:.4f}, ROC-AUC: {best_auc:.4f})")
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
        "leaderboard": leaderboard,
        # backward compatibility fallbacks
        "pipeline": pipelines[best_key],
        "model_name": metrics_summary[best_key]["name"]
    }, model_output_path)
    
    print(f"\nTrained models, CV scores, tuning params, and metadata saved to: {model_output_path}")
    return pipelines, metrics_summary, smote_info


if __name__ == "__main__":
    train_and_evaluate()

