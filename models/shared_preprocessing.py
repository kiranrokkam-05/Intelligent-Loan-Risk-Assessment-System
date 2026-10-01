"""Shared preprocessing definitions for auxiliary model experiments."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLUMNS = [
    "age", "emp_exp_years", "annual_income", "additional_income",
    "loan_amount", "loan_term_months", "existing_debt", "monthly_emi",
    "credit_score", "previous_defaults",
]
CATEGORICAL_COLUMNS = ["emp_type", "loan_purpose", "repayment_status"]


def dataset_path():
    root = Path(__file__).resolve().parent.parent
    path = root / "preprocessing" / "loan_dataset.csv"
    return path if path.exists() else root / "loan_dataset.csv"


def load_dataset():
    df = pd.read_csv(dataset_path()).drop(columns=["application_id"], errors="ignore")
    return df.drop(columns=["defaulted_within_12_months"]), df["defaulted_within_12_months"]


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_COLUMNS),
    ])
