import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD RAW DATASET
# ============================================================

df = pd.read_csv(
    "Semester_project/Intelligent-Loan-Risk-Assessment-System/Project/preprocessing/loan_dataset.csv"
)


# ============================================================
# 2. REMOVE APPLICATION ID
# ============================================================

df = df.drop("application_id", axis=1)


# ============================================================
# 3. SEPARATE INPUTS AND OUTPUT
# ============================================================

X = df.drop("defaulted_within_12_months", axis=1)

y = df["defaulted_within_12_months"]


# ============================================================
# 4. DEFINE COLUMNS
# ============================================================

categorical_columns = [
    "emp_type",
    "loan_purpose",
    "repayment_status"
]

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
    "previous_defaults"
]


# ============================================================
# 5. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_columns
        )
    ]
)


# ============================================================
# 6. KNN MODEL
# ============================================================

knn = KNeighborsClassifier(
    n_neighbors=5
)


# ============================================================
# 7. CREATE PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("knn", knn)
    ]
)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 10. TEST MODEL
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 11. CALCULATE MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# 12. PRINT MODEL PERFORMANCE
# ============================================================

print("====================================")
print("          KNN MODEL RESULTS")
print("====================================")

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 13. KNN PREDICTION FUNCTION
# ============================================================

def predict_knn(
    age,
    emp_type,
    emp_exp_years,
    annual_income,
    additional_income,
    loan_amount,
    loan_term_months,
    loan_purpose,
    existing_debt,
    monthly_emi,
    credit_score,
    previous_defaults,
    repayment_status
):

    # ========================================================
    # CREATE RAW APPLICANT DATAFRAME
    # ========================================================

    new_applicant = pd.DataFrame([{
        "age": age,
        "emp_type": emp_type.lower(),
        "emp_exp_years": emp_exp_years,
        "annual_income": annual_income,
        "additional_income": additional_income,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "loan_purpose": loan_purpose.lower(),
        "existing_debt": existing_debt,
        "monthly_emi": monthly_emi,
        "credit_score": credit_score,
        "previous_defaults": previous_defaults,
        "repayment_status": repayment_status.lower()
    }])


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    prediction = model.predict(
        new_applicant
    )[0]


    # ========================================================
    # GET PREDICTION PROBABILITY
    # ========================================================

    probability = model.predict_proba(
        new_applicant
    )[0]


    # ========================================================
    # EXTRACT PROBABILITIES
    # ========================================================

    not_risk_probability = probability[0]

    risk_probability = probability[1]


    # ========================================================
    # RETURN VALUES
    # ========================================================

    return (
        prediction,
        risk_probability,
        not_risk_probability,
        accuracy
    )