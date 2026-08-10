import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

df = pd.read_csv("Semester_project/Intelligent-Loan-Risk-Assessment-System/Project/preprocessing/loan_dataset.csv")


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
# 9. TRAIN
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 10. TEST
# ============================================================

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 11. ENTER NEW RAW APPLICANT
# ============================================================

print("\n====================================")
print("       NEW APPLICANT DATA")
print("====================================")


age = float(input("Age: "))

emp_type = input(
    "Employment type (salaried/self/other): "
).lower()

emp_exp_years = float(
    input("Employment experience (years): ")
)

annual_income = float(
    input("Annual income: ")
)

additional_income = float(
    input("Additional income: ")
)

loan_amount = float(
    input("Loan amount: ")
)

loan_term_months = int(
    input("Loan term (months): ")
)

loan_purpose = input(
    "Loan purpose (business/personal/home/other/vehicle/education): "
).lower()

existing_debt = float(
    input("Existing debt: ")
)

monthly_emi = float(
    input("Monthly EMI: ")
)

credit_score = float(
    input("Credit score: ")
)

previous_defaults = int(
    input("Previous defaults: ")
)

repayment_status = input(
    "Repayment status (poor/fair/good/excellent): "
).lower()


# ============================================================
# 12. CREATE RAW INPUT DATAFRAME
# ============================================================

new_applicant = pd.DataFrame([{
    "age": age,
    "emp_type": emp_type,
    "emp_exp_years": emp_exp_years,
    "annual_income": annual_income,
    "additional_income": additional_income,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term_months,
    "loan_purpose": loan_purpose,
    "existing_debt": existing_debt,
    "monthly_emi": monthly_emi,
    "credit_score": credit_score,
    "previous_defaults": previous_defaults,
    "repayment_status": repayment_status
}])


# ============================================================
# 13. PREDICT
# ============================================================

prediction = model.predict(new_applicant)[0]


# ============================================================
# 14. DISPLAY RESULT
# ============================================================

print("\n====================================")

if prediction == 1:
    print("             RISK")
    print("====================================")
    print("Prediction: Applicant is likely to default.")
else:
    print("           NOT RISK")
    print("====================================")
    print("Prediction: Applicant is unlikely to default.")

print("====================================")