# ============================================================
# COMBINED LOAN RISK PREDICTION MODEL
# ============================================================

from Model_Logic_KNN import predict_knn
from Model_Logic_Random_Forest import predict_random_forest


# ============================================================
# 1. ENTER NEW APPLICANT DATA
# ============================================================

print("\n====================================")
print("       NEW APPLICANT DATA")
print("====================================")


age = float(
    input("Age: ")
)

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
# 2. SEND APPLICANT TO KNN
# ============================================================

knn_prediction, knn_risk_probability, knn_not_risk_probability, knn_accuracy = predict_knn(

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
)


# ============================================================
# 3. SEND SAME APPLICANT TO RANDOM FOREST
# ============================================================

rf_prediction, rf_risk_probability, rf_not_risk_probability, rf_accuracy = predict_random_forest(

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
)


# ============================================================
# 4. DISPLAY MODEL COMPARISON
# ============================================================

print("\n====================================")
print("         MODEL COMPARISON")
print("====================================")

print(
    f"KNN Accuracy:            {knn_accuracy * 100:.2f}%"
)

print(
    f"Random Forest Accuracy:  {rf_accuracy * 100:.2f}%"
)


# ============================================================
# 5. SELECT BETTER MODEL
# ============================================================

if rf_accuracy > knn_accuracy:

    best_model = "Random Forest"

    final_prediction = rf_prediction

    final_risk_probability = rf_risk_probability

    final_not_risk_probability = rf_not_risk_probability


elif knn_accuracy > rf_accuracy:

    best_model = "KNN"

    final_prediction = knn_prediction

    final_risk_probability = knn_risk_probability

    final_not_risk_probability = knn_not_risk_probability


else:

    # If both models have exactly the same accuracy,
    # use the average of their probabilities.

    best_model = "KNN + Random Forest"

    final_risk_probability = (
        knn_risk_probability +
        rf_risk_probability
    ) / 2

    final_not_risk_probability = (
        knn_not_risk_probability +
        rf_not_risk_probability
    ) / 2

    if final_risk_probability >= 0.5:
        final_prediction = 1
    else:
        final_prediction = 0


# ============================================================
# 6. FINAL OUTPUT
# ============================================================

print("\n====================================")
print("         FINAL PREDICTION")
print("====================================")

print(
    f"Selected Model: {best_model}"
)

print(
    f"Risk Probability: "
    f"{final_risk_probability * 100:.2f}%"
)

print(
    f"Not Risk Probability: "
    f"{final_not_risk_probability * 100:.2f}%"
)

print("====================================")


if final_prediction == 1:

    print("              RISK")

    print("====================================")

    print(
        "Final Prediction: "
        "Applicant is likely to default."
    )

else:

    print("            NOT RISK")

    print("====================================")

    print(
        "Final Prediction: "
        "Applicant is unlikely to default."
    )

print("====================================")