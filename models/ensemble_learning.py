"""Command-line comparison of the auxiliary KNN and random-forest models."""
from .Model_Logic_KNN import predict_knn
from .Model_Logic_Random_Forest import predict_random_forest


def main():
    prompts = [
        ("age", float), ("emp_type", str), ("emp_exp_years", float),
        ("annual_income", float), ("additional_income", float), ("loan_amount", float),
        ("loan_term_months", int), ("loan_purpose", str), ("existing_debt", float),
        ("monthly_emi", float), ("credit_score", float), ("previous_defaults", int),
        ("repayment_status", str),
    ]
    values = [converter(input(f"{name}: ")) for name, converter in prompts]
    knn = predict_knn(*values)
    forest = predict_random_forest(*values)
    selected = forest if forest[3] >= knn[3] else knn
    print(f"KNN accuracy: {knn[3] * 100:.2f}%")
    print(f"Random Forest accuracy: {forest[3] * 100:.2f}%")
    print(f"Selected model risk probability: {selected[1] * 100:.2f}%")


if __name__ == "__main__":
    main()
