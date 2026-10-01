"""Lazy KNN auxiliary model; importing this module performs no training."""
from functools import lru_cache
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from .shared_preprocessing import build_preprocessor, load_dataset


@lru_cache(maxsize=1)
def _trained_model():
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([("preprocessing", build_preprocessor()),
                      ("knn", KNeighborsClassifier(n_neighbors=5))])
    model.fit(x_train, y_train)
    return model, float(accuracy_score(y_test, model.predict(x_test)))


def predict_knn(age, emp_type, emp_exp_years, annual_income, additional_income,
                loan_amount, loan_term_months, loan_purpose, existing_debt,
                monthly_emi, credit_score, previous_defaults, repayment_status):
    model, accuracy = _trained_model()
    applicant = pd.DataFrame([locals()])
    probabilities = model.predict_proba(applicant)[0]
    return model.predict(applicant)[0], probabilities[1], probabilities[0], accuracy
