"""
risk_engine.py
--------------
Shared, dependency-light risk-scoring logic.

This module holds TWO things on purpose:

1. `rule_based_score()` — a transparent, hand-written scoring formula.
   This is what powers predictions before the ML model has enough data,
   and it is also what generates the *pseudo-labels* used to bootstrap
   the very first version of the model (see generate_bootstrap.py).

2. `encode_row()` — turns a cleaned applicant dict into the numeric
   feature vector the ML model trains/predicts on.

IMPORTANT LIMITATION (read this before deploying):
The rule-based score is a stand-in for a real target variable. It is
NOT ground truth about whether a borrower actually repaid or defaulted.
A production model should be retrained against real historical
outcomes (e.g. `defaulted_within_12_months: 0/1`) once your loan
servicing/collections team can supply them. Until then, this system
is best understood as "rule engine wrapped in an ML training loop" —
useful for building the pipeline, not yet for real underwriting.
"""

FEATURE_COLUMNS = [
    'age', 'emp_type', 'emp_exp', 'income', 'add_income',
    'loan_amt', 'loan_term', 'loan_purpose', 'debt', 'emi',
    'credit', 'defaults', 'repay_status'
]


def rule_based_score(d: dict) -> int:
    score = 50

    credit = d['credit']
    if credit >= 750:
        score -= 25
    elif credit >= 700:
        score -= 15
    elif credit >= 650:
        pass
    else:
        score += 20

    monthly_income = (d['income'] + d['add_income']) / 12 or 1
    est_new_emi = d['loan_amt'] / (d['loan_term'] or 1)
    dti = ((d['emi'] + est_new_emi) / monthly_income) * 100
    if dti > 50:
        score += 20
    elif dti > 35:
        score += 10
    else:
        score -= 10

    defaults = d['defaults']
    if defaults == 0:
        score -= 10
    elif defaults == 1:
        score += 10
    else:
        score += 25

    score += {'excellent': -15, 'good': -5, 'fair': 5, 'poor': 20}.get(d['repay_status'], 0)
    score += {'salaried': -5, 'self': 5, 'other': 15}.get(d['emp_type'], 0)

    if d['emp_exp'] > 5:
        score -= 8
    elif d['emp_exp'] < 2:
        score += 8

    lti = d['loan_amt'] / (d['income'] or 1)
    if lti > 5:
        score += 15
    elif lti > 2:
        score += 5
    else:
        score -= 5

    return max(2, min(98, round(score)))


def categorize(score: float) -> str:
    if score <= 35:
        return 'Low Risk'
    if score <= 65:
        return 'Medium Risk'
    return 'High Risk'


def encode_row(d: dict) -> dict:
    """Numeric feature vector for the ML model."""
    return {
        'age': d['age'],
        'emp_exp': d['emp_exp'],
        'income': d['income'],
        'add_income': d['add_income'],
        'loan_amt': d['loan_amt'],
        'loan_term': d['loan_term'],
        'debt': d['debt'],
        'emi': d['emi'],
        'credit': d['credit'],
        'defaults': d['defaults'],
        'emp_salaried': 1 if d['emp_type'] == 'salaried' else 0,
        'emp_self': 1 if d['emp_type'] == 'self' else 0,
        'repay_excellent': 1 if d['repay_status'] == 'excellent' else 0,
        'repay_good': 1 if d['repay_status'] == 'good' else 0,
        'repay_fair': 1 if d['repay_status'] == 'fair' else 0,
    }
