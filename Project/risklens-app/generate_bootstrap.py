"""
generate_bootstrap.py
----------------------
Run this ONCE before first launch (or any time you want to reset the
model): it generates a batch of synthetic, plausible applicant profiles,
scores them with the rule engine, writes them to data/training_data.csv
as source='bootstrap', and trains the first model so the app has
something to predict with from minute one.

    python generate_bootstrap.py

Replace or supplement this data with real historical applications
(ideally carrying a real outcome label) as soon as you have them —
see README.md.
"""
import random
from risk_engine import rule_based_score
from model_utils import append_training_row, train_model, TRAIN_FILE
import os

random.seed(42)

N = 250

EMP_TYPES = ['salaried', 'self', 'other']
REPAY = ['excellent', 'good', 'fair', 'poor']
PURPOSES = ['Home purchase', 'Vehicle', 'Personal / Consumption', 'Education',
            'Business / MSME working capital', 'Debt consolidation']


def random_applicant():
    income = random.randint(200000, 2500000)
    return {
        'age': random.randint(21, 65),
        'emp_type': random.choices(EMP_TYPES, weights=[0.55, 0.35, 0.10])[0],
        'emp_exp': random.randint(0, 30),
        'income': income,
        'add_income': random.choice([0, 0, 0, random.randint(10000, 200000)]),
        'loan_amt': random.randint(50000, int(income * 6)),
        'loan_term': random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240]),
        'loan_purpose': random.choice(PURPOSES),
        'debt': random.randint(0, int(income * 0.6)),
        'emi': random.randint(0, 40000),
        'credit': random.randint(300, 900),
        'defaults': random.choices([0, 1, 2], weights=[0.75, 0.17, 0.08])[0],
        'repay_status': random.choices(REPAY, weights=[0.35, 0.35, 0.2, 0.1])[0],
    }


def main():
    if os.path.exists(TRAIN_FILE):
        print(f'{TRAIN_FILE} already exists — delete it first if you want a clean reset.')
        return

    for _ in range(N):
        applicant = random_applicant()
        score = rule_based_score(applicant)
        append_training_row(applicant, score, source='bootstrap')

    model = train_model()
    if model is not None:
        print(f'Bootstrapped {N} synthetic rows and trained the initial model.')
    else:
        print(f'Bootstrapped {N} rows, but training did not run — check MIN_ROWS_TO_TRAIN.')


if __name__ == '__main__':
    main()
