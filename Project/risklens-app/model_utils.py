"""
model_utils.py
--------------
Everything to do with:
  - reading/writing the flat-file "training data" store (data/training_data.csv)
  - loading, training, and predicting with the scikit-learn model
  - deciding when to retrain

Kept intentionally simple: no database, no message queue, no scheduler.
Retraining is triggered inline, every RETRAIN_EVERY new rows. That is
fine at prototype scale; see README.md for how to graduate this to a
proper scheduled job once submission volume grows.
"""
import os
import csv
from datetime import datetime, timezone

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

from risk_engine import FEATURE_COLUMNS, rule_based_score, encode_row

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model_store')
TRAIN_FILE = os.path.join(DATA_DIR, 'training_data.csv')
MODEL_FILE = os.path.join(MODEL_DIR, 'risk_model.pkl')

RETRAIN_EVERY = 10          # retrain after this many new rows
MIN_ROWS_TO_TRAIN = 15      # don't bother training on too little data

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is None and os.path.exists(MODEL_FILE):
        _model_cache = joblib.load(MODEL_FILE)
    return _model_cache


def train_model():
    """Retrain on everything currently in training_data.csv and persist to disk."""
    global _model_cache
    if not os.path.exists(TRAIN_FILE):
        return None
    df = pd.read_csv(TRAIN_FILE)
    if len(df) < MIN_ROWS_TO_TRAIN:
        return None

    X = df.apply(lambda r: encode_row(r), axis=1, result_type='expand')
    y = df['risk_score']

    model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    _model_cache = model
    return model


def predict_score(clean: dict) -> int:
    """Use the trained model if we have one; otherwise fall back to the rule engine."""
    model = load_model()
    if model is not None:
        X = pd.DataFrame([encode_row(clean)])
        pred = model.predict(X)[0]
        return max(2, min(98, round(float(pred))))
    return rule_based_score(clean)


def append_training_row(clean: dict, risk_score: int, source: str = 'live'):
    """
    Append one applicant record to the flat-file training store.
    `source` is 'bootstrap' for synthetic seed rows, 'live' for real submissions,
    so you can filter them apart later once real outcome labels exist.
    """
    file_exists = os.path.exists(TRAIN_FILE)
    fieldnames = FEATURE_COLUMNS + ['risk_score', 'source', 'timestamp']
    with open(TRAIN_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {k: clean[k] for k in FEATURE_COLUMNS}
        row['risk_score'] = risk_score
        row['source'] = source
        row['timestamp'] = datetime.now(timezone.utc).isoformat()
        writer.writerow(row)


def count_rows() -> int:
    if not os.path.exists(TRAIN_FILE):
        return 0
    with open(TRAIN_FILE) as f:
        return max(0, sum(1 for _ in f) - 1)  # minus header
