"""
app.py
------
Minimal Flask backend for RiskLens.

Auth:  session-cookie login, credentials hashed (never stored in
       plaintext) in data/users.json.
Data:  every assessment a logged-in user submits is appended to
       data/training_data.csv (source='live'), which is what the model
       retrains against every RETRAIN_EVERY submissions.

Run:
    pip install -r requirements.txt
    python generate_bootstrap.py   # one-time: seed data + first model
    python app.py
    open http://localhost:5000
"""
import os
import secrets
from functools import wraps

from flask import Flask, request, jsonify, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash

import json

from risk_engine import categorize
from model_utils import predict_score, append_training_row, count_rows, train_model, RETRAIN_EVERY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
USERS_FILE = os.path.join(DATA_DIR, 'users.json')

os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__, static_folder='static', static_url_path='')

# In production, set RISKLENS_SECRET as a real environment variable so
# sessions survive a server restart. Falls back to a random one here
# so the demo works out of the box (this means logins reset on restart).
app.secret_key = os.environ.get('RISKLENS_SECRET', secrets.token_hex(32))


# ---------------------------------------------------------------- users ----
def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE) as f:
        return json.load(f)


def save_users(users: dict):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


if not os.path.exists(USERS_FILE):
    # Seed one admin user. Password is hashed before it ever touches disk.
    save_users({'admin@aus.in': generate_password_hash('admin123')})


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('user'):
            return jsonify({'error': 'Not authenticated'}), 401
        return fn(*args, **kwargs)
    return wrapper


# --------------------------------------------------------------- routes ----
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'riskLens.html')


@app.route('/api/register', methods=['POST'])
def register():
    """Optional self-serve signup. Remove this route if accounts should
    only be provisioned by an admin."""
    data = request.get_json(force=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not email or len(password) < 6:
        return jsonify({'ok': False, 'error': 'Email and a 6+ character password are required'}), 400

    users = load_users()
    if email in users:
        return jsonify({'ok': False, 'error': 'Account already exists'}), 409

    users[email] = generate_password_hash(password)
    save_users(users)
    return jsonify({'ok': True})


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json(force=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    users = load_users()
    stored_hash = users.get(email)
    if stored_hash and check_password_hash(stored_hash, password):
        session['user'] = email
        return jsonify({'ok': True, 'user': email})
    return jsonify({'ok': False, 'error': 'Invalid email or password'}), 401


@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({'ok': True})


@app.route('/api/session')
def get_session():
    return jsonify({'authenticated': bool(session.get('user')), 'user': session.get('user')})


@app.route('/api/assess', methods=['POST'])
@login_required
def assess():
    d = request.get_json(force=True) or {}
    try:
        clean = {
            'age': int(d['age']),
            'emp_type': d['empType'],
            'emp_exp': int(d['empExp']),
            'income': float(d['income']),
            'add_income': float(d['addIncome']),
            'loan_amt': float(d['loanAmt']),
            'loan_term': int(d['loanTerm']),
            'loan_purpose': d['loanPurpose'],
            'debt': float(d['debt']),
            'emi': float(d['emi']),
            'credit': int(d['credit']),
            'defaults': int(d['defaults']),
            'repay_status': d['repayStatus'],
        }
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid or missing input: {e}'}), 400

    score = predict_score(clean)
    category = categorize(score)

    # Store this submission — this IS the "train the model on user input" step.
    # It's logged with source='live' and tagged to the signed-in analyst so
    # you can trace who submitted what.
    append_training_row(clean, score, source='live')

    if count_rows() % RETRAIN_EVERY == 0:
        train_model()

    return jsonify({'score': score, 'category': category})


@app.route('/api/retrain', methods=['POST'])
@login_required
def retrain():
    """Manual retrain trigger, e.g. for an admin 'Retrain now' button."""
    model = train_model()
    return jsonify({'ok': model is not None, 'rows': count_rows()})


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
