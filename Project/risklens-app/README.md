# RiskLens — Loan Risk Assessment (backend + frontend)

A minimal Flask backend behind the RiskLens single-page UI. No database
server, no message queue — just Flask, flat files, and scikit-learn.

## Run it

```bash
pip install -r requirements.txt
python generate_bootstrap.py   # one-time: seeds synthetic data + trains the first model
python app.py                  # http://localhost:5000
```

Demo login: `analyst@risklens.bank` / `demo1234` (pre-filled on the login screen).

## How it fits together

```
static/riskLens.html   →  the UI you already saw, now calling the API below
app.py                 →  routes: /api/login, /api/logout, /api/session,
                           /api/assess, /api/retrain
risk_engine.py          →  shared scoring formula + feature encoding
model_utils.py          →  training-data storage + model load/train/predict
generate_bootstrap.py   →  one-time synthetic seed data + first model
data/users.json         →  hashed login credentials
data/training_data.csv  →  every applicant record ever submitted (this is
                            what the model retrains on)
model_store/risk_model.pkl → the trained scikit-learn model
```

### The request flow
1. User logs in → Flask sets a signed session cookie. No token to manage on the frontend.
2. User submits the assessment form → `/api/assess`:
   - predicts a score (model if trained, rule-engine fallback otherwise)
   - appends the applicant's data **and** the score to `training_data.csv`
   - every 10 submissions, retrains the model on the accumulated file
3. Next prediction uses the freshly retrained model. No separate training pipeline to run by hand.

## ⚠️ The one thing to fix before this is a real underwriting model

The score written alongside each row in `training_data.csv` is produced by
the same rule-based formula the model is trained to imitate. That's useful
for standing up the *pipeline* (storage → training → serving) end to end,
but it means the model currently just learns to reproduce the rules —
it has no signal about what actually happens to a loan.

To make this a genuine ML system, you need a real target: something like
`defaulted_within_12_months (0/1)` or `days_past_due`, joined back onto
each applicant record once the loan's outcome is known (typically weeks
or months later, from your loan servicing/collections system). Practical
path:

1. Keep logging live submissions as this app already does.
2. Periodically (e.g. monthly), your loan servicing team exports outcomes
   for loans that have matured enough to have a result.
3. A join script matches outcomes back to `training_data.csv` rows (by
   applicant/loan ID — add one if you don't have it yet) and writes a
   `data/labeled_data.csv`.
4. Retrain `model_utils.py`'s model against `labeled_data.csv` instead of
   the rule-engine score once you have enough labeled rows (a few hundred
   at minimum). Swap `RandomForestRegressor` for a classifier
   (`RandomForestClassifier` / `LogisticRegression` / `XGBClassifier`)
   predicting probability of default.

Until then, treat every prediction as "what the rule engine would say,"
not "what a trained credit model says."

## Authentication & authorization — what's here vs. what to upgrade to

**What this build does:**
- Passwords are hashed with Werkzeug's `generate_password_hash`
  (PBKDF2) before they're ever written to disk — `data/users.json` never
  contains a plaintext password.
- Login uses Flask's signed session cookie. No JWT, no token refresh
  logic to build or break.
- `/api/assess` and `/api/retrain` require an active session
  (`@login_required`); unauthenticated requests get a 401.

**Recommended upgrades before this handles real applicant data:**

| Area | Now | Upgrade to |
|---|---|---|
| Credential storage | `data/users.json` (flat file) | SQLite at minimum (still zero extra infra), Postgres for multi-instance deployments |
| Password hashing | PBKDF2 (Werkzeug default) | bcrypt or argon2 (`flask-bcrypt`) if you want a stronger, purpose-built KDF |
| Session secret | random on every restart | a fixed `RISKLENS_SECRET` environment variable, so sessions survive deploys |
| Transport | HTTP (dev server) | HTTPS in front (nginx/Caddy or your cloud LB) + `SESSION_COOKIE_SECURE=True` |
| Brute-force protection | none | rate-limit `/api/login` (e.g. `flask-limiter`), lock account after N failures |
| Roles | everyone is one role | add a `role` field per user (`analyst` / `admin`) and gate `/api/retrain` to admins only |
| Applicant PII | stored in plain CSV alongside score | separate PII (name, ID numbers if you add them) from the modeling features; encrypt or restrict access to the PII table independently |

None of these require a "complex backend" — SQLite + `flask-bcrypt` +
`flask-limiter` are all drop-in additions to the same single Flask app,
not a rewrite.

## Files you can delete/keep as-is
- `generate_bootstrap.py` is safe to re-run any time you want to wipe
  and reseed (delete `data/training_data.csv` and `model_store/risk_model.pkl` first).
- `/api/register` is included for convenience; remove it if accounts
  should only be created by an admin.
