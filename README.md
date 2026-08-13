# Intelligent Loan Risk Assessment System

RiskLens is a Flask application that serves a loan-risk UI and a versioned ML assessment API backed by Logistic Regression, Random Forest, and XGBoost pipelines.

## Canonical application

The only served UI is `Project/static/riskLens.html`; it is delivered by `Project/app.py`. The former root-level HTML duplicate has been removed to prevent UI/API drift.

## Run locally

```bash
cd Project
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000/`. Run `python -m unittest -v test_system.py` to verify the service.

See [Project/README.md](Project/README.md) for API, configuration, and deployment details.
