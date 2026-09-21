# Intelligent Loan Risk Assessment System (RiskLens)

RiskLens is an educational Flask application and ML decision-support prototype for estimating whether a loan may default within 12 months. It serves the browser UI from `static/riskLens.html` and exposes the same trained pipelines through a JSON API.

## Architecture

`preprocessing/loan_dataset.csv` -> feature engineering -> stratified 80/20 split -> train-only preprocessing and SMOTE -> Logistic Regression, Random Forest, and XGBoost -> hold-out evaluation -> `models/model.joblib` -> Flask API -> RiskLens UI.

`models/Model_Logic.py` is the authoritative training pipeline. It derives total income, debt-to-income, loan-to-income, and EMI-to-income features; standardizes numeric fields; one-hot encodes employment and loan purpose; ordinally encodes repayment status; applies SMOTE only to the training split; and persists the fitted preprocessing pipeline with each classifier.

## Evaluation

The deterministic split uses `random_state=42`, and the model with the highest hold-out F1 score is selected. The current artifact was generated from the included 1,000-row dataset:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9550 | 1.0000 | 0.9211 | 0.9589 | 0.9943 |
| Random Forest | 0.9550 | 0.9730 | 0.9474 | 0.9600 | 0.9932 |
| XGBoost | 0.9600 | 0.9818 | 0.9474 | 0.9643 | 0.9971 |

XGBoost is selected for this artifact. These are hold-out metrics, not a guarantee of real-world performance.

## Setup and run

The tested environment is Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe models\Model_Logic.py
.\.venv\Scripts\python.exe app.py
```

Open `http://localhost:8000/`.

Automatic training at application startup is disabled by default. For local development only, set `ALLOW_MODEL_TRAINING=true` if `models/model.joblib` is absent.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_system.py
.\.venv\Scripts\python.exe check_integration.py
```

The tests cover model loading, all API endpoints, model selection, valid/invalid assessment requests, financial calculations, probability bounds, CORS behavior, and frontend/API integration.

## API

- `GET /` and `GET /riskLens.html`: serve the authoritative frontend.
- `GET /api/health`: report service health and loaded models.
- `GET /api/models`: report persisted evaluation metrics and SMOTE metadata.
- `POST /api/assess`: validate an application, select a model, predict default probability, and return score/category, risk factors, recommendations, and financial calculations.

The request field mapping and response contract are documented in [references/MODEL_INTEGRATION.md](references/MODEL_INTEGRATION.md). Dataset columns are documented in [references/DATASET_FIELDS.md](references/DATASET_FIELDS.md).

## Project structure

- `app.py`: Flask API and static-file server.
- `models/Model_Logic.py`: authoritative training and feature-engineering pipeline.
- `models/model.joblib`: persisted fitted pipelines and metadata.
- `static/riskLens.html`: authoritative frontend.
- `preprocessing/`: dataset and preprocessing utilities.
- `references/`: dataset and API documentation.
- `test_system.py`, `check_integration.py`: automated checks.

## Responsible use

This project is a demonstration and decision-support prototype, not an automated lending decision system. Outputs depend on the supplied data and assumptions and must not be treated as a guarantee of repayment or used without human oversight, privacy controls, fairness review, and appropriate governance.
