# Intelligent Loan Risk Assessment System (RiskLens) - Project Overview

RiskLens is an intelligent underwriting support application that utilizes Machine Learning algorithms to analyze applicant metrics (e.g. income, debt, credit history, defaults) and predict the probability of loan default.

---

## Codebase Directory Structure

```
E:/Intelligent-Loan-Risk-Assessment-System/
├── .dockerignore
├── .env.example
├── .git/
├── .gitignore
├── Dockerfile                   # Production container configuration
├── INTEGRATION_TAKEAWAY.md     # Core integration summary
├── PROJECT_OVERVIEW.md         # This overview file
├── Procfile                    # Deployment service definitions (Heroku/Render)
├── README.md                   # Main run guide and onboarding instructions
├── requirements.txt            # Python dependencies
├── app.py                     # Root Flask server backend (APIs & static hosting)
├── check_integration.py       # Local API integration checks
├── test_system.py             # Automated unit tests for ML pipeline & CORS
├── preprocessing/             # Raw data and preprocessing logic
│   ├── preprocess.py
│   ├── loan_dataset.csv
│   └── preprocessed_loan_dataset.csv
├── static/                    # Frontend assets
│   └── riskLens.html           # Risk console web UI (with shimmer skeleton loader)
├── references/                # System documentation & API specifications
│   ├── DATASET_FIELDS.md
│   └── MODEL_INTEGRATION.md
└── models/                    # Machine Learning logic & artifacts
    ├── model.joblib           # Pickled model pipeline files (lr, rf, xgboost)
    ├── Model_Logic.py         # Primary feature engineering and train/evaluate logic
    ├── Model_Logic_KNN.py     # KNN modeling pipeline
    ├── Model_Logic_Random_Forest.py # Random Forest modeling pipeline
    └── ensemble_learning.py   # Command-line model comparison tool
```

---

## System Components

### 1. Flask Web Application (`app.py`)
Serves the responsive single-page web console (`static/riskLens.html`) and provides three core API endpoints:
- `GET /api/health`: Health status and details of loaded models.
- `GET /api/models`: Model training metrics and SMOTE metadata.
- `POST /api/assess`: Computes real-time default risk score and financial ratios.

### 2. Machine Learning Training Pipeline (`models/Model_Logic.py`)
Handles data preprocessing, class imbalance handling, feature engineering, and model evaluation:
- **Feature Engineering**: Calculates Total Income, Debt-To-Income (DTI), Loan-To-Income (LTI), and EMI-To-Income ratios.
- **Imbalance Handling**: Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance default labels in training.
- **Model Comparison**: Trains Logistic Regression (`lr`), Random Forest (`rf`), and XGBoost (`xgboost`) estimators, selecting the best model based on F1-Score. Saves the results to `models/model.joblib`.

### 3. Model Sub-Modules (`models/Model_Logic_KNN.py`, `models/Model_Logic_Random_Forest.py`)
- Independent pipelines for evaluating and predicting loan defaults using K-Nearest Neighbors and standalone Random Forest classifiers.

### 4. Interactive Command-Line Tool (`models/ensemble_learning.py`)
- Prompts developers for applicant values in the terminal and runs side-by-side KNN and Random Forest predictions.

---

## Deployment & Verification

### Deployment Options
- **Docker**: Build using `docker build -t risklens .` and run on port 8000.
- **Procfile**: Configured for WSGI servers using `gunicorn app:app`.

### Testing
- Run automated unit tests using:
  ```bash
  .venv\Scripts\python -m unittest -v test_system.py
  ```
- Run local integration script:
  ```bash
  python check_integration.py
  ```
