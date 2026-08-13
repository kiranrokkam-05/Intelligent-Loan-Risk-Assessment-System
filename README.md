# Intelligent Loan Risk Assessment System (RiskLens)

RiskLens is a Flask application that serves a loan-risk assessment UI and a versioned ML assessment API backed by Logistic Regression, Random Forest, and XGBoost pipelines.

## Project Structure

The project has been organized with the core application files located directly in the root directory:
- `app.py`: Main Flask application serving the underwriting web console frontend and handling ML risk assessment APIs.
- `Model_Logic.py`: Machine Learning training pipeline and feature engineering.
- `Model_Logic_KNN.py` & `Model_Logic_Random_Forest.py`: Sub-modules for model definitions and predictions.
- `ensemble_learning.py`: Combined loan risk prediction model script.
- `test_system.py`: Automated verification suite for the backend and ML pipelines.
- `check_integration.py`: Integration testing script.
- `model.joblib`: Serialized model and metrics data.
- `static/`: Contains the front-end console (`riskLens.html`).
- `preprocessing/`: Scripts and datasets used for data preparation (`preprocess.py`, `loan_dataset.csv`, `preprocessed_loan_dataset.csv`).
- `INTEGRATION_TAKEAWAY.md`: Takeaway and run summary notes.

## Run Locally

Follow these steps to run the application locally:

1. **Recreate and activate virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Flask server**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   Open `http://localhost:8000/` in your browser.

## Run Tests

To verify the service and ensure all endpoints are operating correctly, run:
```bash
python -m unittest -v test_system.py
```

## Configuration

The application supports the following environment variables:
- `PORT`: Server port (default `8000`)
- `HOST`: Server host (default `0.0.0.0`)
- `DEBUG`: Flask debug mode (default `false`)
- `ALLOWED_ORIGINS`: Comma-separated list of allowed origins for CORS.
- `RATE_LIMIT_PER_MINUTE`: Per-client API request rate limit (default `60`)
- `ALLOW_MODEL_TRAINING`: Set to `true` to allow automatic local training if `model.joblib` is missing.

## Production Notes

Deploy behind a TLS reverse proxy, authenticate users, and attach an approved audit store before processing real applications. Do not expose datasets, model artifacts, or credentials to browsers. [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md) defines the API contract.
