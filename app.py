"""RiskLens Application & Machine Learning API Backend.

Serves the underwriting web console frontend and handles ML risk assessment APIs.
Supports model selection across Logistic Regression, Random Forest, and XGBoost with SMOTE resampling.

Run: python app.py
Open: http://localhost:8000/
"""

import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from Model_Logic import add_engineered_features, train_and_evaluate

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "model.joblib"
ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
}
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
REQUEST_LOG = defaultdict(deque)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")


def load_or_train_models():
    if not MODEL_PATH.exists():
        if os.environ.get("ALLOW_MODEL_TRAINING", "false").lower() not in ("true", "1", "t"):
            raise RuntimeError(
                "Model artifact is missing. Deploy model.joblib or set ALLOW_MODEL_TRAINING=true for local development."
            )
        logger.warning("Model file not found; running the local training pipeline.")
        train_and_evaluate()
    
    print(f"Loading ML models from {MODEL_PATH}...")
    saved_data = joblib.load(MODEL_PATH)
    return saved_data

saved_data = load_or_train_models()
pipelines = saved_data.get("pipelines", {"xgboost": saved_data.get("pipeline")})
metrics = saved_data.get("metrics", {})
smote_info = saved_data.get("smote_info", {})
best_key = saved_data.get("best_key", "xgboost")


def normalize_loan_purpose(raw_purpose: str) -> str:
    p = str(raw_purpose).lower()
    if "home" in p:
        return "home"
    elif "vehicle" in p or "car" in p:
        return "vehicle"
    elif "business" in p or "msme" in p:
        return "business"
    elif "education" in p:
        return "education"
    elif "personal" in p:
        return "personal"
    return "other"


def api_error(message: str, status: int):
    return jsonify({"error": message}), status


def parse_number(payload, field, *, minimum=None, maximum=None, integer=False):
    if field not in payload:
        raise ValueError(f"Missing required field: {field}")
    try:
        value = int(payload[field]) if integer else float(payload[field])
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a valid {'integer' if integer else 'number'}") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return value


def parse_assessment_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("JSON object payload required")

    emp_type = str(payload.get("empType", "")).lower()
    if emp_type not in {"salaried", "self", "other"}:
        raise ValueError("empType must be salaried, self, or other")
    repayment_status = str(payload.get("repayStatus", "")).lower()
    if repayment_status not in {"excellent", "good", "fair", "poor"}:
        raise ValueError("repayStatus must be excellent, good, fair, or poor")
    model_choice = str(payload.get("modelChoice") or payload.get("model") or best_key).lower()
    if model_choice not in pipelines:
        raise ValueError(f"model must be one of: {', '.join(sorted(pipelines))}")

    return {
        "model_choice": model_choice,
        "age": parse_number(payload, "age", minimum=18, maximum=100),
        "emp_type": emp_type,
        "emp_exp_years": parse_number(payload, "empExp", minimum=0, maximum=80),
        "annual_income": parse_number(payload, "income", minimum=0, maximum=1000000000),
        "additional_income": parse_number(payload, "addIncome", minimum=0, maximum=1000000000),
        "loan_amount": parse_number(payload, "loanAmt", minimum=1000, maximum=1000000000),
        "loan_term_months": parse_number(payload, "loanTerm", minimum=3, maximum=360, integer=True),
        "loan_purpose": normalize_loan_purpose(payload.get("loanPurpose", "")),
        "existing_debt": parse_number(payload, "debt", minimum=0, maximum=1000000000),
        "monthly_emi": parse_number(payload, "emi", minimum=0, maximum=100000000),
        "credit_score": parse_number(payload, "credit", minimum=300, maximum=900),
        "previous_defaults": parse_number(payload, "defaults", minimum=0, maximum=50, integer=True),
        "repayment_status": repayment_status,
    }


# ============ STATIC ROUTES ============

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "riskLens.html")

@app.route("/riskLens.html")
def risk_lens_html():
    return send_from_directory(STATIC_DIR, "riskLens.html")


# ============ CORS & HEALTH ENDPOINTS ============

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin and origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.before_request
def enforce_api_rate_limit():
    if not request.path.startswith("/api/") or request.method == "OPTIONS":
        return None
    client = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip()
    now = time.monotonic()
    requests = REQUEST_LOG[client]
    while requests and now - requests[0] >= 60:
        requests.popleft()
    if len(requests) >= RATE_LIMIT_PER_MINUTE:
        return api_error("Too many requests. Please try again shortly.", 429)
    requests.append(now)
    return None


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "RiskLens ML Assessment Backend",
        "version": "1.0.0",
        "modelsLoaded": list(pipelines.keys()),
        "smoteResampling": smote_info.get("method", "Active"),
        "bestModel": best_key
    })


# ============ METRICS API ENDPOINT ============

@app.route("/api/models", methods=["GET"])
def get_model_metrics():
    return jsonify({
        "smoteInfo": smote_info,
        "metrics": metrics,
        "bestModelKey": best_key
    })


# ============ ML ASSESSMENT ENDPOINT ============

@app.route("/api/assess", methods=["POST", "OPTIONS"])
def assess_risk():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            return api_error("A JSON request body is required", 400)
        values = parse_assessment_payload(payload)
        model_choice = values["model_choice"]
        selected_pipeline = pipelines[model_choice]
        selected_metrics = metrics.get(model_choice, {})
        model_display_name = selected_metrics.get("name", model_choice.upper())

        age = values["age"]
        emp_type = values["emp_type"]
        emp_exp_years = values["emp_exp_years"]
        annual_income = values["annual_income"]
        additional_income = values["additional_income"]
        loan_amount = values["loan_amount"]
        loan_term_months = values["loan_term_months"]
        loan_purpose = values["loan_purpose"]
        existing_debt = values["existing_debt"]
        monthly_emi = values["monthly_emi"]
        credit_score = values["credit_score"]
        previous_defaults = values["previous_defaults"]
        repayment_status = values["repayment_status"]

        # Build single-row pandas DataFrame
        input_data = pd.DataFrame([{
            "age": age,
            "emp_type": emp_type,
            "emp_exp_years": emp_exp_years,
            "annual_income": annual_income,
            "additional_income": additional_income,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "loan_purpose": loan_purpose,
            "existing_debt": existing_debt,
            "monthly_emi": monthly_emi,
            "credit_score": credit_score,
            "previous_defaults": previous_defaults,
            "repayment_status": repayment_status
        }])

        # Perform Feature Engineering
        engineered_df = add_engineered_features(input_data)

        # ML Prediction using chosen model
        default_prob = float(selected_pipeline.predict_proba(engineered_df)[0][1])
        score = int(np.round(default_prob * 100))
        score = max(0, min(100, score))

        # Risk Category
        if score <= 34:
            category = "Low Risk"
        elif score <= 64:
            category = "Medium Risk"
        else:
            category = "High Risk"

        # Calculate Tiers & Factors for UI Breakdown
        if credit_score >= 750:
            credit_tier = "good"
            credit_text = f"{int(credit_score)} - Strong"
        elif credit_score >= 680:
            credit_tier = "good"
            credit_text = f"{int(credit_score)} - Above Average"
        elif credit_score >= 600:
            credit_tier = "warn"
            credit_text = f"{int(credit_score)} - Fair"
        else:
            credit_tier = "bad"
            credit_text = f"{int(credit_score)} - Weak"

        dti_val = float(engineered_df["dti"].iloc[0])
        if dti_val > 50:
            dti_tier = "bad"
        elif dti_val > 35:
            dti_tier = "warn"
        else:
            dti_tier = "good"

        if previous_defaults == 0:
            def_tier = "good"
            def_text = "None"
        elif previous_defaults == 1:
            def_tier = "warn"
            def_text = "1 default"
        else:
            def_tier = "bad"
            def_text = f"{previous_defaults} defaults"

        repay_map = {
            "excellent": ("good", "Excellent"),
            "good": ("good", "Good"),
            "fair": ("warn", "Fair"),
            "poor": ("bad", "Poor")
        }
        repay_tier, repay_text = repay_map.get(repayment_status, ("warn", repayment_status.capitalize()))

        if emp_type == "salaried":
            emp_tier = "good"
            emp_label = "Salaried"
        elif emp_type == "self":
            emp_tier = "warn"
            emp_label = "Self-employed"
        else:
            emp_tier = "bad"
            emp_label = "Unemployed / Other"

        lti_val = float(engineered_df["lti"].iloc[0])
        if lti_val > 5:
            lti_tier = "bad"
        elif lti_val > 2:
            lti_tier = "warn"
        else:
            lti_tier = "good"

        # Financial & Loan Details Analysis
        tot_income = float(engineered_df["total_income"].iloc[0])
        monthly_income = tot_income / 12.0 if tot_income > 0 else 1.0
        
        # Standard estimated interest rate per year (10.5%)
        annual_rate = 0.105
        monthly_rate = annual_rate / 12.0
        if monthly_rate > 0 and loan_term_months > 0:
            estimated_emi = loan_amount * monthly_rate * ((1 + monthly_rate)**loan_term_months) / (((1 + monthly_rate)**loan_term_months) - 1)
        else:
            estimated_emi = loan_amount / max(1, loan_term_months)

        max_allowed_emi = max(0, (monthly_income * 0.50) - monthly_emi)
        if monthly_rate > 0 and max_allowed_emi > 0:
            max_eligible_loan = max_allowed_emi * (((1 + monthly_rate)**loan_term_months) - 1) / (monthly_rate * ((1 + monthly_rate)**loan_term_months))
        else:
            max_eligible_loan = max_allowed_emi * loan_term_months

        loan_details_summary = {
            "monthlyIncome": round(monthly_income, 2),
            "estimatedEmi": round(estimated_emi, 2),
            "totalLoanObligation": round(estimated_emi * loan_term_months, 2),
            "dtiPercentage": round(dti_val, 1),
            "ltiRatio": round(lti_val, 2),
            "maxEligibleLoan": round(max(0, max_eligible_loan), 2),
            "affordabilityStatus": "Affordable" if (monthly_emi + estimated_emi) <= (monthly_income * 0.45) else "Strained" if (monthly_emi + estimated_emi) <= (monthly_income * 0.60) else "Over-Leveraged"
        }

        factors = [
            {"label": "Credit score", "valueText": credit_text, "tier": credit_tier, "iconKey": "credit"},
            {"label": "Debt-to-income", "valueText": f"{dti_val:.0f}%", "tier": dti_tier, "iconKey": "dti"},
            {"label": "Previous defaults", "valueText": def_text, "tier": def_tier, "iconKey": "defaults"},
            {"label": "Repayment history", "valueText": repay_text, "tier": repay_tier, "iconKey": "repay"},
            {"label": "Employment", "valueText": f"{emp_label} · {int(emp_exp_years)}y exp", "tier": emp_tier, "iconKey": "emp"},
            {"label": "Loan-to-income", "valueText": f"{lti_val:.1f}×", "tier": lti_tier, "iconKey": "loan"}
        ]

        recommendations = {
            "Low Risk": f"{model_display_name} predicts a low default risk ({default_prob*100:.1f}% default probability). Applicant shows a strong repayment profile and manageable debt exposure ({dti_val:.0f}% DTI). Suitable for standard review.",
            "Medium Risk": f"{model_display_name} predicts moderate default risk ({default_prob*100:.1f}% default probability). Profile shows mixed risk indicators. Recommend additional income proof or a shorter loan tenure.",
            "High Risk": f"{model_display_name} predicts an elevated risk of default ({default_prob*100:.1f}% default probability). High debt or poor credit history detected. Recommend manual underwriter review or collateral requirements."
        }

        logger.info("Assessment completed: model=%s category=%s", model_choice, category)
        return jsonify({
            "apiVersion": "1.0",
            "score": score,
            "category": category,
            "probability": round(default_prob, 4),
            "modelChoice": model_choice,
            "modelUsed": model_display_name,
            "allMetrics": metrics,
            "smoteInfo": smote_info,
            "loanDetails": loan_details_summary,
            "factors": factors,
            "recommendation": recommendations[category]
        })

    except ValueError as exc:
        return api_error(str(exc), 422)
    except Exception:
        logger.exception("Assessment failed")
        return api_error("Assessment service encountered an unexpected error", 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")

    print(f"\n==========================================================")
    print(f" RiskLens ML Backend Server Running!")
    print(f" Models Loaded: Logistic Regression, Random Forest, XGBoost")
    print(f" Resampling: SMOTE Balanced ({smote_info.get('resampled_counts')})")
    print(f" Host: {host}:{port}")
    print(f" Health Check: http://localhost:{port}/api/health")
    print(f" Assessment API: http://localhost:{port}/api/assess")
    print(f"==========================================================\n")
    app.run(host=host, port=port, debug=debug)

