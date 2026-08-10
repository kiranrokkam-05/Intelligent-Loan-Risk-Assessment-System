"""RiskLens Application & Machine Learning API Backend.

Serves the underwriting web console frontend and handles POST /api/assess
requests powered by the trained ML pipeline (XGBoost / Random Forest / Logistic Regression).

Run: python app.py
Open: http://localhost:8000/
"""

import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from Model_Logic import add_engineered_features, train_and_evaluate

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = BASE_DIR / "model.joblib"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

# Load model pipeline on startup
def load_or_train_model():
    if not MODEL_PATH.exists():
        print("Model file not found. Running training pipeline...")
        train_and_evaluate()
    
    print(f"Loading ML model from {MODEL_PATH}...")
    saved_data = joblib.load(MODEL_PATH)
    return saved_data["pipeline"], saved_data.get("model_name", "Trained ML Model")

model_pipeline, model_name = load_or_train_model()


# Helper to normalize loan purpose string
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


# ============ STATIC ROUTES ============

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "riskLens.html")

@app.route("/riskLens.html")
def risk_lens_html():
    return send_from_directory(STATIC_DIR, "riskLens.html")


# ============ ML API ENDPOINT ============

@app.route("/api/assess", methods=["POST"])
def assess_risk():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "No JSON payload provided"}), 400

        # Parse & Map fields
        age = float(payload.get("age", 30))
        emp_type = str(payload.get("empType", "salaried")).lower()
        emp_exp_years = float(payload.get("empExp", 5))
        annual_income = float(payload.get("income", 500000))
        additional_income = float(payload.get("addIncome", 0))
        loan_amount = float(payload.get("loanAmt", 500000))
        loan_term_months = int(payload.get("loanTerm", 36))
        loan_purpose = normalize_loan_purpose(payload.get("loanPurpose", "personal"))
        existing_debt = float(payload.get("debt", 0))
        monthly_emi = float(payload.get("emi", 0))
        credit_score = float(payload.get("credit", 650))
        previous_defaults = int(payload.get("defaults", 0))
        repayment_status = str(payload.get("repayStatus", "good")).lower()

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

        # ML Prediction
        default_prob = float(model_pipeline.predict_proba(engineered_df)[0][1])
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
        # Credit Tier
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

        # DTI Tier
        dti_val = float(engineered_df["dti"].iloc[0])
        if dti_val > 50:
            dti_tier = "bad"
        elif dti_val > 35:
            dti_tier = "warn"
        else:
            dti_tier = "good"

        # Defaults Tier
        if previous_defaults == 0:
            def_tier = "good"
            def_text = "None"
        elif previous_defaults == 1:
            def_tier = "warn"
            def_text = "1 default"
        else:
            def_tier = "bad"
            def_text = f"{previous_defaults} defaults"

        # Repayment Tier
        repay_map = {
            "excellent": ("good", "Excellent"),
            "good": ("good", "Good"),
            "fair": ("warn", "Fair"),
            "poor": ("bad", "Poor")
        }
        repay_tier, repay_text = repay_map.get(repayment_status, ("warn", repayment_status.capitalize()))

        # Employment Tier
        if emp_type == "salaried":
            emp_tier = "good"
            emp_label = "Salaried"
        elif emp_type == "self":
            emp_tier = "warn"
            emp_label = "Self-employed"
        else:
            emp_tier = "bad"
            emp_label = "Unemployed / Other"

        # LTI Tier
        lti_val = float(engineered_df["lti"].iloc[0])
        if lti_val > 5:
            lti_tier = "bad"
        elif lti_val > 2:
            lti_tier = "warn"
        else:
            lti_tier = "good"

        factors = [
            {"label": "Credit score", "valueText": credit_text, "tier": credit_tier, "iconKey": "credit"},
            {"label": "Debt-to-income", "valueText": f"{dti_val:.0f}%", "tier": dti_tier, "iconKey": "dti"},
            {"label": "Previous defaults", "valueText": def_text, "tier": def_tier, "iconKey": "defaults"},
            {"label": "Repayment history", "valueText": repay_text, "tier": repay_tier, "iconKey": "repay"},
            {"label": "Employment", "valueText": f"{emp_label} · {int(emp_exp_years)}y exp", "tier": emp_tier, "iconKey": "emp"},
            {"label": "Loan-to-income", "valueText": f"{lti_val:.1f}×", "tier": lti_tier, "iconKey": "loan"}
        ]

        recommendations = {
            "Low Risk": f"ML Model ({model_name}) predicts a low default risk ({default_prob*100:.1f}% default probability). Applicant shows a <b>strong repayment profile</b> and manageable debt exposure. Suitable for <b>standard approval</b>.",
            "Medium Risk": f"ML Model ({model_name}) predicts moderate default risk ({default_prob*100:.1f}% default probability). Profile shows <b>mixed risk indicators</b>. Recommend <b>additional income proof</b> or shorter loan tenure.",
            "High Risk": f"ML Model ({model_name}) predicts an elevated risk of default ({default_prob*100:.1f}% default probability). High debt or poor credit history detected. Recommend <b>manual underwriter review</b> or collateral requirement."
        }

        return jsonify({
            "score": score,
            "category": category,
            "probability": round(default_prob, 4),
            "modelUsed": model_name,
            "factors": factors,
            "recommendation": recommendations[category]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"\n==========================================================")
    print(f" RiskLens ML Backend Server ({model_name}) Running!")
    print(f" URL: http://localhost:8000/riskLens.html")
    print(f" API: http://localhost:8000/api/assess")
    print(f"==========================================================\n")
    app.run(host="0.0.0.0", port=8000, debug=False)
