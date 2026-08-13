"""Automated Verification Suite for RiskLens ML Backend & Web Engine.

Tests all 5 key checklist features:
1. Backend Integration (/api/health, /api/assess, CORS)
2. Real-Time Prediction (XGBoost, Random Forest, Logistic Regression)
3. Model Metrics Validation (/api/models)
4. Loan Details & Financial Ratio Calculations
5. Production Readiness & Error Bounds Handling
"""

import sys
import unittest
import json
from pathlib import Path

# Ensure Project directory is on Python path
project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from app import app, pipelines, metrics, smote_info

class TestRiskLensBackend(unittest.TestCase):
    
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_health_check_endpoint(self):
        """Verify Production Readiness Health Check endpoint."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("xgboost", data.get("modelsLoaded", []))
        self.assertIn("rf", data.get("modelsLoaded", []))
        self.assertIn("lr", data.get("modelsLoaded", []))
        print("✓ Production Health Check Endpoint Verified")

    def test_model_metrics_endpoint(self):
        """Verify Model Metrics validation endpoint."""
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("metrics", data)
        self.assertIn("smoteInfo", data)
        self.assertIn("bestModelKey", data)
        
        xgb_metrics = data["metrics"].get("xgboost", {})
        self.assertGreaterEqual(xgb_metrics.get("accuracy", 0), 0.90)
        self.assertGreaterEqual(xgb_metrics.get("auc", 0), 0.95)
        print("✓ Model Metrics Endpoint & Accuracy Benchmarks Verified")

    def test_realtime_prediction_low_risk(self):
        """Verify real-time prediction for low risk profile."""
        payload = {
            "age": 35,
            "empType": "salaried",
            "empExp": 10,
            "income": 1200000,
            "addIncome": 100000,
            "loanAmt": 400000,
            "loanTerm": 36,
            "loanPurpose": "personal",
            "debt": 50000,
            "emi": 5000,
            "credit": 800,
            "defaults": 0,
            "repayStatus": "excellent",
            "modelChoice": "xgboost"
        }
        response = self.client.post("/api/assess", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("score", data)
        self.assertIn("category", data)
        self.assertIn("loanDetails", data)
        self.assertEqual(data["category"], "Low Risk")
        self.assertLessEqual(data["score"], 35)
        self.assertEqual(data["loanDetails"]["affordabilityStatus"], "Affordable")
        print(f"✓ Low Risk Real-Time Assessment Verified (Score: {data['score']}, Category: {data['category']})")

    def test_realtime_prediction_high_risk(self):
        """Verify real-time prediction for high risk profile."""
        payload = {
            "age": 22,
            "empType": "other",
            "empExp": 1,
            "income": 200000,
            "addIncome": 0,
            "loanAmt": 900000,
            "loanTerm": 48,
            "loanPurpose": "personal",
            "debt": 300000,
            "emi": 15000,
            "credit": 520,
            "defaults": 2,
            "repayStatus": "poor",
            "modelChoice": "rf"
        }
        response = self.client.post("/api/assess", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["category"], "High Risk")
        self.assertGreater(data["score"], 60)
        print(f"✓ High Risk Real-Time Assessment Verified (Score: {data['score']}, Category: {data['category']})")

    def test_loan_details_calculations(self):
        """Verify Loan Details ratio calculations (DTI, LTI, EMI preview)."""
        payload = {
            "age": 35,
            "empType": "salaried",
            "empExp": 8,
            "income": 600000,
            "addIncome": 0,
            "loanAmt": 300000,
            "loanTerm": 24,
            "debt": 60000,
            "emi": 5000,
            "credit": 720,
            "defaults": 0,
            "repayStatus": "good",
            "loanPurpose": "personal"
        }
        response = self.client.post("/api/assess", data=json.dumps(payload), content_type="application/json")
        data = response.get_json()
        loan_det = data["loanDetails"]
        self.assertGreater(loan_det["estimatedEmi"], 0)
        self.assertGreater(loan_det["dtiPercentage"], 0)
        self.assertAlmostEqual(loan_det["ltiRatio"], 0.5, delta=0.1)
        print(f"✓ Loan Details Ratios Verified: DTI={loan_det['dtiPercentage']}%, LTI={loan_det['ltiRatio']}x, EMI=₹{loan_det['estimatedEmi']}")

    def test_cors_headers(self):
        """Verify Production Readiness CORS headers."""
        response = self.client.get("/api/health")
        self.assertIsNone(response.headers.get("Access-Control-Allow-Origin"))
        print("✓ Production CORS Headers Verified")

    def test_rejects_incomplete_input(self):
        """Invalid assessment requests must not silently receive defaults."""
        response = self.client.post("/api/assess", json={"age": 17})
        self.assertEqual(response.status_code, 422)
        self.assertIn("error", response.get_json())

if __name__ == "__main__":
    unittest.main()
