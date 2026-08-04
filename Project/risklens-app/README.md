# RiskLens UI Prototype

This folder is a UI/UX prototype for the Intelligent Loan Risk Assessment System. It deliberately has no trained model, dataset, authentication, or database. All assessment results are mock results intended for testing the user experience only.

## Run the prototype

```bash
python app.py
```

Open http://localhost:8000/riskLens.html. No package installation is required.

## What is included

- A responsive loan-risk assessment screen
- Demo sign-in and sign-out interactions
- A mock risk score, category, explanations, and recommendation
- Form inputs that let the team test low-, medium-, and high-risk UI states

## Model integration handoff

Read [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md) after the ML team has a trained model and a serving API. The UI is designed so the mock scoring block in `static/riskLens.html` can be replaced with one API call while the current form and result rendering remain unchanged.
