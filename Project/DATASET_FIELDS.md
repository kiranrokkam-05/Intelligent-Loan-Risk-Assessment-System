# Dataset Fields for Model Training

This specification matches the inputs in the integrated RiskLens UI and `POST /api/assess` API. Use these fields as the agreed contract when preparing training data and retraining the prediction pipelines.

## Required feature columns

| UI field | Dataset column | Type | Example | Preparation notes |
|---|---|---:|---:|---|
| Applicant age | `age` | integer | `32` | Years; validate a realistic adult range. |
| Employment type | `emp_type` | categorical text | `salaried` | Use one agreed vocabulary: `salaried`, `self`, `other`. |
| Employment experience | `emp_exp_years` | integer | `6` | Total years of experience. |
| Annual income | `annual_income` | number | `850000` | INR; keep the same unit for every record. |
| Additional annual income | `additional_income` | number | `50000` | INR; use `0` where none is declared. |
| Requested loan amount | `loan_amount` | number | `1200000` | INR. |
| Loan term | `loan_term_months` | integer | `60` | Months. |
| Loan purpose | `loan_purpose` | categorical text | `home` | Keep controlled categories such as `home`, `personal`, `education`, `vehicle`, `business`, `other`. |
| Existing total debt | `existing_debt` | number | `150000` | INR. |
| Monthly EMI / debt obligation | `monthly_emi` | number | `8000` | INR per month. |
| Credit score | `credit_score` | integer | `712` | Use the same credit-bureau scoring scale across all rows. |
| Previous defaults | `previous_defaults` | integer | `0` | Count of prior defaults; zero is valid. |
| Previous repayment status | `repayment_status` | categorical text | `good` | Use `excellent`, `good`, `fair`, or `poor`. |

## Required target column

The target must come from actual loan outcomes, not from the UI risk score.

| Column | Type | Example | Definition |
|---|---:|---:|---|
| `defaulted_within_12_months` | integer / boolean | `0` | `1` if the loan defaulted within 12 months of disbursement; otherwise `0`. |

A binary classifier can predict the probability of `defaulted_within_12_months = 1`. Do **not** train on the UI's displayed risk score or category; those are presentation-layer outputs derived from the model probability.

## Recommended operational columns

These fields do not need to be sent from the current UI, but they are important for reliable training, auditing, and joining loan outcomes.

| Column | Purpose |
|---|---|
| `application_id` | Unique ID for joining application and repayment-outcome data. |
| `application_date` | Supports time-based train/test splits and monitoring. |
| `loan_disbursement_date` | Establishes the target observation window. |
| `outcome_observed_date` | Date on which the default outcome was confirmed. |
| `data_source` | Tracks originating system or import. |

## Example CSV row

```csv
application_id,age,emp_type,emp_exp_years,annual_income,additional_income,loan_amount,loan_term_months,loan_purpose,existing_debt,monthly_emi,credit_score,previous_defaults,repayment_status,defaulted_within_12_months
APP-10001,32,salaried,6,850000,50000,1200000,60,home,150000,8000,712,0,good,0
```

## API field mapping

When the model is served, its `/api/assess` request should accept the UI's camelCase names and map them to the dataset columns:

| UI API field | Dataset column |
|---|---|
| `age` | `age` |
| `empType` | `emp_type` |
| `empExp` | `emp_exp_years` |
| `income` | `annual_income` |
| `addIncome` | `additional_income` |
| `loanAmt` | `loan_amount` |
| `loanTerm` | `loan_term_months` |
| `loanPurpose` | `loan_purpose` |
| `debt` | `existing_debt` |
| `emi` | `monthly_emi` |
| `credit` | `credit_score` |
| `defaults` | `previous_defaults` |
| `repayStatus` | `repayment_status` |
