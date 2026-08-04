# Model Integration Guide

The current application is intentionally a frontend-only UX prototype. The browser calculates a mock score only to demonstrate result states. Do not use it for lending decisions.

## 1. ML team's deliverable

Provide a deployed HTTP endpoint, for example `POST /api/assess`.

It should accept JSON with the same fields currently collected by the form:

```json
{
  "age": 32,
  "empType": "salaried",
  "empExp": 6,
  "income": 850000,
  "addIncome": 50000,
  "loanAmt": 1200000,
  "loanTerm": 60,
  "loanPurpose": "home",
  "debt": 150000,
  "emi": 8000,
  "credit": 712,
  "defaults": 0,
  "repayStatus": "good"
}
```

Return a stable response shape:

```json
{
  "score": 28,
  "category": "Low Risk",
  "factors": [{"label": "Credit score", "value": "Strong", "tier": "good"}],
  "recommendation": "Suitable for standard review."
}
```

`category` must be `Low Risk`, `Medium Risk`, or `High Risk`; `tier` must be `good`, `warn`, or `bad`; and `score` must be 0–100.

## 2. Where to connect it

In `static/riskLens.html`, find `MOCK SCORING START`. Replace that mock calculation with a `fetch()` call to the ML endpoint. Keep the DOM updates below `MOCK SCORING END`; they render the gauge, badge, factors, and recommendation.

```js
const response = await fetch('https://your-api.example.com/api/assess', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(payload)
});
if (!response.ok) throw new Error('Assessment service is unavailable');
const result = await response.json();
```

Map `result.score`, `result.category`, `result.factors`, and `result.recommendation` to the existing UI elements.

## 3. Integration checks

1. Enable CORS for the deployed UI origin, or serve the static UI from the same domain as the API.
2. Validate all input on the server; browser validation is only a UX aid.
3. Do not expose model files, credentials, or training data in this UI repository.
4. Add authentication, logging, rate limiting, and privacy controls in the production API layer before handling real applicant information.
5. Test the API with low-, medium-, high-risk, invalid-input, and timeout responses so every UI state is verified.
