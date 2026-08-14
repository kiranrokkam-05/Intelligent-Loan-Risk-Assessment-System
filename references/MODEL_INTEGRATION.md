# Model Integration Guide

The integrated Flask service exposes `POST /api/assess`; the browser uses it from the same origin. Do not use the API as the sole basis for lending decisions.

## API contract

`POST /api/assess` accepts JSON with the required application fields:

```json
{"age":32,"empType":"salaried","empExp":6,"income":850000,"addIncome":50000,"loanAmt":1200000,"loanTerm":60,"loanPurpose":"home","debt":150000,"emi":8000,"credit":712,"defaults":0,"repayStatus":"good"}
```

The optional `model` or `modelChoice` selects `xgboost`, `rf`, or `lr`. Successful responses include `apiVersion`, a 0–100 `score`, `category`, `probability`, `factors` (`label`, `valueText`, `tier`, `iconKey`), `loanDetails`, and `recommendation`.

Invalid input receives `422` with `{ "error": "..." }`; malformed or missing JSON receives `400`; excessive requests receive `429`.

## External UI integration

For a separately hosted UI, configure its exact origin in `ALLOWED_ORIGINS`, then call:

```js
await fetch('https://your-api.example.com/api/assess', {
  method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
});
```

Validate the response before display. The bundled UI is the reference consumer in `static/riskLens.html`.

## Deployment checks

1. Serve the UI from the same domain where possible; otherwise set exact trusted origins in `ALLOWED_ORIGINS`.
2. Keep input validation, authentication, logging, rate limiting, and privacy controls at the API layer.
3. Never expose model artifacts, credentials, or training data to clients.
4. Test valid low-, medium-, and high-risk profiles plus invalid input, rate limits, and service failures.
