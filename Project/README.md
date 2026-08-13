# RiskLens Application

This folder contains the integrated RiskLens web UI and Flask ML API. The UI calls the local `POST /api/assess` endpoint and renders real pipeline predictions. It is an underwriting support tool, not an automatic lending-decision system.

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000/`.

## Included integrations

- A responsive assessment UI served from `static/riskLens.html`
- Live assessment, model-metric, and health endpoints
- Server-side JSON validation, bounded values, origin allow-listing, and per-client rate limiting
- Model metrics and financial-analysis results rendered in the UI

## Configuration

`PORT` (default `8000`), `HOST` (default `0.0.0.0`), `DEBUG` (default `false`), `ALLOWED_ORIGINS` (comma-separated external UI origins), and `RATE_LIMIT_PER_MINUTE` (default `60`) are supported. The model artifact must be deployed as `model.joblib`; automatic local training requires `ALLOW_MODEL_TRAINING=true`.

## Production notes

Deploy behind a TLS reverse proxy, authenticate users, and attach an approved audit store before processing real applications. Do not expose datasets, model artifacts, or credentials to browsers. [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md) defines the API contract.
