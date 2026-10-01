"""Operational model-risk, privacy, and deployment controls for RiskLens.

These checks support governance; they are not a legal or regulatory certification.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
HASH_PATH = MODEL_PATH.with_suffix(".joblib.sha256")
METADATA_PATH = MODEL_PATH.with_suffix(".metadata.json")
REPORT_PATH = BASE_DIR / "evaluation" / "model_risk_report.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_metadata(path: Path = MODEL_PATH, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    hash_path = path.with_suffix(".joblib.sha256")
    hash_path.write_text(digest + "\n", encoding="utf-8")
    result = {
        "artifact": path.name,
        "sha256": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "metadata": metadata or {},
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def verify_artifact_integrity(path: Path = MODEL_PATH, *, require_sidecar: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Model artifact is missing: {path}")
    actual = sha256_file(path)
    hash_path = path.with_suffix(".joblib.sha256")
    metadata_path = path.with_suffix(".metadata.json")
    if require_sidecar and (not hash_path.is_file() or not metadata_path.is_file()):
        raise RuntimeError("Model integrity sidecars are missing")
    expected = hash_path.read_text(encoding="utf-8").strip() if hash_path.is_file() else actual
    if not secrets.compare_digest(actual, expected):
        raise RuntimeError("Model artifact hash verification failed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    if metadata.get("sha256") and not secrets.compare_digest(actual, metadata["sha256"]):
        raise RuntimeError("Model metadata hash verification failed")
    return {"sha256": actual, "metadata": metadata}


def validate_production_config(env: dict[str, str] | None = None) -> list[str]:
    values = os.environ if env is None else env
    if values.get("ENVIRONMENT", "development").strip().lower() != "production":
        return []
    errors = []
    if values.get("ALLOW_MODEL_TRAINING", "false").lower() in {"1", "true", "yes", "t"}:
        errors.append("ALLOW_MODEL_TRAINING must be false in production")
    if not values.get("ALLOWED_ORIGINS", "").strip():
        errors.append("ALLOWED_ORIGINS must be explicitly configured")
    return errors


def run_governance_checks() -> dict[str, Any]:
    errors = validate_production_config()
    integrity = verify_artifact_integrity(require_sidecar=os.environ.get("ENVIRONMENT", "development").lower() == "production")
    if not REPORT_PATH.is_file():
        errors.append("model-risk evaluation report is missing")
    return {"ok": not errors, "errors": errors, "artifact": integrity}
