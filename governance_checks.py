"""Fail-closed deployment governance checks used by run_project.bat."""
import json
import sys

from governance import run_governance_checks


if __name__ == "__main__":
    try:
        result = run_governance_checks()
    except Exception as exc:
        print(f"Governance checks failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
