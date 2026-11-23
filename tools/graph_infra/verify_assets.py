"""Verify the frozen asset manifest; exit non-zero on any drift."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.graph_infra import assets  # noqa: E402

MANIFEST = REPO / "results" / "research" / "infra245" / "assets" / "manifest.json"


def main() -> int:
    result = assets.verify_manifest(MANIFEST)
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(f"issue {result['issue']}: {len(result['verified'])}/{len(payload['assets'])} assets verified")
    for name in result["verified"]:
        print(f"  OK {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
