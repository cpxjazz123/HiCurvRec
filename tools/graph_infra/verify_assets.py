"""Verify the frozen asset manifest; exit non-zero on any drift."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.graph_infra import assets  # noqa: E402

MANIFESTS = {
    "v1": REPO / "results" / "research" / "infra245" / "assets" / "manifest.json",
    "v2": REPO / "results" / "research" / "infra246" / "assets" / "manifest.json",
}


def main(argv) -> int:
    version = argv[1] if len(argv) > 1 else "v2"
    if version not in MANIFESTS:
        raise SystemExit(f"unknown version {version!r}; choose from {sorted(MANIFESTS)}")
    manifest = MANIFESTS[version]
    if not manifest.exists():
        raise SystemExit(f"missing manifest: {manifest}")
    if version == "v2":
        audit = assets.audit_manifest(manifest)
        if not audit["ok"]:
            raise SystemExit(f"v2 manifest audit failed: {audit['issues'][:5]}")
    result = assets.verify_manifest(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    print(f"issue {result['issue']} ({version}): {len(result['verified'])}/{len(payload['assets'])} "
          f"assets verified")
    for name in result["verified"]:
        print(f"  OK {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
