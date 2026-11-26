"""Issue #262: compare the public-infrastructure/asset snapshot after analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/research/issue262"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    start = json.loads((OUT / "INFRA_SNAPSHOT_START.json").read_text())
    end = {"schema": "genrec.issue262.infra_snapshot/1",
           "issue": start["issue"], "issue_body_sha256": start["issue_body_sha256"],
           "mode": "FROZEN", "files": {}, "missing": [], "changed": []}
    for path, before in start["files"].items():
        target = ROOT / path
        if not target.is_file():
            end["files"][path] = {"path": path, "status": "MISSING", "sha256": None,
                                  "bytes": None}
        else:
            end["files"][path] = {"path": path, "status": "PASS", "sha256": sha(target),
                                  "bytes": target.stat().st_size}
        if end["files"][path].get("sha256") != before.get("sha256"):
            end["changed"].append(path)
        if end["files"][path]["status"] != "PASS":
            end["missing"].append(path)
    end["status"] = "PASS" if not end["changed"] and not end["missing"] else "DRIFT"
    (OUT / "INFRA_SNAPSHOT_END.json").write_text(json.dumps(end, indent=2, sort_keys=True) + "\n")
    (OUT / "INFRA_DIFF.json").write_text(json.dumps(
        {"schema": "genrec.issue262.infra_diff/1", "status": end["status"],
         "changed": end["changed"], "missing": end["missing"],
         "start": "INFRA_SNAPSHOT_START.json", "end": "INFRA_SNAPSHOT_END.json"}, indent=2) + "\n")
    print(json.dumps({"status": end["status"], "changed": end["changed"],
                      "missing": end["missing"]}, indent=2))


if __name__ == "__main__":
    main()
