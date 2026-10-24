"""One detached verifier stage, using run-scoped evidence and stdin IPC."""
import json
import sys
import traceback
from pathlib import Path

from ..core.evidence import atomic_json, read_json
from ..verification import workflow
from .handlers import _root


def main():
    request = json.loads(sys.stdin.readline())
    run_id = request["run_id"]
    root = _root(run_id)
    path = root / "verification_requests" / (request["action"] + ".json")
    current = read_json(path)
    if current["request_id"] != request["request_id"]:
        raise SystemExit("superseded verification request")
    try:
        if (root / "cancel.json").exists():
            raise RuntimeError("research cancelled")
        tree = Path(request["source_dir"])
        if request["action"] == "draft":
            report = workflow.draft_test_suites(request["candidate_id"], tree, run_id)
            current["status"] = "drafts_ready"
            current["draft_count"] = sum(len(x["drafts"]) for x in report["authors"].values())
        elif request["action"] == "review":
            workflow.review_committed_suites(request["candidate_id"], tree, run_id)
            if (root / "cancel.json").exists():
                raise RuntimeError("research cancelled")
            manifest = workflow.freeze_reviewed_manifest(request["candidate_id"], tree, run_id)
            current["status"] = "reviewed"
            current["manifest"] = str(manifest)
        else:
            raise ValueError("unknown verification action")
    except Exception as exc:
        current["status"] = "failed"
        current["error"] = type(exc).__name__ + ": " + str(exc)[:400]
        atomic_json(path, current)
        traceback.print_exc()
        return 1
    atomic_json(path, current)
    return 0


if __name__ == "__main__":
    sys.exit(main())
