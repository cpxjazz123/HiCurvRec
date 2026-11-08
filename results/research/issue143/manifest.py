"""The product manifest of one Stage 2 arm: what was run, and what it was.

Every one of Issues #140-#143 asks for the frozen math spec, the reference
contract, the source and the tests with their SHAs, plus the admission record and
the run configuration. This writes that manifest next to the issue, reading the
conformance report from the framework's own cache so a finished run's evidence is
never re-derived by hand.

    python3 results/research/issue143/manifest.py M4_post_sid_af3
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TREE = REPO / "stage2_RQ-VAE"
CACHE = REPO / "results" / "research" / "_conformance"
ISSUE = {"M1_learned_kappa": 140, "M2_radial_manifold": 141,
         "M3_role_profiles": 142, "M4_post_sid_af3": 143}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def arm_directory(arm: str) -> str:
    return arm if arm.startswith("TIGER_") else "TIGER_" + arm


def tree_files(tree: Path) -> dict:
    """Every source and test file with its digest, in a stable order."""
    files = {}
    for pattern in ("train_rqvae.py", "math_spec.json", "reference_contract.json",
                    "model/*.py", "tests/*.py"):
        for path in sorted(tree.glob(pattern)):
            if "__pycache__" in path.parts:
                continue
            files[str(path.relative_to(tree))] = digest(path)
    return files


def conformance(tree_name: str) -> dict:
    """The framework's cached conformance report for this tree, if one exists."""
    candidates = sorted(CACHE.glob("*__" + tree_name + "__*.json"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        return {"cache": str(path),
                "passed": report.get("passed"),
                "admissible": report.get("admissible"),
                "checks": {check["name"]: check["status"]
                           for check in report.get("checks", [])},
                "detail": {check["name"]: str(check.get("detail"))[:220]
                           for check in report.get("checks", [])}}
    return {"cache": None, "passed": None,
            "reason": ("no conformance report is cached for this tree: the launch ran "
                       "the checks live, and they are re-run after training so the "
                       "four-rank probe does not compete with the runs for the card")}


def run_config(arm: str) -> dict:
    path = TREE / arm_directory(arm) / "logs" / "training_metrics.jsonl"
    if not path.is_file():
        return {"available": False, "reason": "no metrics stream"}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "train_start":
            return {"available": True, "metrics": str(path), **row}
    return {"available": False, "reason": "no train_start event"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    args = parser.parse_args()
    tree_name = arm_directory(args.arm)
    tree = TREE / tree_name
    if not tree.is_dir():
        raise SystemExit("no tree at " + str(tree))
    issue = ISSUE.get(args.arm.replace("TIGER_", ""))
    payload = {
        "arm": args.arm,
        "issue": issue,
        "tree": str(tree),
        "files": tree_files(tree),
        "math_spec_sha256": digest(tree / "math_spec.json"),
        "reference_contract_sha256": digest(tree / "reference_contract.json"),
        "conformance": conformance(tree_name),
        "run": run_config(args.arm),
        "stage": "stage2-only",
        "stage3": "not started and not designed (Issues #140-#143)",
    }
    target_dir = REPO / "results" / "research" / ("issue%d" % issue)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / ("manifest_%s.json" % args.arm)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(target)
    print(json.dumps({key: payload[key] for key in
                      ("math_spec_sha256", "reference_contract_sha256")}, indent=2))
    print("conformance:", payload["conformance"].get("passed"),
          payload["conformance"].get("checks"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
