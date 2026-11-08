"""Execute the four independently authored suites of one arm and record the evidence.

The verification runner executes in three process groups because the isolation is
the boundary, not an optimisation: math+gradient can share an interpreter, while
DDP (two real Gloo ranks) and the real entrypoint each keep their own process.

    python3 results/research/issue143/run_arm_suites.py M1_learned_kappa
    python3 results/research/issue143/run_arm_suites.py M1_learned_kappa --groups ddp

Each suite is recorded with its tests_run/failures/errors and ``ok``; a suite that
ran zero tests, skipped, or failed is not a pass. The evidence file is
``results/research/issue<NN>/suites_<ARM>.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TREE = REPO / "stage2_RQ-VAE"
RUNNER = REPO / "tools" / "research" / "verification" / "run_suites.py"
PROJECT_PYTHON = Path("/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9")

ARMS = {"M1_learned_kappa": 140, "M2_radial_manifold": 141,
        "M3_role_profiles": 142, "M4_post_sid_af3": 143}
GROUPS = [("math", "gradient"), ("ddp",), ("entrypoint",)]


def run_group(case_dir: Path, names: tuple[str, ...]) -> dict:
    command = [str(PROJECT_PYTHON), str(RUNNER), str(case_dir)] + list(names)
    done = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    # Torch/Gloo print to stdout before the runner's single JSON document, so the
    # document is the last parseable line, not the whole stdout.
    for line in reversed(done.stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"ok": False,
            "suites": {name: {"ok": False, "tests_run": 0,
                              "output": (done.stdout + done.stderr)[-1500:]}
                       for name in names}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=sorted(ARMS))
    parser.add_argument("--groups", nargs="+", choices=["math", "gradient", "ddp",
                                                       "entrypoint"])
    args = parser.parse_args()
    wanted = set(args.groups) if args.groups else {"math", "gradient", "ddp",
                                                   "entrypoint"}
    case_dir = TREE / ("TIGER_" + args.arm) / "tests"
    suites: dict[str, dict] = {}
    for group in GROUPS:
        names = tuple(name for name in group if name in wanted)
        if not names:
            continue
        payload = run_group(case_dir, names)
        suites.update(payload.get("suites") or {})
    record = {"arm": args.arm, "tree": str(case_dir),
              "interpreter": str(PROJECT_PYTHON),
              "ok": all(suite.get("ok") for suite in suites.values()) and bool(suites),
              "suites": {name: {"path": body.get("path"),
                                "tests_run": body.get("tests_run"),
                                "failures": body.get("failures"),
                                "errors": body.get("errors"),
                                "skipped": body.get("skipped"),
                                "expected_failures": body.get("expected_failures"),
                                "unexpected_successes": body.get("unexpected_successes"),
                                "ok": body.get("ok"),
                                "output": (body.get("output") or "")[-800:]}
                         for name, body in suites.items()}}
    target = (REPO / "results" / "research" / ("issue%d" % ARMS[args.arm])
              / ("suites_%s.json" % args.arm))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
