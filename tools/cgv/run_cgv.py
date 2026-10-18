"""CLI wrapper so the CGV engine can run outside the project interpreter.

The MCP server itself runs on a torch-less interpreter (its other tools only spawn subprocesses),
so ``cgv_check`` shells out to this script under the project's python instead of importing the
engine in-process.

    cgv/run_cgv.py <tree>                 # run the six gates, print JSON, do NOT write a certificate
    cgv/run_cgv.py <tree> --certificate   # ... and write results/stage2_RQ-VAE/_cgv/certificate_<arm>.json

Exit code is 0 when every gate passes, 1 otherwise (so a caller can branch without parsing).
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import engine  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_cgv.py <tree> [--certificate]", file=sys.stderr)
        return 2
    tree = Path(sys.argv[1])
    if not tree.is_dir():
        print(json.dumps({"passed": False, "refused": f"no such tree: {tree}"}), flush=True)
        return 2
    # G6 executes the trainer's real main(), which prints to stdout. Redirect that noise to
    # stderr so stdout carries ONLY the JSON the caller parses.
    with contextlib.redirect_stdout(sys.stderr):
        result = engine.run_cgv(tree)
    certificate = engine.write_certificate(result) if "--certificate" in sys.argv[3:] else None
    trainable, why = engine.certificate_is_valid(tree)
    payload = {
        "passed": bool(result.get("passed")),
        "arm": result.get("arm"),
        "mechanism": result.get("mechanism"),
        "tree": result.get("tree"),
        "trainer_sha256": result.get("trainer_sha256"),
        "checked_utc": result.get("checked_utc"),
        "certificate": str(certificate) if certificate else None,
        "trainable_now": trainable,
        "trainable_note": why,
        "refused": result.get("refused"),
        "summary": result.get("summary"),
        "gates": {
            gate: {
                "passed": result["gates"].get(gate, {}).get("passed"),
                "title": result["gates"].get(gate, {}).get("title"),
                "detail": result["gates"].get(gate, {}).get("detail"),
                "warnings": result["gates"].get(gate, {}).get("warnings"),
                "finding": result["gates"].get(gate, {}).get("finding"),
            }
            for gate in engine.GATES
            if gate in result.get("gates", {})
        },
    }
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
