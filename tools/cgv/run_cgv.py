"""CLI wrapper so the CGV engine can run outside the project interpreter.

The MCP server itself runs on a torch-less interpreter (its other tools only spawn subprocesses),
so ``cgv_check`` shells out to this script under the project's python instead of importing the
engine. The declaration is passed the same way: as data, never as a path.

    run_cgv.py <tree> --spec -                     # declaration as JSON on stdin
    run_cgv.py <tree> --spec spec.json

Exit code is 0 when every gate passes, 1 otherwise. Nothing is written to disk: the result is
the report.
"""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import contract as C  # noqa: E402
import engine  # noqa: E402


def read_spec(argv: list[str]) -> object:
    """The declaration arrives on stdin or as a JSON file, because that is what a caller has."""
    if "--spec" not in argv:
        print(json.dumps({"passed": False, "refused": C.ContractError(
            "no declaration supplied: CGV verifies a mechanism against the mathematical "
            "behaviour the caller declares, so pass it with --spec - (JSON on stdin) or "
            "--spec <file.json>. CGV does not read a declaration from the mechanism's tree.")}
        ), flush=True)
        raise SystemExit(2)
    value = argv[argv.index("--spec") + 1]
    if value == "-":
        return C.normalise_spec(sys.stdin.read(), source="the declaration on stdin")
    return C.normalise_spec(Path(value).read_text(encoding="utf-8"),
                            source=f"the declaration in {value}")


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: run_cgv.py <tree> --spec <file.json|->', file=sys.stderr)
        return 2
    tree = Path(sys.argv[1])
    argv = sys.argv[2:]
    if not tree.is_dir():
        print(json.dumps({"passed": False, "refused": f"no such tree: {tree}"}), flush=True)
        return 2
    spec = read_spec(argv)
    # G6 executes the trainer's real entry point, which prints to stdout. Redirect that noise to
    # stderr so stdout carries ONLY the JSON the caller parses.
    with contextlib.redirect_stdout(sys.stderr):
        result = engine.run_cgv(tree, spec)
    payload = {
        "passed": bool(result.get("passed")),
        "gates_passed": bool(result.get("gates_passed")),
        "arm": result.get("arm"),
        "mechanism": result.get("mechanism"),
        "tree": result.get("tree"),
        "trainer_sha256": result.get("trainer_sha256"),
        "checked_utc": result.get("checked_utc"),
        "schema_version": result.get("schema_version"),
        "spec_sha256": result.get("spec_sha256"),
        "fingerprint": result.get("fingerprint"),
        "declared": result.get("declared"),
        "independence": result.get("independence"),
        "refused": result.get("refused"),
        "summary": result.get("summary"),
        "gates": {
            gate: {
                "state": result["gates"].get(gate, {}).get("state"),
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
