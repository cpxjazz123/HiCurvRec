"""Run reviewed suites in one interpreter and report per-suite evidence.

The MCP executes four independently authored suites. Starting four Python
processes made each of them pay the interpreter and PyTorch import cost, and
that dominated the wall time of a check. This runner is started once per
PROCESS GROUP instead:

    python run_suites.py <case_dir> <suite> [<suite> ...]

The groups the caller forms are the isolation boundary, not an optimisation
detail: ``math`` and ``gradient`` can share an interpreter because neither
spawns ranks nor drives the trainer, while ``ddp`` (real two-rank Gloo) and
``entrypoint`` (real ``trainer.main()``) each keep their own process.

It prints one JSON document and always reports per suite, so every suite keeps
its own PASS/FAIL evidence:

    {"ok": bool,
     "suites": {"<name>": {"path", "tests_run", "failures", "errors", "skipped",
                           "expected_failures", "unexpected_successes", "ok",
                           "output"}}}

A suite is ``ok`` only when it ran at least one test, every test passed, and
nothing was skipped, expected to fail or unexpectedly passed. An import error
is a failed suite, never a missing one.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import traceback
import unittest
from pathlib import Path


def run_suite(name: str, case_dir: Path) -> dict:
    path = case_dir / ("test_" + name + ".py")
    buffer = io.StringIO()
    record = {
        "path": str(path), "tests_run": 0, "failures": 0, "errors": 1,
        "skipped": 0, "expected_failures": 0, "unexpected_successes": 0,
        "ok": False, "output": "",
    }
    try:
        spec = importlib.util.spec_from_file_location("reviewed_" + name, path)
        if spec is None or spec.loader is None:
            raise ImportError("cannot load " + str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
        record.update(
            tests_run=result.testsRun,
            failures=len(result.failures),
            errors=len(result.errors),
            skipped=len(result.skipped),
            expected_failures=len(result.expectedFailures),
            unexpected_successes=len(result.unexpectedSuccesses),
        )
        record["ok"] = bool(
            result.wasSuccessful()
            and result.testsRun >= 1
            and not result.skipped
            and not result.expectedFailures
            and not result.unexpectedSuccesses
        )
    except BaseException:  # noqa: BLE001 - an import failure is a failed suite
        buffer.write(traceback.format_exc())
        record["errors"] = max(record["errors"], 1)
        record["ok"] = False
    record["output"] = buffer.getvalue()
    return record


def main(argv: list) -> int:
    if len(argv) < 3:
        print("usage: run_suites.py <case_dir> <suite> [<suite> ...]", file=sys.stderr)
        return 2
    case_dir = Path(argv[1])
    names = argv[2:]
    suites = {name: run_suite(name, case_dir) for name in names}
    payload = {"ok": all(record["ok"] for record in suites.values()), "suites": suites}
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
