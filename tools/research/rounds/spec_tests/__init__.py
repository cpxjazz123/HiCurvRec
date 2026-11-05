"""The tests a Math Reviewer wrote, kept and run on every later round.

The rule the round follows: a defect found in a specification is never answered
with prose alone. The reviewer that finds it writes a test - a module exposing
``check(spec) -> list[str]`` that returns a non-empty list exactly when the
specification shows the defect - and the framework proves the test really does
fail on the specification it was written against. The test is then written to
this directory and never removed: every later round runs every test here before
it may freeze, so a defect that was caught once cannot come back unnoticed.

The tests are data, not decoration: they are plain Python over the specification
dictionary, they carry the finding they came from, and a test that raises or
times out is a failure rather than a pass. Nothing here imports torch or the
agents SDK: a check reads the specification.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

#: Where the retained tests live: inside the package, so they are committed with
#: the code that runs them.
DIRECTORY = Path(__file__).resolve().parent

#: A check may not take the freeze hostage: it reads a dictionary.
CHECK_TIMEOUT_SECONDS = 120

#: The suffix a retained test file has.
SUFFIX = ".py"

DRIVER = """
import importlib.util, json, sys, traceback
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
spec = payload["spec"]
problems = {}
for entry in payload["paths"]:
    path = Path(entry)
    try:
        module_spec = importlib.util.spec_from_file_location(
            "retained_" + path.stem, path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        check = getattr(module, "check", None)
        if not callable(check):
            problems[str(path)] = ["the test exposes no check(spec) function"]
            continue
        found = check(spec)
        if not isinstance(found, list):
            problems[str(path)] = ["check(spec) returned %s, not a list"
                                   % type(found).__name__]
        elif found:
            problems[str(path)] = [str(item) for item in found]
    except Exception:
        problems[str(path)] = ["the test raised: " +
                               traceback.format_exc().strip().splitlines()[-1][:300]]
print("@@PROBLEMS@@" + json.dumps(problems))
"""


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", str(text).lower())[:8]
    return "-".join(words) or "finding"


def existing() -> list:
    """Every retained test file."""
    return sorted(path for path in DIRECTORY.glob("*" + SUFFIX)
                  if path.stem not in ("__init__",))


def load() -> list:
    """The retained tests as records, without executing them."""
    records = []
    for path in existing():
        text = path.read_text(encoding="utf-8")
        record = {"path": str(path), "name": path.stem}
        for field in ("FINDING", "ROUND", "EVAL_ID"):
            found = re.search(r"^%s\s*=\s*(.+)$" % field, text, re.MULTILINE)
            if found:
                record[field.lower()] = found.group(1).strip().strip("\"'")
        records.append(record)
    return records


def run(spec: dict, paths: list = None, *, python: str = None) -> dict:
    """Run every retained test against one specification."""
    targets = [str(Path(item)) for item in (paths if paths is not None else existing())]
    if not targets:
        return {"passed": True, "checked": 0, "failures": []}
    directory = Path(tempfile.mkdtemp(prefix="spec-tests-"))
    payload = directory / "payload.json"
    payload.write_text(json.dumps({"spec": spec, "paths": targets}),
                       encoding="utf-8")
    script = directory / "run.py"
    script.write_text(DRIVER, encoding="utf-8")
    try:
        done = subprocess.run([python or sys.executable, str(script), str(payload)],
                              capture_output=True, text=True,
                              timeout=CHECK_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        return {"passed": False, "checked": len(targets),
                "failures": [{"id": "retained-tests", "problems":
                              ["the retained tests could not run: %s" % error]}],
                "findings": ["the retained tests could not run: %s" % error]}
    marker = "@@PROBLEMS@@"
    line = next((item for item in reversed(done.stdout.splitlines())
                 if item.startswith(marker)), "")
    if not line:
        reason = (done.stderr or done.stdout or "no output").strip()[-400:]
        return {"passed": False, "checked": len(targets),
                "failures": [{"id": "retained-tests",
                              "problems": ["the retained tests did not finish: " + reason]}],
                "findings": ["the retained tests did not finish: " + reason]}
    problems = json.loads(line[len(marker):])
    failures = [{"id": Path(path).stem, "problems": issues}
                for path, issues in problems.items() if issues]
    return {"passed": not failures, "checked": len(targets), "failures": failures,
            "findings": [problem for failure in failures
                         for problem in failure["problems"]]}


def record(directory: Path, round_id: str, finding: str, code: str,
           *, eval_id: str = "") -> dict:
    """Write one reviewer-written test to the permanent suite.

    The test is written with the finding it came from, so a later reader knows
    what it refuses and why; the same test is deduplicated by finding, so a
    reviewer that restates a finding does not add a second copy of its test.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    slug = _slug(finding)
    for path in sorted(directory.glob("*__%s%s" % (slug, SUFFIX))):
        return {"path": str(path), "written": False, "reason": "already retained"}
    name = "%s__%s%s" % (round_id, slug, SUFFIX)
    path = directory / name
    header = (
        '"""%s\n\n'
        'Round: %s\n'
        'The reviewer that found this defect wrote the test below; it failed on\n'
        'the specification that carried the defect and is run on every later\n'
        'round before it may freeze.\n"""\n\n'
        'FINDING = %s\n'
        'ROUND = %s\n'
        'EVAL_ID = %s\n\n'
        % (str(finding).replace('"', "'")[:400], round_id,
           json.dumps(str(finding)[:400]), json.dumps(round_id),
           json.dumps(str(eval_id or ""))))
    body = str(code or "").strip() + "\n"
    path.write_text(header + body, encoding="utf-8")
    return {"path": str(path), "written": True}
