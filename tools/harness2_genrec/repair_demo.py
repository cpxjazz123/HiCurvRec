"""The failure -> diagnosis -> repair -> recheck demonstration (issue #102).

An intentionally incorrect implementation is rejected by an independent judge, OpenCode
repairs it, and the judge accepts the repair. The point of the demonstration is not that
OpenCode can fix a bug: it is that the **judge** decides, that the agent cannot edit the
judge, and that the evidence of every step survives on disk.

The defect comes from the frozen mutation catalogue (``tools/mvg/mutants.py``): the same
defects the MVG suite uses to prove its gates have teeth. The catalogue is imported, never
edited, and the defect is injected into a scratch copy of the arm -- never into the
repository's tree.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import opencode

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"


def _mutant(mutant_id: str):
    """The frozen defect, from the repository's own catalogue."""
    sys_path = str(REPO_ROOT)
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)
    from tools.mvg import mutants as MUT  # type: ignore  # noqa: E402

    catalogue = MUT.by_id()
    if mutant_id not in catalogue:
        raise ValueError(
            f"unknown mutant {mutant_id!r}; the catalogue has {sorted(catalogue)}"
        )
    return catalogue[mutant_id]


def _copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(Path(source).rglob("*")):
        if not path.is_file():
            continue
        parts = set(path.relative_to(source).parts)
        if "__pycache__" in parts or path.suffix in (".pth", ".npy", ".pyc"):
            continue
        if parts & {"logs", "results", "ckpt", "dataset"}:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def run_judges(tree: Path, spec_path: Path, out_dir: Path) -> Dict[str, Any]:
    """Run the independent judges over the tree and return their report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            PROJECT_PYTHON,
            "-m",
            "tools.mvg.cli",
            "run",
            "--tree",
            str(tree),
            "--spec",
            str(spec_path),
            "--out",
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    report_path = out_dir / "mvg_report.json"
    report: Dict[str, Any] = {}
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "verdict": report.get("verdict", "failed"),
        "returncode": completed.returncode,
        "report_path": str(report_path),
        "diagnostics": {
            gate: result.get("diagnostics")
            for gate, result in (report.get("gates") or {}).items()
            if result.get("state") != "passed"
        },
        "gates": {
            gate: result.get("state") for gate, result in (report.get("gates") or {}).items()
        },
        "stderr_tail": completed.stderr[-800:],
    }


def run(
    tree: Path,
    spec_path: Path,
    mutant_id: str,
    out_dir: Path,
    *,
    model: str = "",
    timeout: int = 1800,
) -> Dict[str, Any]:
    """Inject the defect, let the judge reject it, let OpenCode repair it, recheck."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = out_dir / "arm"
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    _copy_tree(Path(tree), workspace)

    mutant = _mutant(mutant_id)
    integration = json.loads(
        (Path(spec_path).read_text(encoding="utf-8"))
    )["integration"]
    module_path = workspace / f"{integration['module']}.py"
    source = module_path.read_text(encoding="utf-8")
    for old, new in mutant.replacements:
        if old not in source:
            raise RuntimeError(
                f"the defect {mutant.id!r} does not match the current source of "
                f"{module_path}; the catalogue and the implementation have drifted"
            )
        source = source.replace(old, new, 1)
    module_path.write_text(source, encoding="utf-8")
    (out_dir / "defect.json").write_text(
        json.dumps(
            {
                "mutant": mutant.id,
                "why": mutant.why,
                "rationale": mutant.rationale,
                "must_fail_gate": mutant.must_fail_gate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    first = run_judges(workspace, Path(spec_path), out_dir / "judges_before")
    if first["verdict"] == "passed":
        raise RuntimeError(
            f"the judge accepted the defective implementation: the defect {mutant.id!r} is not "
            "caught, and the demonstration cannot be made"
        )

    problems = "\n".join(
        f"- {gate}: {problem}"
        for gate, problems_list in (first["diagnostics"] or {}).items()
        for problem in (problems_list or [])
    )
    prompt = (
        "An independent check rejected your implementation. The findings are below; they come "
        "from a judge that runs outside this tree and that you cannot change.\n\n"
        f"# FINDINGS\n\n{problems}\n\n"
        "# WHAT TO DO\n\n"
        "Diagnose the findings against the code in this tree, then fix the code. Do not "
        "change the judge, the fixtures, the thresholds or the frozen specification: they "
        "are not yours, and a fix that hides the symptom will be caught by the same check. "
        "Report what you changed and why."
    )
    repair = opencode.run_agent(
        workspace,
        prompt,
        model=model or opencode.model_name(),
        timeout=timeout,
        mcp=opencode.mcp_server_block(REPO_ROOT),
        title=f"repair {mutant.id}",
        deny_paths=[REPO_ROOT],
    )
    (out_dir / "repair.json").write_text(
        json.dumps(repair, indent=2, default=str), encoding="utf-8"
    )

    second = run_judges(workspace, Path(spec_path), out_dir / "judges_after")

    return {
        "mutant": mutant.id,
        "why": mutant.why,
        "judged_before": first,
        "repair": {
            "model": repair.get("model"),
            "exit_code": repair.get("exit_code"),
            "cost": repair.get("cost"),
            "tokens": repair.get("tokens"),
            "tool_calls": repair.get("tool_calls"),
            "text": repair.get("text"),
        },
        "judged_after": second,
        "rejected_then_repaired": first["verdict"] != "passed" and second["verdict"] == "passed",
        "workspace": str(workspace),
    }


__all__ = ["run", "run_judges"]
