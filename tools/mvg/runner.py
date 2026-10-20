"""Run the MVG gates over a mechanism tree and write ``mvg_report.json``.

The report is the artefact: per gate a state (``passed`` / ``failed`` / ``blocked``), the
measurements behind it, the diagnostics, and the digests that bind it to the exact sources
and specification it was produced from. A gate that could not run is ``blocked``, and
``blocked`` is never a pass.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import gates as G
from . import mutants as MUT
from . import spec as SPEC

REPORT_VERSION = 1

GATE_FUNCTIONS = {
    "M0": G.gate_m0,
    "M1": G.gate_m1,
    "M2": G.gate_m2,
    "M3": G.gate_m3,
    "M4": G.gate_m4,
}


def _git_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - defensive
        return ""


def _tree_digest(tree: Path) -> Tuple[str, int]:
    """The digest of the tree's code, using the CGV's own implementation."""
    repo_cgv = SPEC.REPO_ROOT / "tools" / "cgv"
    if str(repo_cgv) not in sys.path:
        sys.path.insert(0, str(repo_cgv))
    import contract as cgv_contract  # type: ignore  # noqa: E402

    digest, count = cgv_contract.code_tree_sha256(tree)
    return digest or "", int(count or 0)


def _copy_tree(tree: Path, dest: Path) -> Path:
    """Copy the tree's code into a scratch directory. Nothing else is copied."""
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(tree.rglob("*")):
        if not path.is_file():
            continue
        parts = set(path.relative_to(tree).parts)
        if "__pycache__" in parts or path.suffix in (".pth", ".npy", ".pyc"):
            continue
        if parts & {"logs", "results", "dataset", "ckpt"}:
            continue
        target = dest / path.relative_to(tree)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return dest


def _apply_mutant(
    tree: Path, mutant: MUT.Mutant, integration: Dict[str, Any]
) -> Tuple[bool, str]:
    """Inject the defect into the mechanism module of a scratch tree."""
    module_path = tree / f"{integration['module']}.py"
    if not module_path.is_file():
        return False, "the mechanism module of the scratch tree was not found"
    source = module_path.read_text(encoding="utf-8")
    for old, new in mutant.replacements:
        if old not in source:
            return False, f"the mutant {mutant.id!r} does not match the current source"
        source = source.replace(old, new, 1)
    module_path.write_text(source, encoding="utf-8")
    return True, ""


def gate_m5(mut: G.MechanismUnderTest) -> G.GateResult:
    """Every named defect must be rejected by the gate that is supposed to reject it."""
    scratch_root = Path(tempfile.mkdtemp(prefix="mvg-mutants-"))
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    try:
        for mutant in MUT.MUTANTS:
            copy = _copy_tree(mut.baseline_tree, scratch_root / mutant.id)
            patched, detail = _apply_mutant(copy, mutant, mut.integration)
            if not patched:
                problems.append(
                    f"{mutant.id}: {detail} (the catalogue is injected into the tree the "
                    "specification was written against)"
                )
                rows.append(
                    {
                        "mutant": mutant.id,
                        "why": mutant.why,
                        "must_fail_gate": mutant.must_fail_gate,
                        "caught": False,
                        "error": detail,
                    }
                )
                continue
            mutant_mut = G.MechanismUnderTest(copy, mut.spec)
            states: Dict[str, str] = {}
            details: Dict[str, str] = {}
            for gate, function in GATE_FUNCTIONS.items():
                try:
                    result = function(mutant_mut)
                    states[gate] = result.state
                    details[gate] = result.detail
                except Exception as exc:
                    states[gate] = G.FAILED
                    details[gate] = f"{type(exc).__name__}: {exc}"
            failed = [gate for gate, state in states.items() if state == G.FAILED]
            caught = mutant.must_fail_gate in failed
            if not caught:
                problems.append(
                    f"the mutant {mutant.id!r} was not rejected by {mutant.must_fail_gate}: "
                    f"the gates that failed were {failed or 'none'}"
                )
            rows.append(
                {
                    "mutant": mutant.id,
                    "why": mutant.why,
                    "rationale": mutant.rationale,
                    "must_fail_gate": mutant.must_fail_gate,
                    "gate_states": states,
                    "failed_gates": failed,
                    "caught": caught,
                }
            )
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    measurements = {
        "mutants": rows,
        "mutants_checked": len(rows),
        "mutants_caught": sum(1 for row in rows if row.get("caught")),
        "scratch_root_removed": not scratch_root.exists(),
    }
    if problems:
        return G.GateResult(
            "M5",
            G.FAILED,
            f"{len(problems)} mutant(s) were not rejected",
            measurements,
            problems,
        )
    return G.GateResult(
        "M5",
        G.PASSED,
        f"all {len(rows)} named defect(s) were rejected by the gate that owns them",
        measurements,
    )


def run(
    tree: Path,
    spec_path: Path,
    out_dir: Optional[Path] = None,
    *,
    skip_m5: bool = False,
    baseline_tree: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run every gate over the tree and return the report."""
    doc = SPEC.load_spec(spec_path)
    mut = G.MechanismUnderTest(tree, doc, baseline_tree=baseline_tree)

    results: Dict[str, G.GateResult] = {}
    for gate, function in GATE_FUNCTIONS.items():
        try:
            results[gate] = function(mut)
        except Exception as exc:  # pragma: no cover - defensive
            results[gate] = G.GateResult(
                gate, G.BLOCKED, f"the gate could not run: {type(exc).__name__}: {exc}", {}, [str(exc)]
            )
    if skip_m5:
        results["M5"] = G.GateResult("M5", G.BLOCKED, "the mutation suite was skipped", {}, [])
    else:
        results["M5"] = gate_m5(mut)

    tree_digest, tree_files = _tree_digest(mut.tree)
    failed = [gate for gate, result in results.items() if result.state == G.FAILED]
    blocked = [gate for gate, result in results.items() if result.state == G.BLOCKED]
    verdict = "passed" if not failed and not blocked else "failed"

    report: Dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mechanism": {
            "id": doc["mechanism"]["id"],
            "version": doc["mechanism"]["version"],
            "summary": doc["mechanism"]["summary"],
            "spec_path": str(spec_path),
            "spec_sha256": SPEC.spec_digest(doc),
            "tree": str(mut.tree),
            "tree_code_sha256": tree_digest,
            "tree_code_files": tree_files,
            "module_sha256": results["M0"].measurements.get("module_sha256", ""),
            "commit": _git_commit(SPEC.REPO_ROOT),
            "environment": {
                "python": sys.version.split()[0],
                "torch": _torch_version(),
                "device": "cpu",
            },
        },
        "gates": {gate: result.as_dict() for gate, result in results.items()},
        "verdict": verdict,
        "failed_gates": failed,
        "blocked_gates": blocked,
        "coverage": _coverage(doc, results),
        "residual_uncertainty": (
            "A PASS is evidence that the implementation reproduces the frozen specification on "
            "the declared cases, not a proof of absolute mathematical correctness. The gates "
            "compare against an independent reference implementation and against declared "
            "metamorphic properties; a defect that is invisible to both on the declared cases "
            "is not excluded."
        ),
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "mvg_report.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        report["report_path"] = str(path)
    return report


def _torch_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _coverage(doc: Dict[str, Any], results: Dict[str, G.GateResult]) -> Dict[str, Any]:
    """What the run actually exercised, stated as counts."""
    cases = sum(len(entries) for entries in doc["cases"].values())
    roles = sorted(doc["cases"])
    return {
        "cases_in_specification": cases,
        "case_roles": roles,
        "forbidden_substitutions": len(doc["forbidden_substitutions"]),
        "invariants": len(doc["invariants"]),
        "gradient_targets": sorted(doc["gradient_targets"]),
        "mutants_injected": results["M5"].measurements.get("mutants_checked", 0),
        "mutants_rejected": results["M5"].measurements.get("mutants_caught", 0),
    }


__all__ = ["GATE_FUNCTIONS", "REPORT_VERSION", "gate_m5", "run"]
