"""One research campaign under one policy (issue #105, workstream D).

The policy comparison needs more than a score on held-back questions: it needs to know
whether the policies behave differently when they are allowed to do real research. This
module runs the pipeline end to end under a given policy's harness:

    propose -> freeze -> implement -> independent checks -> active probe and export
    sensitivity -> cold Stage-2 -> frozen Hard-L1 ER -> Stage-3 if eligible

Everything the campaign produced is measured, not narrated: the proposal's validity, the
judges' verdict, the probe's verdict, the export-sensitivity verdict, the frozen ER, the
downstream metrics, and the wall-clock and GPU time it consumed. A policy that admits an
inert arm and spends a cold run on it is charged for those GPU-hours; a policy that
rejects it before the cold run is not.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .evidence import EvidenceStore, RunPlan, now_utc, sha256_file
from .proposal import freeze, proposal_digest
from . import ideation

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

#: The base tree every campaign arm starts from: the frozen host.
BASE_TREE = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_r8_control"

#: The control the export sensitivity compares against.
CONTROL_SID = REPO_ROOT / "results" / "stage2_RQ-VAE" / "r8_control" / "sids_for_hgrec.npy"


def _write_hooks(tree: Path, mechanism_id: str, parameters: Dict[str, Any], weight: float) -> Path:
    """The two verifier hooks, plus the counted production call site."""
    hooks = tree / "mvg_hooks.py"
    hooks.write_text(
        f'''"""CGV instrumentation for this arm's mechanism.

Not part of the mathematics and not a judge: it exposes the tensors the mechanism reads, its
real assembly for the verifier to execute, and a call counter so the verifier can observe the
production loop executing it. The formula itself lives in ``mechanism_{mechanism_id}.py`` and
is not duplicated here.
"""

from __future__ import annotations

import torch

from mechanism_{mechanism_id} import mechanism_loss

MECHANISM_ENABLED = True
MECHANISM_WEIGHT = {weight!r}
MECHANISM_LEVEL = 0
MECHANISM_PARAMETERS = {parameters!r}
MECHANISM_CALLS = 0


def mvg_mechanism_term(*args, **kwargs):
    """The production call site: the mechanism's own function, with its frozen parameters."""
    global MECHANISM_CALLS
    MECHANISM_CALLS += 1
    merged = dict(MECHANISM_PARAMETERS)
    merged.update(kwargs or {{}})
    return mechanism_loss(*args, **merged)


def mvg_mechanism_inputs(observations: dict) -> dict:
    """The tensors the mechanism reads, taken from the verifier's observations."""
    model = observations["model"]
    residuals = observations["residuals"]
    pairs = int(observations["n_src"])
    device = residuals.device
    return {{
        "source": residuals[MECHANISM_LEVEL, :pairs],
        "target": residuals[MECHANISM_LEVEL, pairs : 2 * pairs],
        "codebook": model.rq.vq_layers[0].get_code_embs(),
        "mask": torch.ones(pairs, device=device, dtype=residuals.dtype),
    }}


def mvg_mechanism_executor(
    base_loss,
    l1_loss,
    other_loss,
    encoder,
    weight,
    valid_levels: int = 1,
    mechanism_inputs: dict | None = None,
) -> None:
    """The mechanism's real assembly, as the verifier executes it."""
    if mechanism_inputs is None:
        raise RuntimeError("the CGV executor needs the mechanism's inputs")
    inputs = dict(mechanism_inputs)
    source = inputs.pop("source")
    target = inputs.pop("target")
    codebook = inputs.pop("codebook")
    mask = inputs.pop("mask")
    term = mvg_mechanism_term(source, target, codebook, mask, **inputs)
    (MECHANISM_WEIGHT * term).backward()
''',
        encoding="utf-8",
    )
    trainer = tree / "train_rqvae.py"
    text = trainer.read_text(encoding="utf-8")
    text = re.sub(
        r"^_OUT = OUT_ROOT / EXPERIMENT_ARM$",
        "_OUT = OUT_ROOT / TREE_NAME / EXPERIMENT_ARM",
        text,
        count=1,
        flags=re.M,
    )
    if "from mvg_hooks import" not in text:
        # The implementer may import the mechanism under any alias, and the import may
        # span several lines. The statement is located, the name the call site uses is
        # taken from it, and both are rewritten so the production call goes through the
        # counted wrapper the verifier observes.
        lines = text.split("\n")
        index = None
        for position, line in enumerate(lines):
            if re.match(r"\s*from\s+mechanism_" + re.escape(mechanism_id) + r"\s+import\b", line):
                index = position
                break
        if index is None:
            raise RuntimeError(
                f"the tree does not import its mechanism module; the call site cannot be wired"
            )
        statement = lines[index]
        if "(" in statement and ")" not in statement:
            depth = statement.count("(") - statement.count(")")
            cursor = index + 1
            while cursor < len(lines) and depth > 0:
                statement += "\n" + lines[cursor]
                depth += lines[cursor].count("(") - lines[cursor].count(")")
                cursor += 1
            end = cursor
        else:
            end = index + 1
        names = statement.split("import", 1)[1]
        alias = "mechanism_loss"
        for part in names.replace("(", " ").replace(")", " ").replace(",", " ").split():
            if part == "as":
                continue
        match = re.search(r"mechanism_loss\s+as\s+(\w+)", statement)
        if match:
            alias = match.group(1)
        elif re.search(r"import\s+\(\s*mechanism_loss\s*\)", statement, re.S):
            alias = "mechanism_loss"
        lines[index:end] = [
            "from mvg_hooks import (mvg_mechanism_executor, mvg_mechanism_inputs,  # noqa: F401",
            "                       mvg_mechanism_term)",
        ]
        text = "\n".join(lines)
        text = re.sub(r"\b" + re.escape(alias) + r"\(", "mvg_mechanism_term(", text)
    trainer.write_text(text, encoding="utf-8")
    return hooks


def _write_spec(path: Path, mechanism_id: str, parameters: Dict[str, Any], weight: float) -> Path:
    template = json.loads(
        (REPO_ROOT / "tools/mvg/specs/behavior_conditioned_pairwise_curvature.json").read_text()
    )
    template["mechanism"]["id"] = mechanism_id
    template["mechanism"]["summary"] = (
        f"the campaign arm {mechanism_id}: {template['mechanism']['summary']}"
    )
    template["mechanism"]["mathematics"] = {"rule": "additive", "weight": weight, **parameters}
    path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return path


def _judge(tree: Path, proposal_path: Path, mechanism_id: str, out: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        [
            PROJECT_PYTHON,
            str(REPO_ROOT / "tools/mvg/invariant_check.py"),
            "--tree",
            str(tree),
            "--proposal",
            str(proposal_path),
            "--module",
            f"mechanism_{mechanism_id}",
            "--entrypoint",
            "mechanism_loss",
            "--out",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if out.is_file():
        return json.loads(out.read_text(encoding="utf-8"))
    return {"ok": False, "detail": f"the judge produced no report: {completed.stderr[-300:]}"}


def _probe(tree: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            PROJECT_PYTHON,
            str(REPO_ROOT / "tools/mvg/probe_entry.py"),
            "--tree",
            str(tree),
            "--epochs",
            "50",
            "--baseline-tree",
            str(REPO_ROOT / "stage2_RQ-VAE" / "TIGER_r8_control"),
            "--out",
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=3600,
    )
    report = out_dir / "probe_report.json"
    if report.is_file():
        return json.loads(report.read_text(encoding="utf-8"))
    return {"ok": False, "detail": completed.stderr[-400:]}


def _export_sensitivity(arm_sid: Path, control_sid: Path) -> Dict[str, Any]:
    if not arm_sid.is_file() or not control_sid.is_file():
        return {}
    completed = subprocess.run(
        [
            PROJECT_PYTHON,
            str(REPO_ROOT / "tools/mvg/export_sensitivity.py"),
            "--arm-sid",
            str(arm_sid),
            "--control-sid",
            str(control_sid),
        ],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr[-300:]}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"error": "the export-sensitivity output is not JSON"}


__all__ = ["_export_sensitivity", "_judge", "_probe", "_write_hooks", "_write_spec"]


def run_campaign(
    policy_workspace: Path,
    out_dir: Path,
    *,
    ledger_brief: str,
    constraint: str,
    store: Optional[EvidenceStore] = None,
    model: str = "",
    timeout: int = 2400,
    cold: bool = True,
    port: int = 50211,
) -> Dict[str, Any]:
    """One campaign under one policy, end to end, with every outcome measured."""
    from . import opencode, redesign

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record: Dict[str, Any] = {
        "policy_workspace": str(policy_workspace),
        "started_utc": now_utc(),
        "stages": {},
    }

    # 1. the proposal, under the policy
    proposal = redesign.propose_from_ledger(
        ledger_brief, constraint, out_dir / "proposal", model=model, timeout=timeout
    )
    frozen = proposal["proposal"]
    record["stages"]["propose"] = {
        "mechanism_id": frozen.get("mechanism_id"),
        "proposal_sha256": proposal["record"]["proposal_sha256"],
        "critique": proposal["record"]["critique"],
        "tokens": proposal["record"]["tokens"],
    }
    mechanism_id = frozen.get("mechanism_id")
    if not mechanism_id:
        record["status"] = "failed"
        record["detail"] = "the policy's proposal carries no mechanism id"
        (out_dir / "campaign.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return record

    # 2. the tree and the implementation, under the policy
    tree = REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_c1_policy_{mechanism_id}"
    if tree.is_dir():
        shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(BASE_TREE, tree)
    implementation = redesign.implement(frozen, tree, out_dir, model=model, timeout=timeout)
    record["stages"]["implement"] = {
        "tree": str(tree),
        "tokens": implementation.get("tokens"),
        "exit_code": implementation.get("exit_code"),
    }

    # 3. the independent checks, unchanged
    verification = _judge(
        tree, out_dir / "proposal" / "proposal.json", mechanism_id, out_dir / "verification.json"
    )
    record["stages"]["judges"] = {
        "ok": verification.get("ok"),
        "detail": verification.get("detail"),
        "failed": verification.get("failed"),
    }

    # 4. the active probe and the export sensitivity against the matched control
    probe = _probe(tree, out_dir / "probe")
    record["stages"]["probe"] = {
        "ok": probe.get("ok"),
        "detail": probe.get("detail"),
        "occupancy_ratio_vs_host": probe.get("occupancy_ratio_vs_host"),
    }
    probe_export = next(iter(out_dir.glob("probe/redirected/**/sids_for_hgrec.npy")), None)
    probe_control = next(iter(out_dir.glob("probe/baseline/redirected/**/sids_for_hgrec.npy")), None)
    sensitivity = (
        _export_sensitivity(probe_export, probe_control)
        if probe_export and probe_control
        else {}
    )
    record["stages"]["export_sensitivity"] = {
        "verdict": sensitivity.get("verdict"),
        "changed_fraction": sensitivity.get("changed_fraction"),
        "level0_occupancy_ratio_vs_control": sensitivity.get("level0_occupancy_ratio_vs_control"),
    }

    admitted = bool(probe.get("ok")) and sensitivity.get("verdict") not in ("collapsed",)
    record["admitted"] = admitted
    if not admitted:
        record["status"] = "refused"
        record["detail"] = (
            "the probe or the export sensitivity refused the arm before a cold run; the "
            "campaign spent no cold-run GPU time on it"
        )
        record["wall_clock_s"] = round(time.time() - started, 1)
        record["cold_run_gpu_minutes"] = 0.0
        (out_dir / "campaign.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return record

    # 5. the cold run, through the launch path with the six gates
    parameters = {str(k): float(v) for k, v in (frozen.get("parameters") or {}).items()
                  if isinstance(v, (int, float))}
    weight = float((frozen.get("activation") or {}).get("weight", 0.05))
    _write_hooks(tree, mechanism_id, parameters, weight)
    _write_spec(out_dir / "spec.json", mechanism_id, parameters, weight)
    trainer = tree / "train_rqvae.py"
    trainer.write_text(
        re.sub(r'"master_port":\s*\d+', f'"master_port": {port}', trainer.read_text(), count=1),
        encoding="utf-8",
    )
    if not cold:
        record["status"] = "admitted"
        record["wall_clock_s"] = round(time.time() - started, 1)
        record["cold_run_gpu_minutes"] = 0.0
        return record

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "genrec_mcp_server", REPO_ROOT / "tools" / "genrec-mcp" / "server.py"
    )
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    launch = server.tool_stage2_start(
        source_dir=str(tree),
        spec=json.loads((out_dir / "spec.json").read_text()),
    )
    record["stages"]["cold_launch"] = {
        "started": launch.get("started"),
        "refused": launch.get("refused"),
        "cgv_gate": launch.get("cgv_gate"),
        "result_dir": launch.get("result_dir"),
    }
    if not launch.get("started") and launch.get("status") not in ("smoke_pending", "running"):
        record["status"] = "failed"
        record["detail"] = f"the launch refused: {launch.get('refused')}"
        record["wall_clock_s"] = round(time.time() - started, 1)
        record["cold_run_gpu_minutes"] = 0.0
        (out_dir / "campaign.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return record
    record["status"] = "cold_running"
    record["wall_clock_s"] = round(time.time() - started, 1)
    record["cold_run_gpu_minutes"] = None
    record["result_dir"] = launch.get("result_dir")
    (out_dir / "campaign.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record
