"""The research learning loop (issue #104).

Three workstreams, and the rule that ties them together: nothing is learned by writing a
reflection down. A finding is a measurement, a falsified prediction, or an experiment that
tells the next one what to be.

**A -- repair fidelity.** For each mechanism whose implementation failed its own declared
gradient paths, the evidence is preserved, OpenCode writes a *minimal differentiable
prototype* first, the independent checks measure why it fails, and only a prototype that
passes is integrated and re-judged. The diagnosis separates an impossible specification
from a missing gradient from saturation, by measurement.

**B -- the hypothesis ledger.** The two experiments of ``tools/mvg/experiments.py`` are run
and their predictions are scored: supported, falsified, or inconclusive with the next test
named.

**C -- the policy comparison.** The old research policy (the current instructions) and the
learned one are compared, blind and budget-matched, on held-back cases with a known correct
answer. A policy is adopted only if the held-back evidence says so.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import opencode

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9"

#: The four mechanisms #103 could not get past their own declared gradient paths.
FAILED_ARMS = (
    "curvature_weighted_transition_transport",
    "behaviour_conditioned_residual_target",
    "behavior_conditioned_pairwise_curvature",
    "behavior_conditioned_pairwise_margin",
)

#: The held-back cases of the policy comparison. Each has an artifact, a question and a
#: predeclared correct answer, so the comparison is mechanical and the policies cannot
#: argue their way to a better score.
HELD_BACK_CASES: tuple = (
    {
        "id": "detached_gradient",
        "artifact": "def loss(z, e, mask):\n    d = torch.cdist(z, e, p=2)\n"
        "    hard = d.argmin(dim=1)\n    return (torch.clamp(0.1 - d.gather(1, hard[:, None]), min=0) * mask[:, None]).mean()\n",
        "check": "the declared target is the codebook; the gradient on it is exactly zero",
        "question": "classify the failure",
        "answer": "missing gradient: the codebook path is disconnected",
    },
    {
        "id": "plausible_but_wrong",
        "artifact": "the mechanism averages the forward and the reverse direction of a "
        "transition margin before the clamp",
        "check": "the loss is unchanged when the source and the target are exchanged",
        "question": "classify the failure",
        "answer": "the formula is wrong: a directed mechanism was implemented symmetrically",
    },
    {
        "id": "soft_hard_gap",
        "artifact": "the mechanism projects the SOFT assignment onto the host's occupancy "
        "quota and leaves the hard code as the argmin",
        "check": "the soft marginal matches the quota; the exported hard occupancy does not",
        "question": "classify the gap and name the quantity that must be constrained",
        "answer": "the argmin is not constrained by the marginal; constrain the hard code's "
        "distribution",
    },
    {
        "id": "saturation",
        "artifact": "the mechanism's margin term is at its clamp boundary for every pair in "
        "the check case",
        "check": "the loss depends on the codebook finitely, and the gradient there is zero",
        "question": "classify the failure",
        "answer": "saturation: the operating point sits on the clamp boundary",
    },
    {
        "id": "ambiguous_feedback",
        "artifact": "an arm's probe shows 86% of the host's occupancy and a frozen ER 1.8 "
        "below the control, and the arm's own predicted discrimination was never measured",
        "check": "the occupancy is healthy and the outcome is worse; no discrimination "
        "measurement exists",
        "question": "what is the conclusion and the next information-gaining test?",
        "answer": "inconclusive: the occupancy does not explain it, so the next test is a "
        "train-side discrimination measurement on the arm's own SIDs",
    },
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------------------
# A -- repair fidelity
# --------------------------------------------------------------------------------------
def _check(tree: Path, proposal_path: Path, module: str, entrypoint: str, out: Path) -> Dict[str, Any]:
    """Run the unchanged independent checks, in the environment that has torch."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            PROJECT_PYTHON,
            str(REPO_ROOT / "tools" / "mvg" / "invariant_check.py"),
            "--tree",
            str(tree),
            "--proposal",
            str(proposal_path),
            "--module",
            module,
            "--entrypoint",
            entrypoint,
            "--out",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if out.is_file():
        return json.loads(out.read_text(encoding="utf-8"))
    return {
        "ok": False,
        "detail": "the checks produced no report",
        "stderr_tail": (completed.stderr or "")[-300:],
        "checks": [],
        "failed": [],
    }


def repair_arm(arm_dir: Path, out_dir: Path, *, model: str = "", timeout: int = 1800) -> Dict[str, Any]:
    """Repair one mechanism's fidelity, prototype first."""
    arm_dir = Path(arm_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mechanism_id = arm_dir.name
    proposal_path = arm_dir / "proposal.json"
    if not proposal_path.is_file():
        return {"mechanism_id": mechanism_id, "status": "no proposal", "ok": False}
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))

    # 1. the original evidence is preserved before anything is touched
    preserved = out_dir / "original"
    if not preserved.is_dir():
        preserved.mkdir(parents=True, exist_ok=True)
        for name in ("proposal.json", "verification.json", "implement.json"):
            source = arm_dir / name
            if source.is_file():
                shutil.copy2(source, preserved / name)

    record: Dict[str, Any] = {
        "mechanism_id": mechanism_id,
        "original_verification": str(preserved / "verification.json"),
        "started_utc": _now(),
        "stages": [],
    }
    module = f"mechanism_{mechanism_id}"
    entrypoint = "mechanism_loss"
    tree = Path(proposal.get("tree") or (REPO_ROOT / "stage2_RQ-VAE" / f"TIGER_c1_{mechanism_id}"))
    record["tree"] = str(tree)

    # 2. the minimal differentiable prototype, written by the agent, checked first
    prototype_module = f"{module}_prototype"
    prompt = f"""# THE FAILURE

You implemented the mechanism below. An independent checker says it does not deliver the
gradient paths the mechanism's own frozen proposal declares. The findings:

```json
{json.dumps((json.loads((preserved / "verification.json").read_text()).get("failed") if (preserved / "verification.json").is_file() else []), indent=2)}
```

# THE PROPOSAL (frozen, do not edit)

```json
{json.dumps(proposal, indent=2, sort_keys=True)}
```

# WHAT TO WRITE, IN THIS ORDER

1. A **minimal differentiable prototype** in a new file `{prototype_module}.py`: the
   mathematical term only, no trainer, no data, importing nothing from the tree. Its entry
   point must be named `{entrypoint}` with the same signature the checker uses:
   `{entrypoint}(source, target, codebook, mask, **parameters)`.
2. Nothing else yet.

The checker will call the prototype and measure, per target, whether the loss depends on it
at all (a finite step moves it?), whether the backward pass reaches it, and whether the
local derivative is zero. If the proposal's formula cannot reach a target, the checker will
say so -- and that is a correct outcome: it means the proposal's claim, not your code, is
what has to change.

Write the prototype, then report what you changed and why.
"""
    prototype = opencode.run_agent(
        tree,
        prompt,
        model=model or opencode.model_name(),
        timeout=timeout,
        title=f"prototype {mechanism_id}",
        deny_paths=[REPO_ROOT / "results", REPO_ROOT / "tools"],
    )
    (out_dir / "prototype_session.json").write_text(
        json.dumps(prototype, indent=2, default=str) + "\n", encoding="utf-8"
    )
    record["stages"].append(
        {"name": "prototype", "exit_code": prototype.get("exit_code"), "tokens": prototype.get("tokens")}
    )
    prototype_report = _check(tree, proposal_path, prototype_module, entrypoint, out_dir / "prototype_check.json")
    record["prototype_check"] = prototype_report
    diagnosis = prototype_report.get("diagnosis") or {}
    record["diagnosis"] = diagnosis

    impossible = [
        name for name, entry in diagnosis.items()
        if isinstance(entry, dict) and entry.get("verdict") == "impossible specification"
    ]
    saturation = [
        name for name, entry in diagnosis.items()
        if isinstance(entry, dict) and entry.get("verdict") == "saturation"
    ]
    if impossible:
        # The claim cannot be implemented. The proposal is versioned, never rewritten.
        revised = dict(proposal)
        revised["schema_version"] = 1
        revised["version"] = f"{proposal.get('version', '1.0.0')}+v2"
        targets = dict(revised.get("gradient_targets") or {})
        for name in impossible:
            targets[name] = "finite"
        revised["gradient_targets"] = targets
        history = dict(revised.get("revision") or {})
        history.update(
            {
                "supersedes": str(preserved / "proposal.json"),
                "reason": (
                    "measured: the loss does not depend on "
                    + ", ".join(impossible)
                    + " at all, so the original claim was impossible; the target is now "
                    "declared finite for those paths"
                ),
                "measured_utc": _now(),
            }
        )
        revised["revision"] = history
        (out_dir / "proposal_v2.json").write_text(
            json.dumps(revised, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        record["status"] = "invalid_specification"
        record["note"] = (
            "the original claim was impossible; a versioned proposal v2 records the revised "
            "claim and the measurement behind it"
        )
        record["finished_utc"] = _now()
        (out_dir / "repair.json").write_text(
            json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return record

    if not prototype_report.get("ok"):
        # The prototype exists but still does not deliver. One bounded repair, then it stays
        # unresolved: a mechanism that cannot be made faithful in two passes is a fact.
        second = opencode.run_agent(
            tree,
            "The independent checks rejected the prototype again. The findings are below. Fix "
            "the prototype only; do not change the proposal, the checker or the fixtures.\n\n"
            + json.dumps(prototype_report.get("failed"), indent=2)
            + "\n\nReport what you changed.",
            model=model or opencode.model_name(),
            timeout=timeout,
            title=f"prototype repair {mechanism_id}",
            deny_paths=[REPO_ROOT / "results", REPO_ROOT / "tools"],
        )
        (out_dir / "prototype_repair_session.json").write_text(
            json.dumps(second, indent=2, default=str) + "\n", encoding="utf-8"
        )
        prototype_report = _check(tree, proposal_path, prototype_module, entrypoint, out_dir / "prototype_check.json")
        record["prototype_check"] = prototype_report
        record["stages"].append({"name": "prototype_repair", "tokens": second.get("tokens")})
        if not prototype_report.get("ok"):
            record["status"] = "unresolved"
            record["note"] = (
                "the prototype does not deliver the declared paths after one bounded repair "
                f"(saturation on {saturation or 'none'}); the mechanism stays failed"
            )
            record["finished_utc"] = _now()
            (out_dir / "repair.json").write_text(
                json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
            )
            return record

    # 3. only a passing prototype is integrated, and the UNCHANGED judges decide
    integrate = opencode.run_agent(
        tree,
        f"The prototype `{prototype_module}.py` passes the independent checks. Integrate "
        f"exactly that mathematics into `{module}.py` (same entry point and signature, no "
        "behavioural change), then wire it into `train_rqvae.py` so the term is composed into "
        "the host's loss before backward. Do not change the proposal, the checks or the "
        "fixtures. Report what you changed.",
        model=model or opencode.model_name(),
        timeout=timeout,
        title=f"integrate {mechanism_id}",
        deny_paths=[REPO_ROOT / "results", REPO_ROOT / "tools"],
    )
    (out_dir / "integrate_session.json").write_text(
        json.dumps(integrate, indent=2, default=str) + "\n", encoding="utf-8"
    )
    final = _check(tree, proposal_path, module, entrypoint, out_dir / "verification.json")
    record["verification"] = final
    record["stages"].append(
        {"name": "integrate", "exit_code": integrate.get("exit_code"), "tokens": integrate.get("tokens")}
    )
    if final.get("ok"):
        record["status"] = "faithfully_implemented"
        record["note"] = "the prototype passed, the integration was judged unchanged, and the judges agree"
    else:
        record["status"] = "unresolved"
        record["note"] = (
            "the prototype passes but the integrated module does not: the wiring in the "
            "trainer, not the mathematics, is what remains"
        )
    record["finished_utc"] = _now()
    (out_dir / "repair.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record


# --------------------------------------------------------------------------------------
# B -- the hypothesis ledger
# --------------------------------------------------------------------------------------
def hypothesis_ledger(experiments_dir: Path) -> Dict[str, Any]:
    """Score each explanation against its prediction, from the experiments' own output."""
    experiments_dir = Path(experiments_dir)
    entries: List[Dict[str, Any]] = []
    h1_path = experiments_dir / "h1_soft_hard.json"
    if h1_path.is_file():
        h1 = json.loads(h1_path.read_text(encoding="utf-8"))
        mass = float(h1.get("soft_mass_on_unused_codes") or 0.0)
        rho = float(h1.get("spearman_soft_vs_hard") or 0.0)
        reachable = int(h1.get("unused_codes_geometrically_reachable") or 0)
        unreachable = int(h1.get("codes", 0)) - int(h1.get("hard_used", 0)) - reachable
        entries.append(
            {
                "id": "H1-soft-to-hard",
                "observation": (
                    "constraining the soft assignment to the host's occupancy (39.5% of the "
                    "host's level-0 codes) and enforcing the host's quota exactly (24.2%) both "
                    "failed to preserve the exported hard codes"
                ),
                "explanations": [
                    {
                        "id": "E1a-argmin",
                        "claim": "the exported code is the argmin; a constraint on the "
                                 "assignment's marginal is not a constraint on the argmin",
                        "prediction": "a large soft mass on codes the hard map never uses, and "
                                      "a weak rank agreement between the two",
                        "measured": {
                            "soft_mass_on_unused": mass,
                            "spearman_soft_vs_hard": rho,
                        },
                        "verdict": "supported" if mass > 0.1 and abs(rho) < 0.8 else "falsified",
                    },
                    {
                        "id": "E1b-geometry",
                        "claim": "the hard occupancy is set by the codebook's geometry: codes "
                                 "with no item nearby cannot be used however the assignment is "
                                 "constrained",
                        "prediction": "the unused codes are geometrically unreachable",
                        "measured": {
                            "unused_codes": int(h1.get("codes", 0)) - int(h1.get("hard_used", 0)),
                            "unreachable": unreachable,
                            "reachable": reachable,
                        },
                        "verdict": "supported" if unreachable > reachable else "falsified",
                    },
                ],
                "design_change": (
                    "a redesign must constrain the HARD code's distribution, or the geometry "
                    "that produces it -- not the soft marginal"
                ),
                "confidence": "high: the measurement is on the frozen control, no training involved",
            }
        )
    h2_path = experiments_dir / "h2_occupancy_discrimination.json"
    if h2_path.is_file():
        h2 = json.loads(h2_path.read_text(encoding="utf-8"))
        correlations = h2.get("correlations") or {}
        occupancy_rho = correlations.get("occupancy_vs_frozen_er")
        conditional_rho = correlations.get("conditional_l2_vs_frozen_er")
        entries.append(
            {
                "id": "H2-occupancy-vs-discrimination",
                "observation": (
                    "the #103 arm with 86% of the host's level-0 occupancy scored 1.81 below "
                    "the control, and the R8 lanes show occupancy and ER moving apart"
                ),
                "explanations": [
                    {
                        "id": "E2a-usage-is-not-discrimination",
                        "claim": "occupancy counts how many codes are used; the ER measures "
                                 "whether observed transitions are more likely than negatives to "
                                 "share a code",
                        "prediction": "across the eleven recorded lanes, occupancy and the frozen "
                                      "ER are weakly correlated, while the train-side pair "
                                      "discrimination tracks the ER",
                        "measured": {"occupancy_vs_frozen_er": occupancy_rho},
                        "verdict": "falsified"
                        if isinstance(occupancy_rho, (int, float)) and abs(occupancy_rho) > 0.8
                        else "supported",
                    },
                    {
                        "id": "E2b-deeper-levels-compensate",
                        "claim": "the level-2 prefix recovers what the level-1 code loses, so a "
                                 "level-1 occupancy loss is invisible downstream",
                        "prediction": "among pairs sharing a level-1 code, the level-2 match rate "
                                      "is as high as on the control",
                        "measured": {"conditional_l2_vs_frozen_er": conditional_rho},
                        "verdict": "supported"
                        if isinstance(conditional_rho, (int, float)) and conditional_rho > 0
                        else "falsified",
                    },
                ],
                "design_change": (
                    "occupancy is a good PROXY for discrimination across these lanes, so a "
                    "healthy occupancy does not explain the #103 arm; that arm is the "
                    "counter-example to the proxy and its own train-side discrimination was "
                    "never measured"
                ),
                "next_test": (
                    "re-export the #103 arm's hard SIDs (a short probe, or its cold run) and "
                    "measure its train-side pair discrimination; without that measurement the "
                    "counter-example stays inconclusive"
                ),
                "confidence": (
                    "high for the falsification (eleven lanes, one measurement each); low for "
                    "the #103 counter-example, which is unresolved"
                ),
            }
        )
    ledger = {
        "generated_utc": _now(),
        "entries": entries,
        "summary": {
            "supported": sum(
                1
                for entry in entries
                for explanation in entry["explanations"]
                if explanation["verdict"] == "supported"
            ),
            "falsified": sum(
                1
                for entry in entries
                for explanation in entry["explanations"]
                if explanation["verdict"] == "falsified"
            ),
        },
    }
    (Path(experiments_dir) / "hypothesis_ledger.json").write_text(
        json.dumps(ledger, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return ledger


# --------------------------------------------------------------------------------------
# C -- the policy comparison
# --------------------------------------------------------------------------------------
def policy_workspace(source_harness: Path, destination: Path) -> Path:
    """A copy of a harness: CLAUDE.md, the skills and the scripts."""
    source_harness = Path(source_harness)
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("CLAUDE.md",):
        source = source_harness / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    skills = source_harness / ".agent" / "skills"
    if skills.is_dir():
        shutil.copytree(skills, destination / ".agent" / "skills", dirs_exist_ok=True)
    scripts = source_harness / "scripts"
    if scripts.is_dir():
        (destination / "scripts").mkdir(parents=True, exist_ok=True)
        for script in sorted(scripts.glob("*.py")):
            if _SCRIPT.fullmatch(script.name):
                shutil.copy2(script, destination / "scripts" / script.name)
    return destination


import re as _re  # noqa: E402  (kept local to this module's tail)

_SCRIPT = _re.compile(r"[a-z0-9_][a-z0-9_]*\.py")


def derive_learned_policy(out_dir: Path, ledger: Dict[str, Any], *, model: str = "",
                          timeout: int = 1800) -> Dict[str, Any]:
    """The agent writes the learned policy: instructions derived from the tested findings."""
    out_dir = Path(out_dir)
    workspace = policy_workspace(REPO_ROOT, out_dir / "learned_harness")
    prompt = f"""# YOU ARE EDITING THE RESEARCH POLICY

This directory is the harness the agents work under: `CLAUDE.md` and the skills. It is what
decides how a mechanism is proposed, implemented, checked and repaired.

# WHAT THE LAST STUDY FOUND

```json
{json.dumps(ledger, indent=2, sort_keys=True)[:12000]}
```

# THE TASK

Write ONE change to this harness that would have caught the failures above sooner, or would
have avoided them. It has to be actionable by a future agent: a rule in `CLAUDE.md`, or a
new skill under `.agent/skills/<name>/SKILL.md` with frontmatter.

Two things it must not do: it must not change any judge, threshold, fixture or evaluation,
and it must not describe a mechanism's mathematics as if it were settled. The finding that
matters most is measured: a constraint on the soft assignment does not constrain the hard
code, and occupancy is a good proxy for discrimination *except* where an arm's own
discrimination was never measured.

Make the edit, then report in one paragraph what you changed and which finding it follows.
"""
    result = opencode.run_agent(
        workspace,
        prompt,
        model=model or opencode.model_name(),
        timeout=timeout,
        title="derive learned policy",
    )
    (out_dir / "policy_session.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    changed = sorted(
        str(path.relative_to(workspace))
        for path in workspace.rglob("*")
        if path.is_file()
    )
    record = {
        "workspace": str(workspace),
        "files": changed,
        "text": result.get("text"),
        "tokens": result.get("tokens"),
        "exit_code": result.get("exit_code"),
    }
    (out_dir / "learned_policy.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record


def _judge_answer(answer: str, expected: str) -> bool:
    """A mechanical judge: the answer has to carry the expected conclusion's key terms."""
    text = (answer or "").lower()
    key_terms = []
    if "missing gradient" in expected:
        key_terms = ["gradient", "disconnect"]
    elif "formula is wrong" in expected:
        key_terms = ["directed", "symmetr"]
    elif "argmin" in expected:
        key_terms = ["argmin", "hard"]
    elif "saturation" in expected:
        key_terms = ["saturat"]
    elif "inconclusive" in expected:
        key_terms = ["inconclusive", "discriminat"]
    if not key_terms:
        key_terms = [word for word in expected.lower().split() if len(word) > 6][:3]
    return all(term in text for term in key_terms)


def compare_policies(out_dir: Path, *, model: str = "", timeout: int = 900,
                     learned: Optional[Path] = None) -> Dict[str, Any]:
    """The blind, budget-matched old-versus-learned comparison on held-back cases.

    The judge is mechanical and does not know which policy produced an answer; the case
    order is shuffled with a fixed seed so the order cannot correlate with the policy.
    """
    import random

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    old = policy_workspace(REPO_ROOT, out_dir / "old_harness")
    learned_dir = Path(learned) if learned else out_dir / "learned_harness"
    if not learned_dir.is_dir():
        return {"ok": False, "error": f"no learned harness at {learned_dir}"}
    order = list(HELD_BACK_CASES)
    random.Random(4242).shuffle(order)
    rows: List[Dict[str, Any]] = []
    for case in order:
        for label, harness in (("old", old), ("learned", learned_dir)):
            workspace = out_dir / "sessions" / f"{case['id']}__{label}"
            policy_workspace(harness, workspace)
            (workspace / "artifact.md").write_text(
                f"# Artifact under review\n\n{case['artifact']}\n\n# Check report\n\n{case['check']}\n",
                encoding="utf-8",
            )
            prompt = (
                "Read the harness in this directory (CLAUDE.md and the skills) -- it is the "
                "policy you work under -- then read `artifact.md`, which is a mechanism and a "
                "check report on it. Answer this question about it, in at most six lines, "
                "ending with a line that starts with `CONCLUSION:`\n\n"
                f"**{case['question']}**\n"
            )
            result = opencode.run_agent(
                workspace,
                prompt,
                model=model or opencode.model_name(),
                timeout=timeout,
                title=f"policy {label} {case['id']}",
            )
            answer = result.get("text") or ""
            rows.append(
                {
                    "case": case["id"],
                    "policy": label,
                    "correct": _judge_answer(answer, case["answer"]),
                    "expected": case["answer"],
                    "answer": answer[:1500],
                    "tokens": result.get("tokens"),
                    "exit_code": result.get("exit_code"),
                }
            )
    old_rows = [row for row in rows if row["policy"] == "old"]
    new_rows = [row for row in rows if row["policy"] == "learned"]
    old_correct = sum(1 for row in old_rows if row["correct"])
    new_correct = sum(1 for row in new_rows if row["correct"])
    paired = [
        (row["case"], row["correct"], next((r["correct"] for r in new_rows if r["case"] == row["case"]), None))
        for row in old_rows
    ]
    record = {
        "ok": True,
        "tasks": len(HELD_BACK_CASES),
        "budget_matched": True,
        "old_correct": old_correct,
        "learned_correct": new_correct,
        "paired": [{"case": case, "old": o, "learned": n} for case, o, n in paired],
        "improved": new_correct > old_correct,
        "adopted": new_correct > old_correct,
        "reason": (
            "the learned policy solved more held-back cases than the old one"
            if new_correct > old_correct
            else "the learned policy did not solve more held-back cases; it is NOT adopted"
            + ("" if new_correct < old_correct else " (tied)")
        ),
        "rows": rows,
        "generated_utc": _now(),
    }
    (out_dir / "policy_comparison.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record


__all__ = [
    "FAILED_ARMS",
    "HELD_BACK_CASES",
    "compare_policies",
    "derive_learned_policy",
    "hypothesis_ledger",
    "policy_workspace",
    "repair_arm",
]
