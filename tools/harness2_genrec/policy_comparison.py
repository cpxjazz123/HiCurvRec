"""The blind, budget-matched old-versus-learned policy comparison (issue #105, workstream D).

Two policies, one task set, one model, one budget, one evidence cutoff:

* the **old** policy is the repository's current harness -- `CLAUDE.md` and the skills -- as
  it stands;
* the **learned** policy is the same harness plus an instruction set that the agent
  derives from the ledger's supported findings.

The task set is frozen and hashed BEFORE either policy sees it, the graders are programs
rather than string matching, the case order is shuffled with a fixed seed, and the grader
does not know which policy produced an answer. A policy is adopted only if it solves
strictly more held-back tasks than the baseline; a tie is a failure to adopt, and it is
reported as one.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .evidence import canonical_json, now_utc, sha256_bytes
from .policy_tasks import Task, _answer

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The predeclared adoption criterion. It is fixed before the comparison runs.
ADOPTION_CRITERION = (
    "the learned policy must solve strictly more held-back tasks than the old policy "
    "(8 tasks, one point each); a tie or a loss is a failure to adopt"
)


@dataclass
class Policy:
    """One policy: a workspace with the instructions the agent works under."""

    name: str
    workspace: Path
    digest: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def materialise_policy(name: str, destination: Path, *, extra_instructions: str = "") -> Policy:
    """A copy of the repository's harness: CLAUDE.md, the skills and the scripts."""
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)
    source_claude = REPO_ROOT / "CLAUDE.md"
    if source_claude.is_file():
        shutil.copy2(source_claude, destination / "CLAUDE.md")
    skills = REPO_ROOT / ".agent" / "skills"
    if skills.is_dir():
        shutil.copytree(skills, destination / ".agent" / "skills", dirs_exist_ok=True)
    if extra_instructions:
        # The learned policy's extra rules live in the same file the agent reads, so the
        # two policies differ by exactly the instructions and by nothing else.
        with (destination / "CLAUDE.md").open("a", encoding="utf-8") as handle:
            handle.write("\n" + extra_instructions)
    digest = sha256_bytes(canonical_json(
        {
            "claude_md": sha256_bytes((destination / "CLAUDE.md").read_bytes())
            if (destination / "CLAUDE.md").is_file()
            else None,
            "skills": sorted(
                str(path.relative_to(destination))
                for path in (destination / ".agent" / "skills").rglob("*")
                if path.is_file()
            ),
        }
    ).encode("utf-8"))
    return Policy(name=name, workspace=destination, digest=digest)


def derive_learned_policy(
    ledger_brief: str, out_dir: Path, *, model: str = "", timeout: int = 1800
) -> Dict[str, Any]:
    """The agent writes the learned policy: rules that follow the ledger's findings."""
    from . import opencode

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The agent edits a COPY of the harness: the baseline must stay untouched, and the
    # learned policy must differ from it by exactly the agent's edit.
    materialise_policy("learned", out_dir / "learned_harness")
    prompt = f"""# YOU ARE EDITING THE RESEARCH POLICY

The harness the research agents work under is `CLAUDE.md` and the skills, in THIS
directory. It is what decides how a mechanism is proposed, implemented, checked and
admitted. Edit the copy in this directory; do not edit anything outside it.

# WHAT THE LEDGER ESTABLISHED

{ledger_brief}

# THE CONSTRAINT THE HOST IMPOSES (measured)

The host's training loop overwrites the ENCODER gradient after the backward with
g_base + w_b/valid_levels * (g'_L1 + g_L2 + g_L3), and never touches the decoder or the
codebook gradient. On the host's own mini case an auxiliary term changed the encoder
gradient by 1.66e-05 before that assembly and by exactly 0.0 after it. Two arms whose
mechanisms reached only the encoder were therefore bit-identical to the control in their
trained weights and in their exported SIDs.

# THE TASK

Write ONE addition to the harness: a section of `CLAUDE.md` (or a new skill under
`.agent/skills/<name>/SKILL.md`) that a future agent will follow. It must be actionable
and it must follow from the findings above.

Three things it must not do: it must not change any judge, threshold, fixture or
evaluation; it must not describe a mechanism's mathematics as if it were settled; and it
must not restate the project's objective as the Hard-L1 ER, because twice an above-gate ER
came with no downstream gain and once with a clear downstream regression.

Make the edit, then report in one paragraph what you added and which finding it follows.
"""
    result = opencode.run_agent(
        out_dir,
        prompt,
        model=model or opencode.model_name(),
        timeout=timeout,
        title="derive the learned policy",
    )
    (out_dir / "policy_session.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    # The agent edited the copy of the harness in its workspace; that edited copy IS the
    # learned policy, and the baseline is the untouched repository. Nothing else differs
    # between the two policies: the same model, the same tools, the same task artifacts.
    learned_workspace = out_dir / "learned_harness"
    digest = sha256_bytes(canonical_json(
        {
            "claude_md": sha256_bytes((learned_workspace / "CLAUDE.md").read_bytes())
            if (learned_workspace / "CLAUDE.md").is_file()
            else None,
            "skills": sorted(
                str(path.relative_to(learned_workspace))
                for path in (learned_workspace / ".agent" / "skills").rglob("*")
                if path.is_file()
            ),
        }
    ).encode("utf-8"))
    learned = Policy(name="learned", workspace=learned_workspace, digest=digest)
    record = {
        "text": result.get("text"),
        "tokens": result.get("tokens"),
        "exit_code": result.get("exit_code"),
        "learned_workspace": str(learned.workspace),
        "learned_digest": learned.digest,
    }
    (out_dir / "learned_policy.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record


def run_comparison(
    tasks: List[Task],
    policies: List[Policy],
    out_dir: Path,
    *,
    model: str = "",
    timeout: int = 900,
    seed: int = 4242,
) -> Dict[str, Any]:
    """The blind comparison: every task, every policy, the same model and the same budget."""
    from . import guardrails, opencode

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for policy in policies:
        # An archived policy is evidence, not something to run again (issue #106).
        verdict = guardrails.check_intent(
            guardrails.ExecutionIntent(kind="policy", policy=policy.name)
        )
        if not verdict.allowed:
            raise guardrails.GuardrailRefusal(verdict.rule, verdict.detail)
    order = list(tasks)
    random.Random(seed).shuffle(order)
    rows: List[Dict[str, Any]] = []
    for task in order:
        for policy in policies:
            workspace = out_dir / "sessions" / f"{task.id}__{policy.name}"
            if workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)
            workspace.mkdir(parents=True, exist_ok=True)
            shutil.copy2(policy.workspace / "CLAUDE.md", workspace / "CLAUDE.md")
            skills = policy.workspace / ".agent" / "skills"
            if skills.is_dir():
                shutil.copytree(skills, workspace / ".agent" / "skills", dirs_exist_ok=True)
            prompt = (
                "Read the harness in this directory (CLAUDE.md and the skills): it is the "
                "policy you work under. Then answer the question below.\n\n"
                + task.prompt
                + json.dumps(task.artifact, indent=2, default=str)
            )
            result = opencode.run_agent(
                workspace,
                prompt,
                model=model or opencode.model_name(),
                timeout=timeout,
                title=f"policy {policy.name} {task.id}",
            )
            answer = _answer(result.get("text") or "")
            ok, reason = task.grader(answer, task.artifact)
            rows.append(
                {
                    "task": task.id,
                    "policy": policy.name,
                    "correct": bool(ok),
                    "reason": reason,
                    "answer": answer,
                    "tokens": result.get("tokens"),
                    "exit_code": result.get("exit_code"),
                }
            )
            print(f"  {policy.name:<8} {task.id:<26} {'correct' if ok else 'WRONG'}  {reason[:70]}")
    scores: Dict[str, int] = {}
    for policy in policies:
        scores[policy.name] = sum(1 for row in rows if row["policy"] == policy.name and row["correct"])
    baseline = policies[0].name if policies else ""
    candidate = policies[1].name if len(policies) > 1 else ""
    old_score = scores.get(baseline, 0)
    new_score = scores.get(candidate, 0)
    record = {
        "ok": True,
        "generated_utc": now_utc(),
        "tasks": len(tasks),
        "budget_matched": True,
        "model": model or opencode.model_name(),
        "timeout_s": timeout,
        "seed": seed,
        "adoption_criterion": ADOPTION_CRITERION,
        "scores": scores,
        "old_correct": old_score,
        "learned_correct": new_score,
        "adopted": bool(new_score > old_score),
        "reason": (
            "the learned policy solved strictly more held-back tasks than the old one"
            if new_score > old_score
            else "the learned policy did not solve strictly more held-back tasks; it is NOT "
                 "adopted"
        ),
        "rows": rows,
    }
    (out_dir / "policy_comparison.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record


__all__ = [
    "ADOPTION_CRITERION",
    "Policy",
    "derive_learned_policy",
    "materialise_policy",
    "run_comparison",
]
