"""The four research roles, bound to real OpenCode sessions.

The campaign's record (issue #113's audit) is explicit that the roles named L, A, M and I had
**no** session evidence: their artifacts were committed but nothing recorded a process. This
module is the binding that makes a role a session: each role has

* a fixed role definition and a fixed output contract,
* a prompt built from FROZEN inputs, whose hash goes into the execution record,
* a verifier that decides whether the agent's output counts,
* its own workspace and its own bounded resource limits.

The prompts are declared here but nothing in this module runs an agent: the worker does that,
and only when the operator has authorised it. The role text is the harness's own wording about
how to work, not a mechanism, not a hypothesis and not a literature query.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .executor import (ROLE_ANALYSIS, ROLE_IMPLEMENTATION, ROLE_LITERATURE, ROLE_PROPOSAL,
                       AgentSpec, ExecutionRecord, OpenCodeAgentExecutor)

__all__ = ["RoleBinding", "BINDINGS", "build_spec", "role_limits_for"]


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RoleBinding:
    """How one role works: its output contract, its prompt and its verifier."""

    role: str
    title: str
    output_name: str
    prompt_template: str
    verifier: Callable[[Path], Dict[str, Any]]
    max_seconds: int = 1800
    timeout: int = 900

    def build(self, task_id: str, *, model: str, inputs: Dict[str, Any],
              files: Tuple[Tuple[str, Path], ...] = ()) -> AgentSpec:
        """The immutable spec for one run of this role."""
        rendered = self.prompt_template.format(
            task_id=task_id,
            output=self.output_name,
            inputs=json.dumps(inputs, indent=2, sort_keys=True, default=str),
        )
        return AgentSpec(
            role=self.role,
            task_id=task_id,
            prompt=rendered,
            model=model,
            timeout=self.timeout,
            inputs_hash=_digest(inputs),
            expected_outputs=(self.output_name,),
            verifier=self.verifier,
            max_seconds=self.max_seconds,
            input_files=files,
        )


# -- the verifiers ---------------------------------------------------------
def _require_json(name: str, required: Tuple[str, ...] = ()) -> Callable[[Path], Dict[str, Any]]:
    """The verifier: the file must exist, parse, and carry the fields the role promised."""

    def verify(workspace: Path) -> Dict[str, Any]:
        path = workspace / name
        if not path.is_file():
            return {"ok": False, "error": f"the agent did not write {name}"}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"{name} is not valid JSON: {exc}"}
        missing = [field for field in required if field not in payload]
        if missing:
            return {"ok": False, "error": f"{name} is missing: " + ", ".join(missing)}
        return {"ok": True, "file": name, "bytes": path.stat().st_size}

    return verify


def _require_text(name: str, minimum: int = 200) -> Callable[[Path], Dict[str, Any]]:
    def verify(workspace: Path) -> Dict[str, Any]:
        path = workspace / name
        if not path.is_file():
            return {"ok": False, "error": f"the agent did not write {name}"}
        body = path.read_text(encoding="utf-8")
        if len(body.strip()) < minimum:
            return {"ok": False, "error": f"{name} is too short to be a report"}
        return {"ok": True, "file": name, "bytes": path.stat().st_size}

    return verify


# -- the four roles --------------------------------------------------------
BINDINGS: Dict[str, RoleBinding] = {
    ROLE_LITERATURE: RoleBinding(
        role=ROLE_LITERATURE,
        title="verified literature",
        output_name="literature_matrix.json",
        prompt_template=(
            "# ROLE\n"
            "You are the literature provider of an autonomous research system.\n\n"
            "# TASK\n"
            "Read the frozen search log supplied with this task and produce a literature matrix\n"
            "for the mechanism question it names.\n\n"
            "Every row must carry a verifiable identifier (DOI or arXiv id), the title, the year and\n"
            "what the work claims. A work you cannot verify is not a row; record it under\n"
            "\"unverified\" with the reason.\n\n"
            "# OUTPUT CONTRACT\n"
            "Write `{output}` as JSON with the keys \"question\", \"rows\" and \"unverified\".\n"
            "Do not invent a mechanism and do not recommend one: your job is the record of what\n"
            "exists, not a proposal.\n\n"
            "# TASK ID\n"
            "{task_id}\n\n"
            "# FROZEN INPUTS\n"
            "{inputs}\n"
        ),
        verifier=_require_json("literature_matrix.json", ("question", "rows")),
        max_seconds=1800,
        timeout=900,
    ),
    ROLE_ANALYSIS: RoleBinding(
        role=ROLE_ANALYSIS,
        title="retrospective analysis",
        output_name="failure_analysis.md",
        prompt_template=(
            "# ROLE\n"
            "You are the retrospective analyst of an autonomous research system.\n\n"
            "# TASK\n"
            "Read the frozen measurements supplied with this task and explain what they show about\n"
            "the campaign so far: which claims the evidence supports, which it contradicts, and\n"
            "which measurement would discriminate between the remaining explanations.\n\n"
            "Every number you cite must be one of the supplied numbers. A conclusion the\n"
            "measurements do not carry is marked as a hypothesis, not as a finding.\n\n"
            "# OUTPUT CONTRACT\n"
            "Write `{output}` as a report with the sections \"what the evidence shows\",\n"
            "\"what it contradicts\" and \"the discriminating measurement\".\n\n"
            "# TASK ID\n"
            "{task_id}\n\n"
            "# FROZEN INPUTS\n"
            "{inputs}\n"
        ),
        verifier=_require_text("failure_analysis.md"),
        max_seconds=1800,
        timeout=900,
    ),
    ROLE_PROPOSAL: RoleBinding(
        role=ROLE_PROPOSAL,
        title="novel math proposal",
        output_name="proposal.json",
        prompt_template=(
            "# ROLE\n"
            "You are the mechanism proposer of an autonomous research system.\n\n"
            "# TASK\n"
            "Read the frozen brief supplied with this task and propose ONE mechanism as a\n"
            "mathematical declaration: the operator, its domain, its parameters and the claim it\n"
            "makes about the quantity it is supposed to change.\n\n"
            "The proposal is a declaration, not an implementation: an implementation contract and a\n"
            "seven-group CGV declaration follow, and only then may anything be built.\n\n"
            "# OUTPUT CONTRACT\n"
            "Write `{output}` as JSON with the keys \"mechanism_id\", \"family\", \"title\",\n"
            "\"equation\", \"symbols\", \"parameters\", \"hypotheses\", \"falsifier\" and\n"
            "\"expected_consequences\".\n\n"
            "# TASK ID\n"
            "{task_id}\n\n"
            "# FROZEN INPUTS\n"
            "{inputs}\n"
        ),
        verifier=_require_json("proposal.json",
                                ("mechanism_id", "equation", "hypotheses", "falsifier")),
        max_seconds=2400,
        timeout=1200,
    ),
    ROLE_IMPLEMENTATION: RoleBinding(
        role=ROLE_IMPLEMENTATION,
        title="precise frozen implementation",
        output_name="implementation_contract.json",
        prompt_template=(
            "# ROLE\n"
            "You are the implementation contract writer of an autonomous research system.\n\n"
            "# TASK\n"
            "Read the frozen proposal supplied with this task and write the contract an\n"
            "implementer must satisfy: the exact entry point, the parameters, the gradients that\n"
            "must reach which parameters, and the observable properties the code must have.\n\n"
            "The contract is frozen before any code is written, and an implementation that does\n"
            "not satisfy it is refused. Write only what can be checked mechanically.\n\n"
            "# OUTPUT CONTRACT\n"
            "Write `{output}` as JSON with the keys \"mechanism_id\", \"entrypoint\", \"parameters\",\n"
            "\"gradient_targets\", \"observables\" and \"forbidden_substitutions\".\n\n"
            "# TASK ID\n"
            "{task_id}\n\n"
            "# FROZEN INPUTS\n"
            "{inputs}\n"
        ),
        verifier=_require_json("implementation_contract.json",
                                ("mechanism_id", "entrypoint", "gradient_targets")),
        max_seconds=2400,
        timeout=1200,
    ),
}


def build_spec(role: str, task_id: str, *, model: str = "", inputs: Optional[Dict[str, Any]] = None,
               files: Tuple[Tuple[str, Path], ...] = ()) -> AgentSpec:
    """The spec for one role's run. The prompt is frozen by the inputs' hash."""
    if role not in BINDINGS:
        raise KeyError(f"unknown role {role!r}; the roles are {sorted(BINDINGS)}")
    binding = BINDINGS[role]
    return binding.build(task_id, model=model, inputs=inputs or {}, files=files)


def role_limits_for(concurrency: int = 4) -> Dict[str, int]:
    """How many sessions of each role may run at once.

    L, A and M are independent of each other and may overlap; I reads what M produced and is
    therefore limited to one at a time.
    """
    return {ROLE_LITERATURE: max(1, concurrency // 2), ROLE_ANALYSIS: max(1, concurrency // 2),
            ROLE_PROPOSAL: max(1, concurrency // 2), ROLE_IMPLEMENTATION: 1}
