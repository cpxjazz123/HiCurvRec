"""Step 5b - the three coding agents (`code`): three independent
implementations of the frozen specification.

Moved out of worker.py by the split; the bodies are unchanged.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

from .. import settings
from ..agents.runner import (READ_ONLY_ROLES, boundary_fingerprint, run_agent)
from ..agents import policy, progress_server, runner
from ..core.evidence import atomic_json, digest_file, read_json
from . import agents as agent_contracts
from . import baseline as baseline_module
from . import evals as evals_module
from . import evidence as evidence_module
from . import ideas as ideas_module
from . import literature as literature_module
from . import math_checks as math_checks_module
from . import module_differential
from . import pipeline as pipeline_module
from . import plaintext
from . import probes as probes_module
from . import scoring as scoring_module
from . import spec_tests as spec_tests_module
from . import specification as specification_module


from .agent_calls import (  # noqa: F401  (the shared call)
    _run_agent, _run_agents_concurrently, _round_roots,
    _workspace, geometry_contract, role_of_task,
)



def _coding_prompt(spec: dict) -> str:
    """The prompt of a Coding Agent: the frozen specification, and nothing else."""
    return (
        "Implement this frozen mathematical specification. Do not reinterpret "
        "it: every sign, scale, limit and switch is as written.\n\n"
        + json.dumps(spec, indent=1, ensure_ascii=False))


def _parse_coding(text: str) -> dict:
    """The module an agent answered with."""
    import json as _json

    decoder = _json.JSONDecoder(strict=False)
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[position:])
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("module"):
            return {"module": str(payload.get("module")),
                    "notes": str(payload.get("notes") or "")}
    # A coding agent may answer with the module itself instead of a JSON object
    # carrying it: both are evidence, and a fence around the code is stripped.
    module = text.strip()
    if "```" in module:
        blocks = []
        for part in module.split("```")[1::2]:
            body = part.split("\n", 1)[1] if "\n" in part else part
            blocks.append(body)
        if blocks:
            module = max(blocks, key=len).strip()
    return {"module": module, "notes": ""}


def phase_coding(root: Path, round_id: str, *, log=print) -> dict:
    """Three independent implementations of the frozen specification.

    Each Coding Agent receives the same frozen specification and works in
    isolation; none of them may read another's code. Their agreement is the
    differential evidence, and the verdict blocks when it is not AGREE.
    """
    from . import differential as differential_module

    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    if state["step"] != "coding":
        return {"phase": "coding", "ran": False, "step": state["step"],
                "reason": "the round is no longer in the coding phase"}
    frozen = state.get("math_spec") or {}
    spec = frozen.get("spec") or {}
    if not spec:
        raise pipeline_module.RoundError("no frozen specification to implement")
    written = pipeline_module.check_frozen(directory, frozen.get("sha256"))
    if not written.get("matches"):
        raise pipeline_module.RoundError(
            "the frozen specification changed after the freeze: " + str(written.get("reason")))
    surface = spec.get("continuous_surface") or {}
    implementations = []
    for index in (1, 2, 3):
        task_id = "coding_%d" % index
        answer = _run_agent(directory, task_id, _coding_prompt(spec), "module.py",
                            _parse_coding, role="code", read_paths=[],
                            allowed_roots=_round_roots(directory))
        module = answer.get("module") or ""
        if not module.strip():
            raise pipeline_module.RoundError(
                "%s answered with no module: its answer file is kept as evidence "
                "at agents/%s/module.py and is not overwritten" % (task_id, task_id))
        target = directory / "agents" / task_id / "implementation.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(module, encoding="utf-8")
        implementations.append({
            "name": task_id,
            "symbols": list(surface.get("symbols") or []),
            "args": list(surface.get("args") or []),
            "statements": list(surface.get("statements") or []),
            "output": str(surface.get("output") or ""),
            "code": module,
            "notes": answer.get("notes") or "",
        })
        log("%s: %d bytes of module" % (task_id, len(module)))
    # A coding agent answers with a module, so the three of them are compared as
    # programs: same inputs, same switch arms, elementwise agreement.
    verdict = module_differential.compare_modules(implementations, spec)
    state["coding"] = {"implementations": [{"name": entry["name"],
                                             "bytes": len(entry["code"])}
                                            for entry in implementations],
                       "verdict": verdict}
    if verdict.get("status") != "AGREE":
        pipeline_module.save_state(directory, state)
        raise pipeline_module.RoundError(
            "the three implementations do not agree (" + str(verdict.get("status")) +
            "): " + str(verdict.get("reason"))[:300])
    state["step"] = "implementation"
    pipeline_module.save_state(directory, state)
    log("coding: three implementations agree")
    return {"phase": "coding", "ran": True, "step": state["step"],
            "status": verdict.get("status"),
            "implementations": len(implementations)}
