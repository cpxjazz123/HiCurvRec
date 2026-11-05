"""Step 5c - the three derivation agents (`derive`): three
independent derivations of the same mechanism, compared.

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
    AgentPhaseIncomplete, RoundIncomplete,
    _run_agent, _run_agents_concurrently, _round_roots,
    _workspace, geometry_contract, role_of_task,
)



def _parse_derivation(text: str) -> dict:
    """The first JSON object in an answer, or an empty derivation.

    The derivation's ``code`` field contains dict literals and f-strings, so a
    brace counter sees braces inside strings and closes in the wrong place. The
    decoder knows which braces are quoting: it is the only parser that survives
    a torch implementation embedded as a JSON string.
    """
    import json as _json

    # strict=False: a model writes a raw newline inside a long string (a wrapped
    # torch body, a wrapped expression). The decoder's strict mode refuses the
    # whole answer over one control character; the derivation is then validated
    # for real by sympify and by running it, which is the check that matters.
    decoder = _json.JSONDecoder(strict=False)
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[position:])
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("statements"):
            return payload
    return {}


#: The instance every derivation of a round must use. Two derivations that
#: choose different dimensions are different functions, and the comparison of
#: them says nothing; the instance is therefore fixed by the framework and
#: checked, not chosen by the agents.
DERIVATION_INSTANCE = {"latent_width": 2, "levels": 2, "codes": 2,
                       "latent_name": "z", "codebook_name": "theta",
                       "scale_name": "s", "anchor_name": "v_pi",
                       "output": "loss"}


def derivation_signature() -> tuple:
    """The arguments and the symbols of the fixed instance.

    The framework owns them: the agents fill the statements, the output name
    and the code. An agent that invents its own arguments produces a different
    function, and the comparison of two such answers says nothing.
    """
    instance = DERIVATION_INSTANCE
    width = int(instance["latent_width"])
    args = [{"name": instance["latent_name"], "width": width, "radius": 0.4}]
    symbols = ["%s%d" % (instance["latent_name"], index) for index in range(width)]
    for level in range(int(instance["levels"])):
        for code in range(int(instance["codes"])):
            name = "%s_%d_%d" % (instance["codebook_name"], level + 1, code + 1)
            args.append({"name": name, "width": width, "radius": 0.3})
            symbols.extend("%s%d_%d_%d" % (instance["codebook_name"], level + 1,
                                           code + 1, index) for index in range(width))
    args.append({"name": instance["scale_name"], "scalar": 1.0})
    symbols.append(instance["scale_name"])
    for code in range(int(instance["codes"])):
        name = "%s_%d" % (instance["anchor_name"], code + 1)
        args.append({"name": name, "width": width, "radius": 0.3})
        symbols.extend("%s%d_%d" % (instance["anchor_name"], code + 1, index)
                       for index in range(width))
    return args, symbols


def _instance_codebook_width() -> int:
    instance = DERIVATION_INSTANCE
    return int(instance["levels"]) * int(instance["codes"]) * int(instance["latent_width"])


def _instance_problems(payload: dict) -> list:
    """Where a derivation departs from the fixed instance."""
    instance = DERIVATION_INSTANCE
    problems = []
    arguments = payload.get("args") or []
    widths = {str(spec.get("name")): spec for spec in arguments}
    latent = [spec for spec in arguments
              if str(spec.get("name")).lower().startswith(instance["latent_name"])]
    if not latent:
        problems.append("no latent argument named " + instance["latent_name"])
    else:
        width = int(latent[0].get("width") or 1)
        if width != int(instance["latent_width"]):
            problems.append("the latent width is %d, the round fixes %d"
                            % (width, int(instance["latent_width"])))
    codebook = [spec for spec in arguments
                if str(spec.get("name")).lower().startswith(instance["codebook_name"])]
    if not codebook:
        problems.append("no codebook argument named " + instance["codebook_name"])
    else:
        # The packing of the codebook is a diagnostic, not a failure: a
        # derivation may write one argument per level or one per codeword, and
        # the comparison aligns the arguments by position.
        # The codebook may be one argument or one argument per (level, code)
        # pair: what the round fixes is the TOTAL number of codeword
        # components, not how they are packed.
        total = sum(int(spec.get("width") or 1) for spec in codebook)
    if not any(str(spec.get("name")).lower().startswith(instance["scale_name"])
               and spec.get("scalar") is not None for spec in arguments):
        problems.append("no scalar scale named " + instance["scale_name"])
    output = str(payload.get("output") or "")
    if output != instance["output"]:
        problems.append("the output is %r, the round fixes %r"
                        % (output, instance["output"]))
    return problems


def _code_signature(payload: dict) -> list:
    """The parameter names of a derivation's own code, or an empty list."""
    import ast as _ast

    code = payload.get("code") or ""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "compute":
            names = []
            for argument in node.args.args:
                names.append(argument.arg)
            if node.args.kwarg is not None:
                names = []
            return names
    return []


def _code_accepts(payload: dict, args: list) -> bool:
    """Whether the derivation's code can be called with the round's arguments.

    A code that takes **kwargs accepts everything; a code with named
    parameters must not require a name the round does not declare.
    """
    import ast as _ast

    code = payload.get("code") or ""
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return False
    declared = {str(spec["name"]) for spec in args}
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "compute":
            if node.args.kwarg is not None:
                return True
            required = [argument.arg for argument in node.args.args
                        if argument.arg not in ("self",)]
            return all(name in declared for name in required)
    return False


def _derivation_prompt(idea: dict) -> str:
    """One specification, no other derivation, and no way to ask for one."""
    instance = DERIVATION_INSTANCE
    args, symbols = derivation_signature()
    argument_lines = "\n".join(
        "      %s: %s" % (spec["name"],
                         ("a scalar" if spec.get("scalar") is not None
                          else "a vector of %d components" % int(spec["width"])))
        for spec in args)
    symbol_lines = ", ".join(symbols)
    return (
        "FIXED INSTANCE - the round fixes it and you must use exactly these "
        "arguments and these symbols. A derivation with its own signature is a "
        "different function and cannot be compared with the others.\n"
        "  * the arguments, in this exact order:\n" + argument_lines + "\n"
        "  * the free symbols, in this exact order (a vector argument of width w "
        "consumes w symbols, one per component): " + symbol_lines + "\n"
        "  * the output is a single per-item vector named '" + instance["output"] +
        "' - the quantity the mechanism changes in the training objective.\n"
        "  * the code is a complete Python module defining compute(**kwargs) with "
        "exactly these argument names, and it must compute the SAME formula as "
        "the statements.\n"
        "  * `symbols` and `args` are NOT part of your answer: the round owns "
        "them. You answer with the statements, the output name, the code and the "
        "excluded surfaces.\n\n"
        "Derive the formula of this mechanism from the specification alone.\n\n"
        "TITLE: " + str(idea.get("title")) + "\n"
        "MECHANISM: " + str(idea.get("mechanism")) + "\n"
        "SPECIFICATION: " + str(idea.get("specification")) + "\n"
        "SPEC EQUATIONS: " + str(idea.get("spec_equations")) + "\n"
        "INPUTS AND OUTPUTS: " + str(idea.get("spec_inputs_outputs")) + "\n\n"
        "FIXED INSTANTIATION - every derivation must use exactly this, or the "
        "two cannot be compared:\n"
        "  * latent width D = 2;\n"
        "  * the smallest complete instance: L = 2 levels, K = 2 codes per level;\n"
        "  * arguments, in this exact order and with these exact names:\n"
        "      z (width 2) - the encoder vector,\n"
        "      theta_1_1, theta_1_2 (width 2), theta_2_1, theta_2_2 (width 2) - "
        "the codebook vectors in (level, code) order,\n"
        "      s (scalar) - the frozen scale.\n"
        "  * output: a single per-item vector named `loss` (one value per row of z), "
        "the quantity the mechanism adds or changes in the training objective.\n"
        "  * a switch or a table that the specification needs may be carried as "
        "an extra argument (a scalar 0/1 or a width-2 vector) placed AFTER s, and "
        "must be named in `args` and `symbols` in the same order.\n"
        "  Symbols follow the arguments: a width-w argument consumes w symbols in "
        "column order (z -> z0, z1; theta_1_1 -> t110, t111 ...).\n\n"
        "Turn the equations into SymPy statements and into a complete torch "
        "module defining compute(**kwargs), as the answer format requires.")


def run_derivations(directory: Path, ideas: list, *, log=print) -> list:
    """Three independent derivations of the one selected mechanism."""
    from ..agents import runtime as agent_runtime

    answers = []
    for idea in ideas:
        # A derivation answers in a workspace that a previous pass already used.
        # Its old answer must not be read as this pass's delivery when the new
        # agent times out: it is moved aside before the agents start.
        for index in range(1, 4):
            stale = directory / "agents" / ("derivation_%d" % index) / "derivation.json"
            if stale.is_file():
                archive = stale.parent / "previous" / str(int(time.time()))
                archive.mkdir(parents=True, exist_ok=True)
                stale.replace(archive / "derivation.json")

        def derive(task_id, _idea=idea):
            return _run_agent(
                directory, task_id, _derivation_prompt(_idea), "derivation.json",
                _parse_derivation, role="derive", read_paths=[],
                allowed_roots=_round_roots(directory))

        task_ids = ["derivation_1", "derivation_2", "derivation_3"]
        try:
            outcomes = _run_agents_concurrently([(task_id,) for task_id in task_ids],
                                                lambda task_id: derive(task_id))
        except AgentPhaseIncomplete as exc:
            # A derivation that ran out of its five minutes is not a reason to
            # throw the ones that answered away: the comparison needs two, and
            # the phase records what is missing. The answers keep the task they
            # belong to, so a missing first agent cannot shift the others.
            if not exc.results:
                raise
            outcomes = [entry["result"] for entry in exc.delivered] or exc.results
            task_ids = [entry["task"] for entry in exc.delivered] or task_ids
        delivered, missing = [], []
        for task_id, outcome in zip(task_ids, outcomes):
            payload = {key: outcome.get(key) for key
                       in ("statements", "output", "code", "excluded",
                           "symbols", "args")
                       if outcome.get(key)}
            if payload.get("statements") and not payload.get("args"):
                args, symbols = derivation_signature()
                payload["args"] = args
                payload["symbols"] = symbols
            if payload.get("statements") and payload.get("code"):
                delivered.append({"idea_id": idea["idea_id"],
                                  "index": int(task_id.split("_")[-1]),
                                  "payload": payload, "receipt": outcome.get("receipt")})
            else:
                missing.append(int(task_id.split("_")[-1]))
            log("%s %s: %s" % (idea["idea_id"], task_id,
                               "ok" if payload.get("statements") else "no readable answer"))
        args, symbols = derivation_signature()
        for answer in list(delivered):
            payload = answer.get("payload") or {}
            problems = _instance_problems(payload) if payload else ["no payload"]
            if not problems and not _code_accepts(payload, args):
                problems = ["the code cannot be called with the round's arguments"]
            if not problems:
                continue
            index = int(answer["index"])
            log("%s derivation %d: the instance does not match (%s); asking again"
                % (idea["idea_id"], index, "; ".join(problems)))
            retry_id = "derivation_%d" % index
            correction = (_derivation_prompt(idea) +
                          "\n\nYOUR PREVIOUS DERIVATION DID NOT FIT THE INSTANCE: " +
                          "; ".join(problems) +
                          ". Use exactly the arguments listed in the FIXED INSTANCE "
                          "block above and nothing else, and your code must define "
                          "compute with exactly these parameter names, in this order: "
                          + ", ".join(spec["name"] for spec in args) +
                          ". Your statements must use exactly these symbols: " +
                          ", ".join(symbols) + ".")
            outcome = _run_agent(directory, retry_id, correction, "derivation.json",
                                 _parse_derivation, role="derive", read_paths=[],
                                 allowed_roots=_round_roots(directory))
            payload = {key: outcome.get(key) for key
                       in ("statements", "output", "code", "excluded",
                           "symbols", "args")
                       if outcome.get(key)}
            if payload.get("statements") and not payload.get("args"):
                args, symbols = derivation_signature()
                payload["args"] = args
                payload["symbols"] = symbols
            answer["payload"] = payload
            answer["revision"] = True
        if missing and len(delivered) < 2:
            # A derivation that delivered nothing is asked once more - but only
            # when the round would otherwise have fewer than two. Waiting for a
            # fifth minute of a third derivation while two are on disk holds the
            # comparison for evidence it does not need; the missing one is
            # recorded as a gap in the verdict instead.
            retry = ["derivation_%d" % index for index in missing]
            log("%s: asking %s again" % (idea["idea_id"], ", ".join(retry)))
            try:
                outcomes = _run_agents_concurrently([(task_id,) for task_id in retry],
                                                    lambda task_id: derive(task_id))
            except AgentPhaseIncomplete as exc:
                if not exc.results:
                    continue
                outcomes = exc.results
            for task_id, outcome in zip(retry, outcomes):
                payload = {key: outcome.get(key) for key
                           in ("symbols", "args", "statements", "output", "code", "excluded")
                           if outcome.get(key)}
                delivered.append({"idea_id": idea["idea_id"],
                                  "index": int(task_id.split("_")[-1]),
                                  "payload": payload, "receipt": outcome.get("receipt")})
        answers.extend(delivered)
    return answers


def _run_differential(implementations: list, *, repo: Path) -> dict:
    """The deterministic N-version comparison, in the Stage 2 interpreter."""
    import subprocess
    import tempfile
    from .. import settings

    payload = {"implementations": implementations}
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    handle.write(json.dumps(payload))
    handle.close()
    # The comparison needs sympy, sympytorch and torch: that is the Stage 2
    # interpreter, the one the framework's own checks run in.
    from ..stage2 import contract as stage2_contract

    command = [stage2_contract.STAGE2_PYTHON, "-m",
               "tools.research.rounds.differential",
               "--implementations", handle.name]
    done = subprocess.run(command, cwd=str(repo), capture_output=True, text=True,
                          timeout=1800)
    lines = [line for line in done.stdout.strip().splitlines() if line.startswith("{")]
    if not lines:
        return {"status": "NOT_RUN",
                "reason": "the comparison produced no report: " +
                          (done.stdout + done.stderr).strip()[-300:]}
    return json.loads(lines[-1])


def phase_derivation(root: Path, round_id: str, *, log=print) -> dict:
    """Three independent derivations of the selected mechanism, then compare them."""
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    selected = pipeline_module.selected_ideas(state)
    if not selected:
        return {"phase": "derivation", "ran": False,
                "reason": "the round has no selected idea"}
    if state.get("derivations"):
        return {"phase": "derivation", "ran": False,
                "reason": "the derivations are on disk",
                "status": (state.get("differential") or {}).get("status")}
    answers = run_derivations(directory, selected, log=log)
    implementations = [{"name": "%s_%d" % (answer["idea_id"], answer["index"]),
                        **(answer.get("payload") or {})}
                       for answer in answers if answer.get("payload")]
    mismatched = []
    comparable = []
    for implementation in implementations:
        problems = _instance_problems(implementation)
        if problems:
            implementation["instance_problems"] = problems
            mismatched.append({"name": implementation["name"],
                                "status": "INSTANCE_MISMATCH",
                                "reason": "; ".join(problems)})
        else:
            comparable.append(implementation)
    implementations = comparable
    verdict = (_run_differential(implementations, repo=Path(root))
               if len(implementations) >= 2
               else {"status": "NOT_APPLICABLE",
                     "reason": "fewer than two derivations answered"})
    state["derivation_instance"] = dict(DERIVATION_INSTANCE)
    state["derivations"] = [{"idea_id": answer["idea_id"],
                             "index": answer["index"],
                             "payload": answer.get("payload") or {},
                             "receipt": answer.get("receipt")}
                            for answer in answers]
    state["differential"] = verdict
    if mismatched:
        state["differential"] = dict(verdict)
        state["differential"]["instance_mismatches"] = mismatched
    pipeline_module.save_state(directory, state)
    log("derivation: %s" % verdict.get("status"))
    return {"phase": "derivation", "ran": True, "status": verdict.get("status"),
            "derivations": len(implementations)}
