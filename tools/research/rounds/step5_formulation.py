"""Step 5a - the formula agent (`formulate`) and the two math
reviewers (`review_math`): the mathematics, its review, the eval
gate and the freeze.

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



def _contract_text() -> str:
    """The geometry contract's notation and operators, verbatim.

    The framework owns the geometry; a specification that invents its own
    parameterisation loses a factor, and the factor is what the reviewers then
    have to find.
    """
    path = Path(__file__).resolve().parents[1] / "contracts" / \
        "HYPERBOLIC_GEOMETRY_CONTRACT.md"
    if not path.is_file():
        return "(the geometry contract is missing)"
    text = path.read_text(encoding="utf-8")
    return text[:6000]


def _formula_prompt(idea: dict, findings: list = None) -> str:
    """The prompt of the Formula Agent: the idea, and what the reviewers said."""
    prompt = (
        "Write the mathematics of this mechanism. The mechanism has already been "
        "chosen; you are not choosing it and you may not change it.\n\n"
        "TITLE: " + str(idea.get("title")) + "\n"
        "MECHANISM: " + str(idea.get("mechanism")) + "\n"
        "SPECIFICATION: " + str(idea.get("specification")) + "\n"
        "SPEC EQUATIONS: " + str(idea.get("spec_equations")) + "\n"
        "INPUTS AND OUTPUTS: " + str(idea.get("spec_inputs_outputs")) + "\n"
        "BEHAVIOUR SIGNAL: " + str(idea.get("behavior_signal")) + "\n"
        "BEHAVIOUR COMPUTATION: " + str(idea.get("behavior_computation")) + "\n"
        "COUPLING: " + str(idea.get("coupling_path")) + "\n"
        "ABLATION BEHAVIOUR OFF: " + str(idea.get("ablation_behavior_off")) + "\n"
        "ABLATION GEOMETRY OFF: " + str(idea.get("ablation_geometry_off")) + "\n")
    prompt += ("\n\nTHE GEOMETRY CONTRACT IS THE AUTHORITY on every convention. "
               "Read it and quote its operators exactly; a formula written in an "
               "unstated parameterisation is wrong:\n\n"
               + _contract_text() +
               "\n\nANSWER THESE THREE QUESTIONS EXPLICITLY, each as its own JSON "
               "key (the review refuses a specification that leaves them out):\n"
               "  dimensions: which tensor has which width - the batch, the latent, "
               "the codebook, every intermediate;\n"
               "  gradient_rules: for EVERY operator of the mechanism, what the "
               "backward pass does - the straight-through estimator, the detach "
               "points, the operators with no gradient at all;\n"
               "  training_integration: what the total loss is, what the decoder "
               "receives (the chosen codeword or the transported one), and which "
               "optimizer and schedule train it.\n")
    prompt += ("\n\nThe specification MUST state, as JSON keys: mechanism, "
               "computations, inputs, outputs, parameters, dimensions, "
               "gradient_rules, training_integration, switches, excluded, and the "
               "continuous_surface. Every computation the mechanism touches "
               "belongs in computations; dimensions states every tensor width; "
               "gradient_rules states how the gradient flows for every operator "
               "(including the straight-through estimator and the argmin); "
               "training_integration states the loss, what the decoder receives "
               "and the optimizer.")
    if findings:
        prompt += ("\n\nTHE REVIEWERS FOUND THESE PROBLEMS. Fix them, and nothing "
                   "else:\n- " + "\n- ".join(str(finding) for finding in findings))
    return prompt


def _review_math_prompt(idea: dict, spec: dict) -> str:
    """The prompt of a Math Reviewer: the idea and the specification."""
    return (
        "Check this mathematical specification against the idea it came from.\n\n"
        "IDEA TITLE: " + str(idea.get("title")) + "\n"
        "IDEA SPECIFICATION: " + str(idea.get("specification")) + "\n"
        "IDEA SPEC EQUATIONS: " + str(idea.get("spec_equations")) + "\n"
        "IDEA COUPLING: " + str(idea.get("coupling_path")) + "\n"
        "IDEA ABLATIONS: " + str(idea.get("ablation_behavior_off")) + " | "
        + str(idea.get("ablation_geometry_off")) + "\n\n"
        "THE SPECIFICATION:\n" + json.dumps(spec, indent=1, ensure_ascii=False)
        + "\n\nANSWER CONTRACT. Answer with one JSON object:\n"
        '{"verdict": "PASS" | "REVISE", "findings": ["..."], '
        '"tests": [{"finding": "...", "eval_id": "...", "code": "..."}], '
        '"reason": "..."}\n'
        "Every finding MUST carry a test. A test is a Python module that defines "
        "check(spec) -> list[str]: it returns a non-empty list of problems exactly "
        "when the specification shows the defect you describe, and an empty list "
        "otherwise. It may only read the specification dictionary; it must not "
        "import torch or the agents SDK, run training, or read data. The framework "
        "runs your test against the specification you are reviewing: a test that "
        "does not fail on it is refused as not being a test of that defect. Your "
        "tests are written to tools/research/rounds/spec_tests/ and are run on "
        "every later round before it may freeze, so write them to refuse the "
        "defect, not to pass.")


def phase_formulation(root: Path, round_id: str, *, log=print) -> dict:
    """The formulation stage: the mathematics, reviewed, frozen.

    The Formula Agent writes the mechanism as mathematics, two Math Reviewers
    check it against the idea, and the freeze writes ``math_spec.json`` and its
    hash. After the freeze the specification is the only authority the
    implementation may follow.
    """
    directory = pipeline_module.find_round_directory(root, round_id)
    state = pipeline_module.load_state(directory)
    if state["step"] not in ("formulation", "implementation"):
        return {"phase": "formulation", "ran": False, "step": state["step"],
                "reason": "the round is no longer in the formulation phase"}
    ideas = pipeline_module.selected_ideas(state)
    if not ideas:
        return {"phase": "formulation", "ran": False, "step": state["step"],
                "reason": "the round has no selected idea"}
    idea = ideas[0]
    findings = list(state.get("math_review_findings") or [])
    # A specification the main agent wrote is a legitimate draft: the formula
    # often comes from the round's objective, and the two Math Reviewers judge
    # the draft as they judge the agent's. The draft is never the authority -
    # the freeze is.
    drafted = Path(directory) / "math_spec_draft.json"
    for attempt in range(3):
        if drafted.is_file():
            spec = json.loads(drafted.read_text(encoding="utf-8"))
        else:
            answer = _run_agent(directory, "formulation",
                                _formula_prompt(idea, findings),
                                "math_spec.json", _parse_math_spec, role="formulate",
                                read_paths=[], allowed_roots=_round_roots(directory))
            spec = answer.get("spec") or {}
        if spec.get("incomplete"):
            missing = spec.get("missing") or []
            state["math_specification_incomplete"] = {"missing": missing,
                                                       "note": spec.get("incomplete")}
            pipeline_module.save_state(directory, state)
            raise pipeline_module.RoundError(
                "the specification is incomplete: the idea does not define " +
                "; ".join(str(item) for item in missing))
        spec = specification_module.complete(spec)
        problems = specification_module.validate(spec)
        if problems:
            findings = problems
            continue
        reviews = []
        for index in (1, 2):
            verdict = _run_agent(directory, "math_review_%d" % index,
                                 _review_math_prompt(idea, spec),
                                 "math_review.json", _parse_math_review,
                                 role="review_math", read_paths=[],
                                 allowed_roots=_round_roots(directory))
            reviews.append(verdict)
        state["math_reviews"] = reviews
        rejected = [review for review in reviews
                    if str(review.get("verdict", "")).upper() != "PASS"]
        if not rejected:
            # Every retained test, from every round, runs before the freeze: a
            # defect caught once is refused from then on, whatever this round's
            # reviewers say.
            retained = spec_tests_module.run(spec)
            state["spec_test_suite"] = {"passed": retained["passed"],
                                        "checked": retained["checked"],
                                        "failures": retained["failures"]}
            if not retained["passed"]:
                state["math_review_findings"] = retained["findings"]
                pipeline_module.save_state(directory, state)
                log("formulation: %d retained tests refused the specification"
                    % len(retained["failures"]))
                continue
            # Every defect an earlier attempt caught must have a decision before
            # the specification is frozen: an eval that now refuses it, or a
            # recorded reason it is not mechanically decidable.
            history_findings = [finding
                                for attempt_record in state.get("math_review_history") or []
                                for finding in attempt_record.get("findings") or []]
            undecided = evals_module.undecided(history_findings)
            if undecided:
                state["math_review_findings"] = undecided
                state["eval_ledger_undecided"] = undecided
                pipeline_module.save_state(directory, state)
                log("formulation: %d findings have no eval and no recorded reason"
                    % len(undecided))
                raise pipeline_module.RoundError(
                    "every finding a review caught must be decided before the "
                    "freeze: add an eval to tools/research/rounds/evals.py or an "
                    "entry to tools/research/rounds/eval_ledger.json for: " +
                    "; ".join(str(finding)[:120] for finding in undecided[:3]))
            gate = evals_module.run(spec)
            state["spec_eval_gate"] = {"passed": gate["passed"],
                                       "checked": gate["checked"],
                                       "failures": gate["failures"]}
            if not gate["passed"]:
                findings = gate["findings"]
                state["math_review_findings"] = findings
                pipeline_module.save_state(directory, state)
                log("formulation: the eval gate refused the specification (%s)"
                    % ", ".join(failure["id"] for failure in gate["failures"]))
                continue
            frozen = specification_module.freeze(directory, spec)
            state["math_spec"] = {"sha256": frozen["sha256"], "path": frozen["path"],
                                  "reviewed_by": len(reviews),
                                  "spec": spec}
            state["step"] = "coding"
            state["math_review_findings"] = []
            pipeline_module.save_state(directory, state)
            log("formulation: the specification is frozen (%s)" % frozen["sha256"][:12])
            return {"phase": "formulation", "ran": True, "step": state["step"],
                    "sha256": frozen["sha256"], "reviews": len(reviews)}
        findings = [finding for review in rejected
                    for finding in review.get("findings") or []]
        # A finding without a test is not accepted: the reviewer writes the test
        # that refuses the defect, the framework proves it fails on the
        # specification it was written against, and keeps it forever.
        written_tests, refused_tests = _retain_spec_tests(
            directory, round_id, rejected, spec, log=log)
        state["spec_tests_written"] = written_tests
        if refused_tests:
            findings = findings + refused_tests
        state["math_review_findings"] = findings
        history = state.setdefault("math_review_history", [])
        history.append({"attempt": attempt + 1,
                        "verdicts": [str(review.get("verdict", "")) for review in reviews],
                        "findings": findings,
                        "recorded_at": __import__("datetime").datetime.now(
                            __import__("datetime").timezone.utc).isoformat()})
        pipeline_module.save_state(directory, state)
        log("formulation: the reviewers asked for %d changes" % len(findings))
    raise pipeline_module.RoundError(
        "the mathematical specification did not pass the review: " +
        "; ".join(str(finding) for finding in findings[:3]))


def _retain_spec_tests(directory: Path, round_id: str, rejected: list, spec: dict,
                       *, log=print) -> tuple:
    """Keep the reviewers' tests, and refuse the ones that are not tests.

    A test must fail on the specification it was written against; one that
    passes proves nothing about the defect it claims to describe, and a finding
    that comes with no test is reported as such rather than silently accepted.
    """
    written, refused = [], []
    for review in rejected:
        tests = list(review.get("tests") or [])
        findings = [str(finding) for finding in review.get("findings") or []]
        covered = {str(test.get("finding") or "").strip() for test in tests}
        for finding in findings:
            if str(finding).strip() not in covered:
                refused.append("the finding '%s' came with no test: a finding "
                               "must be answered with a test that refuses it"
                               % str(finding)[:160])
        for test in tests:
            code = str(test.get("code") or "")
            finding = str(test.get("finding") or "") or str(findings[0] if findings else "")
            # prove the test fails on the specification under review
            probe = Path(tempfile.mkdtemp(prefix="spec-test-")) / "candidate.py"
            probe.write_text(code, encoding="utf-8")
            failed = spec_tests_module.run(spec, paths=[str(probe)])
            if not failed["failures"]:
                refused.append("the test for '%s' passes on the specification it "
                               "was written against: it does not refuse the defect"
                               % str(finding)[:140])
                continue
            kept = spec_tests_module.record(spec_tests_module.DIRECTORY, round_id,
                                            finding, code,
                                            eval_id=str(test.get("eval_id") or ""))
            written.append({"finding": str(finding)[:200], "path": kept["path"],
                            "written": kept.get("written")})
            log("formulation: retained a test for '%s'" % str(finding)[:80])
    return written, refused


def _parse_math_spec(text: str) -> dict:
    """The specification an agent answered with, or an empty specification."""
    import json as _json

    decoder = _json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[position:])
        except ValueError:
            continue
        if isinstance(payload, dict) and (payload.get("computations")
                                          or payload.get("incomplete")):
            return {"spec": payload}
    return {"spec": {}}


def _parse_math_review(text: str) -> dict:
    """The verdict an agent answered with."""
    import json as _json

    decoder = _json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[position:])
        except ValueError:
            continue
        if isinstance(payload, dict) and payload.get("verdict"):
            tests = []
            for entry in payload.get("tests") or []:
                if isinstance(entry, dict) and str(entry.get("code") or "").strip():
                    tests.append({"finding": str(entry.get("finding") or ""),
                                  "eval_id": str(entry.get("eval_id") or ""),
                                  "code": str(entry.get("code"))})
            return {"verdict": str(payload.get("verdict")),
                    "findings": [str(finding) for finding in payload.get("findings") or []],
                    "tests": tests,
                    "reason": str(payload.get("reason") or "")}
    return {"verdict": "", "findings": [], "reason": ""}
