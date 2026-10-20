"""Autonomous ideation: the researcher's angles, the critique, the portfolio.

The researcher is OpenCode. Harness2 supplies the angles and the evidence; the model
writes the mechanism. The critique that follows is **deterministic** -- a rubric over
the proposal's own declarations -- so the campaign's decisions are automated and
auditable rather than a matter of taste: a proposal that does not declare its
differentiation from the failed lanes, its collapse risks, its gradient paths and its
machine-checkable invariants does not qualify, whatever it says about itself.

The prior-failure record is the R8 programme's own synthesis: eleven lanes, none of which
beat the fresh cold control, with three failure shapes and one pattern -- mechanisms that
*reorganise* the codebook destroy the hard ID, mechanisms that *reweight* it trade
discrimination away, and none of them added discrimination.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: The lanes that already failed, with their mechanism names. A proposal must differ
#: from these explicitly, and the campaign screens for a rename of one of them.
PRIOR_FAILURES: Dict[str, str] = {
    "r8_01_directed_l1_transition_nce": "Directed L1 code-transition NCE",
    "r8_02_l1_next_item_prediction": "L1 next-item semantic prediction",
    "r8_03_successor_profile_js": "Successor-profile JS co-clustering",
    "r8_04_behavior_conditioned_ot": "Behaviour-conditioned entropic OT quantizer",
    "r8_05_three_level_distillation": "Three-level teacher to L1 distillation",
    "r8_06_train_graph_diffusion": "Train-graph diffusion distance preservation",
    "r8_07_hyperbolic_gyro": "Hyperbolic directed gyro-translation",
    "r8_08_hyperbolic_frechet": "Hyperbolic Frechet successor barycentres",
    "r8_09_hyperbolic_branching": "Hyperbolic branching radial hierarchy",
    "r8_10_hyperbolic_gw": "Hyperbolic graph-code Gromov-Wasserstein",
    "r8_control": "untouched R4-08 host, no auxiliary term",
}

#: The lessons the campaign must not relearn.
PRIOR_LESSONS = (
    "Reorganising the codebook destroys the hard ID: five of the ten lanes collapsed the L1 "
    "codebook (2.78, 3.84, 6.80, 6.85, 1.00) and the worst collapsed every item into one code.",
    "Reweighting the codebook trades discrimination away: the four least damaging lanes still lost "
    "0.45 to 1.32 of ER, and none of the eleven added discrimination.",
    "The frozen gate sits inside the host's own run-to-run band (the untouched host re-run cold "
    "scored 16.3587 against the gate's 16.3895), so the comparison that carries information is "
    "against a FRESH COLD CONTROL, not against the gate.",
    "A mechanism with no gradient to the codebook, or none at all, is caught by a preflight "
    "gradient check before any full run: two lanes were repaired for exactly that.",
    "The reference arm's collapse is the sharpest instance: it was mathematically faithful, passed "
    "every correctness check, and still took the level-0 codebook from 241 used entries to 6.",
)

#: The six angles. They are the campaign's own hypotheses about where a mechanism could
#: still add discrimination, and they are deliberately different from one another.
ANGLES: Dict[str, str] = {
    "discrimination_first": (
        "Add a behaviour-conditioned DISCRIMINATIVE signal that leaves the codebook's occupancy "
        "alone: a term that changes which items are separated at L1, not how many codes are used. "
        "The R8 record says occupancy and discrimination moved in opposite directions every "
        "time; a mechanism that touches only the separation should not pay the occupancy price."
    ),
    "behaviour_conditioned_curvature": (
        "Make the GEOMETRY itself depend on the behaviour: a behaviour-conditioned curvature, a "
        "learned Busemann-style distance, or a per-pair curvature that the behaviour statistics "
        "determine. The hyperbolic lanes that failed did so because they applied a fixed geometry "
        "to a behaviour-dependent structure; the question is what changes when the curvature is "
        "estimated from the behaviour rather than assumed."
    ),
    "occupancy_preserving_codebook_organisation": (
        "Organise the codebook WITH the behaviour while explicitly preserving the host's "
        "occupancy: a constrained or regularised assignment whose constraint is the host's own "
        "code usage. The difference from the failed co-clustering and transport lanes is that the "
        "occupancy is a CONSTRAINT rather than a consequence."
    ),
    "curvature_weighted_transition": (
        "Replace the graph's plain diffusion with a CURVATURE-WEIGHTED transport: the behaviour "
        "graph's local curvature decides how much a transition moves in representation space. The "
        "difference from the diffusion lane is the weight, and the difference from the "
        "Gromov-Wasserstein lane is that the structure is local and the transport is not a "
        "full pairwise coupling."
    ),
    "behaviour_conditioned_residual": (
        "Make the QUANTIZATION TARGET behaviour-dependent: the residual an item is asked to "
        "represent depends on where it sits in the behaviour graph, without a transport plan and "
        "without touching the codebook's usage. The failed OT lane moved the assignment; this "
        "moves the target."
    ),
    "ranking_space_behaviour_regulariser": (
        "Regularise the RANKING of the L1 logits with the behaviour, and nothing else: a term "
        "that only reorders which item is closest, with no effect on the codebook's geometry or "
        "usage. It is the smallest possible behaviour-aware intervention, and it is the one the "
        "R8 pattern leaves open."
    ),
}

_REQUIRED_FIELDS = (
    "mechanism_id",
    "version",
    "title",
    "equation",
    "symbols",
    "parameters",
    "target_paths",
    "hypotheses",
    "expected_failure_modes",
    "tests",
    "judges",
    "forbidden_substitutions",
    "invariants",
    "gradient_targets",
    "inputs",
    "activation",
    "differentiation",
)

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class Critique:
    """The rubric's verdict on one proposal."""

    ok: bool
    problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    scores: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "notes": list(self.notes),
            "scores": dict(self.scores),
        }


def prompt_for(angle: str, *, base_tree: str, evidence: str) -> str:
    """The researcher's brief for one angle."""
    if angle not in ANGLES:
        raise ValueError(f"unknown angle {angle!r}; known: {sorted(ANGLES)}")
    return f"""# ROLE

You are the researcher of an autonomous mechanism campaign. You propose ONE mechanism; a
controller reviews it, freezes it, and gives it to an implementer. You do not write code
and you do not run anything.

# THE HYPOTHESIS SPACE FOR THIS ROUND

{ANGLES[angle]}

# WHAT THE EVIDENCE ALREADY SAYS (do not repeat it)

{evidence}

# WHAT THE MECHANISM MUST BE

A behaviour-aware mechanism for the RQ-VAE of this repository, in which the user-item
BEHAVIOUR changes the geometry or the organisation of the learned item and codebook
representations -- not a detached auxiliary metric. Concretely, the mechanism must:

* have an equation you can write down, with every symbol defined;
* say exactly which paths the loss reaches: the encoder, each affected codebook level, and
  the exported hard L1/L2/L3 SID, and whether the straight-through estimator or a
  stop-gradient blocks any of them;
* declare the hyperparameters it uses and the criteria by which it would be selected;
* declare the invariants that a checker can execute WITHOUT knowing your internals (from:
  the loss is unchanged when the codebook rows are permuted; the loss changes when the
  source and the target are exchanged; the loss is zero when every pair is masked out);
* declare the gradient targets it claims (nonzero, zero or finite for the source residual,
  the target residual and the codebook);
* declare the substitutions that would make it a DIFFERENT mechanism, as regular
  expressions over its own source, with the reason each is forbidden;
* declare how it differs from the historical lanes listed below -- by name, with a reason
  each -- and why it does not repeat the collapse pattern;
* declare its own failure signatures: what a wrong implementation of it would look like.

# THE MECHANISM MUST BE WRITTEN DOWN AS ONE JSON OBJECT

Reply with ONE fenced ```json block and nothing else. The object must have exactly
these keys:

{json.dumps({key: "..." for key in _REQUIRED_FIELDS}, indent=2)}

Notes on the keys:

* `mechanism_id`: snake_case, lower case, describing the mechanism itself, never the
  round or the attempt.
* `equation`: the objective, in text, with the symbols it uses.
* `parameters`: a flat map of name -> number, every hyperparameter you need.
* `target_paths`: a list of "encoder", "codebook", "hard_sid_export" -- the paths your loss
  reaches.
* `inputs`: how a checker builds a tiny synthetic case for your mechanism:
  {{"source": {{"shape": [B, D], "seed": 11}}, "target": {{...}}, "codebook": {{"shape":
  [C, D], "seed": 13}}, "mask": {{"seed": 17, "valid_fraction": 0.75}}}}. The checker
  calls your loss as `loss(source, target, codebook, mask, **parameters)`.
* `invariants`: a list of {{"id", "transform", "expect", "tolerance"}} where `transform` is
  one of "permute_codebook", "swap_source_target", "mask_all" and `expect` is one of
  "unchanged", "changed", "zero".
* `gradient_targets`: {{"source": "nonzero|zero|finite", "target": ..., "codebook": ...}}.
* `activation`: {{"kind": "term", "weight": 0.05}} -- the weight your term enters the host's
  loss with.
* `forbidden_substitutions`: a list of {{"id", "pattern", "why"}} with regular expressions
  over your own source.
* `differentiation`: {{"vs_r8": [{{"lane", "why"}}], "vs_reference_arm": "...",
  "occupancy_preservation": "..."}} -- the lanes you differ from, why, and how your
  mechanism avoids paying the occupancy price the R8 lanes paid.
* `tests`: a list of {{"id", "kind", "statement"}} with kind in numeric, metamorphic,
  gradient, integration.
* `judges`: the independent checks your mechanism needs from
  {{"contract_audit", "forward_oracle", "derivative_evidence", "property_checks",
  "integration_on_off"}}. A mechanism with no reference implementation cannot be checked
  by the forward oracle: say "contract_audit" and "integration_on_off" and declare your
  own invariants.

# THE TREE YOUR IMPLEMENTER WILL WORK IN

{base_tree}
"""


#: The names a proposal may use for the things the campaign needs. A researcher that
#: writes "objective" instead of "equation" has still written the equation.
_ALIASES = {
    "objective": "equation",
    "loss": "equation",
    "hyperparameters": "parameters",
    "hyper_parameters": "parameters",
    "gradient_paths": "target_paths",
    "paths": "target_paths",
    "failure_modes": "expected_failure_modes",
    "expected_failures": "expected_failure_modes",
    "mechanism_name": "title",
    "name": "title",
    "id": "mechanism_id",
    "checks": "tests",
    "forbidden": "forbidden_substitutions",
    "substitutions": "forbidden_substitutions",
    "gradients": "gradient_targets",
    "gradient_target": "gradient_targets",
}

#: The defaults the campaign fills in when a proposal does not declare them. The inputs
#: and the activation are the campaign's own contract with the checker, not the
#: researcher's to invent; the gradient targets default to the weakest claim, so a
#: mechanism that genuinely does not reach the codebook is not failed for saying so.
_DEFAULTS: Dict[str, Any] = {
    "inputs": {
        "source": {"shape": [4, 4], "seed": 11},
        "target": {"shape": [4, 4], "seed": 12},
        "codebook": {"shape": [5, 4], "seed": 13},
        "mask": {"seed": 17, "valid_fraction": 0.75},
    },
    "activation": {"kind": "term", "weight": 0.05},
    "gradient_targets": {"source": "nonzero", "target": "nonzero", "codebook": "finite"},
    "version": "1.0.0",
    "tolerance": 1e-6,
}


def parse_proposal(text: str) -> Dict[str, Any]:
    """Read the researcher's JSON block out of its reply, and normalise its names."""
    block = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    raw = block.group(1) if block else None
    if raw is None:
        start = text.find("{")
        end = text.rfind("}")
        raw = text[start : end + 1] if start >= 0 and end > start else ""
    if not raw:
        raise ValueError("the reply carries no JSON object")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the reply's JSON does not parse: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("the reply's JSON is not an object")
    for alias, canonical in _ALIASES.items():
        if canonical not in document and alias in document:
            document[canonical] = document[alias]
    for key, value in _DEFAULTS.items():
        document.setdefault(key, json.loads(json.dumps(value)))
    return document


def critique(proposal: Dict[str, Any]) -> Critique:
    """The rubric. Deterministic: a proposal either declares what it must or it does not.

    The problems are fatal; the notes are recorded and do not disqualify. A proposal that
    names a lane that does not exist, or that uses a word a failed lane used, is not
    rejected for it: what disqualifies a proposal is a MISSING declaration, because a
    missing declaration cannot be checked and therefore cannot be trusted.
    """
    problems: List[str] = []
    notes: List[str] = []

    for field in _REQUIRED_FIELDS:
        if field not in proposal:
            problems.append(f"the proposal has no {field!r}")

    mechanism_id = str(proposal.get("mechanism_id", ""))
    if mechanism_id and not _NAME.match(mechanism_id):
        problems.append(f"mechanism_id {mechanism_id!r} is not snake_case lower case")

    equation = str(proposal.get("equation", ""))
    for lane, name in PRIOR_FAILURES.items():
        if lane == "r8_control":
            continue
        # A rename screen: the equation may not name a failed lane's mechanism verbatim.
        distinctive = [word for word in name.split() if len(word) > 4]
        for word in distinctive:
            if word.lower() in equation.lower():
                notes.append(
                    f"the equation names {word!r}, which is the failed lane {lane} ({name}); "
                    "check that this is a difference and not a rename"
                )
                break

    differentiation = proposal.get("differentiation") or {}
    vs_r8 = differentiation.get("vs_r8") or []
    if not isinstance(vs_r8, list) or not vs_r8:
        problems.append("the proposal does not say which historical lanes it differs from")
    else:
        lanes = {str(item.get("lane", "")) for item in vs_r8 if isinstance(item, dict)}
        unknown = sorted(lanes - set(PRIOR_FAILURES))
        if unknown:
            notes.append(f"the proposal names lanes that do not exist: {unknown}")
        for item in vs_r8:
            if not isinstance(item, dict) or not str(item.get("why", "")).strip():
                problems.append("a differentiation entry has no reason")
                break
    if not str(differentiation.get("occupancy_preservation", "")).strip():
        problems.append(
            "the proposal does not say how it avoids the occupancy price the R8 lanes paid"
        )
    if not str(differentiation.get("vs_reference_arm", "")).strip():
        problems.append("the proposal does not say how it differs from the reference arm")

    invariants = proposal.get("invariants") or []
    if not isinstance(invariants, list) or not invariants:
        problems.append("the proposal declares no machine-checkable invariant")
    else:
        transforms = {str(item.get("transform", "")) for item in invariants if isinstance(item, dict)}
        unknown = transforms - {"permute_codebook", "swap_source_target", "mask_all"}
        if unknown:
            problems.append(f"the proposal declares transformations a checker cannot run: {sorted(unknown)}")

    targets = proposal.get("gradient_targets") or {}
    if not isinstance(targets, dict) or not targets:
        problems.append("the proposal declares no gradient target")
    else:
        for name, expectation in targets.items():
            if expectation not in ("nonzero", "zero", "finite"):
                problems.append(f"gradient target {name!r} has an unknown expectation {expectation!r}")

    inputs = proposal.get("inputs") or {}
    if not isinstance(inputs, dict) or not {"source", "target", "codebook"} <= set(inputs):
        problems.append("the proposal does not say how a checker builds its synthetic case")

    scores = {
        "declares_differentiation": bool(vs_r8),
        "declares_invariants": len(invariants) if isinstance(invariants, list) else 0,
        "gradient_targets": len(targets) if isinstance(targets, dict) else 0,
        "failure_modes": len(proposal.get("expected_failure_modes") or []),
        "tests": len(proposal.get("tests") or []),
        "targets": len(proposal.get("target_paths") or []),
    }
    return Critique(ok=not problems, problems=problems, notes=notes, scores=scores)


def signature(proposal: Dict[str, Any]) -> str:
    """A fingerprint of what the mechanism DOES, for the diversity constraint."""
    equation = str(proposal.get("equation", "")).lower()
    words = [word for word in re.findall(r"[a-z_]{4,}", equation)]
    return "|".join(sorted(set(words))[:12])


def select_portfolio(
    proposals: Sequence[Dict[str, Any]], *, size: int = 6
) -> List[Dict[str, Any]]:
    """Pick the portfolio: qualified first, then diverse.

    A portfolio of six variants of one idea is not a portfolio. Two proposals with the
    same equation signature are the same idea, and only one of them is taken.
    """
    qualified = []
    for proposal in proposals:
        verdict = critique(proposal)
        if verdict.ok:
            qualified.append((proposal, verdict))
    qualified.sort(key=lambda pair: -sum(pair[1].scores.values()))
    chosen: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for proposal, _verdict in qualified:
        key = signature(proposal)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(proposal)
        if len(chosen) >= size:
            break
    return chosen


__all__ = [
    "ANGLES",
    "Critique",
    "PRIOR_FAILURES",
    "PRIOR_LESSONS",
    "critique",
    "parse_proposal",
    "prompt_for",
    "select_portfolio",
    "signature",
]
