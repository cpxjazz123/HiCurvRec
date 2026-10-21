"""The versioned redesign that follows a tested hypothesis (issue #104).

One campaign arm proposed a mechanism; this module drives the agent that turns a *measured*
finding into a new, frozen proposal, implements it in its own tree, and hands it to the
unchanged judges and the active probe. Nothing here decides whether the mechanism is good:
the frozen Hard-L1 ER decides that, once, after a cold run.

The finding this redesign follows is measured, not assumed: a constraint on the soft
assignment does not constrain the exported hard code (a quarter of the soft mass sits on
codes the hard map never uses), and the codebook's geometry -- not a quota -- bounds which
codes the hard map can reach at all.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from . import ideation, opencode
from .proposal import freeze, proposal_digest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The tree every new arm starts from: the frozen host, unchanged.
BASE_TREE = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_r8_control"

#: What the study measured. Every number here comes from an artifact on disk.
EVIDENCE = """\
The campaign that precedes this proposal measured the following. Every number is a
measurement, not an estimate.

1. SOFT AND HARD DISAGREE. On the frozen control's level 0 (24587 items, 256 codes, 147
   used): 25.2% of the SOFT assignment's mass sits on codes the HARD map never uses, and
   the two distributions rank-correlate at only rho = 0.53. Of the 109 unused codes, 100
   are geometrically unreachable (no item is closer to them than to the nearest used code)
   and only 9 are reachable. Conclusion: a constraint on the soft assignment does not
   constrain the exported code, and the geometry -- not a quota -- bounds which codes the
   exported code can reach at all.

2. OCCUPANCY IS A GOOD PROXY, WITHIN THESE LANES. Across eleven recorded lanes the level-0
   occupancy and the frozen Hard-L1 ER correlate at rho = 0.973, and the train-side pair
   discrimination correlates with the ER at the same 0.973. The conditional level-2 match
   rate among shared-level-1 pairs correlates at -0.727, so the deeper level does NOT
   compensate a collapsed one.

3. FOUR ARMS WERE TRAINED COLD (3000 epochs, the frozen protocol) AND MEASURED:
   - control: Hard-L1 ER 16.334505666275888 (24587 items, 245 level-0 codes, 23657
     distinct 3-token SIDs).
   - a behaviour-conditioned pairwise margin on the continuous latents: ER 16.334505666275888,
     and its exported SIDs are BYTE-IDENTICAL to the control's. The training loss moved; the
     exported codes did not move at all.
   - a curvature-weighted transition transport: ER 16.334505666275888, byte-identical
     export again. The same verdict: the mechanism never reached the export decision.
   - a stop-gradient behaviour-conditioned residual target: ER 16.503445774392453 (above the
     frozen gate 16.38953942232631), but its Stage-3 test recall@10 is 0.061004, BELOW the
     control's 0.0637372 and below this project's 0.065 target. A small ER gain did not
     transfer downstream.
   - a curvature-conditioned ranking margin on the level-0 residual: ER 2.619839290662233.
     It wrecked the code structure (19475 distinct 3-token SIDs against the control's
     23657).

4. TWO EARLIER OCCUPANCY CONSTRAINTS FAILED TOO: projecting the soft assignment onto the
   host's occupancy quota left 81/205 = 39.5% level-0 occupancy, and an exact occupancy
   constraint left 47/194 = 24.2%. Neither preserved the exported hard codes.

So the levers that have NOT yet been tried are: acting on the EXPORTED code distribution
itself (the hard assignment), and moving the GEOMETRY that decides which codes are
reachable at all. A mechanism that only adds a term to the encoder's continuous loss has
now been shown twice to leave the export untouched, and a mechanism that reshapes the
assignment has been shown to destroy it.
"""


#: The second lever H1 names: the geometry that decides which codes are reachable.
GEOMETRY_LEVER = """\
Your mechanism must act on the GEOMETRY that decides which codes the exported hard map can
reach, and it must do so through the behaviour. Concretely: the measured fact is that 100 of
the 109 codes the hard map never uses are geometrically unreachable -- no item is closer to
them than to the nearest used code -- while only 9 are reachable. The quantity that decides
reachability is the geometry: where the items sit relative to the codewords. So your
mechanism must change WHERE the codewords are, in a way the behaviour graph determines, and
it must be a term whose gradient reaches the codebook.

The trap to avoid is the one the R8 record documents: five of the ten lanes reorganised the
codebook and collapsed the level-0 codebook (241 used entries down to 6 in the worst case).
A mechanism that spreads the codewords apart without regard to the items can empty the
codebook just as efficiently. Your mechanism has to keep the codewords where the items are,
while making the reachable set larger and more even -- and you must declare how it does that.
"""

#: The lever already tested, for the record.
EXPORTED_LEVER = """\
Your mechanism must act on the EXPORTED code distribution itself -- the hard assignment the
quantizer actually produces -- with a gradient path that reaches the codebook through a
straight-through estimator or an equivalent, so the balance it enforces is the balance that
is exported.
"""

LEVERS = {"exported_distribution": EXPORTED_LEVER, "geometry_reachability": GEOMETRY_LEVER}


def _brief(lever: str = "exported_distribution") -> str:
    if lever not in LEVERS:
        raise ValueError(f"unknown lever {lever!r}; known: {sorted(LEVERS)}")
    lever_text = LEVERS[lever]
    schema = {key: "..." for key in ideation._REQUIRED_FIELDS}
    return f"""# ROLE

You are the researcher of an autonomous mechanism campaign. You propose ONE mechanism; a
controller freezes it and gives it to an implementer. You do not write code and you do not
run anything.

# THE HYPOTHESIS THIS PROPOSAL MUST FOLLOW

The measurements below say that the soft assignment and the exported hard code are different
objects, that the codebook's geometry bounds which codes the exported code can reach, and
that two terms added to the encoder's continuous loss changed the exported SIDs by exactly
nothing. Therefore the mechanism you propose must act on ONE of these two objects.

{lever_text}

# THE EVIDENCE

{EVIDENCE}

# WHAT THE MECHANISM MUST BE

A behaviour-aware mechanism for the RQ-VAE of this repository, in which the user-item
BEHAVIOUR changes the geometry or the organisation of the learned item and codebook
representations -- not a detached auxiliary metric. Concretely the mechanism must:

* have an equation you can write down, with every symbol defined;
* say exactly which paths the loss reaches: the encoder, each affected codebook level, and
  the exported hard L1/L2/L3 SID, and whether the straight-through estimator or a
  stop-gradient blocks any of them;
* declare its hyperparameters and the criteria by which it would be selected;
* declare the invariants that a checker can execute WITHOUT knowing your internals (from:
  the loss is unchanged when the codebook rows are permuted; the loss changes when the
  source and the target are exchanged; the loss is zero when every pair is masked out);
* declare the gradient targets it claims (nonzero, zero or finite for the source residual,
  the target residual and the codebook);
* declare the substitutions that would make it a DIFFERENT mechanism, as regular expressions
  over its own source, with the reason each is forbidden;
* declare how it differs from the lanes that failed (the four arms measured above, the two
  occupancy constraints, and the ten historical R8 lanes), by name, with a reason each, and
  why it does not repeat their collapse pattern;
* declare its own failure signatures: what a wrong implementation of it would look like.

# THE MECHANISM MUST BE WRITTEN DOWN AS ONE JSON OBJECT

Reply with ONE fenced ```json block and nothing else. The object must have exactly these
keys:

{json.dumps(schema, indent=2)}

The controller VALIDATES this object before it freezes anything, and it refuses anything
that does not have the shape below. These are the shapes, exactly:

* `schema_version`: the number 1.
* `symbols`: an object mapping every symbol of the equation to what it means.
* `parameters`: an object of name -> number.
* `target_paths`: a list drawn from "encoder", "codebook", "hard_sid_export".
* `hypotheses` and `expected_failure_modes`: lists of non-empty strings.
* `tests`: a list of objects, each with an `id`, a `kind` and a `statement`. The kind is one
  of "numeric", "metamorphic", "integration", "gradient", "distributed".
* `judges`: a list drawn ONLY from "contract_audit", "forward_oracle", "derivative_evidence",
  "property_checks", "integration_on_off". Your selection criteria belong in
  `hypotheses` or in a test's statement, NOT in `judges`.
* `forbidden_substitutions`: a list of objects, each with an `id`, a `pattern` (a regular
  expression over your own source) and a `reason`.
* `provenance`: an object with `authored_by` (your name or "opencode") and `authored_at`
  (an ISO timestamp).
* `activation`: an object with `kind` ("term") and `weight` (a number).
* `differentiation`: an object with THREE keys, all required:
  - `occupancy_preservation`: a non-empty string saying how the mechanism avoids the
    occupancy price the R8 lanes paid;
  - `vs_r8`: a non-empty list of objects with a `lane` (one of the exact ids listed below)
    and a `why` (a non-empty string);
  - `vs_reference_arm`: a non-empty string saying how the mechanism differs from the
    reference arm `r8_control`, the untouched host with no auxiliary term.
* `inputs`: the checker's synthetic case, exactly as described below.

Notes on the keys:

* `mechanism_id`: snake_case, lower case, describing the mechanism itself.
* `version`: "2.0.0" -- this is a redesign of a family that has been measured, and the
  version says so.
* `equation`: the objective, in text, with the symbols it uses.
* `parameters`: a flat map of name -> number, every hyperparameter you need.
* `target_paths`: a list of "encoder", "codebook", "hard_sid_export".
* `inputs`: how a checker builds a tiny synthetic case for your mechanism:
  {{"source": {{"shape": [B, D], "seed": 11}}, "target": {{...}}, "codebook": {{"shape":
  [C, D], "seed": 13}}, "mask": {{"seed": 17, "valid_fraction": 0.75}}}}. The checker calls
  your loss as `loss(source, target, codebook, mask, **parameters)`.
* `invariants`: a list of {{"id", "transform", "expect", "tolerance"}} where `transform` is one
  of "permute_codebook", "swap_source_target", "mask_all" and `expect` is one of "unchanged",
  "changed", "zero".
* `gradient_targets`: {{"source": ..., "target": ..., "codebook": ...}}, each "nonzero",
  "zero" or "finite".
* `forbidden_substitutions`: a list of {{"pattern", "reason"}}.
* `activation`: {{"kind": "term", "weight": <number>}}.
"""


def propose(
    out_dir: Path, *, model: str = "", timeout: int = 2400, lever: str = "exported_distribution"
) -> Dict[str, Any]:
    """The agent proposes; the controller parses, critiques and freezes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = opencode.run_agent(
        out_dir,
        _brief(lever),
        model=model or opencode.model_name(),
        timeout=timeout,
        title="redesign proposal",
    )
    (out_dir / "proposal_session.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    raw = result.get("text") or ""
    (out_dir / "proposal_reply.txt").write_text(raw, encoding="utf-8")
    proposal = ideation.parse_proposal(raw)
    proposal.setdefault("version", "2.0.0")
    proposal["version"] = "2.0.0"
    critique = ideation.critique(proposal)
    frozen = freeze(proposal)
    frozen["revision"] = {
        "supersedes": "the four #103 arms measured in this session and the two occupancy "
                      "constraints of issue #103",
        "reason": "designed from the measured finding that a soft-assignment constraint does "
                  "not constrain the exported hard code, and that the geometry bounds which "
                  "codes the hard map can reach",
        "measured_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "proposal.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record = {
        "mechanism_id": frozen.get("mechanism_id"),
        "proposal_sha256": proposal_digest(frozen),
        "critique": {
            "fatal": list(critique.problems),
            "notes": list(critique.notes),
        },
        "tokens": result.get("tokens"),
        "exit_code": result.get("exit_code"),
    }
    (out_dir / "proposal_record.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {"proposal": frozen, "record": record}


def implement(
    proposal: Dict[str, Any], tree: Path, out_dir: Path, *, model: str = "", timeout: int = 3600
) -> Dict[str, Any]:
    """The implementer writes the mechanism into its own tree."""
    tree = Path(tree)
    if not tree.is_dir():
        shutil.copytree(BASE_TREE, tree)
        # The copy carries the base tree's own results directory name; the arm writes into
        # its own tree instead, so the host's artifacts are never touched.
    mechanism_id = proposal.get("mechanism_id")
    module_name = f"mechanism_{mechanism_id}"
    entrypoint = "mechanism_loss"
    brief = (
        f"# TASK\n\n"
        "Implement ONE Stage-2 mechanism in this arm tree, from a proposal that was frozen "
        "before you started. The proposal is below. It is the contract: implement THAT "
        "mechanism, not a different one.\n\n"
        "# THE PROPOSAL\n\n"
        "```json\n"
        + json.dumps(proposal, indent=2, sort_keys=True)
        + "\n```\n\n"
        "# WHERE TO WORK\n\n"
        "This tree is a scratch copy of an experiment arm. Implement the mechanism here: a "
        "module of your own inside the tree, and the call site in `train_rqvae.py` that "
        "composes its term into the host's loss BEFORE backward, so it reaches the optimizer "
        "through the ordinary path.\n\n"
        "The host's own loss graph, the codebook sizes, the KMeans init, the seed, the "
        "3000-epoch budget and the optimizer are NOT yours to change. The mechanism's own "
        "weight is a hardcoded constant of the tree, like the other constants.\n\n"
        "# THE NAMES ARE FIXED\n\n"
        f"Put the mechanism in a module named `{module_name}.py` in the root of this tree, and "
        f"name its entry point `{entrypoint}`. The checker looks for exactly those names. The "
        "entry point must be a module-level function with this signature:\n\n"
        "```python\n"
        f"def {entrypoint}(source, target, codebook, mask, **parameters):\n"
        "    ...\n"
        "```\n\n"
        "where `source` is [B, D], `target` is [B, D], `codebook` is [C, D], `mask` is [B] with "
        "1.0 for a valid pair, and `parameters` are the proposal's parameters. The module must "
        "import on its own, with no dependency on the tree.\n\n"
        "# TWO THINGS THAT HAVE BROKEN ARMS BEFORE\n\n"
        "1. The mechanism block must live inside the host's `if epoch_weight > 0.0:` region, "
        "or read nothing that only exists there: the host draws its directed pairs only once "
        "the behaviour ramp is open, and a block outside that region crashes at epoch 0.\n"
        "2. The mechanism must be composed into the loss BEFORE backward, and its gradient "
        "must reach the parameters it claims: the encoder, the codebook levels it declares, "
        "and nothing else.\n\n"
        "Report what you changed and why.\n"
    )
    result = opencode.run_agent(
        tree,
        brief,
        model=model or opencode.model_name(),
        timeout=timeout,
        title=f"implement {mechanism_id}",
        deny_paths=[REPO_ROOT / "results", REPO_ROOT / "tools"],
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "implement_session.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {
        "mechanism_id": mechanism_id,
        "module": module_name,
        "entrypoint": entrypoint,
        "tokens": result.get("tokens"),
        "exit_code": result.get("exit_code"),
        "text": result.get("text"),
    }


def repair_proposal(
    out_dir: Path, *, model: str = "", timeout: int = 2400, extra: str = ""
) -> Dict[str, Any]:
    """One bounded repair: the agent fixes the proposal's own fatal problem.

    The controller does not edit the proposal's science. It reports what the rubric
    refuses and asks the agent to fix it, once.
    """
    out_dir = Path(out_dir)
    previous = json.loads((out_dir / "proposal.json").read_text(encoding="utf-8"))
    lanes = "\n".join(
        f"* `{lane}` -- {name}" for lane, name in sorted(ideation.PRIOR_FAILURES.items())
    )
    brief = f"""# YOUR PROPOSAL WAS REFUSED BY THE RUBRIC

The controller's rubric is deterministic. Here is what it said about the proposal you wrote:

{json.dumps(json.loads((out_dir / "proposal_record.json").read_text())["critique"]["fatal"], indent=2)}

and here is what it noted:

{json.dumps(json.loads((out_dir / "proposal_record.json").read_text())["critique"]["notes"], indent=2)}

Your proposal was:

```json
{json.dumps(previous, indent=2, sort_keys=True)}
```

{extra}

# WHAT TO FIX

`differentiation.vs_r8` is a list of objects with a `lane` and a `why`. The `lane` must be
one of these exact ids -- the historical lanes that failed, with their names:

{lanes}

For EACH lane you name, the `why` must say concretely what your mechanism does that the lane
did not, and why it therefore does not inherit that lane's failure. A rename of an existing
lane is a refusal, not a redesign: if your mechanism is a directed L1 transition term, say
precisely how it differs from `r8_01_directed_l1_transition_nce`, which is a directed L1
code-transition likelihood and whose level-0 codebook collapsed to 6 used entries.

Also make sure `parameters` contains every number your equation uses, including the weight
`lambda` if your equation has one, and that `provenance` carries `authored_by` and
`authored_at`.

AND make sure `differentiation` has all THREE of these keys: `occupancy_preservation` (a
non-empty string), `vs_r8` (a non-empty list of {{"lane", "why"}}) and `vs_reference_arm`
(a non-empty string). The rubric refuses a proposal that is missing any of them.

Reply with ONE fenced ```json block and nothing else: the FULL corrected proposal object,
with the same keys as before.
"""
    result = opencode.run_agent(
        out_dir,
        brief,
        model=model or opencode.model_name(),
        timeout=timeout,
        title="redesign proposal repair",
    )
    (out_dir / "proposal_repair_session.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    raw = result.get("text") or ""
    (out_dir / "proposal_repair_reply.txt").write_text(raw, encoding="utf-8")
    proposal = ideation.parse_proposal(raw)
    proposal.setdefault("version", "2.0.0")
    proposal["version"] = "2.0.0"
    critique = ideation.critique(proposal)
    frozen = freeze(proposal)
    frozen["revision"] = {
        "supersedes": "the four #103 arms measured in this session and the two occupancy "
                      "constraints of issue #103",
        "reason": "designed from the measured finding that a soft-assignment constraint does "
                  "not constrain the exported hard code, and that the geometry bounds which "
                  "codes the hard map can reach",
        "measured_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "proposal.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record = {
        "mechanism_id": frozen.get("mechanism_id"),
        "proposal_sha256": proposal_digest(frozen),
        "critique": {"fatal": list(critique.problems), "notes": list(critique.notes)},
        "tokens": result.get("tokens"),
        "exit_code": result.get("exit_code"),
    }
    (out_dir / "proposal_record.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {"proposal": frozen, "record": record}


def _ledger_brief(ledger_text: str, constraint: str, lanes: str) -> str:
    """The research system's brief, assembled from the ledger and nothing else."""
    schema = {key: "..." for key in ideation._REQUIRED_FIELDS}
    return f"""# ROLE

You are the researcher of an autonomous research system. You propose ONE mechanism; a
controller freezes it, gives it to an implementer, and puts it through the independent
checks, an active probe and a cold run. You do not write code and you do not run anything.

# THE LEDGER: WHAT THE SYSTEM KNOWS AND WHAT IT HAS TESTED

{ledger_text}

# THE CONSTRAINT THE HOST IMPOSES (measured, not assumed)

{constraint}

# WHAT THE MECHANISM MUST BE

A behaviour-aware mechanism for the RQ-VAE of this repository, in which the user-item
BEHAVIOUR changes the geometry or the organisation of the learned item and codebook
representations -- not a detached auxiliary metric. Concretely the mechanism must:

* have an equation you can write down, with every symbol defined;
* say exactly which paths the loss reaches: the encoder, each affected codebook level, and
  the exported hard L1/L2/L3 SID, and whether the straight-through estimator or a
  stop-gradient blocks any of them;
* declare its hyperparameters and the criteria by which it would be selected;
* declare the invariants that a checker can execute WITHOUT knowing your internals (from:
  the loss is unchanged when the codebook rows are permuted; the loss changes when the
  source and the target are exchanged; the loss is zero when every pair is masked out);
* declare the gradient targets it claims (nonzero, zero or finite for the source residual,
  the target residual and the codebook) -- and be honest about them, because the checker
  measures them;
* declare the substitutions that would make it a DIFFERENT mechanism, as regular
  expressions over its own source, with the reason each is forbidden;
* declare how it differs from the lanes that failed, by name, with a reason each, and why
  it does not repeat their collapse pattern;
* declare its own failure signatures: what a wrong implementation of it would look like.

# THE MECHANISM MUST BE WRITTEN DOWN AS ONE JSON OBJECT

Reply with ONE fenced ```json block and nothing else. The object must have exactly these
keys:

{json.dumps(schema, indent=2)}

The controller VALIDATES this object before it freezes anything, and it refuses anything
that does not have the shape below:

* `schema_version`: the number 1.
* `symbols`: an object mapping every symbol of the equation to what it means.
* `parameters`: an object of name -> number.
* `target_paths`: a list drawn from "encoder", "codebook", "hard_sid_export".
* `hypotheses` and `expected_failure_modes`: lists of non-empty strings.
* `tests`: a list of objects, each with an `id`, a `kind` ("numeric", "metamorphic",
  "integration", "gradient" or "distributed") and a `statement`.
* `judges`: a list drawn ONLY from "contract_audit", "forward_oracle", "derivative_evidence",
  "property_checks", "integration_on_off". Your selection criteria belong in `hypotheses` or
  in a test's statement, NOT in `judges`.
* `forbidden_substitutions`: a list of objects, each with an `id`, a `pattern` (a regular
  expression over your own source) and a `reason`.
* `provenance`: an object with `authored_by` and `authored_at` (an ISO timestamp).
* `activation`: an object with `kind` ("term") and `weight` (a number).
* `differentiation`: an object with THREE keys, all required: `occupancy_preservation`
  (a non-empty string), `vs_r8` (a non-empty list of objects with a `lane` -- one of the
  exact ids listed below -- and a `why`), and `vs_reference_arm` (a non-empty string).
* `inputs`: the checker's synthetic case, exactly as described below.

The historical lanes that failed, with their names:

{lanes}

Notes on the keys:

* `mechanism_id`: snake_case, lower case, describing the mechanism itself.
* `version`: "3.0.0" -- this is the third generation: the first two families were repaired
  and measured, and the third follows what the ledger established.
* `equation`: the objective, in text, with the symbols it uses.
* `parameters`: a flat map of name -> number, every hyperparameter you need.
* `target_paths`: a list of "encoder", "codebook", "hard_sid_export".
* `inputs`: how a checker builds a tiny synthetic case for your mechanism:
  {{"source": {{"shape": [B, D], "seed": 11}}, "target": {{...}}, "codebook": {{"shape":
  [C, D], "seed": 13}}, "mask": {{"seed": 17, "valid_fraction": 0.75}}}}. The checker calls
  your loss as `loss(source, target, codebook, mask, **parameters)`.
* `invariants`: a list of {{"id", "transform", "expect", "tolerance"}} where `transform` is one
  of "permute_codebook", "swap_source_target", "mask_all" and `expect` is one of "unchanged",
  "changed", "zero".
* `gradient_targets`: {{"source": ..., "target": ..., "codebook": ...}}, each "nonzero",
  "zero" or "finite".
* `forbidden_substitutions`: a list of {{"pattern", "reason"}}.
* `activation`: {{"kind": "term", "weight": <number>}}.
"""


def propose_from_ledger(
    ledger_text: str,
    constraint: str,
    out_dir: Path,
    *,
    model: str = "",
    timeout: int = 2400,
) -> Dict[str, Any]:
    """The agent proposes the next mechanism, briefed only by the ledger."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lanes = "\n".join(
        f"* `{lane}` -- {name}" for lane, name in sorted(ideation.PRIOR_FAILURES.items())
    )
    brief = _ledger_brief(ledger_text, constraint, lanes)
    result = opencode.run_agent(
        out_dir,
        brief,
        model=model or opencode.model_name(),
        timeout=timeout,
        title="mechanism proposal from the ledger",
    )
    (out_dir / "proposal_session.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    raw = result.get("text") or ""
    (out_dir / "proposal_reply.txt").write_text(raw, encoding="utf-8")
    proposal = ideation.parse_proposal(raw)
    proposal.setdefault("version", "3.0.0")
    proposal["version"] = "3.0.0"
    proposal.setdefault("schema_version", 1)
    proposal["schema_version"] = 1
    critique = ideation.critique(proposal)
    frozen = freeze(proposal)
    frozen["revision"] = {
        "supersedes": "the two families the ledger measured in issue #105",
        "reason": "proposed from the ledger's supported findings, not from a fixed lever list",
        "measured_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "proposal.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record = {
        "mechanism_id": frozen.get("mechanism_id"),
        "proposal_sha256": proposal_digest(frozen),
        "critique": {"fatal": list(critique.problems), "notes": list(critique.notes)},
        "tokens": result.get("tokens"),
        "exit_code": result.get("exit_code"),
    }
    (out_dir / "proposal_record.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {"proposal": frozen, "record": record}


__all__ = [
    "BASE_TREE",
    "EVIDENCE",
    "implement",
    "propose",
    "propose_from_ledger",
    "repair_proposal",
]
