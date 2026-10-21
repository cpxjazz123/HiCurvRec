"""The candidate search (issue #106, workstream C).

The agent reads the evidence store and the hypothesis ledger and chooses at least three
**mathematically distinct** candidate mechanisms, each with its equation, the parameter
groups it intends to reach, the behaviour signal it uses, how it interacts with the
codebook and the encoder, its risk, and an explicit falsifier. The controller freezes each
proposal before anything is implemented.

The brief states the two measured facts a candidate has to live with:

* the host's assembly **discards** an auxiliary term's encoder gradient, so a candidate that
  reaches only the encoder changes nothing;
* the codebook path is what collapses the codebook when it pulls a transition's two
  codewords together, so a candidate that reaches the codebook must not do that.

and the objective: the downstream Stage-3 quality against the archived control, not the
Hard-L1 ER.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence import EvidenceStore, now_utc, sha256_bytes, canonical_json
from .proposal import freeze, proposal_digest

#: The measured facts every candidate has to respect.
CONSTRAINTS = """\
1. The host's training loop OVERWRITES the encoder gradient after the backward
   (g_base + w_b/valid_levels * (g'_L1 + g_L2 + g_L3)) and never touches the decoder or the
   codebook gradient. Measured: an auxiliary term changes the encoder gradient by 1.66e-05
   before that assembly and by exactly 0.0 after it. A candidate that reaches only the
   encoder changes nothing: two arms whose mechanisms did exactly that trained to weights
   bit-identical to the control's, and exported SIDs bit-identical to the control's.
2. The codebook path is the one that survives, and it is the one that destroys the codebook
   when it pulls a transition's two codewords together. Measured on the archived checkpoints:
   the arm that collapsed to two level-0 codes has a transition's two level-0 codewords
   0.0644 apart on average (the MEDIAN is 0.0 -- the endpoints share one codeword) against
   the control's 0.2732, and 77.2% of its pairs sit below a tenth of the control's
   separation against the control's 7.8%. The used codewords converged onto each other.
3. The magnitude alone does not explain it: the delivered codebook gradient scales exactly
   linearly with the mechanism's weight and the hard assignment's churn is ZERO at every
   weight, the production one included.
4. The objective is the DOWNSTREAM recommendation quality against the archived control
   (Stage-3 recall@10 0.06406796775709883, NDCG@10 0.03530651701940677), not the Hard-L1
   ER. An above-gate ER has twice come with no downstream gain and once with a clear
   downstream regression.
5. The budget: ONE cold 3000-epoch Stage-2 run per admitted candidate, at the single frozen
   seed, and the frozen held-out evaluation is read exactly once per candidate. A candidate
   that is rejected at the probe costs no cold run at all.
"""

CANDIDATE_FIELDS = (
    "id",
    "family",
    "title",
    "equation",
    "symbols",
    "parameters",
    "gradient_targets",
    "target_paths",
    "behaviour_signal",
    "codebook_interaction",
    "encoder_interaction",
    "risk",
    "falsifier",
    "why_not_the_measured_failures",
)


def _transfer_requirements() -> str:
    """The measured requirements a candidate has to meet, read from the transfer analysis."""
    path = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "harness3_runs"
        / "transfer_analysis"
        / "stage3_transfer_report.json"
    )
    if not path.is_file():
        return (
            "(the transfer analysis has not been run; the requirement below is the generic one: "
            "do not degrade the exported codes' balance, their prefix structure, their sequence "
            "information or their train-to-test stability)"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    implications = document.get("design_implications") or []
    if not implications:
        return "(the transfer analysis reported no additional requirement)"
    return "\n".join(f"* {line}" for line in implications)


def _brief(store: EvidenceStore, ledger_text: str) -> str:
    from .hypotheses import store_brief

    return f"""# ROLE

You are the research scientist of an autonomous research system. You choose the next
candidate mechanisms; a controller freezes them, implements them, checks them and trains
them. You do not write code and you do not run anything.

# WHAT THE SYSTEM HAS MEASURED

{store_brief(store)}

# THE HYPOTHESIS LEDGER

{ledger_text}

# THE TWO FACTS A CANDIDATE MUST LIVE WITH

{CONSTRAINTS}

# THE PATHS THAT SURVIVE THE HOST, MEASURED

The host overwrites the ENCODER gradient after the backward, and the decoder's gradient only
changes the decoder. The measured consequence is that **the codebook is the only path whose
gradient can change the exported SIDs**: an auxiliary term that reaches only the encoder
leaves the trained weights bit-identical to the control's, and one that reaches only the
decoder cannot move the export either. A candidate that wants to change the exported codes
must therefore deliver its gradient to the codebook -- and the only question left is *which
direction* it moves the codebook in.

# THE TENSION THE MEASUREMENTS SHOW

The frozen pair-level ER is bought with CODE CONCENTRATION: a behaviour-driven merge of the
level-0 codes raises the positive match rate, and the same merge removes the distinctions the
recommender ranks by. The arm with the best ER of the campaign (17.13 against the control's
16.33) has a level-0 Gini of 0.313 against the control's 0.216 and 62.5 prefix branches
against 65.9, and its Stage-3 recall@10 was the worst measured; the arm with the healthiest
structure had an ER far below the control. So a candidate that repeats the merge will not
help. A candidate has to raise the ER by a better PARTITION of the items over the SAME number
of codes.

# THE MEASURED REQUIREMENT A CANDIDATE HAS TO MEET

The archived arms that raised the pair-level ER did not improve the recommender. The
measurements below say what the exported codes must do instead, and they are requirements a
checker can execute, not adjectives:

{_transfer_requirements()}

# THE TASK

Choose at least **three mathematically distinct** candidate mechanisms, spanning at least
**three different families** (see `family` above). One of them must address the measured
cause of the downstream loss directly: the pair-convergence of the two codewords a behaviour
pair shares. A candidate that keeps the pair's codewords apart while still raising the
pair-level ER is permitted, and the ledger records that no arm has achieved it yet. "Distinct" means
their objectives differ in kind, not in their parameters: one that acts on the codebook's
GEOMETRY, one that acts on the assignment's structure without letting a pair's codewords
converge, and one that acts on the behaviour-conditioned residual geometry are three
different kinds; three weights of the same term are not.

For each candidate, state:

* `id`: snake_case, lower case, no round or attempt number;
* `title`: one line;
* `equation`: the objective, in text, with the symbols it uses;
* `symbols`: every symbol, what it means;
* `parameters`: a flat map of name -> number, every hyperparameter, INCLUDING the weight the
  term is composed with;
* `gradient_targets`: {{"source": ..., "target": ..., "codebook": ...}}, each "nonzero",
  "zero" or "finite" -- and it must be HONEST, because a checker measures it;
* `target_paths`: a list drawn from "encoder", "codebook", "hard_sid_export";
* `behaviour_signal`: which behaviour quantity the term reads;
* `codebook_interaction`: exactly how the term's gradient acts on the codewords, and why it
  cannot pull a transition's two codewords together (or how it is prevented from doing so);
* `encoder_interaction`: what the term does to the encoder, knowing that the host discards
  the encoder's gradient unless the term reaches the codebook;
* `risk`: the most likely way this candidate fails, in one paragraph;
* `falsifier`: the measurement that would show this candidate is wrong;
* `why_not_the_measured_failures`: how it differs from the arms that were inert and from the
  arms that collapsed the codebook.

6. THE INTERFACE AND THE DIRECTEDNESS CHECK, LEARNED FROM THE FIRST ROUND'S REJECTIONS
   - The checker calls the mechanism as `mechanism_loss(source, target, codebook, mask,
     **parameters)`, with source [B, D], target [B, D], codebook [C, D], mask [B] and
     `parameters` EXACTLY the proposal's declared parameters. A term that needs any other
     input cannot be checked and is rejected -- two candidates were rejected for asking for
     one.
   - The checker exchanges the source and the target and requires the loss to CHANGE. A
     candidate that says it is directed must actually be directed; two candidates declared a
     directed mechanism and implemented a symmetric one, and were rejected.
   - The checker also requires the gradient to reach the parameter groups the proposal
     declares, and to be finite everywhere.

# THE FORMAT

Reply with ONE fenced ```json block and nothing else:

```json
{{
  "candidates": [
    {{
      "id": "...",
      "title": "...",
      "equation": "...",
      "symbols": {{"...": "..."}},
      "parameters": {{"...": 0.0}},
      "gradient_targets": {{"source": "nonzero", "target": "nonzero", "codebook": "nonzero"}},
      "target_paths": ["codebook", "hard_sid_export"],
      "behaviour_signal": "...",
      "codebook_interaction": "...",
      "encoder_interaction": "...",
      "risk": "...",
      "falsifier": "...",
      "why_not_the_measured_failures": "..."
    }}
  ]
}}
```
"""


def propose_candidates(
    store: EvidenceStore,
    ledger_text: str,
    out_dir: Path,
    *,
    model: str = "",
    timeout: int = 2400,
) -> Dict[str, Any]:
    """The agent chooses the candidates; the controller freezes each one."""
    from . import opencode

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = opencode.run_agent(
        out_dir,
        _brief(store, ledger_text),
        model=model or opencode.model_name(),
        timeout=timeout,
        title="candidate mechanisms",
    )
    (out_dir / "candidates_session.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    raw = result.get("text") or ""
    (out_dir / "candidates_reply.txt").write_text(raw, encoding="utf-8")
    block = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.S)
    document = json.loads(block.group(1) if block else "{}")
    frozen: List[Dict[str, Any]] = []
    for candidate in document.get("candidates") or ():
        missing = [field for field in CANDIDATE_FIELDS if field not in candidate]
        if missing:
            (out_dir / "rejected_candidates.json").write_text(
                json.dumps(
                    {
                        "rejected": [
                            {"candidate": candidate, "reason": f"missing fields: {missing}"}
                        ]
                    },
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            continue
        proposal = {
            "mechanism_id": candidate["id"],
            "family": candidate.get("family", "unspecified"),
            "version": "1.0.0",
            "title": candidate["title"],
            "equation": candidate["equation"],
            "symbols": candidate["symbols"],
            "parameters": {str(k): float(v) for k, v in (candidate["parameters"] or {}).items()},
            "gradient_targets": candidate["gradient_targets"],
            "target_paths": candidate["target_paths"],
            "hypotheses": [candidate["falsifier"], candidate["why_not_the_measured_failures"]],
            "expected_failure_modes": [candidate["risk"]],
            "tests": [
                {
                    "id": "numeric",
                    "kind": "numeric",
                    "statement": "the term's value on a hand-computable case matches the "
                                 "declared equation",
                },
                {
                    "id": "gradient",
                    "kind": "gradient",
                    "statement": "the gradient reaches exactly the declared parameter groups",
                },
            ],
            "judges": ["contract_audit", "derivative_evidence", "property_checks"],
            "forbidden_substitutions": [
                {
                    "id": "no_occupancy_quota",
                    "pattern": "occupancy|quota|project_to_capacity",
                    "reason": "a quota on the assignment is the family that produced 39.5% and "
                              "24.2% occupancy without preserving the exported codes",
                },
                {
                    "id": "no_pair_codeword_convergence",
                    "pattern": "codebook\\[[^\\]]*\\]\\s*-\\s*codebook",
                    "reason": "a term that pulls a pair's two codewords together is what "
                              "collapsed the level-0 codebook to two codes",
                },
            ],
            "invariants": [
                {
                    "id": "permute_codebook",
                    "transform": "permute_codebook",
                    "expect": "unchanged",
                    "tolerance": 1e-06,
                },
                {
                    "id": "swap_source_target",
                    "transform": "swap_source_target",
                    "expect": "changed",
                    "tolerance": 1e-06,
                },
                {
                    "id": "mask_all",
                    "transform": "mask_all",
                    "expect": "zero",
                    "tolerance": 1e-08,
                },
            ],
            "inputs": {
                "source": {"shape": [4, 3], "seed": 11},
                "target": {"shape": [4, 3], "seed": 12},
                "codebook": {"shape": [5, 3], "seed": 13},
                "mask": {"seed": 17, "valid_fraction": 0.75},
            },
            "activation": {
                "kind": "term",
                "weight": float(candidate["parameters"].get("weight", 0.05)),
            },
            "differentiation": {
                "occupancy_preservation": candidate["codebook_interaction"],
                "vs_r8": [
                    {
                        "lane": "r8_01_directed_l1_transition_nce",
                        "why": candidate["why_not_the_measured_failures"],
                    }
                ],
                "vs_reference_arm": (
                    "the reference arm is the untouched host; this candidate adds one term "
                    "whose gradient reaches the codebook without pulling a transition's two "
                    "codewords together"
                ),
            },
            "provenance": {
                "authored_by": "opencode",
                "authored_at": now_utc(),
            },
            "schema_version": 1,
        }
        try:
            frozen_proposal = freeze(proposal)
        except Exception as exc:  # noqa: BLE001
            (out_dir / "rejected_candidates.json").write_text(
                json.dumps(
                    {"rejected": [{"candidate": candidate, "reason": str(exc)}]}, indent=2, default=str
                )
                + "\n",
                encoding="utf-8",
            )
            continue
        directory = out_dir / frozen_proposal["mechanism_id"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "proposal.json").write_text(
            json.dumps(frozen_proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        frozen.append(
            {
                "mechanism_id": frozen_proposal["mechanism_id"],
                "proposal_sha256": proposal_digest(frozen_proposal),
                "path": str(directory / "proposal.json"),
                "title": candidate["title"],
                "falsifier": candidate["falsifier"],
                "risk": candidate["risk"],
                "codebook_interaction": candidate["codebook_interaction"],
            }
        )
    record = {
        "generated_utc": now_utc(),
        "candidates": frozen,
        "count": len(frozen),
        "tokens": result.get("tokens"),
    }
    (out_dir / "candidates_record.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return record


__all__ = ["CANDIDATE_FIELDS", "CONSTRAINTS", "propose_candidates"]
