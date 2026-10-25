"""The frozen evidence bundle handed to every reviewer (Issue #122).

A review is only as good as what it can see, so the bundle is assembled once,
before anyone scores, and the same bytes go to all three reviewers. Every claim
carries its source and a status: a measured value, a file, or an explicit
``UNKNOWN`` / ``NOT_CHECKED``. Nothing is invented to fill a section.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import rubric as rubric_module

#: The frozen data contract the mechanisms are built on. These are facts about
#: the repository, not measurements the bundle has to invent.
DATA_CONTRACT = {
    "stage2_train": "results/stage0_build_parquet/train.parquet",
    "stage1_embeddings": "stage1_GeneEmbedding/output/sentence_t5.npy",
    "baseline_tree": "stage2_RQ-VAE/TIGER_RQ-VAE",
    "valid_pairs": "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz",
    "stage2_epochs": 3000,
    "stage2_recipe": "4 GPU DDP, total batch 4096, 0.12-0.16 s/epoch",
    "stage3_recipe": "NUM_EPOCHS=150, EVAL_INTERVAL=5, EVAL_START_EPOCH=75, "
                     "SEED=42, BEAM_SIZE=20",
}


def _claim(claim: str, source_type: str, source: str, status: str = "OBSERVED",
           value=None) -> dict:
    return {"claim": claim, "source_type": source_type, "source": source,
            "status": status, "value": value}


def measured_outcomes(root: Path) -> list:
    """Every real Stage 3 outcome on disk, with the recipe it was produced under.

    A number is only comparable when the recipe matches the frozen one, so the
    recipe is read from the run's own metrics and carried with the number.
    """
    root = Path(root)
    outcomes = []
    for test_file in sorted(root.glob("results/stage3_T5Train/*/logs/*/*/test_final.json")):
        arm = test_file.relative_to(root / "results/stage3_T5Train").parts[0]
        metrics = sorted(test_file.parent.parent.parent.glob("**/training_metrics.jsonl"))
        start = None
        for path in metrics:
            try:
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    row = json.loads(line)
                    if row.get("event") == "train_start":
                        start = row
                        break
            except (OSError, ValueError):
                continue
            if start:
                break
        try:
            payload = json.loads(test_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        launch = (start or {}).get("launch_mode")
        batch = (start or {}).get("global_train_batch")
        comparable = launch in ("ddp4", "ddp") and batch == 4096
        outcomes.append({
            "arm": arm,
            "test_recall_at_10": payload.get("test_recall@10"),
            "test_ndcg_at_10": payload.get("test_ndcg@10"),
            "n_eval": payload.get("n_eval"),
            "launch_mode": launch,
            "global_batch": batch,
            "comparable_to_frozen": bool(comparable),
            "source": str(test_file.relative_to(root)),
            "source_type": "measured",
        })
    return outcomes


def build_bundle(idea: dict, *, root: Path, round_state: dict = None,
                 prior_rounds: list = None) -> dict:
    """The evidence bundle of one proposal, assembled from what exists.

    Sections that cannot be grounded say so; a reviewer that wants more can ask
    for one bounded clarification instead of guessing.
    """
    root = Path(root)
    outcomes = measured_outcomes(root)
    ledger = []
    for outcome in outcomes:
        ledger.append(_claim(
            "Stage 3 test_R@10 " + str(outcome["test_recall_at_10"]) + " for " +
            outcome["arm"],
            "measured", outcome["source"], "OBSERVED", outcome))
    for earlier in (prior_rounds or []):
        for entry in earlier.get("explored", []):
            ledger.append(_claim("a prior round explored " + str(entry), "repo",
                                 "results/research/issue121/rounds", "OBSERVED"))
    behavioural = []
    for field, label in (("behavior_signal", "the proposed training-only signal"),
                         ("behavior_computation", "how it is computed"),
                         ("leakage_boundary", "why it cannot leak")):
        value = str(idea.get(field, "")).strip()
        behavioural.append(_claim(label, "proposal", "idea:" + str(idea.get("idea_id")),
                                  "PROPOSED" if value else rubric_module.UNKNOWN, value or
                                  rubric_module.UNKNOWN))
    behavioural.append(_claim("the frozen training data contract", "repo",
                              DATA_CONTRACT["stage2_train"], "OBSERVED"))
    geometric = []
    for field, label in (("hyperbolic_operator", "the curved operation"),
                         ("hyperbolic_location", "where it acts"),
                         ("coupling_equation", "the coupling equation")):
        value = str(idea.get(field, "")).strip()
        geometric.append(_claim(label, "proposal", "idea:" + str(idea.get("idea_id")),
                                "PROPOSED" if value else rubric_module.UNKNOWN,
                                value or rubric_module.UNKNOWN))
    geometric.append(_claim("a matched Euclidean control is specified", "proposal",
                            "idea:" + str(idea.get("idea_id")),
                            "PROPOSED" if idea.get("ablation_geometry_off") else
                            rubric_module.UNKNOWN))
    cost = [ _claim("declared cost of the frozen run", "proposal",
                    "idea:" + str(idea.get("idea_id")),
                    "PROPOSED" if idea.get("cost_estimate") else rubric_module.UNKNOWN,
                    idea.get("cost_estimate") or rubric_module.UNKNOWN),
             _claim("a timed real-data probe", "repo", "no probe recorded",
                    rubric_module.UNKNOWN) ]
    bundle = {
        "schema": 1,
        "built_at": time.time(),
        "idea_id": idea.get("idea_id"),
        "rubric_version": rubric_module.load_rubric()["version"],
        "behavioral_evidence": behavioural,
        "geometric_necessity": geometric,
        "prior_work": [
            _claim("literature search for this proposal", "agent",
                   "no literature pass was run in this round", rubric_module.NOT_CHECKED),
            _claim("proposal novelty statement", "proposal",
                   "idea:" + str(idea.get("idea_id")), "PROPOSED",
                   str(idea.get("novelty", ""))[:400]),
        ],
        "experiment_ledger": ledger,
        "implementation_cost": cost,
        "data_contract": dict(DATA_CONTRACT),
        "evidence_ids": {},
    }
    counter = 1
    for section in ("behavioral_evidence", "geometric_necessity", "prior_work",
                    "experiment_ledger", "implementation_cost"):
        for entry in bundle[section]:
            key = "E" + str(counter)
            entry["id"] = key
            bundle["evidence_ids"][key] = section
            counter += 1
    bundle["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in bundle.items() if k not in ("sha256", "built_at")},
                   sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return bundle


def render_bundle(bundle: dict, limit: int = 6000) -> str:
    """The bundle as the plain text a reviewer reads."""
    lines = ["EVIDENCE BUNDLE " + str(bundle.get("sha256", ""))[:16],
             "rubric " + str(bundle.get("rubric_version", ""))]
    for section in ("behavioral_evidence", "geometric_necessity", "prior_work",
                    "experiment_ledger", "implementation_cost"):
        lines.append("")
        lines.append(section.upper() + ":")
        for entry in bundle.get(section, []):
            value = entry.get("value")
            if isinstance(value, dict):
                value = json.dumps(value, ensure_ascii=False)[:220]
            lines.append("  [" + str(entry.get("id")) + "] " + str(entry.get("status")) +
                         " | " + str(entry.get("claim")) +
                         (" -> " + str(value)[:220] if value not in (None, "") else ""))
    text = "\n".join(lines)
    return text[:limit]
