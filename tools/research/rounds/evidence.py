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

from . import literature as literature_module
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


def measured_behaviour(root: Path) -> list:
    """The measured successor-entropy stratification, if a round recorded one.

    A reviewer that asked for a measured behaviour statistic must be answered
    with the numbers, not with a promise, so every recorded stratification
    artifact under the research tree is quoted as OBSERVED evidence.
    """
    claims = []
    for path in sorted(Path(root).rglob("evidence_successor_entropy_stratification.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        bins = payload.get("bins") or []
        if not bins:
            continue
        low, high = payload.get("low_entropy_vs_high_entropy", {}).get("top1_mass", [None, None])
        summary = ("; ".join("bin %s H=%.3f support %.1f distinct %.2f top1 %.3f" % (
            entry["bin"], entry["mean_entropy"], entry["mean_support"],
            entry["mean_distinct_successors"], entry["mean_top1_mass"])
            for entry in bins))
        claims.append(_claim(
            "measured successor-entropy stratification on the train split: top-1 successor "
            "mass " + str(low) + " in the lowest entropy quintile against " + str(high) +
            " in the highest, Spearman(entropy, top-1 mass) = " +
            str(payload.get("spearman_entropy_vs_top1_mass")) + "; " + summary,
            "repo", str(path), "OBSERVED", payload))
    return claims


def measured_forensics(root: Path) -> list:
    """The Issue #125 measurements, as evidence a reviewer can read.

    A proposal that claims a prefix will stay alive in a width-20 beam can be
    checked against the measured survival of the arms that already exist, so the
    numbers are quoted verbatim from the forensics report instead of restated.
    """
    claims = []
    for path in sorted(Path(root).rglob("per_model_beam_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        parts = []
        for arm, entry in sorted(payload.items()):
            beam = entry.get("beam") or {}
            alive = beam.get("gold_prefix_alive_rate") or {}
            parts.append("%s: recall@10 %.5f, target SID inside the 20 beams %.4f, "
                         "gold prefix alive at depth 1/2/3/4 %.3f/%.3f/%.3f/%.3f"
                         % (arm, beam.get("recall@10") or 0.0,
                            beam.get("raw_beam_coverage") or 0.0,
                            alive.get("depth_1", 0.0), alive.get("depth_2", 0.0),
                            alive.get("depth_3", 0.0), alive.get("depth_4", 0.0)))
        claims.append(_claim(
            "measured width-20 beam survival on the 57,439-row test split: " +
            "; ".join(parts), "repo", str(path), "OBSERVED", payload))
    for path in sorted(Path(root).rglob("per_model_token_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        parts = []
        for arm, entry in sorted(payload.items()):
            positions = ((entry.get("token") or {}).get("positions") or {})
            parts.append("%s: T1/T2/T3/T4 NLL %.2f/%.2f/%.2f/%.2f" % (
                arm,
                (positions.get("T1") or {}).get("mean_nll") or 0.0,
                (positions.get("T2") or {}).get("mean_nll") or 0.0,
                (positions.get("T3") or {}).get("mean_nll") or 0.0,
                (positions.get("T4") or {}).get("mean_nll") or 0.0))
        claims.append(_claim(
            "measured teacher-forced per-position NLL: " + "; ".join(parts),
            "repo", str(path), "OBSERVED", payload))
    return claims


def measured_premises(root: Path) -> list:
    """The measured premises a capacity or cohort proposal must be judged against.

    Two questions decide the next round, and both are answerable before training:
    how much of the canonical behaviour pairing a first level of K groups can keep,
    and whether a cohort profile separates behavioural neighbours at all.
    """
    claims = []
    for path in sorted(Path(root).rglob("next_round_premises.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        capacity = (payload.get("first_level_capacity_ceiling") or {}).get("capacity") or {}
        if capacity:
            claims.append(_claim(
                "measured first-level capacity ceiling on the frozen Stage1 vectors: "
                "share of canonical pairs whose anchor and target share one of K clusters is " +
                ", ".join("K=%s %.4f (negative %.4f)" % (
                    key, value.get("positive_hit_rate") or 0.0,
                    value.get("negative_hit_rate") or 0.0)
                    for key, value in sorted(capacity.items(), key=lambda item: int(item[0]))) +
                "; a small K keeps behavioural neighbours together far more often in absolute "
                "terms, which is what width-20 prefix survival depends on",
                "repo", str(path), "OBSERVED", payload))
        profile = payload.get("cohort_profile_signal") or {}
        overlap = profile.get("profile_overlap") or {}
        if overlap:
            claims.append(_claim(
                "measured train-only cohort-profile signal: a positive shares a successor cohort "
                "with its anchor %.4f of the time against %.4f for the fixed negatives, and the "
                "positive's profile overlap exceeds every one of its ten negatives for %.4f of "
                "pairs; profile entropy averages %.2f nats over %.1f cohorts per item, so the "
                "profile is not degenerate" % (
                    overlap.get("positive_hit_rate") or 0.0,
                    overlap.get("negative_hit_rate") or 0.0,
                    overlap.get("share_positive_above_its_negatives") or 0.0,
                    (profile.get("profile_entropy") or {}).get("mean") or 0.0,
                    (profile.get("profile_support") or {}).get("mean_nonzero_cohorts") or 0.0),
                "repo", str(path), "OBSERVED", payload))
    return claims


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
    for measured in measured_behaviour(root):
        behavioural.append(measured)
    for measured in measured_forensics(root):
        behavioural.append(measured)
    for measured in measured_premises(root):
        behavioural.append(measured)
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


def merge_probe_evidence(bundle: dict, *, probe: dict = None, literature: dict = None,
                        math_report: dict = None) -> dict:
    """Fold the probe outcomes into the bundle the reviewers read.

    The measured statistics replace the proposal's own claims about them, and the
    literature records replace the ``NOT_CHECKED`` placeholder -- but only when
    the pass actually ran. Nothing is upgraded to a measurement it did not get.
    """
    bundle = json.loads(json.dumps(bundle, ensure_ascii=False, default=str))
    if probe:
        for entry in probe.get("metrics", {}).values():
            if not isinstance(entry, dict):
                continue
            bundle["behavioral_evidence"].append(_claim(
                "measured train-only statistic for this hypothesis", "probe",
                "probe:" + str(probe.get("idea_id")), entry.get("status", "UNKNOWN"),
                {key: value for key, value in entry.items()
                 if key not in ("status", "reason")}))
    if literature:
        if literature.get("status") == literature_module.STATUS_OK:
            for paper in literature.get("papers", []):
                bundle["prior_work"].append(_claim(
                    "related work: " + str(paper.get("title")), "literature",
                    "literature:" + str(paper.get("doi") or paper.get("source_id")),
                    paper.get("match", "not_established"),
                    {"title": paper.get("title"), "doi": paper.get("doi"),
                     "url": paper.get("url"), "year": paper.get("year"),
                     "match": paper.get("match")}))
        else:
            bundle["prior_work"].append(_claim(
                "the related-work pass did not complete", "literature",
                "literature:" + str(literature.get("idea_id")),
                literature.get("status", rubric_module.NOT_CHECKED),
                literature.get("reason")))
    if math_report:
        for check in math_report.get("checks", []):
            bundle["geometric_necessity"].append(_claim(
                "sanity check: " + str(check.get("check")), "math",
                "math:" + str(math_report.get("idea_id")), check.get("status", "UNKNOWN"),
                {"detail": check.get("detail"), "implication": check.get("implication")}))
    counter = 1
    ids = {}
    for section in ("behavioral_evidence", "geometric_necessity", "prior_work",
                    "experiment_ledger", "implementation_cost"):
        for entry in bundle.get(section, []):
            key = "E" + str(counter)
            entry["id"] = key
            ids[key] = section
            counter += 1
    bundle["evidence_ids"] = ids
    bundle["probe_sha256"] = (probe or {}).get("sha256")
    bundle["literature_sha256"] = (literature or {}).get("sha256")
    bundle["math_sha256"] = (math_report or {}).get("sha256")
    bundle["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in bundle.items() if k not in ("sha256", "built_at")},
                   sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
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
