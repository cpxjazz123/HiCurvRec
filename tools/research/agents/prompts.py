"""Review a specific current business problem, never a cached old formula."""
import json
from pathlib import Path

ROLE_GUIDANCE = {
    "LITERATURE": (
        "Find independently verifiable FULL-TEXT research for the intersection "
        "of user next-item behavior, hyperbolic geometry and generative semantic IDs. "
        "Provide DOI/arXiv/title, exact overlap and FULL_TEXT/ABSTRACT/UNKNOWN evidence labels. "
        "Write a JSON object with sources, behavior_problem_candidates and limitations."
    ),
    "GEOMETRY": (
        "Read the MEASURED behavior_evidence and literature_evidence below. Pick ONE "
        "demonstrated next-item failure, justify negative curvature for it, and freeze ONE "
        "mechanism as a Stage2 item-SID builder: content embedding -> 32-d latent -> 3 levels "
        "of 256 codes -> one global three-token SID per item. The downstream T5 Stage3 ranks "
        "next items; Stage2 does not. Every requirement below is mandatory.\n"
        "1. No recommender parts inside the candidate: no prefix Transformer, no beam search, "
        "no NDCG/Recall, no user bootstrap. No new training protocol: epochs, batch, schedule, "
        "seeds and codebook initialisation are frozen.\n"
        "1b. Measured constraint on the assignment: the admission reference family "
        "(KMeans-initialised codebooks with Sinkhorn-balanced assignment) reaches ER "
        "16.4-20.8 and 15k-19k level-2 prefixes, while two real candidates with a "
        "distance-based hard argmin landed at ER 2.9-6.1 and 6k-8k prefixes and were "
        "refused. A candidate that keeps a calibrated, balanced assignment and changes only "
        "the metric is therefore far more likely to reach the gate than one that replaces "
        "the assignment rule; if you do replace it, say how prefix diversity is preserved. "
        "2. The mechanism is a change to code assignment, codebook geometry or the "
        "reconstruction/commitment objective. Say which Stage2 diagnostics it should move. Use "
        "these FROZEN definitions and do not redefine them: (a) valid L1 positive hit rate = "
        "over the 52855 frozen validation pairs (1 positive + 10 negatives per anchor, seed 42, "
        "history[-1] -> target), the share where the anchor item's first SID token equals the "
        "positive's first token; (b) valid Hard-L1 ER = 10 * positive_hits / negative_hits with "
        "negative_hits counted the same way on the ten negatives, so the Resample reference is "
        "(see FROZEN_THRESHOLDS below); (c) level2_distinct_prefixes = |{(k1,k2)}| over all "
        "items; (d) L3 collision = share of items whose (k1,k2,k3) is shared with "
        "another item; (e) L3 refinement gain = mean over items of log2(|bucket(k1,k2)| / "
        "|bucket(k1,k2,k3)|); (f) conditional G3, defined to the estimator: with alpha = 1 and "
        "K = 256, for each validation pair (anchor, target), p_pref(l) = (number of items that "
        "share the target's level-two prefix and have third code l, minus one when l is the "
        "target's own third code, plus alpha) / (bucket size minus one plus alpha*K); "
        "p_hist(l) = (train-only longest-prefix successor count of the anchor restricted to "
        "successors that share the target's level-two prefix and have third code l, plus "
        "alpha*p_pref(l)) / (that restricted successor total plus alpha); G3 is the mean over "
        "pairs of log2(p_hist(target third code) / p_pref(target third code)); a pair with no "
        "restricted successor contributes 0 and a non-finite pair is NULL.\n"
        "3a0. One gradient rule per tensor: state for every tensor whether it is frozen "
        "(stop-gradient) or differentiable, and never claim both. An assignment frozen for the "
        "epoch cannot carry a gradient, so an anti-collapse term written on that assignment is "
        "constant and cannot train anything; either the assignment is differentiable in the "
        "loss that needs it, or the term is computed on a differentiable quantity.\n"
        "3a. A balanced assignment is allowed ONLY if it is computed deterministically over "
        "the whole training item set and frozen for the epoch: a Sinkhorn or other "
        "normalisation that depends on which items share a batch is forbidden as "
        "batch-composition signal and will be rejected. State how the balanced assignment is "
        "computed, over which population, and how it preserves level-two prefix diversity.\n"
        "3. Train-split item transitions are legal supervision (CLAUDE.md rule 16). "
        "Validation/test behaviour, future events, prediction-time user context and batch "
        "composition are not.\n"
        "4. Freeze every number: c, tau, epsilon/clamp, the numerically stable arcosh form and "
        "the domain policy. State the c->0 limit of EVERY c-dependent rule: the distance is "
        "4*||x-y||^2, the exponential map is the identity, and the projection radius "
        "(1-epsilon)/sqrt(c) is undefined at c=0, so say what the flat arm applies instead.\n"
        "4b. Stabilise monotonically: clamp the arcosh argument to at least 1.0, never to "
        "1+delta, because a fixed floor makes the distance constant below the floor and then "
        "jump, which the preflight reports as NON_MONOTONE_CLAMP. Use log1p for the stable "
        "form and keep the curvature-dependent projection out of the scored region's "
        "monotonicity.\n"
        "4e. State the assignment rule once, without contradiction, and make sure the hard "
        "labels and the transport Q are UNIQUELY determined by that rule and the frozen "
        "constants: if two rules are written, the proposal is not recomputable and will be "
        "blocked. Name the exact vector subtracted from the residual at each level (the raw "
        "code vector, its projected image, or the level's residual), in the order the steps "
        "happen, so two readers cannot implement two different residual chains.\n"
        "4d. Your assignment rule must be EXACTLY computable at the frozen scale: the whole "
        "training population is 24556 items over 256 codes, so a rule that needs an optimal "
        "transportation solve at that size is not implementable and will be blocked. Say how "
        "the hard labels are obtained by a rule whose result can be reproduced exactly, and "
        "how the lexicographic or tie-break requirement is satisfied by that rule. An "
        "approximation (auction, greedy repair, local swaps) is not acceptable: state the "
        "exact rule or choose a different assignment rule.\n"
        "4c. Every loss must have a real gradient path to the parameters it is meant to "
        "train. A loss computed only from one-hot argmin assignments has none; either relax "
        "the assignment (softmax, Sinkhorn) or declare an explicit straight-through estimator "
        "and say what gradient it passes.\n"
        "5a. Declare math_definition.objective.anti_collapse: the term or mechanism that keeps "
        "the latent informative, and where it enters the loss. An objective that can be "
        "lowered by mapping every item to the same code has already collapsed a real "
        "3000-epoch run into one distinct SID, and it will be rejected before GPU time. "
        "5. Every loss and target distribution is ONE equation over named tensors, with shapes, "
        "index sets, normalization, reduction, the empty-group rule, and an explicit statement "
        "of how any target-depends-on-parameters circularity is broken.\n"
        "5b. Write the scored distance once, squared: D^2(x,y) = d_c(x,y)^2 = "
        "(1/c) * arcosh(1 + 2c*||x-y||^2/((1-c*||x||^2)*(1-c*||y||^2)))^2. Its exact flat limit "
        "is 4*||x-y||^2, while the distance itself tends to 2*||x-y||; state which power the "
        "objective uses and keep the same power in both arms. Do not claim 4*||x-y||^2 as the "
        "limit of the unsquared distance.\n"
        "5b2. Do NOT apply an exponential map in the scoring path. Score the projected latent "
        "and the projected code vectors directly. An exp_0 makes the curved arm's coordinates "
        "differ from the flat arm's identity coordinates, so the arms no longer differ only in "
        "the metric and the review rejects the control (this has blocked three proposals). The "
        "curvature then appears where it belongs: in the distance formula on identical "
        "coordinates.\n"
        "5c. Both arms project every vector that enters the distance into the SAME fixed bounded "
        "region ||x|| <= R with one frozen R < 1, so the feasible domain and the representation "
        "map are identical and the arms differ only in the distance formula. Any residual update "
        "is projected before it is scored; state that order.\n"
        "6. The curvature and flat arms are ONE implementation with curvature as a parameter: "
        "the flat arm is the c->0 limit of the same code path. Same architecture, parameter "
        "count, initialisation, optimizer, schedule, seeds and budget; a graph arm, if any, "
        "trains WITH the graph signal so the comparison isolates geometry.\n"
        "7. If a level conditions on earlier codes, name exactly which earlier codes.\n"
        "8. The falsifier is ONE object with exactly these keys and no numeric threshold "
        "repeated anywhere else in the JSON: stage2_diagnostics=[{name, direction, "
        "minimum_effect, seed_rule}], stage3_behavioral_test={metric (test_R@10 or NDCG@10, "
        "chosen once), direction, minimum_effect, seed_rule}, classification={falsified_if, "
        "passed_if, mixed_if} where each entry is a boolean expression over those registered "
        "thresholds only. If another field mentions a threshold, refer to the falsifier "
        "instead of restating a number. State the operation order explicitly for every "
        "residual step (map, project, assign, subtract) so no reader can apply the projection "
        "at a different point. Define a NULL outcome rule: if a seed-wise or aggregated change "
        "is undefined or non-finite (zero denominator, missing evaluation, empty stratum), it "
        "is classified NULL, and a comparison with any NULL seed is inconclusive rather than "
        "passed or falsified.\n"
        "Shape reference (structure only; every number below is an example, and the "
        "mechanism must be your own): math_definition = {geometry_constants: \"freeze c=1, "
        "clamp arcosh at 1.0\", operator: {family: \"poincare_squared_distance\", "
        "curvature_c: 1.0, delta: 0.0, radius: 0.99999, flat_limit: \"4*||x-y||^2\"}, "
        "objective: {anti_collapse: \"<the term and where it enters>\", loss_equations: [...]}, "
        "loss_equations: [...]}; falsifier = {stage2_diagnostics: [{name: "
        "\"valid_l1_hard_er\", direction: \"curved minus flat > 0\", minimum_effect: 0.1, "
        "seed_rule: \"...\"}], stage3_behavioral_test: {metric: \"test_R@10\", direction: "
        "\"curved minus flat > 0\", minimum_effect: 0.005, seed_rule: \"...\"}, "
        "classification: {falsified_if: \"...\", passed_if: \"...\", mixed_if: "
        "\"otherwise, or any seed is NULL\"}}, and euclidean_control states the one operational "
        "fact the reviewer keeps checking: the same implementation and the same fixed "
        "projection radius for BOTH arms, with only D_c^2 replaced by 4*||x-y||^2. "
        "Diagnostic names must be registry ids or "
        "documented labels; thresholds live only in the falsifier. "
        "Write JSON with candidate_id, behavior_problem_id, behavior_evidence_sha256, "
        "literature_evidence_sha256, next_item_hypothesis, math_definition, hyperbolic_necessity, "
        "euclidean_control, effect_site, real_trigger_measurement, training_only_inputs, "
        "stable_global_sid=true, protocol={latent_dim:32,levels:3,codes_per_level:256,"
        "hyperparameter_sweep:false,uses_future_events:false}, falsifier."
    ),
    "MATH_VERIFY": (
        "Independently derive numeric hyperbolic reference values from this candidate's "
        "frozen mathematics; write a concrete failing-first unittest test_math.py proposal "
        "without calling the candidate implementation to compute expected results. "
        "Reply JSON with candidate_id, source_sha256, test_source and falsifiers."
    ),
    "GRADIENT_VERIFY": (
        "Independently author an executable test_gradient.py for true PyTorch gradient "
        "flow, finite differences/gradcheck where appropriate, nonzero targeted "
        "gradients and NaN/graph-disconnection negatives; no trivially true assertions. "
        "Reply JSON with candidate_id, source_sha256 and test_source."
    ),
    "RUNTIME_VERIFY": (
        "Author test_ddp.py and test_entrypoint.py that really exercise multi-rank "
        "gradient/parameter consistency and the source tree's actual trainer mechanism "
        "call and optimizer update; never accept simulations or one rank as DDP. "
        "The entrypoint suite must CALL the real trainer entry point (trainer.main() or the "
        "real training loop) on a synthetic train-only fixture and assert what that run "
        "really produced (the checkpoint it wrote, the SID it exported, the mechanism in "
        "the loss path); a reconstructed model/optimizer step plus substring checks on the "
        "source are NOT entrypoint evidence. The DDP suite must let every rank compute its "
        "OWN value and then GATHER the per-rank values (torch.distributed.all_gather) and "
        "compare the gathered list; a check that broadcasts rank 0's value and then "
        "compares is vacuous, because the broadcast overwrites the receiver's buffer. "
        "The tree directory itself must be on sys.path, because the trainer imports its own "
        "model package as a top-level module. "
        "Reply JSON with candidate_id, source_sha256, ddp_test_source and "
        "entrypoint_test_source."
    ),
    "VERIFY_REVIEW": (
        "Independently inspect the frozen candidate, the actual trainer source and the four "
        "committed test suites. You are a STATIC reviewer. The manifest's source and test "
        "hashes are given as identity only: do NOT block because you cannot run commands or "
        "recompute hashes yourself, because hash recomputation and suite execution are "
        "performed deterministically by verification_check immediately after your verdict. "
        "Judge the substance: does each suite contain independent numeric or behavioural "
        "oracles rather than restatements of the implementation; does the gradient suite use "
        "the real model and batch; does the DDP suite spawn real ranks that gather their own "
        "gradients and post-step parameters; does the entrypoint suite call the real trainer "
        "entry point and check what it actually produced; is the mechanism the proposal froze "
        "the one the source implements; and is any suite trivially true, skipped or dependent "
        "on the implementation to compute its own expected value? "
        "Reply JSON with candidate_id, source_sha256, proposal_sha256, test_sha256 mapping and "
        "verdict APPROVE_TESTS or BLOCK with rationale. Keep the rationale under 120 words."
    ),
    "REVIEW": (
        "Read ONLY the candidate in THIS manifest. Audit its next-item causal link as a "
        "design claim, its math types, its real nontrivial hyperbolic effect, its numerical "
        "domain, its same-capacity control, user-future leakage, the frozen model protocol "
        "and the behavioural test it registers.\n"
        "Scope rules. (1) The candidate is not implemented yet: do not require the mechanism "
        "to exist in any repository tree, and do not treat an existing Stage2 tree as this "
        "candidate's implementation; implementation identity and execution evidence belong to "
        "the independent test authors, the four suites and verification_check.\n"
        "(2) Stage2 builds semantic IDs and never ranks. Do NOT block because the declared "
        "Stage2 diagnostics cannot by themselves prove a next-item gain: the behavioural "
        "question is answered downstream by Stage3 test_R@10. What you must check is that the "
        "candidate registers that behavioural test (metric, direction, minimum effect, seed "
        "combination, and how mixed or null outcomes are classified) and that its Stage2 "
        "diagnostics are the right leading indicators (validation Hard-L1 ER, distinct "
        "two-token prefixes, L3 statistics).\n"
        "(3) DO block on: ill-typed or unsound math, an undefined numerical domain or flat "
        "limit, a control that differs in more than the metric, use of validation/test "
        "behaviour, future events or prediction-time user context, a loss or target that "
        "cannot be recomputed uniquely, or a falsifier that cannot be decided mechanically.\n"
        "Write JSON with candidate_id, candidate_sha256, verdict (FATAL/MATERIAL/CPU_ELIGIBLE/"
        "PARKED), blockers and reproducible evidence. The right to CPU study is NOT GPU approval."
    ),
}

#: The suites are executed by `python -m unittest discover -s <dir> -p test_<suite>.py`,
#: so a suite must be a unittest module. A pytest-style function is discovered as ZERO
#: tests and the suite is refused; every authoring role is told the contract.
UNITTEST_CONTRACT = (
    "The test source must be a self-contained unittest module: import unittest, define "
    "one or more unittest.TestCase subclasses with test_* methods, and contain no pytest "
    "fixtures (no tmp_path, no monkeypatch) and no pytest-only imports. It is executed as "
    "`python -m unittest discover -s <dir> -p test_<suite>.py -v`, which collects only "
    "unittest.TestCase classes, so a plain test function would run ZERO tests and be "
    "refused. It must import the candidate implementation from the source tree by path."
)


def build_prompt(role: str, manifest: dict, output_name: str, root=None) -> str:
    if role not in ROLE_GUIDANCE:
        raise ValueError("undefined role")
    if not manifest.get("task_id") or not manifest.get("inputs_sha256"):
        raise ValueError("unbound task/inputs")
    if role in ("GEOMETRY", "REVIEW") and not manifest.get("behavior_evidence_sha256"):
        raise ValueError("missing behavior evidence")
    if role == "REVIEW" and not (manifest.get("candidate_id")
                                  and manifest.get("candidate_sha256")):
        raise ValueError("candidate identity missing")
    # Name the ABSOLUTE output path. A relative path is resolved against the git
    # root, which is the repository, so the Agent would write its answer outside
    # the workspace it is allowed to write in.
    task_id = manifest["task_id"]
    if root is not None:
        destination = str(Path(root) / "agents" / task_id / output_name)
        instruction = ("Write the answer to " + destination + " (this absolute path). "
                       "Write nowhere else.")
    else:
        destination = output_name
        instruction = ("Write the answer to " + destination + " in your working directory.")
    guidance = ROLE_GUIDANCE[role]
    if role in ("MATH_VERIFY", "GRADIENT_VERIFY", "RUNTIME_VERIFY"):
        guidance = guidance + "\n" + UNITTEST_CONTRACT
    if role in ("GEOMETRY", "REVIEW"):
        # The numbers come from the contract compiler, so a prompt can never
        # disagree with the gate that enforces them.
        from ..science import contract as _contract
        guidance = guidance + (
            "\nFROZEN_THRESHOLDS: valid_l1_hard_er strictly above "
            f"{_contract.THRESHOLDS['valid_l1_hard_er_strict_gt']}; "
            f"level2_distinct_prefixes strictly above "
            f"{_contract.THRESHOLDS['level2_prefixes_strict_gt']}; "
            f"schema_version {_contract.SCHEMA_VERSION}.")
    return ("ROLE: " + role + "\n" + guidance
            + "\nFROZEN CURRENT-ONLY MANIFEST:\n"
            + json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2)
            + "\n" + instruction
            + " Do not use prior-candidate prompts or any unseen future/test interactions.")
