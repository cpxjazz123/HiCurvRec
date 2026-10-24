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
        "2. The mechanism is a change to code assignment, codebook geometry or the "
        "reconstruction/commitment objective. Say which Stage2 diagnostics it should move. Use "
        "these FROZEN definitions and do not redefine them: (a) valid L1 positive hit rate = "
        "over the 52855 frozen validation pairs (1 positive + 10 negatives per anchor, seed 42, "
        "history[-1] -> target), the share where the anchor item's first SID token equals the "
        "positive's first token; (b) valid Hard-L1 ER = 10 * positive_hits / negative_hits with "
        "negative_hits counted the same way on the ten negatives, so the Resample reference is "
        "16.38953942232631; (c) level2_distinct_prefixes = |{(k1,k2)}| over all items, "
        "reference 16145; (d) L3 collision = share of items whose (k1,k2,k3) is shared with "
        "another item; (e) L3 refinement gain = mean over items of log2(|bucket(k1,k2)| / "
        "|bucket(k1,k2,k3)|); (f) conditional G3 = mean over the validation pairs of "
        "log2(p_hist(L3 target | anchor) / p_pref(L3 target | target prefix2)) with the "
        "train-transition history term mixed into a Laplace-smoothed (alpha = 1 over 256 codes) "
        "prefix prior.\n"
        "3. Train-split item transitions are legal supervision (CLAUDE.md rule 16). "
        "Validation/test behaviour, future events, prediction-time user context and batch "
        "composition are not.\n"
        "4. Freeze every number: c, tau, epsilon/clamp, the numerically stable arcosh form and "
        "the domain policy. State the c->0 limit of EVERY c-dependent rule: the distance is "
        "4*||x-y||^2, the exponential map is the identity, and the projection radius "
        "(1-epsilon)/sqrt(c) is undefined at c=0, so say what the flat arm applies instead.\n"
        "5. Every loss and target distribution is ONE equation over named tensors, with shapes, "
        "index sets, normalization, reduction, the empty-group rule, and an explicit statement "
        "of how any target-depends-on-parameters circularity is broken.\n"
        "5b. Write the scored distance once, squared: D^2(x,y) = d_c(x,y)^2 = "
        "(1/c) * arcosh(1 + 2c*||x-y||^2/((1-c*||x||^2)*(1-c*||y||^2)))^2. Its exact flat limit "
        "is 4*||x-y||^2, while the distance itself tends to 2*||x-y||; state which power the "
        "objective uses and keep the same power in both arms. Do not claim 4*||x-y||^2 as the "
        "limit of the unsquared distance.\n"
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
        "Independently inspect the frozen candidate, actual trainer source, the four "
        "concrete reviewed test files and independent Agent logs; reject mock-only, "
        "self-referential math, missing DDP/entrypoint evidence and test/hash drift. "
        "The manifest carries the source digest together with the exact recipe that "
        "produced it (the algorithm and the per-file sha256 list), so you can recompute "
        "the digest yourself instead of comparing single file hashes with it. The author "
        "reports in this manifest are the CURRENT evidence; any other Agent session under "
        "the run directory belongs to an earlier attempt against a source that has since "
        "changed, so it is archived history and NOT competing evidence. "
        "Your sandbox has no shell and no torch, so you CANNOT execute these suites; do not "
        "block on that. Execution proof belongs to verification_check, which re-runs all four "
        "suites with the project Stage2 interpreter and refuses on any failure, on zero tests "
        "or on a skip. Judge what static inspection can settle: the oracles are independent, "
        "the DDP suite really gathers per-rank values, the entrypoint suite really calls the "
        "trainer, the hashes match. State in the rationale what you verified statically and "
        "what execution still has to prove. "
        "Reply JSON with candidate_id, source_sha256, proposal_sha256, "
        "test_sha256 mapping and verdict APPROVE_TESTS or BLOCK with rationale. "
        "Keep the rationale under 120 words and finish it; an incomplete rationale "
        "cannot justify either verdict."
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
    return ("ROLE: " + role + "\n" + guidance
            + "\nFROZEN CURRENT-ONLY MANIFEST:\n"
            + json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2)
            + "\n" + instruction
            + " Do not use prior-candidate prompts or any unseen future/test interactions.")
