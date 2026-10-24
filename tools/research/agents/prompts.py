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
        "reconstruction/commitment objective. Say which Stage2 diagnostics it should move: "
        "validation Hard-L1 ER, distinct two-token prefixes, and L3 collision, refinement gain "
        "and conditional G3.\n"
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
        "6. Controls are runnable arms with the same architecture, parameter count, "
        "initialisation, optimizer, schedule, seeds and budget: the Euclidean arm differs only "
        "in the metric (4*||x-y||^2), and a graph arm trains WITH the graph signal so the "
        "comparison isolates geometry.\n"
        "7. If a level conditions on earlier codes, name exactly which earlier codes.\n"
        "8. The falsifier must be mechanical: per-diagnostic direction, minimum effect, seed "
        "combination rule, and how mixed or null outcomes are classified.\n"
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
        "Read ONLY the candidate in THIS manifest. Audit the next-item causal link, math "
        "types, real nontrivial hyperbolic effect, numerical domain, same-capacity control, "
        "user-future leakage, frozen model protocol and novelty of the PROPOSAL. "
        "Scope: the candidate is not implemented yet at this stage. Do not require the "
        "mechanism to exist in any repository tree, and do not treat an existing Stage2 tree "
        "as this candidate's implementation; implementation identity, hash binding and "
        "execution evidence are checked later by the independent test authors, the four "
        "suites and verification_check. Judge whether the proposed Stage2 SID mechanism is "
        "implementable inside the frozen contract and whether its falsifier is decidable. "
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
