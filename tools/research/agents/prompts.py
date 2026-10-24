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
        "Read the MEASURED behavior_evidence and literature_evidence below. "
        "Choose one demonstrated next-item behavior failure, justify why NEGATIVE CURVATURE "
        "might help that failure against fixed-capacity Euclidean/graph controls, then define "
        "one sound numeric hyperbolic operator and a falsifiable SID and ranking prediction. "
        "Do NOT invent a mechanism disconnected from the measured behavior. "
        "Write a JSON object with candidate_id, behavior_problem_id, "
        "behavior_evidence_sha256, literature_evidence_sha256, next_item_hypothesis, "
        "math_definition, hyperbolic_necessity, euclidean_control, effect_site, "
        "real_trigger_measurement, training_only_inputs, stable_global_sid=true, "
        "protocol={latent_dim:32,levels:3,codes_per_level:256,hyperparameter_sweep:false,"
        "uses_future_events:false}, falsifier."
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
        "Reply JSON with candidate_id, source_sha256, ddp_test_source and "
        "entrypoint_test_source."
    ),
    "VERIFY_REVIEW": (
        "Independently inspect the frozen candidate, actual trainer source, the four "
        "concrete reviewed test files and independent Agent logs; reject mock-only, "
        "self-referential math, missing DDP/entrypoint evidence and test/hash drift. "
        "Reply JSON with candidate_id, source_sha256, proposal_sha256, "
        "test_sha256 mapping and verdict APPROVE_TESTS or BLOCK with rationale."
    ),
    "REVIEW": (
        "Read ONLY the candidate in THIS manifest. Audit actual behavior source, "
        "next-item causal link, math types, real nontrivial hyperbolic effect, numerical "
        "domain, same-capacity control, user-future leakage, frozen model protocol and novelty. "
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
