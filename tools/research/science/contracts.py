"""Machine-checkable candidate identity, user-behavior grounding and SID policy."""
from pathlib import Path

from ..core.evidence import digest_file


REQUIRED = (
    "candidate_id", "behavior_problem_id", "behavior_evidence_sha256",
    "literature_evidence_sha256", "next_item_hypothesis", "math_definition",
    "hyperbolic_necessity", "euclidean_control", "effect_site",
    "real_trigger_measurement", "training_only_inputs", "stable_global_sid",
    "protocol", "falsifier",
)


def check_candidate(candidate: dict) -> None:
    missing = [key for key in REQUIRED if not candidate.get(key)]
    if missing:
        raise ValueError("incomplete behavior-first contract: " + ", ".join(missing))
    if candidate["stable_global_sid"] is not True:
        raise ValueError("SID must be globally fixed per item")
    policy = candidate["protocol"]
    if not isinstance(policy, dict) or (policy.get("latent_dim"),
       policy.get("levels"), policy.get("codes_per_level")) != (32, 3, 256):
        raise ValueError("frozen Stage2 SID capacity mismatch")
    if policy.get("hyperparameter_sweep") or policy.get("uses_future_events"):
        raise ValueError("sweep/test leakage forbidden")


def check_review_identity(candidate_path: Path, manifest: dict, prompt: str) -> None:
    """Fail before agent launch; never count a stale review as a scientific verdict."""
    expected = digest_file(Path(candidate_path))
    ident = str(manifest.get("candidate_id") or "")
    if not ident or manifest.get("candidate_sha256") != expected:
        raise ValueError("candidate identity or digest does not match staged bytes")
    if ident not in prompt or expected not in prompt:
        raise ValueError("review prompt does not bind current candidate ID and SHA")
    forbidden = tuple(manifest.get("forbidden_stale_formulas", ()))
    if any(text and text in prompt for text in forbidden):
        raise ValueError("stale candidate formula found in review prompt")
