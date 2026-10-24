"""Machine-checkable candidate identity, user-behavior grounding and SID policy."""
import json
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
    check_falsifier(candidate)


def check_review_identity(candidate_path: Path, manifest: dict, prompt: str) -> None:
    """Fail before agent launch; never count a stale review as a scientific verdict."""
    expected = digest_file(Path(candidate_path))
    ident = str(manifest.get("candidate_id") or "")
    if not ident or manifest.get("candidate_sha256") != expected:
        raise ValueError("candidate identity or digest does not match staged bytes")
    payload = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("candidate_id") != ident:
        raise ValueError("candidate identity differs from the actual frozen proposal")
    if ident not in prompt or expected not in prompt:
        raise ValueError("review prompt does not bind current candidate ID and SHA")
    forbidden = tuple(manifest.get("forbidden_stale_formulas", ()))
    if any(text and text in prompt for text in forbidden):
        raise ValueError("stale candidate formula found in review prompt")

FALSIFIER_KEYS = ("stage2_diagnostics", "stage3_behavioral_test", "classification")
CLASSIFICATION_KEYS = ("falsified_if", "passed_if", "mixed_if")


def check_falsifier(candidate: dict) -> None:
    """The falsifier must be decidable from numbers registered exactly once.

    A proposal that names a threshold twice, omits the mixed/null rule or leaves
    a diagnostic without a direction or minimum effect cannot be decided by two
    independent readers, and the review blocks it. Reject it here instead of
    spending a whole research run to find out.
    """
    falsifier = candidate.get("falsifier")
    if not isinstance(falsifier, dict):
        raise ValueError("falsifier must be an object")
    missing = [key for key in FALSIFIER_KEYS if not falsifier.get(key)]
    if missing:
        raise ValueError("falsifier is incomplete: " + ", ".join(missing))
    classification = falsifier["classification"]
    if not isinstance(classification, dict) or any(
            not classification.get(key) for key in CLASSIFICATION_KEYS):
        raise ValueError("falsifier classification must define " + ", ".join(CLASSIFICATION_KEYS))
    diagnostics = falsifier["stage2_diagnostics"]
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ValueError("falsifier requires at least one Stage2 diagnostic")
    for entry in diagnostics:
        if not isinstance(entry, dict) or any(
                not entry.get(key) for key in ("name", "direction", "minimum_effect", "seed_rule")):
            raise ValueError("each Stage2 diagnostic needs name, direction, minimum_effect, seed_rule")
    behavioural = falsifier["stage3_behavioral_test"]
    if not isinstance(behavioural, dict) or any(
            not behavioural.get(key) for key in ("metric", "direction", "minimum_effect", "seed_rule")):
        raise ValueError("stage3_behavioral_test needs metric, direction, minimum_effect, seed_rule")
    if behavioural["metric"] not in ("test_R@10", "test_NDCG@10"):
        raise ValueError("stage3 behavioural metric must be test_R@10 or test_NDCG@10")
    registered = {entry["name"] for entry in diagnostics}
    text = " ".join(str(classification.get(key)) for key in CLASSIFICATION_KEYS)
    # Only bare snake_case identifiers can be diagnostic names. Anything carrying
    # a dot or slash is a field reference (stage3_behavioral_test.metric), and an
    # all-caps token is an operator (OR/AND/NOT), so neither is a name claim.
    words = {word.strip("()[],") for word in text.replace("+", " ").split()}
    # A classification may name a registered diagnostic or any key that the
    # falsifier itself defines; anything else written as a bare snake_case
    # identifier is a claim about a diagnostic that was never registered.
    allowed = set(registered) | _keys_of(falsifier)
    suspicious = sorted(
        word for word in words
        if "_" in word and "." not in word and "/" not in word and not word.isupper()
        and word not in allowed
    )
    if suspicious:
        raise ValueError("classification references unregistered diagnostics: "
                         + ", ".join(suspicious))
    if "NULL" not in text and "null" not in text:
        raise ValueError("classification must define the NULL outcome")

def _keys_of(value) -> set:
    """Every mapping key reachable in a nested JSON value."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key))
            found |= _keys_of(item)
    elif isinstance(value, list):
        for item in value:
            found |= _keys_of(item)
    return found
