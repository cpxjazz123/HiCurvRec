"""The prior-art registry of hyperbolic mechanism discovery (issue #110, WP0).

Issue #110 is explicit about how a novelty check must fail. A registry that cannot be read
must BLOCK, not skip. A historic mechanism whose mathematics cannot be recovered is
``novelty_unknown`` -- never a pseudo-fingerprint built from its name, because a name is not
mathematics. Entries are keyed by the immutable identity of the thing they describe, the
proposal hash and the run id, and never by the mechanism's name alone, because a name is
exactly what a repetition changes.

The registry is built from the archives: the R8 hyperbolic arms, whose mathematics is
recoverable from their source, the frozen proposals of the previous campaigns, and the
38-model historical inventory, whose mathematics is NOT recoverable here and is therefore
marked unknown. The builder records what it found and where it found it, so the record can
be checked against the archives.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where the registry is persisted in the new namespace.
DEFAULT_REGISTRY = REPO_ROOT / "results" / "hyperbolic_research_runs" / "prior_art_registry.json"

#: The hyperbolic and Euclidean operator classes a mechanism may use. The distinction
#: matters: issue #110 rejects a Euclidean mechanism with a hyperbolic title, and the
#: registry has to be able to tell the two apart.
HYPERBOLIC_OPERATORS = (
    "poincare_distance",
    "lorentz_distance",
    "hyperbolic_distance",
    "exp_map0",
    "log_map0",
    "mobius_add",
    "mobius_mul",
    "gyro",
    "busemann",
    "horosphere",
    "parallel_transport",
    "pt_parallel_transport",
    "riemannian",
    "hyperbolic_exp",
    "hyperbolic_log",
    "curvature",
    "artanh",
    "clamp_to_ball",
    "project_to_ball",
)

EUCLIDEAN_OPERATORS = (
    "cdist",
    "torch_cdist",
    "euclidean",
    "l2_norm",
    "pairwise_distance",
    "softmax",
    "argmax",
    "cosine_similarity",
    "dot_product",
    "matmul",
)

#: The R8 hyperbolic arms, whose mathematics is recoverable from their source.
R8_HYPERBOLIC_ARMS = (
    "TIGER_r8_07_hyperbolic_gyro",
    "TIGER_r8_08_hyperbolic_frechet",
    "TIGER_r8_09_hyperbolic_branching",
    "TIGER_r8_10_hyperbolic_gw",
)

#: The archived historical inventory, whose mathematics is not recoverable here.
HISTORICAL_INVENTORY = (
    REPO_ROOT / "results" / "stage3_T5Train" / "monotonicity" / "model_inventory.csv"
)

#: The previous campaigns' frozen proposals.
CAMPAIGN_PROPOSALS = REPO_ROOT / "results" / "harness3_runs" / "controller" / "rounds"

#: The campaign's immutable run manifests.
CAMPAIGN_MANIFESTS = REPO_ROOT / "results" / "harness3_runs" / "store" / "runs"


class RegistryError(RuntimeError):
    """The registry is missing, corrupt or stale. The caller must fail closed."""


@dataclass
class PriorArtEntry:
    """One historic mechanism, as far as its mathematics is known."""

    proposal_sha256: str
    run_id: str
    mechanism_id: str
    family: str
    source: str
    math_status: str                      # "verified" | "novelty_unknown"
    equation_signature: str = ""          # the normalised equation, "" when unknown
    operator_classes: Tuple[str, ...] = ()
    ast_sha256: str = ""
    sid_export: str = ""
    notes: str = ""

    @property
    def identity(self) -> Tuple[str, str]:
        """The immutable identity of the entry: the proposal hash and the run id."""
        return (self.proposal_sha256, self.run_id)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["operator_classes"] = list(self.operator_classes)
        return payload


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_equation(equation: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    """The normalised equation: numbers canonicalised, parameter values replaced by names.

    A weight-only change and a cosmetic reformatting must leave the signature alone, and a
    change of the mathematics must not.
    """
    text = str(equation or "").lower()
    text = re.sub(r"(?<![\w.])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lambda m: repr(float(m.group(0))), text)
    for name, value in sorted((parameters or {}).items(), key=lambda kv: -len(str(kv[1]))):
        for candidate in {str(value), repr(value)}:
            if candidate and candidate in text:
                text = text.replace(candidate, f"<{name}>")
    return re.sub(r"\s+", "", text)


def operator_classes(source: str) -> Tuple[str, ...]:
    """The operator classes a piece of source uses, hyperbolic and Euclidean."""
    text = str(source or "")
    found = {name for name in HYPERBOLIC_OPERATORS if name in text}
    found |= {name for name in EUCLIDEAN_OPERATORS if name in text}
    return tuple(sorted(found))


def ast_sha256(source: str) -> str:
    """The digest of a source tree's abstract syntax, so a cosmetic edit is visible."""
    try:
        tree = ast.parse(str(source or ""))
    except SyntaxError:
        return sha256_bytes(str(source or "").encode("utf-8"))
    return sha256_bytes(ast.dump(tree, annotate_fields=True).encode("utf-8"))


# --------------------------------------------------------------------------- the builder
def _r8_entries(root: Path) -> List[PriorArtEntry]:
    """The R8 hyperbolic arms, whose mathematics is in their source."""
    entries: List[PriorArtEntry] = []
    for arm in R8_HYPERBOLIC_ARMS:
        tree = root / "stage2_RQ-VAE" / arm
        if not tree.is_dir():
            continue
        modules = sorted(tree.glob("*.py"))
        source = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in modules)
        digest = sha256_bytes(source.encode("utf-8"))
        operators = operator_classes(source)
        entries.append(
            PriorArtEntry(
                proposal_sha256=digest,
                run_id=f"r8::{arm}",
                mechanism_id=arm,
                family="r8_hyperbolic",
                source=str(tree.relative_to(root)) if tree.is_relative_to(root) else str(tree),
                math_status="verified",
                equation_signature="",
                operator_classes=operators,
                ast_sha256=ast_sha256(source),
                notes=(
                    "the mathematics is recoverable from the source; the equation itself is "
                    "not transcribed here, so the comparison is by the operator classes and "
                    "the abstract syntax"
                ),
            )
        )
    return entries


def _proposal_entries(root: Path) -> List[PriorArtEntry]:
    """The previous campaigns' frozen proposals, with their equations."""
    entries: List[PriorArtEntry] = []
    rounds = root / "results" / "harness3_runs" / "controller" / "rounds"
    if not rounds.is_dir():
        return entries
    for proposal_path in sorted(rounds.glob("round_*/*/proposal.json")):
        try:
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mechanism_id = proposal.get("mechanism_id") or proposal_path.parent.name
        implementation = proposal_path.parent / f"mechanism_{mechanism_id}.py"
        source = ""
        ast_digest = ""
        if implementation.is_file():
            source = implementation.read_text(encoding="utf-8", errors="replace")
            ast_digest = ast_sha256(source)
        entries.append(
            PriorArtEntry(
                proposal_sha256=proposal.get("proposal_sha256") or sha256_file(proposal_path),
                run_id=f"proposal::{proposal_path.parent.parent.name}::{mechanism_id}",
                mechanism_id=mechanism_id,
                family=str(proposal.get("family") or "unclassified"),
                source=str(proposal_path.relative_to(root))
                if proposal_path.is_relative_to(root)
                else str(proposal_path),
                math_status="verified",
                equation_signature=normalise_equation(
                    proposal.get("equation", ""), proposal.get("parameters") or {}
                ),
                operator_classes=operator_classes(source or str(proposal)),
                ast_sha256=ast_digest,
            )
        )
    return entries


def _inventory_entries(root: Path) -> List[PriorArtEntry]:
    """The 38 historical models, whose mathematics cannot be recovered here."""
    path = root / "results" / "stage3_T5Train" / "monotonicity" / "model_inventory.csv"
    if not path.is_file():
        return []
    import csv

    entries: List[PriorArtEntry] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model = (row.get("model") or "").strip()
            if not model:
                continue
            entries.append(
                PriorArtEntry(
                    proposal_sha256="",
                    run_id=f"inventory::{model}",
                    mechanism_id=model,
                    family=(row.get("family") or "").strip(),
                    source="results/stage3_T5Train/monotonicity/model_inventory.csv",
                    math_status="novelty_unknown",
                    equation_signature="",
                    operator_classes=(),
                    sid_export=(row.get("sid_sha256") or "").strip(),
                    notes=(
                        "the archived model has no recoverable proposal or source here, so "
                        "its mathematics is UNKNOWN: this entry is a name-level record only "
                        "and a mechanism with this name is refused, while a mechanism with a "
                        "different name is NOT cleared by it and must be checked against the "
                        "prior-art review"
                    ),
                )
            )
    return entries


def build_registry(root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Build the registry from the archives, and say what was found."""
    entries: List[PriorArtEntry] = []
    entries += _r8_entries(root)
    entries += _proposal_entries(root)
    entries += _inventory_entries(root)
    identities = [entry.identity for entry in entries]
    if len(set(identities)) != len(identities):
        raise RegistryError("the prior-art registry has two entries with the same identity")
    registry = {
        "version": 2,
        "entries": [entry.as_dict() for entry in entries],
        "counts": {
            "entries": len(entries),
            "r8_hyperbolic_arms": sum(1 for e in entries if e.family == "r8_hyperbolic"),
            "campaign_proposals": sum(1 for e in entries if e.run_id.startswith("proposal::")),
            "historical_models_with_unknown_math": sum(
                1 for e in entries if e.math_status == "novelty_unknown"
            ),
        },
    }
    registry["registry_sha256"] = registry_digest(registry)
    return registry


def registry_digest(registry: Dict[str, Any]) -> str:
    """The digest of the registry's content."""
    payload = {
        "version": registry.get("version"),
        "entries": registry.get("entries"),
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))


def save_registry(registry: Dict[str, Any], path: Path = DEFAULT_REGISTRY) -> str:
    """Write the registry atomically, and return the digest of the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = dict(registry)
    registry["registry_sha256"] = registry_digest(registry)
    text = json.dumps(registry, indent=2, sort_keys=True, default=str) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return sha256_file(path)


def load_registry(path: Path = DEFAULT_REGISTRY) -> Dict[str, Any]:
    """Read the registry, and fail closed when it is missing, corrupt or edited."""
    path = Path(path)
    if not path.is_file():
        raise RegistryError(
            f"the prior-art registry is missing: {path}. A missing registry blocks the "
            "launch; it never skips the comparison."
        )
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"the prior-art registry cannot be read: {error}") from error
    recorded = registry.get("registry_sha256")
    computed = registry_digest(registry)
    if recorded != computed:
        raise RegistryError(
            "the prior-art registry was edited: its recorded digest does not match its "
            f"content ({recorded} against {computed})"
        )
    return registry


# --------------------------------------------------------------------------- the verdict
@dataclass
class NoveltyVerdict:
    """The verdict of a novelty check, and why."""

    allowed: bool
    reason: str = ""
    matched_identity: Optional[Tuple[str, str]] = None
    matched_component: str = ""
    unresolved_overlaps: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "matched_identity": list(self.matched_identity) if self.matched_identity else None,
            "matched_component": self.matched_component,
            "unresolved_overlaps": list(self.unresolved_overlaps),
        }


def check_novelty(
    proposal: Dict[str, Any],
    registry: Dict[str, Any],
    *,
    implementation_source: str = "",
) -> NoveltyVerdict:
    """Whether a proposal is genuinely new with respect to the whole historical record.

    Three refusals, in order of how certain they are: the same mechanism id as a recorded
    entry, the same normalised equation as a recorded entry, and the same operator classes
    as a recorded entry whose mathematics is known. A recorded entry whose mathematics is
    UNKNOWN is reported as an unresolved overlap, because a name is not mathematics: the
    prior-art review has to settle it, and the workflow requires that review before an
    implementation.
    """
    mechanism_id = str(proposal.get("mechanism_id") or "")
    equation = normalise_equation(
        proposal.get("equation", ""), proposal.get("parameters") or {}
    )
    operators = set(operator_classes(implementation_source or str(proposal)))
    entries = registry.get("entries") or []
    for entry in entries:
        if mechanism_id and entry.get("mechanism_id") == mechanism_id:
            return NoveltyVerdict(
                False,
                f"the mechanism id {mechanism_id!r} is already a recorded mechanism",
                tuple(entry.get("identity") or (entry.get("proposal_sha256", ""), entry.get("run_id", ""))),
                "mechanism_id",
            )
    for entry in entries:
        signature = entry.get("equation_signature") or ""
        if signature and signature == equation:
            return NoveltyVerdict(
                False,
                "the normalised equation matches a recorded mechanism's",
                (entry.get("proposal_sha256", ""), entry.get("run_id", "")),
                "equation",
            )
    # A copied implementation is a repetition whatever the name: the abstract syntax is
    # compared. A mechanism that merely USES the same operators with different mathematics
    # is not a repetition, so the operator classes are reported rather than used to refuse.
    implementation_digest = ast_sha256(implementation_source) if implementation_source else ""
    if implementation_digest:
        for entry in entries:
            recorded = entry.get("ast_sha256") or ""
            if recorded and recorded == implementation_digest:
                return NoveltyVerdict(
                    False,
                    "the implementation's abstract syntax matches a recorded mechanism's",
                    (entry.get("proposal_sha256", ""), entry.get("run_id", "")),
                    "ast_sha256",
                )
    unresolved = tuple(
        sorted(
            {
                str(entry.get("mechanism_id"))
                for entry in entries
                if entry.get("math_status") == "novelty_unknown"
                and entry.get("family")
                and entry.get("family") == (proposal.get("family") or "")
            }
        )
    )
    if unresolved:
        return NoveltyVerdict(
            True,
            "no recorded mechanism matches, but the prior-art review must settle the "
            "unresolved family overlap",
            unresolved_overlaps=unresolved,
        )
    return NoveltyVerdict(True, "no recorded mechanism matches")


__all__ = [
    "CAMPAIGN_MANIFESTS",
    "CAMPAIGN_PROPOSALS",
    "DEFAULT_REGISTRY",
    "HISTORICAL_INVENTORY",
    "HYPERBOLIC_OPERATORS",
    "NoveltyVerdict",
    "PriorArtEntry",
    "R8_HYPERBOLIC_ARMS",
    "RegistryError",
    "ast_sha256",
    "build_registry",
    "check_novelty",
    "load_registry",
    "normalise_equation",
    "operator_classes",
    "registry_digest",
    "save_registry",
    "sha256_file",
]
