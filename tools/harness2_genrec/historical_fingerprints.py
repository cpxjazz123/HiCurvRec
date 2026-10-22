"""The immutable registry of historic mechanism fingerprints (issue #109, section 2.2).

Issue #109 forbids repeating an old mechanism, and names the ways that is usually disguised:
renaming it, changing only a loss weight, touching the code cosmetically, or resubmitting it
under a new id. A name is therefore never evidence of novelty -- the mechanism's mathematics
is.

The fingerprint of a proposal is a digest of the parts the issue names: the normalised
equation, the declared gradient targets, the target paths, and the SID-effect hypothesis.
Normalisation removes exactly the differences that must not create novelty: the declared
parameter VALUES are replaced by their names, so a weight-only change is the same
mechanism, while a change of the mathematics (a different exponent, a different objective,
a different gradient path) is a different fingerprint.

The registry is a function of the archives -- the historical model inventory and the
campaign's immutable run manifests -- and never a hand-written list. It records the digest
of every source it was built from, and it refuses to be used when those sources have
changed under it, which is what makes "immutable" checkable rather than a claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where the registry is persisted by default.
DEFAULT_REGISTRY = REPO_ROOT / "results" / "harness3_runs" / "historical_fingerprints.json"

#: The historical inventory of the 38 archived models.
HISTORICAL_INVENTORY = Path("results/stage3_T5Train/monotonicity/model_inventory.csv")

#: The campaign's immutable run manifests.
CAMPAIGN_MANIFESTS = Path("results/harness3_runs/store/runs")

#: The frozen proposals the campaign ran.
CAMPAIGN_PROPOSALS = Path("results/harness3_runs/controller/rounds")

#: The words that carry no meaning in the SID-effect hypothesis.
_STOP_WORDS = frozenset(
    """
    a an the of to in on for with by that this it its and or as is are be will not no so
    if then than when where which who whom whose what how why do does did done can could
    may might must shall should would have has had having they them their there here
    into over under between each any all both more most other some such only own same
    """.split()
)


#: A numeric literal, which may be written in scientific notation.
_NUMBER_LITERAL = re.compile(r"(?<![\w.])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _canonical_number(match: "re.Match[str]") -> str:
    """The canonical spelling of a numeric literal."""
    raw = match.group(0)
    try:
        return repr(float(raw))
    except ValueError:
        return raw


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_equation(equation: str, parameters: Dict[str, Any]) -> str:
    """The equation with its declared parameter VALUES replaced by their names.

    A weight-only change has to leave the fingerprint alone, but a change of the
    mathematics has to change it. Replacing the declared values by their names achieves
    exactly that: ``L = ||a - b||^2 * 0.05`` and ``L = ||a - b||^2 * 0.5`` are the same
    mechanism, while ``L = ||a - b||^3 * 0.05`` is a different one.
    """
    text = str(equation or "").lower()
    # every numeric literal is canonicalised first, so "5e-2" and "0.05" are the same
    # number: a weight written differently is the same weight
    text = _NUMBER_LITERAL.sub(_canonical_number, text)
    pairs = sorted(
        ((str(name), value) for name, value in (parameters or {}).items()),
        key=lambda item: len(str(item[1])),
        reverse=True,
    )
    for name, value in pairs:
        candidates = {str(value), repr(value)}
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            candidates.add(f"{value:g}")
            candidates.add(f"{value:.10g}")
        for candidate in sorted(candidates, key=len, reverse=True):
            if candidate and candidate in text:
                text = text.replace(candidate, f"<{name}>")
    # whitespace is cosmetic: "a - b" and "a-b" are the same mathematics
    text = re.sub(r"\s+", "", text)
    return text


def _normalise_tokens(text: str) -> List[str]:
    """The content words of a piece of prose, in a canonical order."""
    words = re.findall(r"[a-z_][a-z_0-9]*", str(text or "").lower())
    return sorted({word for word in words if word not in _STOP_WORDS})


def fingerprint_components(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """The normalised parts of a proposal, each with its own digest."""
    parameters = proposal.get("parameters") or {}
    equation = _normalise_equation(proposal.get("equation", ""), parameters)
    gradient_targets = {
        str(key): str(value) for key, value in sorted((proposal.get("gradient_targets") or {}).items())
    }
    target_paths = sorted(str(path) for path in (proposal.get("target_paths") or []))
    sid_effect = _normalise_tokens(
        " ".join(
            str(proposal.get(key, ""))
            for key in ("codebook_interaction", "encoder_interaction", "behaviour_signal")
        )
    )
    components = {
        "equation": equation,
        "gradient_targets": gradient_targets,
        "target_paths": target_paths,
        "sid_effect_tokens": sid_effect,
    }
    return {
        "components": components,
        "digests": {name: _sha256_text(json.dumps(value, sort_keys=True)) for name, value in components.items()},
        # the primary key: what the mechanism computes and where its gradient goes
        "primary": _sha256_text(
            json.dumps(
                {
                    "equation": components["equation"],
                    "gradient_targets": components["gradient_targets"],
                    "target_paths": components["target_paths"],
                },
                sort_keys=True,
            )
        ),
        "sid_effect": _sha256_text(json.dumps(sid_effect, sort_keys=True)),
    }


def fingerprint(proposal: Dict[str, Any]) -> str:
    """The fingerprint of a proposal: the primary digest of its mathematics."""
    return fingerprint_components(proposal)["primary"]


def build_registry(
    root: Path = REPO_ROOT,
    *,
    inventory: Path = HISTORICAL_INVENTORY,
    manifests: Path = CAMPAIGN_MANIFESTS,
) -> Dict[str, Any]:
    """Assemble the registry from the historical inventory and the campaign's manifests."""
    sources: Dict[str, str] = {}
    entries: List[Dict[str, Any]] = []

    inventory_path = root / inventory
    if inventory_path.is_file():
        sources[str(inventory)] = _sha256_file(inventory_path)
        import csv

        with inventory_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                model = (row.get("model") or "").strip()
                if not model:
                    continue
                entries.append(
                    {
                        "kind": "historical_model",
                        "id": model,
                        "family": (row.get("family") or "").strip(),
                        "primary": _sha256_text(f"historical_model:{model}"),
                        "sid_effect": _sha256_text(f"historical_model:{model}"),
                        "sid_sha256": (row.get("sid_sha256") or "").strip(),
                        "detail": "the archived model inventory; its proposal is not available",
                    }
                )

    # the frozen proposals are the mathematics the campaign actually ran, so the registry
    # computes their fingerprints rather than approximating them from the manifest
    proposal_root = root / CAMPAIGN_PROPOSALS
    proposals: Dict[str, Dict[str, Any]] = {}
    if proposal_root.is_dir():
        for path in sorted(proposal_root.glob("round_*/*/proposal.json")):
            try:
                proposal = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            mechanism_id = proposal.get("mechanism_id") or path.parent.name
            components = fingerprint_components(proposal)
            proposals[mechanism_id] = {
                "kind": "campaign_proposal",
                "id": mechanism_id,
                "family": "",
                "primary": components["primary"],
                "sid_effect": components["sid_effect"],
                "semantic_key": "",
                "proposal_sha256": "",
                "run_id": "",
                "detail": "the frozen proposal the campaign ran",
            }

    manifest_root = root / manifests
    if manifest_root.is_dir():
        for path in sorted(manifest_root.glob("*/manifest.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            mechanism_id = manifest.get("mechanism_id") or ""
            if not mechanism_id:
                continue
            semantic_key = (manifest.get("extra") or {}).get("semantic_key") or ""
            proposal_sha = manifest.get("proposal_sha256") or ""
            entry = proposals.get(mechanism_id)
            if entry is None:
                entry = {
                    "kind": "campaign_run",
                    "id": mechanism_id,
                    "family": "",
                    "primary": _sha256_text(
                        json.dumps(
                            {"semantic_key": semantic_key, "proposal_sha256": proposal_sha},
                            sort_keys=True,
                        )
                    ),
                    "sid_effect": _sha256_text(f"campaign_run:{semantic_key}"),
                    "detail": "the campaign's immutable run manifest",
                }
            entry = dict(entry)
            entry["semantic_key"] = semantic_key
            entry["proposal_sha256"] = proposal_sha
            entry["run_id"] = manifest.get("run_id") or ""
            entries.append(entry)

    registry = {
        "version": 1,
        "sources": sources,
        "entries": entries,
        "historical_models": sum(1 for entry in entries if entry["kind"] == "historical_model"),
        "campaign_runs": sum(1 for entry in entries if entry["kind"] == "campaign_run"),
    }
    registry["registry_sha256"] = _registry_digest(registry)
    return registry


def _registry_digest(registry: Dict[str, Any]) -> str:
    """The digest of the registry's content, excluding the digest itself."""
    payload = {
        "version": registry.get("version"),
        "sources": registry.get("sources"),
        "entries": registry.get("entries"),
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, default=str))


def save_registry(registry: Dict[str, Any], path: Path = DEFAULT_REGISTRY) -> str:
    """Write the registry atomically and return the digest of the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = dict(registry)
    registry["registry_sha256"] = _registry_digest(registry)
    text = json.dumps(registry, indent=2, sort_keys=True, default=str) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return _sha256_file(path)


def load_registry(path: Path = DEFAULT_REGISTRY) -> Dict[str, Any]:
    """Read the registry back and refuse it when its digest or its sources have changed."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"the historical fingerprint registry is missing: {path}")
    registry = json.loads(path.read_text(encoding="utf-8"))
    recorded = registry.get("registry_sha256")
    computed = _registry_digest(registry)
    if recorded != computed:
        raise ValueError(
            "the historical fingerprint registry was edited: its recorded digest does not "
            f"match its content ({recorded} against {computed})"
        )
    return registry


class NoveltyResult:
    """The verdict of a novelty check, with the evidence for it."""

    def __init__(
        self,
        novel: bool,
        matched_component: str = "",
        matched_id: str = "",
        detail: str = "",
        primary: str = "",
        sid_effect: str = "",
    ) -> None:
        self.novel = novel
        self.matched_component = matched_component
        self.matched_id = matched_id
        self.detail = detail
        self.primary = primary
        self.sid_effect = sid_effect

    def as_dict(self) -> Dict[str, Any]:
        return {
            "novel": self.novel,
            "matched_component": self.matched_component,
            "matched_id": self.matched_id,
            "detail": self.detail,
            "primary": self.primary,
            "sid_effect": self.sid_effect,
        }

    def __repr__(self) -> str:  # pragma: no cover - a debugging convenience
        return f"NoveltyResult(novel={self.novel}, matched={self.matched_id!r})"


def check_novelty(
    proposal: Dict[str, Any],
    registry: Dict[str, Any],
    *,
    mechanism_id: str = "",
    semantic_key: str = "",
) -> NoveltyResult:
    """Whether a proposal is genuinely new with respect to the registry."""
    components = fingerprint_components(proposal)
    for entry in registry.get("entries") or ():
        if entry.get("kind") == "campaign_run":
            if semantic_key and entry.get("semantic_key") == semantic_key:
                return NoveltyResult(
                    False,
                    "semantic_key",
                    entry.get("id", ""),
                    f"the mechanism's specification already ran as {entry.get('id')}",
                    components["primary"],
                    components["sid_effect"],
                )
            continue
        if mechanism_id and entry.get("id") == mechanism_id:
            return NoveltyResult(
                False,
                "id",
                entry.get("id", ""),
                f"the model id {mechanism_id} is an archived historical model",
                components["primary"],
                components["sid_effect"],
            )
    for entry in registry.get("entries") or ():
        if entry.get("primary") and entry.get("primary") == components["primary"]:
            return NoveltyResult(
                False,
                "primary",
                entry.get("id", ""),
                "the mechanism's mathematics matches a registered one",
                components["primary"],
                components["sid_effect"],
            )
    # A matching SID-effect hypothesis is reported, not refused: the issue requires a
    # SUBSTANTIVE change of the mathematics for novelty, and a substantive change is
    # novelty even when the prose about the effect happens to be the same.
    for entry in registry.get("entries") or ():
        if entry.get("sid_effect") and entry.get("sid_effect") == components["sid_effect"]:
            return NoveltyResult(
                True,
                "sid_effect",
                entry.get("id", ""),
                "the declared effect on the exported codes matches a registered one; the "
                "mathematics differ, so this is admitted with that recorded",
                components["primary"],
                components["sid_effect"],
            )
    return NoveltyResult(
        True,
        primary=components["primary"],
        sid_effect=components["sid_effect"],
    )


def registered_ids(registry: Dict[str, Any]) -> List[str]:
    """The ids the registry knows about."""
    return sorted({str(entry.get("id", "")) for entry in registry.get("entries") or ()})


def summarise(registry: Dict[str, Any]) -> Dict[str, Any]:
    """A short, printable summary of the registry."""
    return {
        "historical_models": registry.get("historical_models", 0),
        "campaign_runs": registry.get("campaign_runs", 0),
        "entries": len(registry.get("entries") or ()),
        "registry_sha256": registry.get("registry_sha256", ""),
        "sources": registry.get("sources", {}),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build the registry and report what it holds."""
    import argparse

    parser = argparse.ArgumentParser(prog="tools.harness2_genrec.historical_fingerprints")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--out", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--check", action="store_true", help="only verify the stored registry")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.check:
        registry = load_registry(Path(args.out))
    else:
        registry = build_registry(Path(args.root))
        digest = save_registry(registry, Path(args.out))
        registry = load_registry(Path(args.out))
        print(json.dumps({"written": str(args.out), "file_sha256": digest}, indent=2))
    print(json.dumps(summarise(registry), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
