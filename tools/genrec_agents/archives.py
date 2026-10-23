"""The read-only import of the prior campaigns (issue #110, WP0).

Issue #110 asks for a forensic import: the earlier campaigns' records are read, counted and
hashed, and the import says exactly what it found and where it found it. Three kinds of
material have to be told apart, because counting them together is how a campaign comes to
report progress it did not make:

* OLD ARCHIVE IMPORTS -- the manifests, the proposals, the exports and the reports of the
  previous campaigns and of R8. Read-only, never rewritten, never counted as new science.
* NEW SCIENTIFIC WORK -- the arms this campaign has produced, identified by a proposal hash
  that is not in the archive.
* SYNTHETIC MATH TESTS -- the verifier's own fixtures and certificates, which are tests and
  never measurements.

The frozen controls and the data are hashed here so that a later comparison can state
exactly what it was compared against.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The new campaign's namespace. Nothing from the earlier campaigns is written here.
NAMESPACE = REPO_ROOT / "results" / "hyperbolic_research_runs"

#: The archives that are imported, read-only.
ARCHIVES: Dict[str, str] = {
    "historical_inventory": "results/stage3_T5Train/monotonicity/model_inventory.csv",
    "historical_features": "results/stage3_T5Train/monotonicity/joint_prefix_predictors/stage2_valid_features.csv",
    "historical_probe_scores": "results/stage3_T5Train/monotonicity/history_hierarchy_predictors_fix48/probe_scores.csv",
    "historical_per_model_features": "results/stage3_T5Train/monotonicity/history_hierarchy_predictors_fix48/per_model_features.csv",
    "campaign_store": "results/harness3_runs/store/runs",
    "campaign_proposals": "results/harness3_runs/controller/rounds",
    "campaign_audit": "results/harness3_runs/controller/audit.jsonl",
    "campaign_ledger": "results/harness3_runs/hypotheses/ledger.json",
    "prior_art_registry": "results/hyperbolic_research_runs/prior_art_registry.json",
    "frozen_control_export": "results/stage2_RQ-VAE/r8_control/sids_for_hgrec.npy",
    "frozen_control_checkpoint": "results/stage2_RQ-VAE/r8_control/rqvae_best.pth",
    "frozen_pair_archive": "results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz",
    "train_parquet": "results/stage0_build_parquet/train.parquet",
}

#: The R8 hyperbolic arms, imported read-only as studied source.
R8_HYPERBOLIC_TREES = (
    "TIGER_r8_07_hyperbolic_gyro",
    "TIGER_r8_08_hyperbolic_frechet",
    "TIGER_r8_09_hyperbolic_branching",
    "TIGER_r8_10_hyperbolic_gw",
)


def sha256_file(path: Path) -> Optional[str]:
    """The digest of a file, or None when it is absent."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_csv(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def _count_jsonl(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def import_archives(root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Import the archives read-only, with their counts and their digests."""
    entries: List[Dict[str, Any]] = []
    for name, relative in ARCHIVES.items():
        path = root / relative
        record: Dict[str, Any] = {
            "name": name,
            "path": relative,
            "kind": "old_archive_import",
            "exists": path.exists(),
        }
        if path.is_file():
            record["sha256"] = sha256_file(path)
            record["bytes"] = path.stat().st_size
            if path.suffix == ".csv":
                record["rows"] = _count_csv(path)
            elif path.suffix == ".jsonl":
                record["rows"] = _count_jsonl(path)
            elif path.suffix == ".json":
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                    record["top_level_keys"] = sorted(document)[:12] if isinstance(document, dict) else None
                except (OSError, json.JSONDecodeError):
                    record["readable"] = False
        elif path.is_dir():
            files = [p for p in path.rglob("*") if p.is_file()]
            record["files"] = len(files)
            record["bytes"] = sum(p.stat().st_size for p in files)
            if name == "campaign_store":
                manifests = [p for p in files if p.name == "manifest.json"]
                record["manifests"] = len(manifests)
                kinds: Dict[str, int] = {}
                mechanisms = set()
                for manifest_path in manifests:
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    kind = str(manifest.get("kind") or "unknown")
                    kinds[kind] = kinds.get(kind, 0) + 1
                    mechanisms.add(manifest.get("mechanism_id") or "")
                record["kinds"] = kinds
                record["mechanisms"] = len(mechanisms)
            if name == "campaign_proposals":
                proposals = [p for p in files if p.name == "proposal.json"]
                record["proposals"] = len(proposals)
        entries.append(record)

    for tree in R8_HYPERBOLIC_TREES:
        path = root / "stage2_RQ-VAE" / tree
        entries.append(
            {
                "name": f"r8_tree::{tree}",
                "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                "kind": "old_archive_import",
                "exists": path.is_dir(),
                "files": len([p for p in path.rglob("*") if p.is_file()]) if path.is_dir() else 0,
            }
        )

    return {
        "version": 1,
        "namespace": str(NAMESPACE.relative_to(root)) if NAMESPACE.is_relative_to(root) else str(NAMESPACE),
        "imports": entries,
        "counts": {
            "archives_imported": sum(1 for e in entries if e["exists"]),
            "archives_missing": sum(1 for e in entries if not e["exists"]),
            "old_archive_imports": sum(1 for e in entries if e["kind"] == "old_archive_import"),
            "new_scientific_arms": 0,
            "synthetic_math_tests": 0,
        },
        "policy": (
            "the archives are read-only: this import writes nothing into the earlier "
            "campaigns, and an imported record is never counted as new science. A new "
            "scientific arm is one whose proposal hash is absent from the archive import; a "
            "synthetic math test is a verifier fixture and is never a measurement."
        ),
    }


def write_import(document: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Write the import manifest into the new namespace."""
    path = Path(path) if path else NAMESPACE / "archive_import.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, sort_keys=True, default=str) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


__all__ = [
    "ARCHIVES",
    "NAMESPACE",
    "R8_HYPERBOLIC_TREES",
    "import_archives",
    "sha256_file",
    "write_import",
]
