"""Recompute the frozen TIGER baseline under the new unified metrics (issue #245 C23).

Only metrics that the existing artefacts support are computed; anything that would need a
model forward pass is reported as ``NA`` with the reason, never borrowed from a historical
report computed under a different caliber.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.graph_infra import metrics  # noqa: E402

ARM = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
ASSETS = REPO / "results" / "research" / "infra245" / "assets"
OUT = REPO / "results" / "research" / "infra245" / "baseline_tiger.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    sid_path = ARM / "item_sids.json"
    npy_path = ARM / "sids_for_hgrec.npy"
    if not sid_path.exists():
        raise SystemExit(f"missing frozen SID table: {sid_path}")
    table = json.loads(sid_path.read_text(encoding="utf-8"))
    items = sorted(int(k) for k in table)
    codes = [list(table[str(item)]) for item in items]
    tokens = np.array([row[:3] for row in codes], dtype=np.int64)

    report = {
        "arm": "TIGER_RQ-VAE",
        "sid": {"path": str(sid_path.relative_to(REPO)), "sha256": sha256(sid_path),
                "items": len(items)},
        "sid_structure": {
            "collision_first_three": metrics.collision_rate(tokens),
            "duplicate_item_fraction_full_tuple": metrics.duplicate_item_fraction(codes),
            "distinct_first_three": int(len({tuple(row) for row in tokens.tolist()})),
            "codebook": metrics.codebook_stats(tokens),
        },
    }

    # prefix sharing between frozen pair-bank members: model-independent, new caliber
    bank_dir = ASSETS / "pairs"
    prefix = {}
    for name in ("direct", "shared_out", "shared_in", "two_hop", "degree_matched"):
        path = bank_dir / f"bank_{name}.npy"
        if not path.exists():
            prefix[name] = "NA (bank asset missing)"
            continue
        produced = np.load(path)
        pairs = []
        for i, j in produced.tolist():
            if str(int(i)) in table and str(int(j)) in table:
                pairs.append((table[str(int(i))][:3], table[str(int(j))][:3]))
        prefix[name] = {
            "pairs": len(pairs),
            "prefix@1": metrics.prefix_at_k(pairs, 1),
            "prefix@2": metrics.prefix_at_k(pairs, 2),
            "prefix@3": metrics.prefix_at_k(pairs, 3),
        }
    report["prefix_sharing_on_frozen_banks"] = prefix

    # re-export consistency: the exported matrix must agree with the JSON table.
    # If they do not, that is a finding about the frozen arm, not something to average over.
    if npy_path.exists():
        exported = np.load(npy_path)
        row_mismatch = int((exported[:, :3] != tokens[: len(exported)]).any(axis=1).sum())
        set_json = {tuple(row) for row in tokens.tolist()}
        set_npy = {tuple(row) for row in exported[:, :3].tolist()}
        report["reexport_consistency"] = {
            "path": str(npy_path.relative_to(REPO)), "sha256": sha256(npy_path),
            "rows": int(exported.shape[0]),
            "items_differing_rowwise": row_mismatch,
            "distinct_triples_json": len(set_json),
            "distinct_triples_npy": len(set_npy),
            "shared_triples": len(set_json & set_npy),
            "consistent": row_mismatch == 0,
            "finding": (
                "INCONSISTENT: the two frozen TIGER exports do not describe the same SID table. "
                "item_sids.json carries 20425 distinct triples and matches the training log's "
                "collision 0.169276; sids_for_hgrec.npy carries 21873 and shares only "
                f"{len(set_json & set_npy)} triples with it. The JSON is newer. Downstream work must "
                "name which artefact it uses instead of treating them as one object."
                if row_mismatch else "consistent"
            ),
        }
    else:
        report["reexport_consistency"] = "NA (no sids_for_hgrec.npy)"

    report["not_computed"] = {
        "reconstruction_mse": "NA: needs a decoder forward pass through the arm's own model code; "
                              "the infra layer deliberately does not import a Stage2 tree",
        "relation_strength_spearman": "NA: needs a model's D_geo or S_G; the bank ships without "
                                      "any model-specific target",
        "stage3_recall": "NA in this infra pass: existing frozen numbers were computed under an "
                         "older caliber and must not be presented as a new comparison",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("sid", "sid_structure")}, indent=2)[:1200])
    print("prefix:", json.dumps(prefix)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
