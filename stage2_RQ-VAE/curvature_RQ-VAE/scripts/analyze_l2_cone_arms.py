"""Compare the three L2 assignment arms on prefix sharing and quantisation health.

A smaller prefix count is only an improvement if the sharing is *semantic*: the
report therefore separates within-category sharing from cross-category sharing,
and a run only wins if within-category sharing rises while cross-category
sharing stays low. Recall and reconstruction quality are reported next to it so
a collapse into duplicate prefixes cannot masquerade as progress.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "scripts"))

from category_structure import (  # noqa: E402
    codebook_sizes_from_checkpoint,
    load_category_supervision,
    structure_metrics,
)
from prefix_sharing_detail import (  # noqa: E402
    fine_sharing_within_l1,
    third_level_in_shared_prefixes,
)

RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/l2_cone_arms"
SEED = 42
ARMS = ["A_euclid_nocone", "B_poincare_nocone", "C_euclid_cone", "D_poincare_cone"]
def prefix_ids(tokens: np.ndarray, depth: int) -> np.ndarray:
    _, inverse = np.unique(tokens[:, :depth], axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def sharing(tokens: np.ndarray, prefix: np.ndarray, labels: np.ndarray) -> dict:
    """Split the pairs that share an L1+L2 prefix into same-label and other.

    A run only improves if the within-label share rises while the cross-label
    share stays low, so both are reported and their ratio is the headline: the
    ratio is 1 when sharing carries no category information at all.
    """
    n_labels = int(labels.max()) + 1
    _, group_counts = np.unique(prefix * n_labels + labels, return_counts=True)
    within_pairs = int(np.sum(group_counts * (group_counts - 1) // 2))
    label_counts = np.bincount(labels, minlength=n_labels)
    total_within = int(np.sum(label_counts * (label_counts - 1) // 2))
    n = len(labels)
    total_pairs = n * (n - 1) // 2
    total_cross = total_pairs - total_within
    _, prefix_counts = np.unique(prefix, return_counts=True)
    shared_pairs = int(np.sum(prefix_counts * (prefix_counts - 1) // 2))
    cross_shared = shared_pairs - within_pairs
    within_share = within_pairs / max(total_within, 1)
    cross_share = cross_shared / max(total_cross, 1)
    return {
        "n_pairs": int(total_pairs),
        "shared_pairs": shared_pairs,
        "share_all": shared_pairs / max(total_pairs, 1),
        "within_share": within_share,
        "cross_share": cross_share,
        "within_over_cross": within_share / max(cross_share, 1e-12),
    }


def prefix_capacity(tokens: np.ndarray) -> dict:
    """How hard the third level would have to work to undo prefix sharing.

    L3 has one code per item inside a prefix, so any prefix holding more items
    than L3 has codes forces collisions no matter what the third level learns.
    """
    l12 = prefix_ids(tokens, 2)
    _, counts = np.unique(l12, return_counts=True)
    sizes = counts.astype(np.int64)
    return {
        "max_items_in_one_prefix": int(sizes.max()),
        "mean_items_per_prefix": float(sizes.mean()),
        "prefixes_over_256_items": int((sizes > 256).sum()),
        "items_in_prefixes_over_256": int(sizes[sizes > 256].sum()),
        "l3_codes": 256,
        "collision_forced": bool(sizes.max() > 256),
    }


def main() -> None:
    coarse, fine, _ = load_category_supervision()
    summary: dict = {"arms": {}}
    for name in ARMS:
        arm_dir = RESULTS / f"{name}_s{SEED}"
        tokens = np.load(arm_dir / "out/rqvae/instruments/sids_raw.npy")
        report = structure_metrics(
            tokens, coarse, fine, codebook_sizes_from_checkpoint(arm_dir)
        )
        l12 = prefix_ids(tokens, 2)
        coarse_labels = coarse[coarse >= 0]
        fine_labels = fine[fine >= 0]
        l12_coarse = l12[coarse >= 0]
        l12_fine = l12[fine >= 0]
        train_rows = [
            json.loads(line)
            for line in (arm_dir / "logs/training_metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        final_train = [row for row in train_rows if row.get("event") == "train"][-1]
        entry = {
            "arm": name,
            "structure": {
                "l1l2_prefix_clusters": report["l1l2_prefix_clusters"],
                "l1l2_prefix_singleton_fraction": report[
                    "l1l2_prefix_singleton_fraction"
                ],
                "coarse_vs_L1_ami": report["coarse_vs_L1"]["ami"],
                "coarse_vs_L1_ami_control": report["coarse_vs_L1"]["ami_control"],
                "coarse_vs_L1_purity": report["coarse_vs_L1"][
                    "purity_cluster_to_label"
                ],
                "fine_vs_L1L2_ami": report["fine_vs_L1L2"]["ami"],
                "full_sid_unique": report["full_sid_unique"],
                "full_sid_collision_rate": report["full_sid_collision_rate"],
                "codeword_usage": {
                    level: {
                        "used_codewords": block["used_codewords"],
                        "gini": block["gini"],
                        "normalized_entropy": block["normalized_entropy"],
                    }
                    for level, block in report["codeword_usage"].items()
                },
            },
            "prefix_capacity": prefix_capacity(tokens),
            "fine_sharing_within_l1": fine_sharing_within_l1(
                tokens, coarse, fine
            ),
            "third_level_in_shared_prefixes": third_level_in_shared_prefixes(
                tokens
            ),
            "training": {
                "global_step": final_train["global_step"],
                "loss": final_train["loss"],
                "recon": final_train["recon"],
                "collision": final_train["collision"],
                "codebook_used": final_train["codebook_used"],
            },
        }
        # Both views use only labelled items so the two levels share rows.
        entry["sharing_coarse"] = sharing(
            tokens[coarse >= 0], l12_coarse, coarse_labels
        )
        entry["sharing_fine"] = sharing(tokens[fine >= 0], l12_fine, fine_labels)
        summary["arms"][name] = entry
        structure = entry["structure"]
        coarse_share = entry["sharing_coarse"]
        print(
            f"[l2] {name}: prefixes={structure['l1l2_prefix_clusters']} "
            f"singleton={structure['l1l2_prefix_singleton_fraction']:.3f} "
            f"collision={structure['full_sid_collision_rate']:.5f} | "
            f"coarse AMI={structure['coarse_vs_L1_ami']:.4f} "
            f"fine AMI={structure['fine_vs_L1L2_ami']:.4f} | "
            f"within={coarse_share['within_share']:.4f} "
            f"cross={coarse_share['cross_share']:.4f} "
            f"ratio={coarse_share['within_over_cross']:.2f} | "
            f"max_prefix={entry['prefix_capacity']['max_items_in_one_prefix']} "
            f"collision_forced={entry['prefix_capacity']['collision_forced']} | "
            f"recon={entry['training']['recon']:.7f} "
            f"used={[b['used_codewords'] for b in structure['codeword_usage'].values()]}",
            flush=True,
        )

    tracked = (
        ("structure", "coarse_vs_L1_ami"),
        ("structure", "fine_vs_L1L2_ami"),
        ("structure", "l1l2_prefix_clusters",),
        ("structure", "full_sid_collision_rate",),
        ("fine_sharing_within_l1", "within_l1_same_fine_share"),
        ("fine_sharing_within_l1", "within_l1_different_fine_share"),
        ("third_level_in_shared_prefixes", "mean_distinct_l3_in_shared_prefix"),
        ("third_level_in_shared_prefixes", "l3_collision_pairs"),
    )

    def value(label, path):
        node = summary["arms"][label]
        for key in path:
            node = node[key]
        return float(node)

    comparisons = {}
    for name, left, right in (
        ("delta_E", "C_euclid_cone", "A_euclid_nocone"),
        ("delta_H", "D_poincare_cone", "B_poincare_nocone"),
        ("D_minus_C", "D_poincare_cone", "C_euclid_cone"),
    ):
        row = {".".join(path): value(left, path) - value(right, path) for path in tracked}
        if name == "D_minus_C":
            row["_note"] = (
                "direct comparison under identical supervision; a smaller "
                "regression is not an improvement on its own"
            )
        comparisons[name] = row
    comparisons["delta_H_minus_delta_E"] = {
        ".".join(path): comparisons["delta_H"][".".join(path)]
        - comparisons["delta_E"][".".join(path)]
        for path in tracked
    }
    summary["comparisons"] = comparisons
    with (RESULTS / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"\n[l2] wrote {RESULTS / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
