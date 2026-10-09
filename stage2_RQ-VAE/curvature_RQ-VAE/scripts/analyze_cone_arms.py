"""Compare the four quantized-cone arms on structure, containment and behaviour.

Reads each arm's native snapshot, recomputes the cone diagnostics directly from
the trained checkpoint rather than from the training loop's interval averages,
and reports the two paired differences the question asks for:

    Delta_E = C - A        (what the cone supervision buys in Euclidean space)
    Delta_H = D - B        (what it buys in the Poincare ball)
    Delta_H - Delta_E      (whether hyperbolic geometry adds anything on top)

The held-out items here are items excluded from the *cone supervision* only; they
still train the RQ-VAE itself, so this measures generalisation of the category
supervision, not of an unseen catalogue.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "scripts"))

import category_structure as structure  # noqa: E402
from category_structure import load_category_supervision, structure_metrics  # noqa: E402
from curvature_config import GEOMETRY as DEFAULT_GEOMETRY  # noqa: E402
from model.category_cone import CategoryCone  # noqa: E402
from model.model import RQVAE  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import curvature_config as experiment  # noqa: E402

RESULTS = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/category_cone_arms"
SEED = 42
ARMS = {
    "A": ("A_euclid_nocone", "euclid", False),
    "B": ("B_poincare_nocone", "poincare", False),
    "C": ("C_euclid_cone", "euclid", True),
    "D": ("D_poincare_cone", "poincare", True),
}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _model_config(geometry: str, cone: bool):
    config = SimpleNamespace(
        hidden_sizes=(512, 256, 128),
        codebook_num=3,
        codebook_size=(256, 256, 256),
        codebook_dim=32,
        dropout=0.0,
        beta=float(experiment.__dict__.get("BETA", 0.25)),
        vq_type="vq",
        ema_decay=0.99,
        fix_code_embs=False,
        sk_epsilon=float(experiment.__dict__.get("SK_EPSILON", 0.003)),
        sk_iters=int(experiment.__dict__.get("SK_ITERS", 50)),
        layer_curvatures=(1.0, 1.0, 1.0),
        layer_working_radii=(0.2, 0.2, 0.2),
        pin_in_s_coordinates=True,
        geometry=geometry,
    )
    return config


def _load_trainer():
    """The trainer owns the encoder's two input channels; reuse them verbatim."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generec_train_rqvae_analysis", PKG / "train_rqvae.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_encoder_inputs() -> torch.Tensor:
    """Rebuild exactly what the trained encoder consumed.

    The encoder reads the raw embedding concatenated with the curvature context
    channel, while the decoder reconstructs the raw embedding alone. So the two
    widths differ and the model has to be rebuilt with in_dim/context_dim split,
    the same way the trainer constructs it.
    """
    trainer = _load_trainer()
    train_frame = trainer.pd.read_parquet(trainer.TRAIN_FILE)
    train_ids = np.unique(train_frame["target"].to_numpy(dtype=np.int64))
    embeddings = trainer.maybe_apply_pca(
        trainer.load_embeddings(trainer.EMBEDDING_FILE), trainer.PCA_DIM, trainer.SEED
    )
    all_embeddings = torch.from_numpy(embeddings)
    sources, successors = trainer._transition_pairs(train_frame)
    centroid = trainer._behaviour_context(all_embeddings, sources, successors)
    channel = trainer._curvature_context_channel(all_embeddings, centroid)
    return (
        torch.cat([all_embeddings, channel], dim=1).to(DEVICE),
        int(all_embeddings.shape[1]),
        int(channel.shape[1]),
    )


def _cone_diagnostics(
    arm_dir: Path,
    geometry: str,
    cone: bool,
    features: torch.Tensor,
    in_dim: int,
    context_dim: int,
) -> dict:
    """Recompute containment from the trained model, not the loop averages."""
    if not cone:
        return {}
    checkpoint = torch.load(
        arm_dir / "out/rqvae/instruments/rqvae_best.pth", map_location=DEVICE
    )
    config = _model_config(geometry, cone)
    model = RQVAE(config, in_dim=in_dim, context_dim=context_dim).to(DEVICE)
    model.category_cone = CategoryCone(
        geometry=geometry,
        codebook_dim=int(config.codebook_dim),
        holdout_fraction=float(experiment.CATEGORY_CONE_HOLDOUT_FRACTION),
        data_seed=int(experiment.CATEGORY_CONE_DATA_SEED),
        margin=float(experiment.CATEGORY_CONE_MARGIN),
        radial_weight=float(experiment.CATEGORY_CONE_RADIAL_WEIGHT),
        radial_margin=float(experiment.CATEGORY_CONE_RADIAL_MARGIN),
        device=torch.device(DEVICE),
    )
    missing, unexpected = model.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    if missing:
        raise RuntimeError(f"checkpoint is missing parameters: {missing[:4]}")
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(features)
        _, _, _, prefixes = model.rq(encoded, return_prefixes=True)
        curvature = model.rq.vq_layers[0].get_curvature()
        items = torch.arange(features.shape[0], device=DEVICE)
        _, diagnostics = model.category_cone.loss(
            items, prefixes, curvature, 0
        )
        evaluation = model.category_cone.evaluate(prefixes, curvature)
        report = model.category_cone.state_report()
    return {
        "recomputed": {
            key: value for key, value in diagnostics.items()
        },
        "containment_by_split": evaluation,
        "state": report,
    }


def _loop_diagnostics(arm_dir: Path) -> dict:
    path = arm_dir / "logs/training_metrics.jsonl"
    if not path.is_file():
        return {}
    events: dict[str, list] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        events.setdefault(row.get("event"), []).append(row)
    out: dict = {}
    if events.get("train"):
        last = events["train"][-1]
        out["final_train"] = {
            "global_step": last["global_step"],
            "loss": last["loss"],
            "recon": last["recon"],
            "collision": last["collision"],
            "codebook_used": last["codebook_used"],
            "codebook_usage_counts": last.get("codebook_usage_counts"),
        }
    for name in ("cone_calibration", "cone_train", "cone_containment"):
        if events.get(name):
            out[name] = events[name][-1]
    return out


def main() -> None:
    coarse, fine, _ = load_category_supervision()
    features, in_dim, context_dim = _load_encoder_inputs()
    summary: dict = {"arms": {}, "comparisons": {}}
    for label, (name, geometry, cone) in ARMS.items():
        arm_dir = RESULTS / f"{name}_s{SEED}"
        tokens = np.load(arm_dir / "out/rqvae/instruments/sids_raw.npy")
        structure_report = structure_metrics(tokens, coarse, fine)
        entry = {
            "arm": name,
            "geometry": geometry,
            "cone": cone,
            "seed": SEED,
            "structure": {
                "coarse_vs_L1": structure_report["coarse_vs_L1"],
                "fine_vs_L1L2": structure_report["fine_vs_L1L2"],
                "l1l2_prefix_clusters": structure_report["l1l2_prefix_clusters"],
                "l1l2_prefix_singleton_fraction": structure_report[
                    "l1l2_prefix_singleton_fraction"
                ],
                "full_sid_unique": structure_report["full_sid_unique"],
                "full_sid_collision_rate": structure_report[
                    "full_sid_collision_rate"
                ],
                "codeword_usage": {
                    level: {
                        "used_codewords": block["used_codewords"],
                        "gini": block["gini"],
                        "normalized_entropy": block["normalized_entropy"],
                    }
                    for level, block in structure_report["codeword_usage"].items()
                },
            },
            "training_log": _loop_diagnostics(arm_dir),
        }
        entry["cone"] = _cone_diagnostics(
            arm_dir, geometry, cone, features, in_dim, context_dim
        )
        summary["arms"][label] = entry
        print(
            f"[analyse] {label} {name}: "
            f"coarse~L1 AMI={structure_report['coarse_vs_L1']['ami']:.4f} "
            f"(control {structure_report['coarse_vs_L1']['ami_control']:.4f}) "
            f"fine~L1L2 AMI={structure_report['fine_vs_L1L2']['ami']:.4f} "
            f"prefixes={structure_report['l1l2_prefix_clusters']} "
            f"singleton={structure_report['l1l2_prefix_singleton_fraction']:.3f} "
            f"collision={structure_report['full_sid_collision_rate']:.6f} "
            f"used={[b['used_codewords'] for b in structure_report['codeword_usage'].values()]} "
            f"gini={[round(b['gini'], 4) for b in structure_report['codeword_usage'].values()]}",
            flush=True,
        )
        train_block = entry["training_log"].get("final_train", {})
        print(
            f"[analyse] {label} training: step={train_block.get('global_step')} "
            f"loss={train_block.get('loss'):.5f} recon={train_block.get('recon'):.7f} "
            f"collision={train_block.get('collision'):.6f}",
            flush=True,
        )
        if entry["cone"]:
            recomputed = entry["cone"]["recomputed"]
            splits = entry["cone"]["containment_by_split"]
            state = entry["cone"]["state"]
            print(
                f"[analyse] {label} cone (recomputed from checkpoint): "
                f"pos_containment={recomputed['positive_containment']:.4f} "
                f"neg_containment={recomputed['negative_containment']:.4f} "
                f"nonzero_neg={recomputed['nonzero_negative_ratio']:.4f} "
                f"radial_ok={recomputed['radial_order_rate']:.4f} | "
                f"sup_q2={splits['supervised']['coarse_q2_containment']:.4f} "
                f"held_q2={splits['heldout']['coarse_q2_containment']:.4f} | "
                f"sup_q3={splits['supervised']['fine_q3_containment']:.4f} "
                f"held_q3={splits['heldout']['fine_q3_containment']:.4f} | "
                f"k_q2={state['k_q2']:.4f} k_q3={state['k_q3']:.4f} "
                f"max_kg={state['max_kg_proto']:.3f} "
                f"aperture_med_deg={state['aperture_deg_proto']['median']:.1f}",
                flush=True,
            )

    def metric(label: str, path: tuple[str, ...]) -> float:
        node = summary["arms"][label]
        for key in path:
            node = node[key]
        return float(node)

    pairs = {
        "delta_E_cone_gain_in_euclid": ("C", "A"),
        "delta_H_cone_gain_in_poincare": ("D", "B"),
    }
    tracked = (
        ("structure", "coarse_vs_L1", "ami"),
        ("structure", "coarse_vs_L1", "ami_control"),
        ("structure", "coarse_vs_L1", "purity_cluster_to_label"),
        ("structure", "coarse_vs_L1", "concentration_label_to_cluster"),
        ("structure", "fine_vs_L1L2", "ami"),
        ("structure", "l1l2_prefix_clusters",),
        ("structure", "l1l2_prefix_singleton_fraction",),
        ("structure", "full_sid_unique",),
        ("structure", "full_sid_collision_rate",),
    )
    for name, (left, right) in pairs.items():
        row = {}
        for path in tracked:
            key = ".".join(path)
            row[key] = metric(left, path) - metric(right, path)
        summary["comparisons"][name] = {"left": left, "right": right, "deltas": row}
    delta_e = summary["comparisons"]["delta_E_cone_gain_in_euclid"]["deltas"]
    delta_h = summary["comparisons"]["delta_H_cone_gain_in_poincare"]["deltas"]
    summary["comparisons"]["delta_H_minus_delta_E"] = {
        key: delta_h[key] - delta_e[key] for key in delta_e
    }
    summary["comparisons"]["D_minus_C"] = {
        ".".join(path): metric("D", path) - metric("C", path) for path in tracked
    }
    # Containment only exists for the cone arms, so its paired difference is
    # computed between those two rather than against a no-cone baseline.
    summary["comparisons"]["D_minus_C_cone_containment"] = {
        split: {
            key: (
                summary["arms"]["D"]["cone"]["containment_by_split"][split][key]
                - summary["arms"]["C"]["cone"]["containment_by_split"][split][key]
            )
            for key in ("coarse_q2_containment", "fine_q3_containment")
        }
        for split in ("supervised", "heldout")
    }

    with (RESULTS / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("\n[analyse] paired differences", flush=True)
    print(json.dumps(summary["comparisons"], indent=2, sort_keys=True, default=str))
    print(f"\n[analyse] wrote {RESULTS / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()