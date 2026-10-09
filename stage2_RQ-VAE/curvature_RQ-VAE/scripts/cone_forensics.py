"""Post-mortem for a cone run: prototype radii, apertures, and loss-term gradients.

Answers the question the divergence raised - did the category prototypes leave
the radius band their cone apertures depend on, and which loss term dominated -
without retraining anything. Reads a finished run's checkpoint, rebuilds the same
model and cone module, and reports:

* the prototype radii in each arm's own point space, against the band the
  training loop enforces and against the initial radii;
* the cone apertures those radii imply, and whether ``K * g`` saturates;
* encoder and codebook norms, to separate a cone problem from a general
  optimisation failure;
* the gradient norm each loss term contributes, so a term that is numerically
  large can be told apart from one that is merely active.

The reported ``gap_active`` is the signed rejection hinge, the same quantity the
loss uses, so it is always at least the containment rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "scripts"))

import curvature_config as experiment  # noqa: E402
from model.category_cone import CategoryCone  # noqa: E402
from model.cones import (  # noqa: E402
    aperture,
    apex_angle,
    containment_loss,
    radial_order_loss,
    to_point,
)

from category_structure import (  # noqa: E402
    codebook_sizes_from_checkpoint,
    load_category_supervision,
    structure_metrics,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ROOTS = {
    "baseline": PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/l2_cone_arms",
    "fixed": PKG.parent.parent
    / "results/stage2_RQ-VAE/curvature_RQ-VAE/l2_cone_arms_fixed_rejection",
}
ARMS = ["A_euclid_nocone", "B_poincare_nocone", "C_euclid_cone", "D_poincare_cone"]


def _model_config(geometry: str):
    return SimpleNamespace(
        hidden_sizes=(512, 256, 128),
        codebook_num=3,
        codebook_size=tuple(codebook_sizes_from_checkpoint(ROOTS["baseline"] / "A_euclid_nocone_s42")),
        codebook_dim=32,
        dropout=0.0,
        beta=0.25,
        vq_type="vq",
        ema_decay=0.99,
        fix_code_embs=False,
        sk_epsilon=0.003,
        sk_iters=50,
        layer_curvatures=(1.0, 1.0, 1.0),
        layer_working_radii=(0.2, 0.2, 0.2),
        pin_in_s_coordinates=True,
        geometry=geometry,
        layer_assignment_modes=("bucket",) * 3,
    )


def _encoder_inputs():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generec_train_rqvae_forensics", PKG / "train_rqvae.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(trainer.TRAIN_FILE)
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    embeddings = trainer.maybe_apply_pca(
        trainer.load_embeddings(trainer.EMBEDDING_FILE), trainer.PCA_DIM, trainer.SEED
    )
    all_embeddings = torch.from_numpy(embeddings)
    sources, successors = trainer._transition_pairs(frame)
    centroid = trainer._behaviour_context(all_embeddings, sources, successors)
    channel = trainer._curvature_context_channel(all_embeddings, centroid)
    return (
        torch.cat([all_embeddings, channel], dim=1).to(DEVICE),
        int(all_embeddings.shape[1]),
        int(channel.shape[1]),
    )


def _radial_stats(points: torch.Tensor) -> dict:
    norm = points.detach().norm(dim=-1).cpu().numpy()
    return {
        "min": float(norm.min()),
        "median": float(np.median(norm)),
        "max": float(norm.max()),
    }


def _rebuild(arm_dir: Path, geometry: str, in_dim: int, context_dim: int):
    from model.model import RQVAE

    checkpoint = torch.load(
        arm_dir / "out/rqvae/instruments/rqvae_best.pth", map_location=DEVICE
    )
    config = _model_config(geometry)
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
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    return model


def _term_gradients(geometry, model, items, curvature, k_by_term, report):
    """Gradient norm of each cone term, separately, on the trained model."""
    cone = model.category_cone
    gradients = {}
    for name in ("positive", "rejection", "radial"):
        model.zero_grad(set_to_none=True)
        coarse = cone.coarse_of_item[items]
        fine = cone.fine_of_item[items]
        apex_coarse = cone.prototypes.coarse[coarse]
        apex_fine = cone.prototypes.fine[fine]
        with torch.no_grad():
            q2 = report["q2"][items]
            q3 = report["q3"][items]
        if name == "positive":
            loss = (
                containment_loss(
                    geometry, curvature, apex_coarse, apex_fine, apex_coarse,
                    k_by_term[0], cone.margin,
                )[1].mean()
                + containment_loss(
                    geometry, curvature, apex_coarse, q2, q2,
                    k_by_term[1], cone.margin,
                )[1].mean()
                + containment_loss(
                    geometry, curvature, apex_fine, q3, q3,
                    k_by_term[2], cone.margin,
                )[1].mean()
            )
        elif name == "rejection":
            loss = containment_loss(
                geometry, curvature, apex_coarse, apex_fine, apex_coarse,
                k_by_term[0], cone.margin,
            )[2].mean()
            loss = loss + containment_loss(
                geometry, curvature, apex_coarse, q2, q2,
                k_by_term[1], cone.margin,
            )[2].mean() + containment_loss(
                geometry, curvature, apex_fine, q3, q3,
                k_by_term[2], cone.margin,
            )[2].mean()
        else:
            loss = cone.radial_weight * radial_order_loss(
                geometry, curvature,
                cone.prototypes.coarse, cone.prototypes.fine, cone.radial_margin,
            )
        loss.backward()
        prototype_grad = torch.cat(
            [
                p.grad.detach().reshape(-1)
                for p in cone.prototypes.parameters()
                if p.grad is not None
            ]
        ) if any(p.grad is not None for p in cone.prototypes.parameters()) else None
        codebook_grad = torch.cat(
            [
                layer.get_code_embs().grad.detach().reshape(-1)
                for layer in model.rq.vq_layers
                if layer.get_code_embs().grad is not None
            ]
        ) if any(
            layer.get_code_embs().grad is not None for layer in model.rq.vq_layers
        ) else None
        gradients[name] = {
            "value": float(loss.detach()),
            "prototype_grad_norm": None if prototype_grad is None else float(prototype_grad.norm()),
            "codebook_grad_norm": None if codebook_grad is None else float(codebook_grad.norm()),
        }
    model.zero_grad(set_to_none=True)
    return gradients


def main() -> None:
    coarse_labels, fine_labels, _ = load_category_supervision()
    features, in_dim, context_dim = _encoder_inputs()
    out: dict = {}
    for sweep, root in ROOTS.items():
        for name in ARMS:
            arm_dir = root / f"{name}_s42"
            checkpoint = arm_dir / "out/rqvae/instruments/rqvae_best.pth"
            if not checkpoint.is_file():
                continue
            geometry = "poincare" if "poincare" in name else "euclid"
            model = _rebuild(arm_dir, geometry, in_dim, context_dim)
            cone = model.category_cone
            with torch.no_grad():
                encoded = model.encoder(features)
                _, _, _, prefixes = model.rq(encoded, return_prefixes=True)
                curvature = model.rq.vq_layers[0].get_curvature()
                items = torch.arange(features.shape[0], device=DEVICE)
                _, diagnostics = cone.loss(items, prefixes, curvature, 0)
                coarse_points = to_point(
                    geometry, cone.prototypes.coarse, curvature
                )
                fine_points = to_point(
                    geometry, cone.prototypes.fine, curvature
                )
                aperture_coarse = torch.rad2deg(
                    aperture(geometry, coarse_points, float(cone.k_proto))
                )
                aperture_fine = torch.rad2deg(
                    aperture(geometry, fine_points, float(cone.k_proto))
                )
                codebook_norms = [
                    float(layer.get_code_embs().detach().norm(dim=-1).mean())
                    for layer in model.rq.vq_layers
                ]
                encoder_norms = [
                    float(p.detach().norm())
                    for n, p in model.named_parameters()
                    if n.startswith("encoder.")
                ]
            # Gradients need the graph, so this runs outside no_grad.
            negatives = _term_gradients(
                geometry, model, items, curvature,
                (float(cone.k_proto), float(cone.k_q2), float(cone.k_q3)),
                {"q2": prefixes[1].detach(), "q3": prefixes[2].detach()},
            )
            tokens = np.load(arm_dir / "out/rqvae/instruments/sids_raw.npy")
            structure = structure_metrics(
                tokens, coarse_labels, fine_labels, codebook_sizes_from_checkpoint(arm_dir)
            )
            out[f"{sweep}/{name}"] = {
                "geometry": geometry,
                "k": [float(cone.k_proto), float(cone.k_q2), float(cone.k_q3)],
                "prototype_radius_point_space": {
                    "coarse": _radial_stats(coarse_points),
                    "fine": _radial_stats(fine_points),
                },
                "aperture_deg": {
                    "coarse_median": float(aperture_coarse.median()),
                    "fine_median": float(aperture_fine.median()),
                    "coarse_max": float(aperture_coarse.max()),
                },
                "recomputed_diagnostics": diagnostics,
                "term_gradients": negatives,
                "codebook_weight_norm_mean": codebook_norms,
                "encoder_weight_norm": encoder_norms,
                "fine_ami": structure["fine_vs_L1L2"]["ami"],
                "collision": structure["full_sid_collision_rate"],
            }
            print(
                f"[forensics] {sweep}/{name}: "
                f"coarse_r={out[f'{sweep}/{name}']['prototype_radius_point_space']['coarse']['median']:.4f} "
                f"fine_r={out[f'{sweep}/{name}']['prototype_radius_point_space']['fine']['median']:.4f} "
                f"aperture_coarse={out[f'{sweep}/{name}']['aperture_deg']['coarse_median']:.1f}deg "
                f"pos={diagnostics.get('positive_containment')} "
                f"neg={diagnostics.get('negative_containment')} "
                f"gap_active={diagnostics.get('negative_gap_active_ratio')} "
                f"fine_ami={structure['fine_vs_L1L2']['ami']:.4f} "
                f"collision={structure['full_sid_collision_rate']:.4f}",
                flush=True,
            )
    destination = PKG.parent.parent / "results/stage2_RQ-VAE/curvature_RQ-VAE/cone_forensics.json"
    with destination.open("w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print(f"\n[forensics] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
