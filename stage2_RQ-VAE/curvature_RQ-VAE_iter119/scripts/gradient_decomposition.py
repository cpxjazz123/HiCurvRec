"""What does the behaviour ranking loss actually do to the encoder latent?

The whole gain of the accepted parent over pure geometry is the behaviour
ranking term: 0.060081 against 0.053274, about +12.8% recall, while curvature,
working depth and residual geometry were all measured to have no usable direction
(iter96-113) and coarser L2 capacity was measured to trade discrimination for
coverage (iter118). What has never been asked is what the loss does to the raw
encoder latent, even though that latent is what produces the gain.

The loss is a hinge over a hyperbolic distance between the anchor and its
successor, against the anchor and a random item:

    L = relu( d_c(A, B+) + margin - d_c(A, X-) )

Its gradient with respect to the anchor's tangent vector is split into the part
that changes the anchor's magnitude and the part that rotates it. The split is
taken against the anchor's own radial direction rather than against an arbitrary
axis, because that is the decomposition the pin downstream will act on: the
quantization path overwrites magnitude at every level, so a radial component of
the gradient is the part the RQ stack does not consume directly.

Two sets are reported, and they answer different questions. Over all pairs the
hinge is zero for most of them, so that average describes the loss as evaluated.
Over the active pairs, the ones violating the margin, the average describes the
loss as trained, since those are the only ones contributing gradient. Only the
active set can justify a change to the loss itself.

Pairs are also split by how strong the behaviour signal already is, since a hard
pair is where a better-behaved objective would have to earn its keep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import curvature_config as experiment
from model import RQVAE
from model.layers import _poincare_distance_tangent_pairs


def tokenizer_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_sizes=experiment.HIDDEN_SIZES,
        codebook_num=3,
        codebook_size=experiment.CODEBOOK_SIZE,
        codebook_dim=experiment.CODEBOOK_DIM,
        dropout=0.0,
        beta=experiment.BETA,
        vq_type=experiment.VQ_TYPE,
        ema_decay=experiment.EMA_DECAY,
        fix_code_embs=False,
        sk_epsilon=experiment.SK_EPSILON,
        sk_iters=experiment.SK_ITERS,
        layer_curvatures=experiment.LAYER_CURVATURES,
        layer_working_radii=experiment.LAYER_WORKING_RADII,
        pin_in_s_coordinates=experiment.PIN_IN_S_COORDINATES,
    )


def transition_pairs(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    successors: list[int] = []
    for history, target in zip(
        train_frame["seen_history"].to_numpy(),
        train_frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is None or len(history) == 0:
            continue
        sources.append(int(history[-1]))
        successors.append(int(target))
    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(successors, dtype=np.int64),
    )


def decompose(grad: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split each row's gradient into radial and tangential parts.

    The radial unit is the latent's own direction, so the radial component is the
    one that scales the anchor and the tangential component is everything that
    rotates it.
    """
    norm = torch.linalg.vector_norm(latent, dim=-1, keepdim=True).clamp_min(1e-12)
    unit = latent / norm
    radial = (grad * unit).sum(-1, keepdim=True) * unit
    return radial, grad - radial


def summarize(radial: torch.Tensor, tangent: torch.Tensor) -> dict:
    r = radial.norm(dim=-1)
    t = tangent.norm(dim=-1)
    total = torch.sqrt(r**2 + t**2).clamp_min(1e-20)
    return {
        "pairs": int(r.numel()),
        "radial_share": float((r / total).mean()),
        "tangential_share": float((t / total).mean()),
        "grad_norm_mean": float(total.mean()),
        "radial_norm_mean": float(r.mean()),
        "tangential_norm_mean": float(t.mean()),
    }


def main() -> None:
    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = np.random.default_rng(experiment.DIAGNOSTIC_SEED)
    torch.manual_seed(experiment.DIAGNOSTIC_SEED)

    embeddings = torch.from_numpy(
        np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    ).to(device)
    n_items = embeddings.shape[0]
    checkpoint = torch.load(
        experiment.PARENT_CKPT, map_location=device, weights_only=False
    )
    model = RQVAE(tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    pick = generator.choice(
        len(source_ids), size=experiment.GRADIENT_PAIRS, replace=False
    )
    source_ids = np.ascontiguousarray(source_ids[pick])
    successor_ids = np.ascontiguousarray(successor_ids[pick])
    negative_ids = generator.integers(0, n_items, size=len(source_ids))
    negative_ids[negative_ids == successor_ids] = (
        negative_ids[negative_ids == successor_ids] + 1
    ) % n_items

    # The anchor is the only latent the loss differentiates through: the
    # successor and the negative are targets, and the model never sees them as
    # gradients elsewhere. The encoder is therefore re-enabled for the anchor
    # path only.
    latent = model.encoder(embeddings[torch.from_numpy(source_ids).to(device)])
    with torch.no_grad():
        successor = model.encoder(
            embeddings[torch.from_numpy(successor_ids).to(device)]
        )
        negative = model.encoder(
            embeddings[torch.from_numpy(negative_ids).to(device)]
        )

    latent = latent.detach().requires_grad_(True)
    curvature = float(experiment.LAYER_CURVATURES[0])
    positive_distance = _poincare_distance_tangent_pairs(
        latent, successor, curvature
    )
    negative_distance = _poincare_distance_tangent_pairs(
        latent, negative, curvature
    )
    violation = positive_distance + experiment.BEHAVIOUR_MARGIN - negative_distance
    active = (violation > 0).detach()
    loss = torch.relu(violation).sum()
    (grad,) = torch.autograd.grad(loss, latent)
    radial, tangential = decompose(grad, latent.detach())

    behaviour_margin = (
        negative_distance - positive_distance
    ).detach()

    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "pairs": int(len(source_ids)),
        "active_pairs": int(active.sum()),
        "behaviour_margin": experiment.BEHAVIOUR_MARGIN,
        "curvature": curvature,
    }
    lines.append(
        f"pairs={len(source_ids)}  active (violating the margin)={int(active.sum())} "
        f"({active.float().mean() * 100:.1f}%)"
    )
    lines.append(
        f"latent ||z|| p50={float(latent.detach().norm(dim=-1).median()):.4f} "
        f"(unpinned, so the radial axis is a real degree of freedom)"
    )
    lines.append(
        f"behaviour margin (d_neg - d_pos): p50="
        f"{float(behaviour_margin.median()):+.6f}  "
        f"fraction positive={float((behaviour_margin > 0).float().mean()) * 100:.2f}%"
    )

    positive = behaviour_margin > 0
    hard_threshold = float(
        torch.quantile(
            behaviour_margin[positive].float(),
            experiment.HARD_MARGIN_QUANTILE,
        )
    )
    hard = positive & (behaviour_margin <= hard_threshold)
    easy = positive & (behaviour_margin > hard_threshold)
    lines.append(
        f"hard band (0 < margin <= {hard_threshold:.6f}): {int(hard.sum())}   "
        f"easy band: {int(easy.sum())}"
    )

    result["hard_threshold"] = hard_threshold
    result["hard_pairs"] = int(hard.sum())
    result["easy_pairs"] = int(easy.sum())

    def report(mask: torch.Tensor, label: str) -> dict:
        if int(mask.sum()) == 0:
            lines.append(f"   {label:<34} n=0")
            return {"pairs": 0}
        entry = summarize(radial[mask], tangential[mask])
        entry.update(
            {
                "active_fraction": float(active[mask].float().mean()),
                "margin_mean": float(behaviour_margin[mask].mean()),
            }
        )
        lines.append(
            f"   {label:<34} n={entry['pairs']:<6} "
            f"radial={entry['radial_share'] * 100:5.2f}%  "
            f"tangential={entry['tangential_share'] * 100:5.2f}%  "
            f"active={entry['active_fraction'] * 100:5.1f}%  "
            f"margin={entry['margin_mean']:+.6f}"
        )
        return entry

    lines.append("")
    lines.append("radial vs tangential share of the anchor gradient")
    result["all_pairs"] = report(
        torch.ones_like(active), "all pairs"
    )
    result["active_pairs"] = report(active, "active pairs (margin violated)")
    result["satisfied_pairs"] = report(~active, "satisfied pairs")
    result["hard_active_pairs"] = report(hard & active, "hard & active")
    result["easy_active_pairs"] = report(easy & active, "easy & active")

    # The hinge only pushes when the margin is violated, so the radial share that
    # actually trains the model is the active one. Restate the verdict on that
    # basis rather than on the all-pairs average.
    active_entry = result["active_pairs"]
    if active_entry.get("pairs", 0) > 0:
        dominant = (
            "tangential"
            if active_entry["tangential_share"] > active_entry["radial_share"]
            else "radial"
        )
        lines.append("")
        lines.append(
            f"on the active set the gradient is {active_entry['tangential_share'] * 100:.2f}% "
            f"tangential and {active_entry['radial_share'] * 100:.2f}% radial, so the "
            f"dominant channel is {dominant}"
        )
        result["dominant_channel_active"] = dominant
        result["radial_share_active"] = active_entry["radial_share"]
        result["tangential_share_active"] = active_entry["tangential_share"]

    report_text = "\n".join(lines)
    print(report_text, flush=True)
    experiment.REPORT_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.REPORT_LOG.write_text(report_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()