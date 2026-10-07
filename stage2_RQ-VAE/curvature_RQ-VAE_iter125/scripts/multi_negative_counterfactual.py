"""Does a multi-negative objective propose a better separation direction than the pairwise hinge?

Four candidates have now been excluded for the quarter of behaviour pairs that
never converge: curvature (iter96-113), capacity (iter118), opposition from
reconstruction and quantization (iter123, orthogonal rather than opposed), and
interference between the behaviour constraints themselves (iter124, consistent
with C_A = 0.64-0.69 and only 1.05 points of gradient budget lost to summation).
What remains is that a pairwise objective may simply not express the direction
these pairs need.

Testing that on a frozen representation cannot be done by asking whether the
margin improves, because a frozen representation does not change on its own. It
has to be a counterfactual: take the same anchor latent, compute the gradient of
each objective with respect to it, take one probe step along each, and read the
angular margin afterwards. The gradients are normalised to unit length first, so
the comparison is between directions and not between magnitudes; otherwise the
objective with the larger gradient wins trivially and the answer means nothing.

The margin read out is the angular one, cos(A, B+) - cos(A, X-), because
iter122 established that this is what the behaviour term actually acts on. The
question is whether InfoNCE over K negatives gives the persistent pairs a
different and better direction than the single-negative hinge, and how much of
the two gradients agree, since near-parallel gradients would mean the change of
objective is cosmetic.

Nothing is trained.
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


def pairwise_gradient(
    anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, curvature: float
) -> torch.Tensor:
    """The parent's own hinge, differentiated at the anchor."""
    variable = anchor.detach().clone().requires_grad_(True)
    d_pos = _poincare_distance_tangent_pairs(variable, positive, curvature)
    d_neg = _poincare_distance_tangent_pairs(variable, negative, curvature)
    # sum, not mean: each row depends only on its own positive and negative,
    # so the summed gradient is the per-pair gradient, and the step is
    # normalised before use anyway.
    loss = torch.relu(d_pos + experiment.BEHAVIOUR_MARGIN - d_neg).sum()
    (grad,) = torch.autograd.grad(loss, variable)
    return grad.detach()


def infonce_gradient(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    curvature: float,
) -> torch.Tensor:
    """InfoNCE over the anchor's positives against K negatives.

    The score is the negative hyperbolic distance, which is the same quantity the
    hinge ranks on, so the only thing that changes is the objective's shape: one
    denominator covering every negative instead of a separate hinge per negative.
    """
    # Broadcast the anchor against its K negatives so the distance helper gets
    # matched rows: (pairs, dim) against (pairs, K, dim).
    variable = anchor.detach().clone().requires_grad_(True)
    expanded = variable.unsqueeze(1).expand(-1, negatives.shape[1], -1)
    scores_pos = -_poincare_distance_tangent_pairs(
        variable, positive, curvature
    ) / experiment.NCE_TEMPERATURE
    scores_neg = -_poincare_distance_tangent_pairs(
        expanded, negatives, curvature
    ) / experiment.NCE_TEMPERATURE
    denominator = torch.logsumexp(
        torch.cat([scores_pos.unsqueeze(1), scores_neg], dim=1), dim=1
    )
    loss = (denominator - scores_pos).sum()
    (grad,) = torch.autograd.grad(loss, variable)
    return grad.detach()


def main() -> None:
    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = np.random.default_rng(experiment.DECOMP_SEED)
    torch.manual_seed(experiment.DECOMP_SEED)

    embeddings = torch.from_numpy(
        np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    ).to(device)
    n_items = embeddings.shape[0]
    model = RQVAE(tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    model.load_state_dict(
        torch.load(
            experiment.PARENT_CKPT, map_location=device, weights_only=False
        )["state_dict"],
        strict=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    curvature = float(experiment.LAYER_CURVATURES[0])

    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    pick = generator.choice(
        len(source_ids), size=experiment.DECOMP_PAIRS, replace=False
    )
    source_ids = np.ascontiguousarray(source_ids[pick])
    successor_ids = np.ascontiguousarray(successor_ids[pick])
    # Negatives for the InfoNCE denominator. Drawn as K independent shuffles of
    # the sampled sources, which is the parent's own negative logic repeated;
    # the first shuffle doubles as the single negative the hinge is compared
    # against so both objectives see the same items.
    k = experiment.NCE_NEGATIVES
    nce_negatives = np.stack(
        [source_ids[generator.permutation(len(source_ids))] for _ in range(k)],
        axis=1,
    )
    single_negative = np.ascontiguousarray(nce_negatives[:, 0])

    idx = torch.from_numpy(source_ids).to(device)
    with torch.no_grad():
        anchor = model.encoder(embeddings[idx])
        positive = model.encoder(embeddings[torch.from_numpy(successor_ids).to(device)])
        single = model.encoder(
            embeddings[torch.from_numpy(single_negative).to(device)]
        )
        multi = model.encoder(
            embeddings[torch.from_numpy(
                np.ascontiguousarray(nce_negatives.reshape(-1))
            ).to(device)]
        ).reshape(len(source_ids), k, -1)

    # Populations from the same five checkpoints as iter121-124.
    trajectory_dir = (
        experiment.REPO_ROOT
        / "results/stage2_RQ-VAE/curvature_RQ-VAE_iter120/trajectory"
    )
    paths = sorted(
        trajectory_dir.glob("step_*.pth"),
        key=lambda q: int(q.stem.split("_")[1]),
    ) + [experiment.PARENT_CKPT]
    columns = []
    for path in paths:
        other = RQVAE(tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
        other.load_state_dict(
            torch.load(path, map_location=device, weights_only=False)["state_dict"],
            strict=True,
        )
        other.eval()
        with torch.no_grad():
            a = other.encoder(embeddings[idx])
            b = other.encoder(embeddings[torch.from_numpy(successor_ids).to(device)])
            c = other.encoder(embeddings[torch.from_numpy(single_negative).to(device)])
            active = (
                _poincare_distance_tangent_pairs(a, c, curvature)
                - _poincare_distance_tangent_pairs(a, b, curvature)
            ) < experiment.BEHAVIOUR_MARGIN
            columns.append(active.cpu().numpy())
        del other
        torch.cuda.empty_cache()
    stack = np.stack(columns, axis=1)
    persistent = stack.all(axis=1)
    never = (~stack).all(axis=1)
    rotating = ~persistent & ~never

    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "pairs": int(len(source_ids)),
        "nce_negatives": experiment.NCE_NEGATIVES,
        "nce_temperature": experiment.NCE_TEMPERATURE,
        "step_size": experiment.STEP_SIZE,
        "populations": {
            "persistent": int(persistent.sum()),
            "rotating": int(rotating.sum()),
            "never": int(never.sum()),
        },
    }
    lines.append(
        f"pairs: persistent={int(persistent.sum())}  "
        f"rotating={int(rotating.sum())}  never={int(never.sum())}"
    )
    lines.append(
        f"one probe step of {experiment.STEP_SIZE} along each unit-normalised "
        f"gradient; K={experiment.NCE_NEGATIVES}, tau={experiment.NCE_TEMPERATURE}"
    )

    chunk = 2048
    margin_base = np.zeros(len(source_ids), dtype=np.float64)
    margin_pair = np.zeros(len(source_ids), dtype=np.float64)
    margin_multi = np.zeros(len(source_ids), dtype=np.float64)
    cos_grad = np.zeros(len(source_ids), dtype=np.float64)

    def unit(v: torch.Tensor) -> torch.Tensor:
        return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-12)

    for start in range(0, len(source_ids), chunk):
        stop = min(start + chunk, len(source_ids))
        a = anchor[start:stop]
        p = positive[start:stop]
        n1 = single[start:stop]
        nk = multi[start:stop]
        # Per pair, not per chunk: taking the chunk mean here would broadcast one
        # number across the slice and erase every difference between populations.
        g_pair = unit(pairwise_gradient(a, p, n1, curvature))
        g_multi = unit(infonce_gradient(a, p, nk, curvature))
        moved_pair = unit(a - experiment.STEP_SIZE * g_pair)
        moved_multi = unit(a - experiment.STEP_SIZE * g_multi)
        margin_base[start:stop] = (
            ((unit(a) * unit(p)).sum(-1) - (unit(a) * unit(n1)).sum(-1))
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        margin_pair[start:stop] = (
            ((moved_pair * unit(p)).sum(-1) - (moved_pair * unit(n1)).sum(-1))
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        margin_multi[start:stop] = (
            ((moved_multi * unit(p)).sum(-1) - (moved_multi * unit(n1)).sum(-1))
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        cos_grad[start:stop] = (
            (g_pair * g_multi).sum(-1).cpu().numpy().astype(np.float64)
        )

    delta_pair = margin_pair - margin_base
    delta_multi = margin_multi - margin_base

    lines.append("")
    lines.append("1. angular margin change after one unit-normalised step")
    lines.append(
        f"   {'population':<12}{'n':>7}{'base margin':>14}"
        f"{'d(pairwise)':>14}{'d(multi-neg)':>15}{'multi - pair':>14}"
    )
    for label, mask in (
        ("persistent", persistent),
        ("rotating", rotating),
        ("never", never),
    ):
        if mask.sum() == 0:
            continue
        entry = {
            "pairs": int(mask.sum()),
            "base_margin": float(margin_base[mask].mean()),
            "delta_pairwise": float(delta_pair[mask].mean()),
            "delta_multi": float(delta_multi[mask].mean()),
            "multi_minus_pair": float(
                delta_multi[mask].mean() - delta_pair[mask].mean()
            ),
        }
        result.setdefault("counterfactual", {})[label] = entry
        lines.append(
            f"   {label:<12}{entry['pairs']:>7}{entry['base_margin']:>14.6f}"
            f"{entry['delta_pairwise']:>14.6f}{entry['delta_multi']:>15.6f}"
            f"{entry['multi_minus_pair']:>14.6f}"
        )

    lines.append("")
    lines.append("2. do the two objectives even point the same way?")
    lines.append(
        f"   {'population':<12}{'cos(g_pair, g_multi) p50':>26}"
        f"{'mean':>10}"
    )
    for label, mask in (
        ("persistent", persistent),
        ("rotating", rotating),
        ("never", never),
    ):
        if mask.sum() == 0:
            continue
        entry = {
            "cosine_p50": float(np.median(cos_grad[mask])),
            "cosine_mean": float(cos_grad[mask].mean()),
        }
        result.setdefault("gradient_alignment", {})[label] = entry
        lines.append(
            f"   {label:<12}{entry['cosine_p50']:>26.4f}{entry['cosine_mean']:>10.4f}"
        )

    lines.append("")
    lines.append("verdict")
    persistent_entry = result.get("counterfactual", {}).get("persistent", {})
    rotating_entry = result.get("counterfactual", {}).get("rotating", {})
    persistent_gain = persistent_entry.get("multi_minus_pair", 0.0)
    rotating_gain = rotating_entry.get("multi_minus_pair", 0.0)
    alignment = result.get("gradient_alignment", {}).get("persistent", {}).get(
        "cosine_p50", 1.0
    )
    new_direction = abs(alignment) < 0.9
    helps_persistent = persistent_gain > 0.0 and persistent_gain > rotating_gain
    if helps_persistent and new_direction:
        verdict = (
            "InfoNCE proposes a different direction and moves the persistent pairs' "
            "angular margin further than the hinge does, so training it is justified"
        )
    elif helps_persistent:
        verdict = (
            "InfoNCE helps the persistent pairs but along much the same direction, "
            "so the gain would be a magnitude effect rather than a new signal"
        )
    else:
        verdict = (
            "InfoNCE does not give the persistent pairs a better separation "
            "direction locally, so training it is not justified"
        )
    result["verdict"] = verdict
    lines.append(
        f"   persistent multi-minus-pair = {persistent_gain:+.6f}   "
        f"rotating = {rotating_gain:+.6f}   "
        f"cosine at persistent = {alignment:+.4f}"
    )
    lines.append(f"   {verdict}")

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.DECOMP_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.DECOMP_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()