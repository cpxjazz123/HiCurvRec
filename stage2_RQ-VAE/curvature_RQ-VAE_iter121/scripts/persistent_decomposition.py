"""Why do a quarter of the behaviour pairs stay active for the whole run?

iter120 followed 20000 fixed pairs through five checkpoints of the parent replay
and found the active set is not churning: 37.4% never violate the margin, 27.1%
violate it at every checkpoint, and the four intermediate buckets are nearly even.
The persistent quarter has a median margin near zero at the end of training, so
those pairs are not converging. What is not yet known is whether the difficulty
comes from the positive, the negative, or both.

The negative sampler has to be right or the answer is meaningless. The parent does
not draw negatives from the catalogue; it takes

    negatives = encoded_source[randperm(batch)]

so a pair's negative is another pair's source item from the same batch of 1024.
Every negative is therefore an item somebody acted on, which makes it a harder and
more legitimate distractor than a uniform draw, and makes a persistent pair much
more likely to be a genuine near-neighbour problem than a sampling artefact. This
replays that sampler rather than substituting a simpler one.

Each persistent pair is then placed in a quadrant by how unusual its positive and
negative distances are against the population's own robust spread, so the cut is
relative rather than an absolute distance that would carry no meaning. Gradient
norms are reported alongside, because a pair's share of the update is its loss
times its gradient norm, not its share of the active set.
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
from model.layers import (
    _hyperbolic_residual,
    _pairwise_poincare_distance_tangents,
    _poincare_distance_tangent_pairs,
)


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


@torch.no_grad()
def encode_all(model: RQVAE, embeddings: torch.Tensor, chunk: int = 4096):
    parts = []
    for start in range(0, embeddings.shape[0], chunk):
        parts.append(model.encoder(embeddings[start : start + chunk]))
    return torch.cat(parts, dim=0)


def batch_shuffled_distances(
    latent: torch.Tensor,
    successor_latent: torch.Tensor,
    batch: int,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the parent's negative sampler and average over draws.

    The parent's negative is another pair's source from the same shuffled batch,
    so the negative for a pair is a different item each epoch. Averaging over
    several draws is what makes the measured distance representative rather than
    one arbitrary shuffle.
    """
    n = latent.shape[0]
    curvature = float(experiment.LAYER_CURVATURES[0])
    generator = torch.Generator(device=latent.device).manual_seed(seed)
    positive_sum = np.zeros(n, dtype=np.float64)
    negative_sum = np.zeros(n, dtype=np.float64)
    for _ in range(draws):
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            chunk = slice(start, stop)
            perm = torch.randperm(stop - start, generator=generator, device=latent.device)
            with torch.no_grad():
                positive_sum[chunk] += (
                    _poincare_distance_tangent_pairs(
                        latent[chunk], successor_latent[chunk], curvature
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                negative_sum[chunk] += (
                    _poincare_distance_tangent_pairs(
                        latent[chunk], latent[chunk][perm], curvature
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
    return positive_sum / draws, negative_sum / draws


def gradient_norms(
    latent: torch.Tensor,
    successor_latent: torch.Tensor,
    negative_latent: torch.Tensor,
) -> np.ndarray:
    """|dL/dz| per pair, so the update share is measured rather than assumed."""
    curvature = float(experiment.LAYER_CURVATURES[0])
    variable = latent.detach().requires_grad_(True)
    positive = _poincare_distance_tangent_pairs(
        variable, successor_latent, curvature
    )
    negative = _poincare_distance_tangent_pairs(
        variable, negative_latent, curvature
    )
    violation = positive + experiment.BEHAVIOUR_MARGIN - negative
    (grad,) = torch.autograd.grad(torch.relu(violation).sum(), variable)
    return grad.norm(dim=-1).cpu().numpy().astype(np.float64)


def robust_scale(values: np.ndarray) -> float:
    """Median absolute deviation, converted to a standard-deviation equivalent."""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return 1.4826 * mad


@torch.no_grad()
def full_corpus_tokens(model: RQVAE, embeddings: torch.Tensor) -> np.ndarray:
    layers = model.rq.vq_layers
    out = np.empty((embeddings.shape[0], len(layers)), dtype=np.int64)
    for start in range(0, embeddings.shape[0], 4096):
        batch = embeddings[start : start + 4096]
        residual = model.encoder(batch)
        previous_codes = None
        for level, layer in enumerate(layers):
            c = float(layer.get_curvature())
            target_norm = model.rq._radius_for_level(level, c, residual)
            pinned = model.rq._pin_to_radius(residual, target_norm)
            distances = _pairwise_poincare_distance_tangents(
                pinned, layer.get_code_embs(), c
            )
            chosen = layer._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            out[start : start + batch.shape[0], level] = chosen.cpu().numpy()
            residual = model.rq._restore_norm(
                _hyperbolic_residual(pinned, layer.get_code_embs()[chosen], c),
                residual,
                target_norm,
            )
            previous_codes = chosen
    return out


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

    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    pick = generator.choice(
        len(source_ids), size=experiment.DECOMP_PAIRS, replace=False
    )
    source_ids = np.ascontiguousarray(source_ids[pick])
    successor_ids = np.ascontiguousarray(successor_ids[pick])

    latent = encode_all(model, embeddings)
    anchor = latent[torch.from_numpy(source_ids).to(device)]
    successor = latent[torch.from_numpy(successor_ids).to(device)]

    d_pos, d_neg = batch_shuffled_distances(
        anchor,
        successor,
        experiment.PAIR_BATCH_SIZE,
        experiment.NEGATIVE_DRAWS,
        experiment.DECOMP_SEED,
    )
    margin = d_neg - d_pos
    active = margin < experiment.BEHAVIOUR_MARGIN

    # The trajectory run established which of these pairs are active at every
    # checkpoint, so the three populations are taken from the same sample.
    persistent = np.zeros(len(source_ids), dtype=bool)
    persistent[: int(0.2713 * len(source_ids))] = False
    trajectory = experiment.REPLAY_ROOT / "trajectory"
    checkpoints = sorted(
        trajectory.glob("step_*.pth"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    active_columns = []
    for path in checkpoints + [experiment.PARENT_CKPT]:
        if str(path) == str(experiment.PARENT_CKPT) and path in checkpoints:
            continue
        other = RQVAE(tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
        other.load_state_dict(
            torch.load(path, map_location=device, weights_only=False)["state_dict"],
            strict=True,
        )
        other.eval()
        for parameter in other.parameters():
            parameter.requires_grad_(False)
        a = encode_all(other, embeddings)[torch.from_numpy(source_ids).to(device)]
        s = encode_all(other, embeddings)[
            torch.from_numpy(successor_ids).to(device)
        ]
        p, n = batch_shuffled_distances(
            a, s, experiment.PAIR_BATCH_SIZE, 2, experiment.DECOMP_SEED
        )
        active_columns.append((n - p) < experiment.BEHAVIOUR_MARGIN)
        del other
        torch.cuda.empty_cache()
    stack = np.stack(active_columns, axis=1)
    persistent = stack.all(axis=1)
    never = (~stack).all(axis=1)
    rotating = ~persistent & ~never

    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "pairs": int(len(source_ids)),
        "negative_sampler": experiment.NEGATIVE_SAMPLER,
        "checkpoints_tracked": int(stack.shape[1]),
        "populations": {
            "persistent": int(persistent.sum()),
            "rotating": int(rotating.sum()),
            "never": int(never.sum()),
        },
    }
    lines.append(
        f"negative sampler: {experiment.NEGATIVE_SAMPLER} "
        f"(batch of {experiment.PAIR_BATCH_SIZE}, {experiment.NEGATIVE_DRAWS} draws)"
    )
    lines.append(
        f"populations over {stack.shape[1]} checkpoints: "
        f"persistent={int(persistent.sum())}  rotating={int(rotating.sum())}  "
        f"never={int(never.sum())}"
    )

    pos_scale = robust_scale(d_pos)
    neg_scale = robust_scale(d_neg)
    lines.append(
        f"population robust spread: d_pos MAD-scale={pos_scale:.6f}  "
        f"d_neg MAD-scale={neg_scale:.6f}"
    )

    lines.append("")
    lines.append("1. is the positive weak or the negative hard?")
    lines.append(
        f"   {'population':<12}{'n':>7}{'d_pos p50':>12}{'d_neg p50':>12}"
        f"{'margin p50':>12}{'d_pos p90':>12}"
    )
    for label, mask in (
        ("persistent", persistent),
        ("rotating", rotating),
        ("never", never),
    ):
        if mask.sum() == 0:
            continue
        lines.append(
            f"   {label:<12}{int(mask.sum()):>7}"
            f"{np.median(d_pos[mask]):>12.6f}"
            f"{np.median(d_neg[mask]):>12.6f}"
            f"{np.median(margin[mask]):>12.6f}"
            f"{np.percentile(d_pos[mask], 90):>12.6f}"
        )

    lines.append("")
    lines.append("2. quadrant split of the persistent pairs")
    weak_positive = d_pos > np.median(d_pos) + experiment.OUTLIER_Z * pos_scale
    hard_negative = d_neg < np.median(d_neg) - experiment.OUTLIER_Z * neg_scale
    quadrants = {
        "weak_positive_only": persistent & weak_positive & ~hard_negative,
        "hard_negative_only": persistent & ~weak_positive & hard_negative,
        "both": persistent & weak_positive & hard_negative,
        "neither": persistent & ~weak_positive & ~hard_negative,
    }
    total_persistent = max(int(persistent.sum()), 1)
    for label, mask in quadrants.items():
        lines.append(
            f"   {label:<20}{int(mask.sum()):>7}  "
            f"{mask.sum() / total_persistent * 100:5.2f}% of persistent"
        )
    result["persistent_quadrants"] = {
        label: {
            "pairs": int(mask.sum()),
            "share_of_persistent": float(mask.sum() / total_persistent),
        }
        for label, mask in quadrants.items()
    }
    lines.append(
        f"   thresholds: d_pos > median + {experiment.OUTLIER_Z}*{pos_scale:.4f}, "
        f"d_neg < median - {experiment.OUTLIER_Z}*{neg_scale:.4f}"
    )

    lines.append("")
    lines.append("3. gradient share, not just pair share")
    generator_b = torch.Generator(device=device).manual_seed(experiment.DECOMP_SEED)
    perm = torch.randperm(
        anchor.shape[0], generator=generator_b, device=device
    )
    grads = gradient_norms(anchor, successor, anchor[perm])
    result["gradient_share"] = {
        "all_pairs_sum": float(grads.sum()),
    }
    for label, mask in (
        ("persistent", persistent),
        ("rotating", rotating),
        ("never", never),
    ):
        if mask.sum() == 0:
            continue
        share = grads[mask].sum() / grads.sum()
        result["gradient_share"][label] = float(share)
        lines.append(
            f"   {label:<12} pairs {mask.sum() / len(mask) * 100:5.2f}% of the sample  "
            f"gradient {share * 100:5.2f}% of the total"
        )

    lines.append("")
    lines.append("4. what is the persistent negative, really?")
    tokens = full_corpus_tokens(model, embeddings)
    negative_items = source_ids[perm.cpu().numpy()]
    same_prefix = (
        tokens[source_ids, :3] == tokens[negative_items, :3]
    ).all(axis=1)
    partner_of: list[set[int]] = [set() for _ in range(n_items)]
    for a, b in zip(source_ids, successor_ids):
        partner_of[int(a)].add(int(b))
    negative_is_partner = np.array(
        [int(n) in partner_of[int(a)] for a, n in zip(source_ids, negative_items)]
    )
    unit = latent / torch.linalg.vector_norm(
        latent, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    cosine = (
        unit[torch.from_numpy(source_ids).to(device)]
        * unit[torch.from_numpy(negative_items).to(device)]
    ).sum(-1).cpu().numpy()
    for label, mask in (
        ("persistent", persistent),
        ("rotating", rotating),
        ("never", never),
    ):
        if mask.sum() == 0:
            continue
        lines.append(
            f"   {label:<12} shares the full SID with its negative "
            f"{same_prefix[mask].mean() * 100:5.2f}%   "
            f"is a known behaviour partner {negative_is_partner[mask].mean() * 100:5.2f}%   "
            f"latent cosine p50={np.median(cosine[mask]):.5f} p99={np.percentile(cosine[mask], 99):.5f}"
        )
    result["negative_character"] = {
        "full_sid_match": {
            "persistent": float(same_prefix[persistent].mean()),
            "rotating": float(same_prefix[rotating].mean()),
            "never": float(same_prefix[never].mean()),
        },
        "known_partner": {
            "persistent": float(negative_is_partner[persistent].mean()),
            "rotating": float(negative_is_partner[rotating].mean()),
            "never": float(negative_is_partner[never].mean()),
        },
    }

    lines.append("")
    lines.append("verdict")
    quadrant = result["persistent_quadrants"]
    negative_share = (
        quadrant["hard_negative_only"]["share_of_persistent"]
        + quadrant["both"]["share_of_persistent"]
    )
    positive_share = (
        quadrant["weak_positive_only"]["share_of_persistent"]
        + quadrant["both"]["share_of_persistent"]
    )
    if negative_share > 0.5 and negative_share > positive_share:
        verdict = "hard negatives dominate: the batch-shuffled source sampler is feeding in distractors that are genuinely close"
    elif positive_share > 0.5:
        verdict = "weak positives dominate: the real transitions are far apart in the learned space and no negative change would help"
    else:
        verdict = "neither distance is the outlier; the persistent set is a representation or capacity conflict"
    result["verdict"] = verdict
    lines.append(
        f"   persistent pairs: hard-negative component {negative_share * 100:.1f}%, "
        f"weak-positive component {positive_share * 100:.1f}%"
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