"""Frozen diagnostic: is a codeword's radius a cause, a symptom, or a control knob?

Three questions, in order, and no training anywhere.

Step 1. iter105 found corr(usage, s_e) = -0.69 at L2 and -0.63 at L3: heavily
used codewords sit closer to the centre. That is only interesting if the radius
is doing something rather than just tracking coverage, so this step measures
each codeword's angular spread and asks whether usage, spread and radius form
the chain

    usage up  =>  angular spread up  =>  radius down

If the chain holds, the radius is where coverage is being expressed.

Step 2. Only one level's curvature moves, the others stay at the parent, and
three things are read: the codeword radius distribution, the usage-radius
correlation, and how much the assignment actually moves. A knob that reshapes
the radial hierarchy in an orderly way, while leaving the behaviour direction
signal intact, is a real control.

Step 3. That verdict is printed, not acted on. Training happens only if these
frozen numbers justify it.
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
)


def tokenizer_config(curvatures: tuple[float, ...]) -> SimpleNamespace:
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
        layer_curvatures=curvatures,
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


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def traverse(
    model: RQVAE,
    embeddings: torch.Tensor,
    curvatures: tuple[float, ...],
    batch_size: int = 4096,
) -> dict:
    """Run the stack at the given per-level curvatures and read everything out.

    The codebook is used exactly as trained; only the geometry the assignment is
    measured in changes, which is what makes this a frozen readout rather than
    a retrain.
    """
    device = embeddings.device
    n_items = embeddings.shape[0]
    layers = model.rq.vq_layers
    n_levels = len(layers)
    n_codes = layers[0].n_embed
    tokens = np.empty((n_items, n_levels), dtype=np.int64)
    margins = np.empty((n_items, n_levels), dtype=np.float64)
    spread_sum = np.zeros((n_levels, n_codes), dtype=np.float64)
    spread_cos = np.zeros((n_levels, n_codes), dtype=np.float64)
    radii = np.empty((n_levels, n_codes), dtype=np.float64)

    for level, layer in enumerate(layers):
        curvature = curvatures[level]
        code = layer.get_code_embs()
        radii[level] = (
            curvature**0.5
        ) * torch.linalg.vector_norm(code, dim=-1).cpu().numpy()
        code_unit = (
            code
            / torch.linalg.vector_norm(code, dim=-1, keepdim=True).clamp_min(1e-12)
        )
        for start in range(0, n_items, batch_size):
            batch = embeddings[start : start + batch_size]
            residual = model.encoder(batch)
            previous_codes = None
            for level2, layer2 in enumerate(layers):
                c2 = curvatures[level2]
                code2 = layer2.get_code_embs()
                target_norm = model.rq._radius_for_level(level2, c2, residual)
                pinned = model.rq._pin_to_radius(residual, target_norm)
                distances = _pairwise_poincare_distance_tangents(
                    pinned, code2, c2
                )
                two = torch.topk(distances, k=2, dim=1, largest=False).values
                if level2 == level:
                    margins[start : start + batch.shape[0], level] = (
                        two[:, 1] - two[:, 0]
                    ).cpu().numpy()
                chosen = layer2._indices(
                    distances, infer_use_sk=True, bucket=previous_codes
                )
                if level2 == level:
                    tokens[start : start + batch.shape[0], level] = (
                        chosen.cpu().numpy()
                    )
                    # Directional concentration of what this codeword absorbs:
                    # mean cosine between the residuals it attracts and its own
                    # unit direction, accumulated per codeword.
                    units = (
                        pinned
                        / torch.linalg.vector_norm(
                            pinned, dim=-1, keepdim=True
                        ).clamp_min(1e-12)
                    )
                    picked = torch.nn.functional.embedding(chosen, code_unit)
                    rows = (picked * units).sum(-1).cpu().numpy()
                    idx = chosen.cpu().numpy()
                    spread_cos[level] += np.bincount(
                        idx, weights=rows, minlength=n_codes
                    )
                    spread_sum[level] += np.bincount(
                        idx, minlength=n_codes
                    ).astype(np.float64)
                residual = model.rq._restore_norm(
                    _hyperbolic_residual(pinned, code2[chosen], c2),
                    residual,
                    target_norm,
                )
                previous_codes = chosen

    usage = spread_sum
    mean_cos = np.where(usage > 0, spread_cos / np.maximum(usage, 1), np.nan)
    unique = int(len(np.unique(tokens, axis=0)))
    return {
        "tokens": tokens,
        "margins": margins,
        "radii": radii,
        "usage": usage,
        "mean_cos_to_codeword": mean_cos,
        "raw_unique": unique,
        "collision": 1.0 - unique / n_items,
        "mean_margin": margins.mean(axis=0),
    }


def angular_spread(
    model: RQVAE,
    embeddings: torch.Tensor,
    curvatures: tuple[float, ...],
    level: int,
    batch_size: int = 4096,
) -> np.ndarray:
    """Mean pairwise cosine spread of the residuals a codeword absorbs.

    ``mean_cos_to_codeword`` says how aligned the crowd is with the codeword's
    own direction. This says how wide the crowd is on its own, which is the
    quantity the hypothesis is about.
    """
    device = embeddings.device
    layers = model.rq.vq_layers
    n_codes = layers[0].n_embed
    total = np.zeros(n_codes, dtype=np.float64)
    weight = np.zeros(n_codes, dtype=np.float64)
    cap = 24
    for start in range(0, embeddings.shape[0], batch_size):
        batch = embeddings[start : start + batch_size]
        residual = model.encoder(batch)
        previous_codes = None
        for level2, layer2 in enumerate(layers):
            c2 = curvatures[level2]
            target_norm = model.rq._radius_for_level(level2, c2, residual)
            pinned = model.rq._pin_to_radius(residual, target_norm)
            distances = _pairwise_poincare_distance_tangents(
                pinned, layers[level2].get_code_embs(), c2
            )
            chosen = layer2._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            if level2 == level:
                units = (
                    pinned
                    / torch.linalg.vector_norm(
                        pinned, dim=-1, keepdim=True
                    ).clamp_min(1e-12)
                )
                # Mean cosine to the level's own residual mean direction, which
                # is the codeword's catchment direction.
                idx = chosen.cpu().numpy()
                flat = units.cpu().numpy()
                for code in np.unique(idx):
                    rows = flat[idx == code]
                    if len(rows) == 0:
                        continue
                    take = rows[:cap]
                    gram = take @ take.T
                    off = gram[~np.eye(len(take), dtype=bool)]
                    total[code] += off.mean() if off.size else np.nan
                    weight[code] += len(rows)
            residual = model.rq._restore_norm(
                _hyperbolic_residual(
                    pinned, layers[level2].get_code_embs()[chosen], c2
                ),
                residual,
                target_norm,
            )
            previous_codes = chosen
    with np.errstate(invalid="ignore"):
        return np.where(weight > 0, total / weight, np.nan)


def main() -> None:
    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = np.random.default_rng(experiment.DIAGNOSTIC_SEED)
    torch.manual_seed(experiment.DIAGNOSTIC_SEED)

    embeddings = torch.from_numpy(
        np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    ).to(device)
    n_items = embeddings.shape[0]
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    pick = generator.choice(
        len(source_ids),
        size=min(experiment.DIAGNOSTIC_PAIRS, len(source_ids)),
        replace=False,
    )
    source_ids, successor_ids = source_ids[pick], successor_ids[pick]
    negative_ids = generator.integers(0, n_items, size=len(source_ids))
    clash = negative_ids == successor_ids
    negative_ids[clash] = (negative_ids[clash] + 1) % n_items

    model = RQVAE(
        tokenizer_config(experiment.LAYER_CURVATURES), in_dim=embeddings.shape[1]
    ).to(device)
    checkpoint = torch.load(
        experiment.PARENT_CKPT, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    parent_curvatures = experiment.LAYER_CURVATURES
    base = traverse(model, embeddings, parent_curvatures)
    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "diagnostic_pairs": int(len(source_ids)),
        "items": int(n_items),
    }

    # ---- step 1: usage -> angular spread -> radius
    lines.append("1. is 'more used' explained by 'covers more directions'?")
    result["step1"] = {}
    for level in range(3):
        spread = angular_spread(
            model, embeddings, parent_curvatures, level
        )
        usage = base["usage"][level].astype(np.float64)
        radius = base["radii"][level].astype(np.float64)
        valid = np.isfinite(spread) & (usage > 0)
        r_usage = pearson(usage[valid], radius[valid])
        r_spread_radius = pearson(spread[valid], radius[valid])
        r_usage_spread = pearson(usage[valid], spread[valid])
        entry = {
            "corr_usage_radius": r_usage,
            "corr_spread_radius": r_spread_radius,
            "corr_usage_spread": r_usage_spread,
            "codewords_measured": int(valid.sum()),
        }
        result["step1"][f"L{level + 1}"] = entry
        lines.append(
            f"   L{level + 1}  corr(usage,s_e)={r_usage:+.4f}  "
            f"corr(spread,s_e)={r_spread_radius:+.4f}  "
            f"corr(usage,spread)={r_usage_spread:+.4f}  "
            f"(n={int(valid.sum())})"
        )

    # ---- step 2: sweep one level's curvature
    level = experiment.SWEEP_LEVEL
    lines.append(
        f"2. frozen sweep of c at L{level + 1}, others pinned at "
        f"{parent_curvatures}"
    )
    result["step2"] = {}
    parent_tokens = base["tokens"].copy()
    for value in experiment.SWEEP_CURVATURES:
        curvatures = list(parent_curvatures)
        curvatures[level] = value
        curvatures = tuple(curvatures)
        if value == parent_curvatures[level]:
            current = base
        else:
            current = traverse(model, embeddings, curvatures)
        flip = float(
            (current["tokens"][:, level] != parent_tokens[:, level]).mean()
        )
        radius = current["radii"][level].astype(np.float64)
        usage = current["usage"][level].astype(np.float64)
        valid = usage > 0
        cos_gap = float(
            (current["tokens"][source_ids, level]
             == current["tokens"][successor_ids, level]).mean()
            - (current["tokens"][source_ids, level]
               == current["tokens"][negative_ids, level]).mean()
        )
        entry = {
            "curvature": value,
            "radius_p50": float(np.median(radius)),
            "radius_p5": float(np.percentile(radius, 5)),
            "radius_p95": float(np.percentile(radius, 95)),
            "radius_spread": float(radius.max() - radius.min()),
            "corr_usage_radius": pearson(usage[valid], radius[valid]),
            "assignment_flip": flip,
            "mean_margin": float(current["mean_margin"][level]),
            "collision": current["collision"],
            "behaviour_prefix1_gain": cos_gap,
        }
        result["step2"][f"{value}"] = entry
        lines.append(
            f"   c={value:<5} s_e p5/p50/p95="
            f"{entry['radius_p5']:.5f}/{entry['radius_p50']:.5f}/"
            f"{entry['radius_p95']:.5f}  spread={entry['radius_spread']:.5f}  "
            f"corr(usage,s_e)={entry['corr_usage_radius']:+.4f}  "
            f"flip={flip * 100:5.2f}%  gap={entry['mean_margin']:.6f}  "
            f"beh_gain={cos_gap:+.4f}  collision={entry['collision']:.6f}"
        )

    # ---- step 3: verdict, from the frozen numbers only
    parent_entry = result["step2"][f"{parent_curvatures[level]}"]
    reshapes = [
        abs(e["radius_p50"] - parent_entry["radius_p50"])
        for key, e in result["step2"].items()
        if key != f"{parent_curvatures[level]}"
    ]
    behaviour_drift = [
        abs(e["behaviour_prefix1_gain"] - parent_entry["behaviour_prefix1_gain"])
        for key, e in result["step2"].items()
        if key != f"{parent_curvatures[level]}"
    ]
    reordered = any(
        (e["corr_usage_radius"] < 0) != (parent_entry["corr_usage_radius"] < 0)
        for key, e in result["step2"].items()
        if key != f"{parent_curvatures[level]}"
    )
    verdict = (
        "curvature moves the radial hierarchy in an orderly way and the "
        "behaviour signal survives, so it is a real control"
        if max(reshapes) > 1e-4 and max(behaviour_drift) < 0.01 and not reordered
        else "curvature does not reshape the radial hierarchy, or the "
        "reshuffle costs the behaviour signal; do not train on it"
    )
    result["verdict"] = verdict
    lines.append("3. frozen verdict")
    lines.append(
        f"   max |d median radius| = {max(reshapes):.6f}   "
        f"max |d behaviour gain| = {max(behaviour_drift):.6f}   "
        f"usage-radius sign flipped = {reordered}"
    )
    lines.append(f"   {verdict}")

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.DIAGNOSTIC_JSON.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    experiment.DIAGNOSTIC_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
