"""Estimate per-level curvature from frozen behaviour ranking (iter96).

iter95 measured a median distance gap and found the level's own behaviour
separation preferred a smaller curvature than the parent's. That metric moves
with the hyperbolic distance scale itself, so a curvature change could have
rescaled every distance without improving any behaviour relation. This estimator
drops the distance magnitudes and looks only at the ordering:

    R_l(A, B+; c) = (1/K) * sum_k 1[ d_c(A, B+) < d_c(A, X_k-) ]
    Q_l(c)       = E[R_l]

Q_l is the fraction of cases in which the true behaviour successor ranks
before a negative for the same anchor, so it is invariant to any monotone
rescaling of the distance. Negatives are drawn once per pair with a fixed seed
and reused across curvature candidates, and the same set is used for every
level so the three candidates are compared on identical evidence.

Two negative families are reported. Global negatives are sampled from the whole
item catalogue and test whether behaviour survives against arbitrary items.
Prefix-hard negatives are sampled only from items sharing the anchor's parent
prefix at that level, so they test the local resolution task the level is
actually responsible for; that is the reported signal.

Nothing is trained. The parent Stage2 checkpoint is loaded, encoder and
codebooks stay frozen, and only the curvature in the distance computation moves.
Assignment margin and collision are recorded as diagnostics; they do not enter
the decision.
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

PARENT_CURVATURE = 1.0
MAX_HARD_NEGATIVES = 64


def tokenizer_config(curvatures: tuple[float, ...]) -> SimpleNamespace:
    """Build the parent tokenizer config at a given per-level curvature."""
    return SimpleNamespace(
        hidden_sizes=(512, 256, 128),
        codebook_num=3,
        codebook_size=(256, 256, 256),
        codebook_dim=32,
        dropout=0.0,
        beta=0.25,
        vq_type="vq",
        ema_decay=0.99,
        fix_code_embs=False,
        sk_epsilon=0.003,
        sk_iters=50,
        layer_curvatures=curvatures,
        layer_working_radii=(0.2, 0.2, 0.2),
        pin_in_s_coordinates=True,
    )


def transition_pairs(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Source and successor item ids from consecutive train-sequence events."""
    sources: list[int] = []
    successors: list[int] = []
    for history, target in zip(
        train_frame["seen_history"].to_numpy(),
        train_frame["target"].to_numpy(dtype=np.int64),
    ):
        if history is None or len(history) == 0:
            continue
        source = int(history[-1])
        if source < 0:
            raise ValueError("Transition source item is negative")
        sources.append(source)
        successors.append(int(target))
    if not sources:
        raise ValueError("Training data has no usable item transitions")
    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(successors, dtype=np.int64),
    )


@torch.no_grad()
def level_latents(
    model: RQVAE,
    embeddings: torch.Tensor,
    item_ids: np.ndarray,
    curvatures: tuple[float, ...],
    batch_size: int = 4096,
) -> list[torch.Tensor]:
    """Per-level pinned input latents for the given items, one level per entry.

    The traversal mirrors training at inference: each level quantizes its
    residual at the working point and the level's own curvature, then the
    hyperbolic residual is restored to the encoder's norm. Because the pin is in
    metric coordinates, changing a level's curvature moves only the metric
    factor and the depth stays fixed.
    """
    device = embeddings.device
    collected: list[list[torch.Tensor]] = [[] for _ in curvatures]
    for start in range(0, len(item_ids), batch_size):
        chunk = item_ids[start : start + batch_size]
        if not isinstance(chunk, torch.Tensor):
            chunk = torch.from_numpy(np.ascontiguousarray(chunk))
        batch = embeddings[chunk.to(device)]
        residual = model.encoder(batch)
        previous_codes = None
        for level, layer in enumerate(model.rq.vq_layers):
            curvature = curvatures[level]
            source = residual
            target_norm = model.rq._radius_for_level(level, curvature, source)
            pinned = model.rq._pin_to_radius(source, target_norm)
            collected[level].append(pinned)
            distances = _pairwise_poincare_distance_tangents(
                pinned, layer.get_code_embs(), curvature
            )
            codes = layer._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            quantized = layer.embed_code(codes)
            residual = model.rq._restore_norm(
                _hyperbolic_residual(pinned, quantized, curvature),
                source,
                target_norm,
            )
            previous_codes = codes
    return [torch.cat(parts, dim=0) for parts in collected]


def sample_global_negatives(
    item_count: int, pair_count: int, negatives: int, seed: int
) -> np.ndarray:
    """(pair_count, negatives) catalogue-wide negatives, never a successor."""
    generator = np.random.default_rng(seed)
    drawn = generator.integers(0, item_count, size=(pair_count, negatives))
    return drawn.astype(np.int64)


def prefix_hard_negatives(
    pair_prefix: np.ndarray,
    bucket_members: dict[bytes, np.ndarray],
    item_count: int,
    negatives: int,
    seed: int,
) -> np.ndarray:
    """Negatives drawn from the anchor's own parent-prefix bucket at that level.

    Buckets are the prefixes the parent's own assignment produced, so a hard
    negative differs from the anchor only at the level under test. A bucket too
    small to serve K distinct alternatives falls back to the whole catalogue,
    which keeps every level measurable instead of dropping the pair.
    """
    generator = np.random.default_rng(seed)
    pair_count = len(pair_prefix)
    keys = _prefix_keys(pair_prefix)
    out = np.empty((pair_count, negatives), dtype=np.int64)
    for index in range(pair_count):
        members = bucket_members.get(keys[index])
        if members is None or len(members) < negatives + 1:
            out[index] = generator.integers(0, item_count, size=negatives)
            continue
        out[index] = members[
            generator.integers(0, len(members), size=negatives)
        ]
    return out


def build_bucket_members(item_prefix: np.ndarray) -> dict[bytes, np.ndarray]:
    """Map each parent prefix to the item ids carrying it.

    A zero-width prefix means the level has no parent prefix to share yet, so
    every item lands in one bucket keyed by the empty string.
    """
    keys = _prefix_keys(item_prefix)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(sorted_keys)]))
    return {
        sorted_keys[start]: np.sort(order[start:end].astype(np.int64))
        for start, end in zip(starts, ends)
    }


def _prefix_keys(prefix: np.ndarray) -> np.ndarray:
    """Row-wise byte keys, so a prefix array can be grouped and looked up."""
    rows = np.ascontiguousarray(prefix)
    width = rows.shape[1]
    if width == 0:
        return np.array([b""] * len(rows), dtype=object)
    return (
        rows.view(np.dtype((np.void, width * rows.dtype.itemsize)))
        .ravel()
        .astype(object)
    )


def level_masks(source_tokens: np.ndarray, successor_tokens: np.ndarray) -> list[np.ndarray]:
    """Pairs that still belong to each level: agreeing on all earlier codes."""
    masks = []
    for level in range(source_tokens.shape[1]):
        keep = np.ones(len(source_tokens), dtype=bool)
        for upper in range(level):
            keep &= source_tokens[:, upper] == successor_tokens[:, upper]
        masks.append(keep)
    return masks


@torch.no_grad()
def ranking_scores(
    model: RQVAE,
    embeddings: torch.Tensor,
    level: int,
    curvatures: tuple[float, ...],
    source_ids: np.ndarray,
    successor_ids: np.ndarray,
    negative_ids: np.ndarray,
    row_index: np.ndarray,
    curvature: float,
    negatives: int,
    chunk: int = 1024,
) -> np.ndarray:
    """Fraction of negatives the true successor outranks, per pair.

    Latents are recomputed per chunk because the negative set is far larger than
    the pair set; materialising every negative latent at once would be tens of
    gigabytes. The traversal is deterministic, so chunking does not change the
    result.
    """
    device = embeddings.device
    source_latent = level_latents(
        model, embeddings, source_ids, curvatures
    )[level]
    successor_latent = level_latents(
        model, embeddings, successor_ids, curvatures
    )[level]
    source_latent = source_latent[torch.from_numpy(row_index).to(device)]
    successor_latent = successor_latent[torch.from_numpy(row_index).to(device)]
    positive = _poincare_distance_tangent_pairs(
        source_latent, successor_latent, curvature
    )
    scores = np.empty(len(row_index), dtype=np.float64)
    for start in range(0, len(row_index), chunk):
        stop = min(start + chunk, len(row_index))
        flat = torch.from_numpy(
            negative_ids[row_index[start:stop]].reshape(-1)
        ).to(device)
        negative_latent = level_latents(
            model, embeddings, flat, curvatures
        )[level].reshape(stop - start, negatives, -1)
        distance = _poincare_distance_tangent_pairs(
            source_latent[start:stop].unsqueeze(1), negative_latent, curvature
        )
        wins = (positive[start:stop].unsqueeze(1) < distance).to(torch.float64)
        scores[start:stop] = wins.mean(dim=1).cpu().numpy()
    return scores


@torch.no_grad()
def corpus_quantization(
    model: RQVAE,
    embeddings: torch.Tensor,
    curvatures: tuple[float, ...],
) -> dict:
    """Assignment margin, collision, and code usage as diagnostics."""
    with torch.no_grad():
        residual = model.encoder(embeddings)
        previous_codes = None
        tokens = []
        margins = []
        usage = []
        for level, layer in enumerate(model.rq.vq_layers):
            curvature = curvatures[level]
            source = residual
            target_norm = model.rq._radius_for_level(level, curvature, source)
            pinned = model.rq._pin_to_radius(source, target_norm)
            distances = _pairwise_poincare_distance_tangents(
                pinned, layer.get_code_embs(), curvature
            )
            two_smallest = torch.topk(
                distances, k=2, dim=1, largest=False
            ).values
            margins.append(
                float((two_smallest[:, 1] - two_smallest[:, 0]).mean())
            )
            codes = layer._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            usage.append(int(torch.unique(codes).numel()))
            tokens.append(codes.cpu().numpy())
            quantized = layer.embed_code(codes)
            residual = model.rq._restore_norm(
                _hyperbolic_residual(pinned, quantized, curvature),
                source,
                target_norm,
            )
            previous_codes = codes
    stacked = np.stack(tokens, axis=1)
    unique = int(len(np.unique(stacked, axis=0)))
    return {
        "assignment_margins": margins,
        "codebook_used": usage,
        "collision": 1.0 - unique / len(stacked),
        "raw_unique": unique,
    }


def bootstrap_interval(
    values: np.ndarray,
    samples: int,
    seed: int,
    z: float,
) -> tuple[float, float]:
    """Mean and a bootstrap CI half-width over the pair population."""
    generator = np.random.default_rng(seed)
    size = len(values)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = values[
            generator.integers(0, size, size)
        ].mean()
    return float(values.mean()), float(z * means.std(ddof=1))


def main() -> None:
    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(experiment.ESTIMATOR_BOOTSTRAP_SEED)
    np.random.seed(experiment.ESTIMATOR_BOOTSTRAP_SEED)

    embedding_array = np.asarray(
        np.load(experiment.EMBEDDING_FILE), dtype=np.float32
    )
    embeddings = torch.from_numpy(embedding_array).to(device)
    item_count = embeddings.shape[0]
    item_ids = np.arange(item_count, dtype=np.int64)

    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)

    model = RQVAE(
        tokenizer_config((PARENT_CURVATURE,) * 3), in_dim=embeddings.shape[1]
    ).to(device)
    checkpoint = torch.load(
        experiment.PARENT_CKPT, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if [float(value) for value in checkpoint["curvatures"]] != [1.0, 1.0, 1.0]:
        raise RuntimeError(
            f"Parent checkpoint is not the fixed-curvature run: "
            f"{checkpoint['curvatures']}"
        )

    parent_curvatures = (PARENT_CURVATURE,) * 3
    corpus_tokens = model.get_indices(embeddings).cpu().numpy()
    source_tokens = corpus_tokens[source_ids]
    successor_tokens = corpus_tokens[successor_ids]

    negatives = experiment.ESTIMATOR_NEGATIVES_PER_PAIR
    masks = level_masks(source_tokens, successor_tokens)
    pair_count = len(source_ids)

    global_negatives = sample_global_negatives(
        item_count, pair_count, negatives, experiment.ESTIMATOR_BOOTSTRAP_SEED
    )
    # The true successor must never also serve as its own negative.
    for index in range(pair_count):
        clash = global_negatives[index] == successor_ids[index]
        if clash.any():
            global_negatives[index][clash] = (
                global_negatives[index][clash] + 1
            ) % item_count

    hard_negatives: list[np.ndarray] = []
    for level in range(3):
        # Level l's hard negatives must share the anchor's prefix over the levels
        # above l, so the pair differs from the anchor only at the level under
        # test. At L1 that prefix is empty and the bucket is the whole catalogue.
        prefix_width = level
        pair_prefix = source_tokens[:, :prefix_width]
        item_prefix = corpus_tokens[:, :prefix_width]
        bucket_members = build_bucket_members(item_prefix)
        drawn = prefix_hard_negatives(
            pair_prefix,
            bucket_members,
            item_count,
            negatives,
            experiment.ESTIMATOR_BOOTSTRAP_SEED + 100 + level,
        )
        # The true successor must never also serve as its own negative.
        for index in range(pair_count):
            clash = drawn[index] == successor_ids[index]
            if clash.any():
                drawn[index][clash] = (drawn[index][clash] + 1) % item_count
        hard_negatives.append(drawn)

    candidates = list(experiment.ESTIMATOR_CURVATURES)
    negative_families = {"global": global_negatives, "hard": hard_negatives}

    levels: list[dict] = []
    for level in range(3):
        keep = masks[level]
        keep_index = np.flatnonzero(keep)
        level_report: dict = {
            "level": level + 1,
            "behavior_pair_count": int(keep.sum()),
            "families": {},
        }
        for family, negative_ids in negative_families.items():
            rows = []
            for candidate in candidates:
                curvatures = [PARENT_CURVATURE] * 3
                curvatures[level] = candidate
                chosen = (
                    negative_ids[level][keep_index]
                    if family == "hard"
                    else negative_ids[keep_index]
                )
                scores = ranking_scores(
                    model,
                    embeddings,
                    level,
                    tuple(curvatures),
                    source_ids,
                    successor_ids,
                    chosen,
                    np.arange(len(keep_index)),
                    candidate,
                    negatives,
                )
                mean, ci95 = bootstrap_interval(
                    scores,
                    experiment.ESTIMATOR_BOOTSTRAP_SAMPLES,
                    experiment.ESTIMATOR_BOOTSTRAP_SEED + 10 * level,
                    experiment.ESTIMATOR_CONFIDENCE_Z,
                )
                rows.append(
                    {
                        "curvature": candidate,
                        "ranking_quality": mean,
                        "ranking_ci95": ci95,
                    }
                )
            best = max(rows, key=lambda row: row["ranking_quality"])
            parent_row = next(
                row
                for row in rows
                if row["curvature"] == PARENT_CURVATURE
            )
            level_report["families"][family] = {
                "candidates": rows,
                "parent_quality": parent_row["ranking_quality"],
                "parent_ci95": parent_row["ranking_ci95"],
                "best_candidate": best["curvature"],
                "best_quality": best["ranking_quality"],
                "best_ci95": best["ranking_ci95"],
                "parent_significantly_beaten": bool(
                    best["ranking_quality"] - best["ranking_ci95"]
                    > parent_row["ranking_quality"]
                    and best["ranking_quality"] - parent_row["ranking_quality"]
                    > best["ranking_ci95"] + parent_row["ranking_ci95"]
                ),
                "parent_is_local_max": bool(
                    rows[1]["ranking_quality"] >= rows[0]["ranking_quality"]
                    and rows[1]["ranking_quality"] >= rows[2]["ranking_quality"]
                ),
            }
        hard = level_report["families"]["hard"]
        selected_curvature = (
            float(hard["best_candidate"])
            if hard["parent_significantly_beaten"]
            else PARENT_CURVATURE
        )
        if hard["parent_is_local_max"]:
            reason = "parent is the local maximum of the hard-negative ranking"
        elif hard["parent_significantly_beaten"]:
            reason = (
                f"hard ranking prefers {hard['best_candidate']} by "
                f"{hard['best_quality'] - hard['parent_quality']:+.5f}"
            )
        else:
            reason = (
                f"no candidate beats the parent beyond noise "
                f"(best {hard['best_candidate']} "
                f"{hard['best_quality'] - hard['parent_quality']:+.5f})"
            )
        level_report["selected_curvature"] = selected_curvature
        level_report["reason"] = reason
        levels.append(level_report)
        print(
            f"  L{level + 1} N={int(keep.sum())} "
            + "  ".join(
                f"Q_{family}({row['curvature']:.1f})={row['ranking_quality']:.4f}"
                f"+-{row['ranking_ci95']:.4f}"
                for family, block in level_report["families"].items()
                for row in block["candidates"]
            ),
            flush=True,
        )

    for candidate in candidates:
        curvatures = [candidate] * 3
        print(
            f"  diagnostics c={candidate}: "
            f"margin={corpus_quantization(model, embeddings, curvatures)['assignment_margins']}",
            flush=True,
        )

    diagnostics = {
        f"{candidate:.1f}": corpus_quantization(
            model, embeddings, [candidate] * 3
        )
        for candidate in candidates
    }
    selection = [level["selected_curvature"] for level in levels]
    result = {
        "geometry": "poincare_behavior_ranking_estimator",
        "mechanism": experiment.MECHANISM_NAME,
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "parent_curvatures": [PARENT_CURVATURE] * 3,
        "probe_curvatures": candidates,
        "negatives_per_pair": negatives,
        "bootstrap_samples": experiment.ESTIMATOR_BOOTSTRAP_SAMPLES,
        "confidence_z": experiment.ESTIMATOR_CONFIDENCE_Z,
        "behavior_pairs_total": int(pair_count),
        "levels": levels,
        "selected_curvatures": selection,
        "recovers_parent": selection == [PARENT_CURVATURE] * 3,
        "quantization_diagnostics": diagnostics,
    }

    lines = [
        "iter96 frozen behaviour-ranking curvature estimator",
        f"parent checkpoint: {experiment.PARENT_CKPT}",
        f"probe curvatures: {candidates}  negatives/pair: {negatives}",
        f"behaviour pairs: {pair_count}",
    ]
    for level in levels:
        lines.append(f"L{level['level']}  N={level['behavior_pair_count']}")
        for family, block in level["families"].items():
            cells = "  ".join(
                f"Q({row['curvature']:.1f})={row['ranking_quality']:.4f}"
                f"+-{row['ranking_ci95']:.4f}"
                for row in block["candidates"]
            )
            lines.append(f"  {family:<7} {cells}")
        lines.append(
            f"  chosen={level['selected_curvature']:.2f}  {level['reason']}"
        )
    lines.append(f"selected curvatures: {selection}")
    lines.append(f"recovers parent (1,1,1): {result['recovers_parent']}")
    report = "\n".join(lines)
    print(report, flush=True)
    experiment.ESTIMATOR_JSON.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    experiment.ESTIMATOR_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
