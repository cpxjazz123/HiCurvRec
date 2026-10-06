"""Estimate per-level curvature from frozen behaviour separation (iter95).

Nothing is trained here. The parent Stage2 checkpoint is loaded, encoder and
codebooks stay frozen, and only the curvature used in the distance computation
is varied. For each level the estimator measures

    G_l(c) = median[ d_c(A, X-) - d_c(A, B+) ]

over the behaviour pairs that reach that level, bootstraps a confidence interval
over the pair population, and keeps the parent's curvature unless a candidate is
both significantly better in behaviour separation and safe on the quantization
side. The sign of the curvature change comes only from behaviour separation;
pair counts only set how much evidence is required; margin and collision can
only refuse a candidate, never propose one.

Phase 1 of the mechanism is this estimator alone: it has to recover the parent's
curvatures from data. A phase 2 that retrains at the selected curvatures is not
wired up, because the selection came out equal to the parent and a retrain at
the parent's own setting would only reproduce the parent.
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
        chunk = torch.from_numpy(item_ids[start : start + batch_size]).to(device)
        batch = embeddings[chunk]
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


@torch.no_grad()
def negative_ids(
    source_ids: np.ndarray, successor_ids: np.ndarray, seed: int
) -> np.ndarray:
    """A fixed in-batch-style permutation: no negative equals its own positive."""
    generator = np.random.default_rng(seed)
    shifted = generator.permutation(len(source_ids))
    for _ in range(16):
        clash = shifted == np.arange(len(source_ids))
        if not clash.any():
            break
        shifted[clash] = generator.permutation(len(source_ids))[clash]
    return successor_ids[shifted]


@torch.no_grad()
def separation_and_quantization(
    model: RQVAE,
    embeddings: torch.Tensor,
    source_ids: np.ndarray,
    successor_ids: np.ndarray,
    negative_item_ids: np.ndarray,
    curvatures: tuple[float, ...],
) -> dict[str, list]:
    """Per-level behaviour gap, assignment margin, and corpus quantization stats.

    The level's pairs are fixed by the parent's own assignment: a pair belongs to
    level l only when the levels above it already gave A and B+ the same code.
    That keeps the pair population identical across the curvature probe, so the
    three candidates are compared on the same evidence.
    """
    source_tokens = model.get_indices(
        embeddings[torch.from_numpy(source_ids).to(embeddings.device)]
    ).cpu().numpy()
    successor_tokens = model.get_indices(
        embeddings[torch.from_numpy(successor_ids).to(embeddings.device)]
    ).cpu().numpy()

    source_latents = level_latents(model, embeddings, source_ids, curvatures)
    successor_latents = level_latents(
        model, embeddings, successor_ids, curvatures
    )
    negative_latents = level_latents(
        model, embeddings, negative_item_ids, curvatures
    )

    corpus_tokens = []
    corpus_margins = []
    corpus_usage = []
    with torch.no_grad():
        encoded = model.encoder(embeddings)
        residual = encoded
        previous_codes = None
        for level, layer in enumerate(model.rq.vq_layers):
            curvature = curvatures[level]
            source = residual
            target_norm = model.rq._radius_for_level(level, curvature, source)
            pinned = model.rq._pin_to_radius(source, target_norm)
            distances = _pairwise_poincare_distance_tangents(
                pinned, layer.get_code_embs(), curvature
            )
            two_smallest = torch.topk(distances, k=2, dim=1, largest=False).values
            corpus_margins.append(
                float((two_smallest[:, 1] - two_smallest[:, 0]).mean())
            )
            codes = layer._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            corpus_usage.append(int(torch.unique(codes).numel()))
            corpus_tokens.append(codes.cpu().numpy())
            quantized = layer.embed_code(codes)
            residual = model.rq._restore_norm(
                _hyperbolic_residual(pinned, quantized, curvature),
                source,
                target_norm,
            )
            previous_codes = codes

    tokens = np.stack(corpus_tokens, axis=1)
    unique = int(len(np.unique(tokens, axis=0)))
    stats = {
        "collision": 1.0 - unique / len(tokens),
        "raw_unique": unique,
        "raw_total": int(len(tokens)),
        "assignment_margins": corpus_margins,
        "codebook_used": corpus_usage,
    }

    gaps: list[np.ndarray] = []
    counts: list[int] = []
    for level in range(len(curvatures)):
        keep = np.ones(len(source_ids), dtype=bool)
        for upper in range(level):
            keep &= source_tokens[:, upper] == successor_tokens[:, upper]
        curvature = curvatures[level]
        gap = (
            _poincare_distance_tangent_pairs(
                source_latents[level][torch.from_numpy(keep).to(source_latents[level].device)],
                negative_latents[level][torch.from_numpy(keep).to(source_latents[level].device)],
                curvature,
            )
            - _poincare_distance_tangent_pairs(
                source_latents[level][torch.from_numpy(keep).to(source_latents[level].device)],
                successor_latents[level][torch.from_numpy(keep).to(source_latents[level].device)],
                curvature,
            )
        )
        gaps.append(gap.cpu().numpy().astype(np.float64))
        counts.append(int(keep.sum()))
    stats["behavior_gaps"] = gaps
    stats["behavior_pair_counts"] = counts
    return stats


def bootstrap_interval(
    values: np.ndarray,
    samples: int,
    seed: int,
    z: float,
) -> tuple[float, float, float]:
    """Median of the per-pair gap plus a bootstrap CI half-width for a median."""
    generator = np.random.default_rng(seed)
    size = len(values)
    medians = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        medians[index] = np.median(
            values[generator.integers(0, size, size)]
        )
    return (
        float(np.median(values)),
        float(z * medians.std(ddof=1)),
        float(medians.std(ddof=1) / np.sqrt(samples)),
    )


def curvature_table(report: dict, level: int) -> str:
    row = [f"L{level + 1}"]
    for candidate in report["candidates"]:
        row.append(
            f"G({candidate['curvature']:.1f})="
            f"{candidate['behavior_separation']:+.5f}"
            f"+-{candidate['separation_ci95']:.5f}"
        )
    row.append(f"g={report['first_derivative']:+.4f}")
    row.append(f"h={report['second_derivative']:+.3f}")
    row.append(f"chosen={report['selected_curvature']:.2f}")
    row.append(f"reason={report['reason']}")
    return "  ".join(row)


def main() -> None:
    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(experiment.ESTIMATOR_BOOTSTRAP_SEED)
    np.random.seed(experiment.ESTIMATOR_BOOTSTRAP_SEED)

    embeddings = torch.from_numpy(
        np.asarray(np.load(experiment.EMBEDDING_FILE), dtype=np.float32)
    ).to(device)
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    negative_item_ids = negative_ids(
        source_ids, successor_ids, experiment.ESTIMATOR_BOOTSTRAP_SEED
    )

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

    candidates = list(experiment.ESTIMATOR_CURVATURES)
    probes: dict[float, dict] = {}
    for level_curvatures in [
        (candidate,) * 3 for candidate in candidates
    ]:
        candidate = level_curvatures[0]
        probes[candidate] = separation_and_quantization(
            model,
            embeddings,
            source_ids,
            successor_ids,
            negative_item_ids,
            level_curvatures,
        )

    parent = probes[PARENT_CURVATURE]
    levels: list[dict] = []
    for level in range(3):
        parent_gap = parent["behavior_gaps"][level]
        level_rows = []
        for candidate in candidates:
            gap = probes[candidate]["behavior_gaps"][level]
            median, ci95, stderr = bootstrap_interval(
                gap,
                experiment.ESTIMATOR_BOOTSTRAP_SAMPLES,
                experiment.ESTIMATOR_BOOTSTRAP_SEED + level,
                experiment.ESTIMATOR_CONFIDENCE_Z,
            )
            level_rows.append(
                {
                    "curvature": candidate,
                    "behavior_separation": median,
                    "separation_ci95": ci95,
                    "separation_stderr": stderr,
                }
            )
        low, mid, high = (row["behavior_separation"] for row in level_rows)
        first = (high - low) / (candidates[2] - candidates[0])
        second = (high - 2.0 * mid + low) / (
            (candidates[2] - candidates[1]) ** 2
        )
        levels.append(
            {
                "level": level + 1,
                "behavior_pair_count": parent["behavior_pair_counts"][level],
                "candidates": level_rows,
                "first_derivative": first,
                "second_derivative": second,
            }
        )

    for level_report in levels:
        level = level_report["level"] - 1
        parent_row = next(
            row
            for row in level_report["candidates"]
            if row["curvature"] == PARENT_CURVATURE
        )
        parent_median = parent_row["behavior_separation"]
        parent_ci = parent_row["separation_ci95"]
        parent_margin = parent["assignment_margins"][level]
        parent_collision = parent["collision"]

        best = max(
            level_report["candidates"],
            key=lambda row: row["behavior_separation"],
        )
        improvement = best["behavior_separation"] - parent_median
        # The candidate's own CI must clear the parent, and the two medians
        # must be separated by more than their noise.
        separated = (
            best["behavior_separation"] - best["separation_ci95"] > parent_median
            and improvement > parent_ci + best["separation_ci95"]
        )
        collision_ok = (
            probes[best["curvature"]]["collision"]
            <= parent_collision
            + experiment.ESTIMATOR_MAX_COLLISION_INCREASE
        )
        margin_ok = (
            probes[best["curvature"]]["assignment_margins"][level]
            >= parent_margin * experiment.ESTIMATOR_MIN_MARGIN_RETENTION
        )

        if best["curvature"] == PARENT_CURVATURE:
            reason = "parent is the best of the probed candidates"
        elif not separated:
            reason = (
                f"best={best['curvature']} gain={improvement:+.5f} within "
                f"noise (ci95 {best['separation_ci95']:.5f} + {parent_ci:.5f})"
            )
        elif not collision_ok:
            reason = (
                f"best={best['curvature']} rejected: collision "
                f"{probes[best['curvature']]['collision']:.6f} > parent "
                f"{parent_collision:.6f}"
            )
        elif not margin_ok:
            reason = (
                f"best={best['curvature']} rejected: margin "
                f"{probes[best['curvature']]['assignment_margins'][level]:.6f} "
                f"< parent {parent_margin:.6f}"
            )
        else:
            reason = (
                f"best={best['curvature']} significant gain={improvement:+.5f}"
            )
        level_report["parent_separation"] = parent_median
        level_report["parent_separation_ci95"] = parent_ci
        level_report["best_candidate"] = best["curvature"]
        level_report["significant_improvement"] = bool(
            separated and collision_ok and margin_ok
        )
        level_report["selected_curvature"] = (
            float(best["curvature"])
            if separated and collision_ok and margin_ok
            else PARENT_CURVATURE
        )
        level_report["reason"] = reason
        level_report["parent_assignment_margin"] = parent_margin
        level_report["parent_collision"] = parent_collision
        level_report["candidate_assignment_margins"] = {
            f"{candidate:.1f}": probes[candidate]["assignment_margins"][level]
            for candidate in candidates
        }
        level_report["candidate_collisions"] = {
            f"{candidate:.1f}": probes[candidate]["collision"]
            for candidate in candidates
        }
        level_report["candidate_codebook_used"] = {
            f"{candidate:.1f}": probes[candidate]["codebook_used"][level]
            for candidate in candidates
        }

    selection = [level_report["selected_curvature"] for level_report in levels]
    result = {
        "geometry": "poincare_behavior_separation_estimator",
        "mechanism": experiment.MECHANISM_NAME,
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "parent_curvatures": [PARENT_CURVATURE] * 3,
        "probe_curvatures": candidates,
        "bootstrap_samples": experiment.ESTIMATOR_BOOTSTRAP_SAMPLES,
        "confidence_z": experiment.ESTIMATOR_CONFIDENCE_Z,
        "behavior_pairs_total": int(len(source_ids)),
        "levels": levels,
        "selected_curvatures": selection,
        "recovers_parent": selection == [PARENT_CURVATURE] * 3,
    }

    lines = [
        "iter95 frozen curvature estimator",
        f"parent checkpoint: {experiment.PARENT_CKPT}",
        f"parent curvatures: {[PARENT_CURVATURE] * 3}",
        f"probe curvatures: {candidates}",
        f"behaviour pairs: {len(source_ids)}",
    ]
    for level_report in levels:
        lines.append(curvature_table(level_report, level_report["level"] - 1))
    lines.append(f"selected curvatures: {selection}")
    lines.append(f"recovers parent (1,1,1): {result['recovers_parent']}")
    report = "\n".join(lines)
    print(report, flush=True)

    experiment.STAGE2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    experiment.ESTIMATOR_JSON.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    experiment.ESTIMATOR_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
