"""Frozen diagnostic: does curvature act on the codewords' radial geometry?

Nothing is trained. The parent checkpoint is loaded and read, and the
questions asked are:

  1. What is the radial distribution of each level's codewords, s_e = sqrt(c)*||e_j||,
     and do the three levels sit at different radii?
  2. Is a codeword's usage correlated with its radius?
  3. Is a behaviour transition's radial codeword difference smaller than a
     random one?
  4. Does behaviour live in the radius or in the direction?
  5. If codewords are moved onto the same shell as the residual, does the
     assignment actually change, and does that change help behaviour?

The last one is the decisive question and is reported as a diagnostic only: the
assignment is recomputed with both the residual and the codeword pinned to the
same working point, exactly as the parent's residual pin already does, and the
result is compared against the parent's own assignment on the same rows.
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
    _expmap0_tangent,
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


def percentiles(values: np.ndarray) -> dict[str, float]:
    quantiles = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "p5": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


@torch.no_grad()
def traverse(
    model: RQVAE,
    embeddings: torch.Tensor,
    pin_codewords: bool,
    batch_size: int = 4096,
) -> dict:
    """Run the RQ stack and keep what each question needs.

    ``pin_codewords`` moves every codeword onto the same working point as the
    residual before the assignment. Nothing is written back to the model: this
    is a readout of an alternative geometry, not a new parameterisation.
    """
    device = embeddings.device
    n_items = embeddings.shape[0]
    layers = model.rq.vq_layers
    tokens = np.empty((n_items, len(layers)), dtype=np.int64)
    radii = np.empty((len(layers), len(layers[0].get_code_embs())), dtype=np.float64)
    margins = np.empty((len(layers), n_items), dtype=np.float64)
    codebook: list[torch.Tensor] = []
    for level, layer in enumerate(layers):
        curvature = float(layer.get_curvature())
        codes = layer.get_code_embs()
        radii[level] = (
            curvature**0.5
        ) * torch.linalg.vector_norm(codes, dim=-1).cpu().numpy()
        if pin_codewords:
            # ArcVQ-style: put the codeword on the same working point as the
            # residual, so only its direction is left to decide the assignment.
            target = experiment.LAYER_WORKING_RADII[level] / curvature**0.5
            codes = (
                codes
                / torch.linalg.vector_norm(codes, dim=-1, keepdim=True).clamp_min(1e-12)
                * target
            )
        codebook.append(codes)
    for start in range(0, n_items, batch_size):
        batch = embeddings[start : start + batch_size]
        residual = model.encoder(batch)
        previous_codes = None
        for level, layer in enumerate(layers):
            curvature = float(layer.get_curvature())
            target_norm = model.rq._radius_for_level(level, curvature, residual)
            pinned = model.rq._pin_to_radius(residual, target_norm)
            distances = _pairwise_poincare_distance_tangents(
                pinned, codebook[level], curvature
            )
            two = torch.topk(distances, k=2, dim=1, largest=False).values
            margins[level, start : start + batch.shape[0]] = (
                two[:, 1] - two[:, 0]
            ).cpu().numpy()
            chosen = layer._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            tokens[start : start + batch.shape[0], level] = chosen.cpu().numpy()
            residual = model.rq._restore_norm(
                _hyperbolic_residual(
                    pinned, codebook[level][chosen], curvature
                ),
                residual,
                target_norm,
            )
            previous_codes = chosen
    unique = int(len(np.unique(tokens, axis=0)))
    return {
        "tokens": tokens,
        "radii": radii,
        "margins": margins,
        "raw_unique": unique,
        "collision": 1.0 - unique / n_items,
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
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    pick = generator.choice(
        len(source_ids), size=min(experiment.DIAGNOSTIC_PAIRS, len(source_ids)),
        replace=False,
    )
    source_ids = source_ids[pick]
    successor_ids = successor_ids[pick]
    # A negative is a random item that is not the pair's own successor.
    negative_ids = generator.integers(0, n_items, size=len(source_ids))
    clash = negative_ids == successor_ids
    negative_ids[clash] = (negative_ids[clash] + 1) % n_items

    model = RQVAE(tokenizer_config(), in_dim=embeddings.shape[1]).to(device)
    checkpoint = torch.load(
        experiment.PARENT_CKPT, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    base = traverse(model, embeddings, pin_codewords=False)
    repinned = traverse(model, embeddings, pin_codewords=True)
    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "diagnostic_pairs": int(len(source_ids)),
        "items": int(n_items),
    }

    # (1) radial distribution per level
    lines.append("1. codeword radial distribution  s_e = sqrt(c)*||e_j||")
    radial: dict = {}
    for level in range(3):
        stats = percentiles(base["radii"][level])
        radial[f"L{level + 1}"] = stats
        lines.append(
            f"   L{level + 1}  min={stats['min']:.6f} p5={stats['p5']:.6f} "
            f"p25={stats['p25']:.6f} p50={stats['p50']:.6f} "
            f"p75={stats['p75']:.6f} p95={stats['p95']:.6f} "
            f"max={stats['max']:.6f}  (residual shell s=0.2)"
        )
    medians = [radial[f"L{l + 1}"]["p50"] for l in range(3)]
    result["codeword_radial"] = radial
    result["level_median_radii"] = medians
    lines.append(
        "   level medians differ by "
        f"{max(medians) - min(medians):.6f} absolute "
        f"({(max(medians) - min(medians)) / min(medians) * 100:.1f}%)"
    )

    # (2) usage vs radius
    lines.append("2. corr(usage, s_e) per level")
    usage: list[np.ndarray] = []
    for level in range(3):
        counts = np.bincount(base["tokens"][:, level], minlength=256)
        usage.append(counts)
        r = pearson(base["radii"][level], counts.astype(np.float64))
        result.setdefault("usage_radius_corr", {})[f"L{level + 1}"] = r
        lines.append(
            f"   L{level + 1}  corr={r:+.4f}  used={int((counts > 0).sum())}/256  "
            f"usage[min={counts.min()} max={counts.max()}]"
        )
    result["code_usage"] = [u.tolist() for u in usage]

    # (3)(4) behaviour in radius vs direction
    lines.append("3/4. behaviour signal by radius and by direction")
    result["behaviour"] = {}
    for level in range(3):
        tok = base["tokens"]
        s = base["radii"][level]
        r_pos = np.abs(
            s[tok[source_ids, level]] - s[tok[successor_ids, level]]
        )
        r_neg = np.abs(
            s[tok[source_ids, level]] - s[tok[negative_ids, level]]
        )
        codes = model.rq.vq_layers[level].get_code_embs()
        unit = codes / torch.linalg.vector_norm(
            codes, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        unit = unit.cpu().numpy()
        c_pos = (unit[tok[source_ids, level]] * unit[tok[successor_ids, level]]).sum(-1)
        c_neg = (unit[tok[source_ids, level]] * unit[tok[negative_ids, level]]).sum(-1)
        entry = {
            "radial_diff_positive": float(r_pos.mean()),
            "radial_diff_negative": float(r_neg.mean()),
            "radial_diff_gap": float(r_neg.mean() - r_pos.mean()),
            "cosine_positive": float(c_pos.mean()),
            "cosine_negative": float(c_neg.mean()),
            "cosine_gap": float(c_pos.mean() - c_neg.mean()),
        }
        result["behaviour"][f"L{level + 1}"] = entry
        lines.append(
            f"   L{level + 1}  |ds| pos={entry['radial_diff_positive']:.6f} "
            f"neg={entry['radial_diff_negative']:.6f} "
            f"gap={entry['radial_diff_gap']:+.6f}   "
            f"cos pos={entry['cosine_positive']:+.6f} "
            f"neg={entry['cosine_negative']:+.6f} "
            f"gap={entry['cosine_gap']:+.6f}"
        )

    # (5) does re-pinning the codewords change the assignment?
    lines.append("5. assignment with the codeword pinned to the same shell")
    flips = (base["tokens"] != repinned["tokens"]).mean(axis=0)
    per_level_flip = flips.tolist()
    result["assignment_flip_rate"] = per_level_flip
    result["assignment_flip_rate_overall"] = float(flips.mean())
    for level in range(3):
        used = int(len(np.unique(repinned["tokens"][:, level])))
        lines.append(
            f"   L{level + 1}  flip={per_level_flip[level] * 100:.2f}%  "
            f"used={used}/256  "
            f"top1-top2 gap parent={base['margins'][level].mean():.6f} "
            f"repinned={repinned['margins'][level].mean():.6f}"
        )
    result["parent_collision"] = base["collision"]
    result["repinned_collision"] = repinned["collision"]
    result["parent_raw_unique"] = base["raw_unique"]
    result["repinned_raw_unique"] = repinned["raw_unique"]
    lines.append(
        f"   collision parent={base['collision']:.6f} "
        f"repinned={repinned['collision']:.6f}   "
        f"raw_unique parent={base['raw_unique']} "
        f"repinned={repinned['raw_unique']}"
    )

    # (7) did the change help behaviour or only churn the assignment?
    # With every codeword on the shell the radial difference between two
    # assignments is zero by construction, so the honest question is whether
    # the re-pinned assignment still puts behaviour pairs together more often
    # than random pairs. Prefix width, not level, is the varying axis here: a
    # behaviour pair that agrees on the first l codes was grouped by the stack
    # up to level l regardless of which level is being reported.
    lines.append("7. does the re-pinned assignment still group behaviour pairs?")
    tp = repinned["tokens"]
    tb = base["tokens"]
    result["behaviour_agreement"] = {}
    for width in (1, 2, 3):
        block = {
            "parent_positive": float(
                np.all(
                    tb[source_ids, :width] == tb[successor_ids, :width], axis=1
                ).mean()
            ),
            "parent_negative": float(
                np.all(
                    tb[source_ids, :width] == tb[negative_ids, :width], axis=1
                ).mean()
            ),
            "repinned_positive": float(
                np.all(
                    tp[source_ids, :width] == tp[successor_ids, :width], axis=1
                ).mean()
            ),
            "repinned_negative": float(
                np.all(
                    tp[source_ids, :width] == tp[negative_ids, :width], axis=1
                ).mean()
            ),
        }
        block["parent_gain"] = block["parent_positive"] - block["parent_negative"]
        block["repinned_gain"] = (
            block["repinned_positive"] - block["repinned_negative"]
        )
        result["behaviour_agreement"][f"prefix{width}"] = block
        lines.append(
            f"   prefix{width}: parent pos/neg="
            f"{block['parent_positive'] * 100:5.2f}/{block['parent_negative'] * 100:5.2f}%  "
            f"repinned pos/neg="
            f"{block['repinned_positive'] * 100:5.2f}/{block['repinned_negative'] * 100:5.2f}%  "
            f"behaviour gain {block['parent_gain']:+.4f} -> {block['repinned_gain']:+.4f}"
        )

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.DIAGNOSTIC_JSON.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    experiment.DIAGNOSTIC_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
