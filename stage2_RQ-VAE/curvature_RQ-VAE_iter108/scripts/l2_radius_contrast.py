"""Frozen contrast: did c_2 = 2 actually leave the L2 codebook further out?

iter106 measured the effect of c on a frozen parent codebook, where L2's median
radius went from 0.0845 to 0.1195. That was the prediction for a retrained run.
But the codebook is re-initialised by KMeans and then trained, so it may absorb
a larger curvature by shrinking its own norms instead, which would mean the
"codebook moved outward" explanation of iter107's -3.48% never actually
happened.

This reads the trained iter107 checkpoint against the parent's, both at their
own curvatures, and reports:

  - s_e percentiles per level, so the outward claim can be checked directly
  - corr(usage, s_e), to see whether the radial hierarchy survived retraining
  - the angular gap between a behaviour successor and a random item, to see
    whether the direction signal moved

Nothing is trained and nothing is written under a results subtree.
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


def canonical_collision(model: RQVAE, embeddings: torch.Tensor) -> tuple[int, float]:
    """The collision the trainer itself reports, from the final checkpoint.

    Three different numbers exist for the same model and they must never be
    compared against each other:

      final_checkpoint_collision  this one, via get_indices_with_stats, which is
                                  the path trainer.py calls at every eval and
                                  export
      training_time_collision     the value in training_metrics.jsonl, read off
                                  the model mid-training rather than at the end
      disk_sid_collision          recomputed from the exported sids_raw.npy,
                                  which went through collision extension

    get_indices (no per-bucket Sinkhorn) is a fourth number again and is only
    kept here as a contrast, never as the canonical value.
    """
    with torch.no_grad():
        tokens, _ = model.get_indices_with_stats(embeddings)
    stacked = tokens.cpu().numpy()
    unique = int(len(np.unique(stacked, axis=0)))
    return unique, 1.0 - unique / len(stacked)


@torch.no_grad()
def read(model: RQVAE, embeddings: torch.Tensor, curvatures, batch_size: int = 4096):
    device = embeddings.device
    n_items = embeddings.shape[0]
    layers = model.rq.vq_layers
    n_codes = layers[0].n_embed
    tokens = np.empty((n_items, len(layers)), dtype=np.int64)
    usage = np.zeros((len(layers), n_codes), dtype=np.float64)
    cos_sum = np.zeros((len(layers), n_codes), dtype=np.float64)
    # Read every level's radius once, outside the batch loop. Computing it inside
    # meant the last batch's traversal overwrote it per level and the per-level
    # numbers depended on iteration order.
    radii = np.empty((len(layers), n_codes), dtype=np.float64)
    units_by_level = []
    for level, layer in enumerate(layers):
        code = layer.get_code_embs().detach()
        radii[level] = (
            float(curvatures[level]) ** 0.5
        ) * torch.linalg.vector_norm(code, dim=-1).cpu().numpy()
        units_by_level.append(
            code / torch.linalg.vector_norm(
                code, dim=-1, keepdim=True
            ).clamp_min(1e-12)
        )
    for start in range(0, n_items, batch_size):
        batch = embeddings[start : start + batch_size]
        residual = model.encoder(batch)
        previous_codes = None
        for lv, layer2 in enumerate(layers):
            c2 = float(curvatures[lv])
            code2 = layer2.get_code_embs()
            target_norm = model.rq._radius_for_level(lv, c2, residual)
            pinned = model.rq._pin_to_radius(residual, target_norm)
            distances = _pairwise_poincare_distance_tangents(
                pinned, code2, c2
            )
            chosen = layer2._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            tokens[start : start + batch.shape[0], lv] = chosen.cpu().numpy()
            units = pinned / torch.linalg.vector_norm(
                pinned, dim=-1, keepdim=True
            ).clamp_min(1e-12)
            picked = torch.nn.functional.embedding(
                chosen, units_by_level[lv]
            )
            idx = chosen.cpu().numpy()
            cos_sum[lv] += np.bincount(
                idx,
                weights=(picked * units).sum(-1).cpu().numpy(),
                minlength=n_codes,
            )
            usage[lv] += np.bincount(
                idx, minlength=n_codes
            ).astype(np.float64)
            residual = model.rq._restore_norm(
                _hyperbolic_residual(pinned, code2[chosen], c2),
                residual,
                target_norm,
            )
            previous_codes = chosen
    return tokens, radii, usage, cos_sum


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

    runs = {
        "parent": (experiment.PARENT_CKPT, experiment.LAYER_CURVATURES),
        "iter107": (experiment.ITER107_CKPT, (1.0, 2.0, 1.0)),
    }
    lines: list[str] = []
    result: dict = {}
    for name, (path, curvatures) in runs.items():
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = RQVAE(
            tokenizer_config(tuple(float(c) for c in curvatures)),
            in_dim=embeddings.shape[1],
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        tokens, radii, usage, cos_sum = read(
            model, embeddings, tuple(float(c) for c in curvatures)
        )
        entry: dict = {"curvatures": list(curvatures), "levels": {}}
        for level in range(3):
            s = radii[level]
            valid = usage[level] > 0
            mean_cos = cos_sum[level][valid] / usage[level][valid]
            unit = (
                model.rq.vq_layers[level].get_code_embs()
                / torch.linalg.vector_norm(
                    model.rq.vq_layers[level].get_code_embs(),
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
            ).cpu().numpy()
            cos_gap = float(
                (unit[tokens[source_ids, level]] * unit[tokens[successor_ids, level]]).sum(-1).mean()
                - (unit[tokens[source_ids, level]] * unit[tokens[negative_ids, level]]).sum(-1).mean()
            )
            entry["levels"][f"L{level + 1}"] = {
                "s_e_p5": float(np.percentile(s, 5)),
                "s_e_p50": float(np.median(s)),
                "s_e_p95": float(np.percentile(s, 95)),
                "s_e_min": float(s.min()),
                "s_e_max": float(s.max()),
                "s_e_spread": float(s.max() - s.min()),
                "corr_usage_radius": pearson(
                    usage[level][valid], s[valid]
                ),
                "mean_cos_to_codeword": float(mean_cos.mean()),
                "behaviour_cosine_gap": cos_gap,
            }
        unique = int(len(np.unique(tokens, axis=0)))
        canonical_unique, canonical = canonical_collision(model, embeddings)
        disk_path = (
            experiment.PARENT_CKPT.parent / "sids_raw.npy"
            if name == "parent"
            else experiment.ITER107_CKPT.parent / "sids_raw.npy"
        )
        disk = np.load(disk_path)
        disk_unique = int(len(np.unique(disk, axis=0)))
        entry["final_checkpoint_collision"] = canonical
        entry["final_checkpoint_raw_unique"] = canonical_unique
        entry["diagnostic_traversal_collision"] = 1.0 - unique / n_items
        entry["disk_sid_collision"] = 1.0 - disk_unique / len(disk)
        entry["disk_sid_raw_unique"] = disk_unique
        entry["codebook_used"] = [int((usage[l] > 0).sum()) for l in range(3)]
        result[name] = entry
        lines.append(
            f"{name}  c={tuple(float(c) for c in curvatures)}  "
            f"used={entry['codebook_used']}"
        )
        lines.append(
            f"   collision: final_checkpoint={canonical:.6f}  "
            f"diagnostic_traversal={entry['diagnostic_traversal_collision']:.6f}  "
            f"disk_sid={entry['disk_sid_collision']:.6f}"
        )
        for level in range(3):
            block = entry["levels"][f"L{level + 1}"]
            lines.append(
                f"   L{level + 1}  s_e p5/p50/p95="
                f"{block['s_e_p5']:.5f}/{block['s_e_p50']:.5f}/"
                f"{block['s_e_p95']:.5f}  spread={block['s_e_spread']:.5f}  "
                f"corr(usage,s_e)={block['corr_usage_radius']:+.4f}  "
                f"mean_cos={block['mean_cos_to_codeword']:+.4f}  "
                f"beh_cos_gap={block['behaviour_cosine_gap']:+.5f}"
            )
        del model

    p = result["parent"]["levels"]["L2"]
    q = result["iter107"]["levels"]["L2"]
    observed_ratio = q["s_e_p50"] / p["s_e_p50"] if p["s_e_p50"] else float("nan")
    predicted_ratio = np.sqrt(2.0)
    observed_growth = observed_ratio - 1.0
    predicted_growth = predicted_ratio - 1.0
    # Fraction of the outward push that survived retraining. Using the ratio
    # itself here would overstate it: a codebook that ignored curvature entirely
    # would sit at the predicted ratio, and one that fully cancelled it would
    # sit at 1.0, so the growth fraction is the honest scale.
    realised_fraction = observed_growth / predicted_growth
    absorbed_fraction = 1.0 - realised_fraction
    lines.append("L2 comparison")
    lines.append(
        f"   s_e_p50 parent={p['s_e_p50']:.5f}  iter107={q['s_e_p50']:.5f}  "
        f"ratio={observed_ratio:.4f}  (unabsorbed prediction sqrt(2)="
        f"{predicted_ratio:.4f})"
    )
    lines.append(
        f"   outward growth: observed {observed_growth * 100:+.1f}%  "
        f"predicted {predicted_growth * 100:+.1f}%  "
        f"realised {realised_fraction * 100:.1f}%  "
        f"absorbed {absorbed_fraction * 100:.1f}%"
    )
    lines.append(
        f"   radius shrank instead of moving out: {observed_ratio < 1.0}"
    )
    lines.append(
        f"   corr(usage,s_e) {p['corr_usage_radius']:+.4f} -> "
        f"{q['corr_usage_radius']:+.4f}"
    )
    lines.append(
        f"   behaviour cosine gap {p['behaviour_cosine_gap']:+.5f} -> "
        f"{q['behaviour_cosine_gap']:+.5f}  "
        f"({100 * (q['behaviour_cosine_gap'] / p['behaviour_cosine_gap'] - 1):+.1f}%)"
    )
    result["l2_radius_ratio"] = observed_ratio
    result["predicted_ratio_unabsorbed"] = predicted_ratio
    result["outward_growth_realised_fraction"] = realised_fraction
    result["outward_growth_absorbed_fraction"] = absorbed_fraction
    result["collision_definition"] = "final_checkpoint_collision"

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.DIAGNOSTIC_JSON.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    experiment.DIAGNOSTIC_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
