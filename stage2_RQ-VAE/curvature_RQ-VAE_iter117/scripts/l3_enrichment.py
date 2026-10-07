"""Complete the hierarchy: how much behaviour enrichment survives each RQ level?

The level curve so far, each measured against a negative drawn from the anchor's
own bucket at that level so both sides are inside the same cluster:

  L1   P(same L1 | +)                    = 5.4789%  vs chance 0.3906%   14.0x
  L2   P(same L2 | same L1, +)           = 1.1020%  vs matched 0.6612%    1.67x

Both are positive, so the stack is not working against behaviour; the enrichment
decays with depth. L3 completes the curve, and it is the level most likely to be
unmeasurable: prefixes of length two group the catalogue into almost singletons,
so the matched negative pool may not exist at a useful size.

That is reported rather than worked around. If the pool is too small the script
says so and stops, because the alternative failure mode is to fall back to a
weaker control (a uniform negative) and read the result as if it were the matched
one, which is exactly the error that made L2 look adversarial in iter114.

The negative for an L3-eligible positive must share the anchor's first two codes
and be neither the anchor nor any of its behaviour partners, in either direction.
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
        source = int(history[-1])
        if source < 0:
            raise ValueError("Transition source item is negative")
        sources.append(source)
        successors.append(int(target))
    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(successors, dtype=np.int64),
    )


@torch.no_grad()
def codes_for(model: RQVAE, embeddings: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
    device = embeddings.device
    layers = model.rq.vq_layers
    out = np.empty((embeddings.shape[0], len(layers)), dtype=np.int64)
    for start in range(0, embeddings.shape[0], batch_size):
        batch = embeddings[start : start + batch_size]
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
                _hyperbolic_residual(
                    pinned, layer.get_code_embs()[chosen], c
                ),
                residual,
                target_norm,
            )
            previous_codes = chosen
    return out


def measure_level(
    codes: np.ndarray,
    partners: list[set[int]],
    bucket_of: np.ndarray,
    prefix_width: int,
    eligible_pairs: np.ndarray,
    generator: np.random.Generator,
) -> dict:
    """P(same next code | prefix agreed, +) against the same bucket's negatives."""
    positives: list[int] = []
    negatives: list[int] = []
    for index in eligible_pairs:
        anchor = int(codes_source_ids[index])
        prefix = codes[anchor, :prefix_width]
        members = np.flatnonzero(np.all(codes[:, :prefix_width] == prefix, axis=1))
        forbidden = partners[anchor] | {anchor}
        candidates = np.array(
            [int(m) for m in members if int(m) not in forbidden], dtype=np.int64
        )
        if len(candidates) == 0:
            continue
        take = min(experiment.MATCHED_NEGATIVES, len(candidates))
        chosen = generator.choice(candidates, size=take, replace=False)
        for c in chosen:
            positives.append(index)
            negatives.append(int(c))
    positives_arr = np.asarray(positives)
    negatives_arr = np.asarray(negatives)
    if len(negatives_arr) == 0:
        return {
            "eligible_positives": int(len(eligible_pairs)),
            "matched_pairs": 0,
            "feasible": False,
        }
    anchor_codes = codes[codes_source_ids[positives_arr], prefix_width]
    positive_codes = codes[codes_successor_ids[positives_arr], prefix_width]
    negative_codes = codes[negatives_arr, prefix_width]
    kept_pos = anchor_codes == positive_codes
    kept_neg = anchor_codes == negative_codes
    rate_pos = float(kept_pos.mean())
    rate_neg = float(kept_neg.mean())
    n = len(kept_pos)
    p_bar = (kept_pos.sum() + kept_neg.sum()) / (2 * n)
    z = (
        float((rate_pos - rate_neg) / np.sqrt(2.0 * p_bar * (1.0 - p_bar) / n))
        if 0.0 < p_bar < 1.0
        else float("nan")
    )
    return {
        "eligible_positives": int(len(eligible_pairs)),
        "matched_pairs": int(n),
        "positive_rate": rate_pos,
        "negative_rate": rate_neg,
        "difference": rate_pos - rate_neg,
        "enrichment": rate_pos / rate_neg if rate_neg > 0 else float("inf"),
        "z_score": z,
        "feasible": bool(n >= experiment.MIN_PAIRS_FOR_VERDICT),
    }


codes_source_ids = None
codes_successor_ids = None


def main() -> None:
    global codes_source_ids, codes_successor_ids
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

    codes = codes_for(model, embeddings)
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    global codes_source_ids, codes_successor_ids
    codes_source_ids, codes_successor_ids = transition_pairs(train_frame)

    partners: list[set[int]] = [set() for _ in range(n_items)]
    for a, b in zip(codes_source_ids, codes_successor_ids):
        partners[a].add(int(b))
        partners[b].add(int(a))

    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "pairs_total": int(len(codes_source_ids)),
        "levels": {},
    }
    lines.append(
        f"catalogue={n_items}  behaviour pairs={len(codes_source_ids)}  "
        f"min pairs for a verdict={experiment.MIN_PAIRS_FOR_VERDICT}"
    )

    # How the prefixes group the catalogue, which bounds the feasible pool at
    # each depth before any measurement is attempted.
    lines.append("")
    lines.append("prefix group sizes over the catalogue")
    for width in (1, 2, 3):
        keys = codes[:, :width]
        _, counts = np.unique(keys, axis=0, return_counts=True)
        lines.append(
            f"   prefix {width}: {len(counts)} groups, size p5/p50/p95 = "
            f"{np.percentile(counts, 5):.0f}/{np.percentile(counts, 50):.0f}/"
            f"{np.percentile(counts, 95):.0f}, singleton groups="
            f"{int((counts == 1).sum())}"
        )
        result.setdefault("prefix_groups", {})[f"prefix{width}"] = {
            "groups": int(len(counts)),
            "p50": float(np.percentile(counts, 50)),
            "singletons": int((counts == 1).sum()),
        }

    # The hierarchy, measured the same way at every depth: the negative shares
    # the anchor's whole prefix of that depth.
    lines.append("")
    lines.append("behaviour enrichment by level (matched negative at the same depth)")
    curve = []
    for width in (1, 2, 3):
        # eligible positives are those whose successor already agrees on the
        # prefix being conditioned on; for width 1 there is no condition.
        if width == 1:
            eligible = np.arange(len(codes_source_ids))
        else:
            eligible = np.flatnonzero(
                np.all(
                    codes[codes_source_ids, : width - 1]
                    == codes[codes_successor_ids, : width - 1],
                    axis=1,
                )
            )
        measured = measure_level(
            codes, partners, codes[:, 0], width - 1, eligible, generator
        )
        measured["level"] = width
        if width == 1:
            chance = float(len(codes) and (counts_p1 := np.bincount(
                codes[:, 0], minlength=256
            ).astype(np.float64).mean()) / n_items)
            measured["chance_rate"] = chance
            measured["enrichment"] = measured["positive_rate"] / chance
            measured["negative_rate"] = chance
            measured["difference"] = measured["positive_rate"] - chance
        result["levels"][f"L{width}"] = measured
        curve.append(measured)
        if measured["matched_pairs"] == 0:
            lines.append(
                f"   L{width}: eligible positives={measured['eligible_positives']}, "
                f"matched negatives=0 -> the prefix has no non-partner to draw, "
                f"this level cannot be measured with a matched control"
            )
            continue
        lines.append(
            f"   L{width}: eligible={measured['eligible_positives']}  "
            f"matched pairs={measured['matched_pairs']}  "
            f"P+={measured['positive_rate'] * 100:.4f}%  "
            f"P-={measured['negative_rate'] * 100:.4f}%  "
            f"E={measured['enrichment']:.2f}x  z={measured['z_score']:+.2f}"
        )

    lines.append("")
    lines.append("hierarchy")
    for measured in curve:
        if measured["matched_pairs"] == 0:
            lines.append(
                f"   L{measured['level']}: unmeasurable (no matched control)"
            )
        else:
            lines.append(
                f"   L{measured['level']}: {measured['enrichment']:.2f}x"
                + ("" if measured["feasible"] else "   [pool too small to trust]")
            )
    result["curve"] = [
        {
            "level": m["level"],
            "enrichment": m["enrichment"],
            "feasible": m["feasible"],
        }
        for m in curve
    ]

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.REPORT_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.REPORT_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()