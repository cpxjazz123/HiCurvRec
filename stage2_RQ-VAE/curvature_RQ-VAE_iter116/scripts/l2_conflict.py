"""Does L2's refinement work against behaviour, once L1 has grouped a pair?

iter114 measured the whole stack on real transitions and found the asymmetry
worth explaining: among behaviour pairs that already share their L1 code, the
share going on to share L2 is 2.50%, while random pairs sharing the same L1 code
reach 3.05%. A level that separates true pairs more than chance is doing the
opposite of what the residual quantizer is for.

That comparison used uniformly random negatives, which is not the control this
question needs. A random negative rarely shares the anchor's L1 code, so most of
the random pairs were never in the position where L2 has to act, and the 3.05%
came from a different population than the 2.50%. The matched negative here is
drawn from the anchor's own L1 bucket and is not one of its behaviour partners,
which puts both sides of the comparison inside exactly the same cluster: the
anchor's bucket, already agreed on at L1, differing only in whether the other
end is a real transition target.

L1 is doing real work in the meantime. Bucket membership alone is 0.39%
(96 items out of 24587) while 5.51% of behaviour pairs share it, so L1 captures
behavioural structure at roughly 14x enrichment; the absolute rate stays low only
because the behaviour graph itself is sparse.

The geometry of the split is recomputed here on every available pair, since the
earlier version had ten kept pairs and could not distinguish a real margin
difference from sampling noise.
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
def stack_pass(model: RQVAE, embeddings: torch.Tensor, batch_size: int = 4096) -> dict:
    device = embeddings.device
    layers = model.rq.vq_layers
    n = embeddings.shape[0]
    codes = np.empty((n, len(layers)), dtype=np.int64)
    l2_margin = np.empty(n, dtype=np.float64)
    l2_pinned = np.empty((n, experiment.CODEBOOK_DIM), dtype=np.float32)
    l2_codeword = np.empty((n, experiment.CODEBOOK_DIM), dtype=np.float32)
    l1_margin = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
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
            two = torch.topk(distances, k=2, dim=1, largest=False).values
            chosen = layer._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            codes[start : start + batch.shape[0], level] = chosen.cpu().numpy()
            if level == 0:
                l1_margin[start : start + batch.shape[0]] = (
                    two[:, 1] - two[:, 0]
                ).cpu().numpy()
            if level == 1:
                l2_margin[start : start + batch.shape[0]] = (
                    two[:, 1] - two[:, 0]
                ).cpu().numpy()
                l2_pinned[start : start + batch.shape[0]] = pinned.cpu().numpy()
                l2_codeword[start : start + batch.shape[0]] = (
                    layer.get_code_embs()[chosen].cpu().numpy()
                )
            residual = model.rq._restore_norm(
                _hyperbolic_residual(
                    pinned, layer.get_code_embs()[chosen], c
                ),
                residual,
                target_norm,
            )
            previous_codes = chosen
    return {
        "codes": codes,
        "l1_margin": l1_margin,
        "l2_margin": l2_margin,
        "l2_pinned": l2_pinned,
        "l2_codeword": l2_codeword,
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

    run = stack_pass(model, embeddings)
    codes = run["codes"]
    bucket = codes[:, 0]

    # Everything an anchor's negative must avoid: the anchor itself and every
    # item it is behaviourally related to, in either direction.
    partners: list[set[int]] = [set() for _ in range(n_items)]
    train_frame = pd.read_parquet(experiment.TRAIN_FILE)
    source_ids, successor_ids = transition_pairs(train_frame)
    for a, b in zip(source_ids, successor_ids):
        partners[a].add(int(b))
        partners[b].add(int(a))

    bucket_members: dict[int, np.ndarray] = {
        int(code): np.flatnonzero(bucket == code) for code in np.unique(bucket)
    }
    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "pairs_total": int(len(source_ids)),
        "matched_negatives_per_anchor": experiment.MATCHED_NEGATIVES,
    }
    lines.append(
        f"catalogue={n_items}  behaviour pairs={len(source_ids)}  "
        f"L1 buckets={len(bucket_members)}  "
        f"matched negatives/anchor={experiment.MATCHED_NEGATIVES}"
    )
    sizes = np.array([len(m) for m in bucket_members.values()])
    lines.append(
        f"bucket size p5/p50/p95 = {np.percentile(sizes, 5):.0f}/"
        f"{np.percentile(sizes, 50):.0f}/{np.percentile(sizes, 95):.0f}"
    )

    # ---- L1 enrichment, stated against the chance rate rather than in absolute terms
    same_l1_pos = bucket[source_ids] == bucket[successor_ids]
    chance = float(sizes.mean() / n_items)
    enrichment = float(same_l1_pos.mean()) / chance
    lines.append("")
    lines.append("1. L1 enrichment, against chance rather than against zero")
    lines.append(
        f"   P(same L1 | +) = {same_l1_pos.mean() * 100:.4f}%   "
        f"chance = {chance * 100:.4f}%   enrichment = {enrichment:.1f}x"
    )
    result["l1_same_positive"] = float(same_l1_pos.mean())
    result["l1_chance"] = chance
    result["l1_enrichment"] = enrichment

    # ---- matched negatives, drawn only for anchors whose positive shares L1
    eligible = np.flatnonzero(same_l1_pos)
    lines.append("")
    lines.append("2. matched negatives: same L1 bucket as the anchor, not a partner")
    lines.append(
        f"   eligible positives (already share L1): {len(eligible)}"
    )

    neg_anchor: list[int] = []
    neg_item: list[int] = []
    pos_anchor: list[int] = []
    for index in eligible:
        anchor = int(source_ids[index])
        members = bucket_members[int(bucket[anchor])]
        forbidden = partners[anchor] | {anchor}
        candidates = np.array(
            [m for m in members if int(m) not in forbidden], dtype=np.int64
        )
        if len(candidates) == 0:
            continue
        chosen = generator.choice(
            candidates, size=experiment.MATCHED_NEGATIVES, replace=False
        )
        for c in chosen:
            pos_anchor.append(index)
            neg_anchor.append(anchor)
            neg_item.append(int(c))
    pos_anchor_arr = np.asarray(pos_anchor)
    neg_anchor_arr = np.asarray(neg_anchor)
    neg_item_arr = np.asarray(neg_item)
    lines.append(
        f"   matched negative pairs built: {len(neg_item_arr)} "
        f"({len(eligible)} positives x up to {experiment.MATCHED_NEGATIVES})"
    )
    if len(neg_item_arr) == 0:
        raise RuntimeError("No matched negatives could be drawn")

    tok_a = codes[source_ids[pos_anchor_arr]]
    tok_p = codes[successor_ids[pos_anchor_arr]]
    tok_n = codes[neg_item_arr]
    # Sanity: the match must hold by construction.
    assert np.all(tok_a[:, 0] == tok_n[:, 0]), "matched negative left the L1 bucket"
    lines.append(
        f"   verified: every matched negative is in the anchor's L1 bucket"
    )

    kept_pos = tok_a[:, 1] == tok_p[:, 1]
    kept_neg = tok_a[:, 1] == tok_n[:, 1]
    rate_pos = float(kept_pos.mean())
    rate_neg = float(kept_neg.mean())
    lines.append("")
    lines.append("3. the comparison the earlier version could not make")
    lines.append(
        f"   P(same L2 | same L1, +) = {rate_pos * 100:.4f}%   "
        f"({int(kept_pos.sum())} / {len(kept_pos)})"
    )
    lines.append(
        f"   P(same L2 | same L1, matched -) = {rate_neg * 100:.4f}%   "
        f"({int(kept_neg.sum())} / {len(kept_neg)})"
    )
    lines.append(
        f"   difference = {(rate_pos - rate_neg) * 100:+.4f} percentage points"
    )
    result["same_L2_given_L1_positive"] = rate_pos
    result["same_L2_given_L1_matched_negative"] = rate_neg
    result["difference"] = rate_pos - rate_neg
    result["l2_works_against_behaviour"] = bool(rate_pos < rate_neg)

    # A binomial check on the difference, so the verdict does not rest on a
    # difference that could be sampling noise.
    n = len(kept_pos)
    p_bar = (kept_pos.sum() + kept_neg.sum()) / (2 * n)
    if 0.0 < p_bar < 1.0:
        se = np.sqrt(2.0 * p_bar * (1.0 - p_bar) / n)
        z = (rate_pos - rate_neg) / se if se > 0 else float("nan")
        lines.append(
            f"   pooled p={p_bar:.4f}  se={se:.5f}  z={z:+.2f}  "
            f"(|z|>1.96 would be a real difference at 95%)"
        )
        result["z_score"] = float(z)
        result["significant"] = bool(abs(z) > 1.96)

    # ---- geometry of kept versus split, now with full sample size
    lines.append("")
    lines.append("4. geometry of the split, on every pair")
    def describe(mask: np.ndarray, label: str) -> dict:
        if not mask.any():
            return {}
        idx = np.flatnonzero(mask)
        anchor_l1 = source_ids[pos_anchor_arr[idx]]
        margin = run["l2_margin"][anchor_l1]
        residual = run["l2_pinned"][anchor_l1]
        codeword = run["l2_codeword"][anchor_l1]
        cosine = (residual * codeword).sum(-1) / (
            np.linalg.norm(residual, axis=-1)
            * np.linalg.norm(codeword, axis=-1)
        ).clip(1e-12)
        out = {
            "pairs": int(len(idx)),
            "l2_margin_p50": float(np.median(margin)),
            "cosine_to_codeword_p50": float(np.median(cosine)),
            "codeword_radius_p50": float(np.median(np.linalg.norm(codeword, axis=-1))),
        }
        lines.append(
            f"   {label:<26} n={out['pairs']:<7} margin p50={out['l2_margin_p50']:.6f}  "
            f"cos p50={out['cosine_to_codeword_p50']:.6f}  "
            f"radius p50={out['codeword_radius_p50']:.6f}"
        )
        return out

    result["kept_geometry"] = describe(kept_pos, "kept at L2 (positive)")
    result["split_geometry"] = describe(~kept_pos, "split at L2 (positive)")

    # The same split for the matched negatives: if L2 behaves identically on
    # both populations, it is not discriminating between them at all.
    result["kept_geometry_negative"] = describe(kept_neg, "kept at L2 (negative)")
    result["split_geometry_negative"] = describe(~kept_neg, "split at L2 (negative)")

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.SEPARATION_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.SEPARATION_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()