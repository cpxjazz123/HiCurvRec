"""Virtual coarsening of L2: the behaviour-capacity trade-off, measured cheaply.

iter114-117 established that the RQ stack keeps behaviour pairs at every level it
can be measured at, 14.03x at L1 and 1.77x at L2, and that L3 is unmeasurable
because prefixes of length two already leave 75% of groups as singletons and
length three leaves 99.6%. The reading is that the stack is not fighting
behaviour but over-refining it: each level's job is to make items more
distinguishable, and the behaviour community thins out as the prefixes get
finer. So the axis worth probing is capacity, not curvature.

This coarsens L2 without retraining anything. The parent's 256 trained codewords
are grouped by geometric similarity in the tangent frame and each group is
replaced by the centroid of its members, so L2 can emit fewer distinct codes
while the encoder, the L1 and L3 codebooks, the pin and every loss stay exactly
as the parent has them. Nothing is trained; this reads a trade-off curve that a
real run would otherwise cost a full Stage2 plus Stage3 to sample one point of.

Three quantities are reported per capacity: behaviour enrichment against a
negative matched to the anchor's own L1 bucket, the singleton rate of the L1+L2
prefixes, and the collision rate over the corpus. Enrichment says whether
behaviour survives better, singletons say how much of the catalogue is still
groupable, and collision says what uniqueness is paid for it. A capacity only
matters if all three move together.
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


def merge_codewords(codes: torch.Tensor, capacity: int) -> torch.Tensor:
    """Group codewords by tangent-frame similarity and return the centroids.

    k-means over the cosine embedding gives a grouping that does not depend on
    how the codebook happens to be ordered, so the merge reflects the learned
    geometry rather than an artefact of codebook indexing.
    """
    n = codes.shape[0]
    if capacity >= n:
        return codes
    unit = codes / torch.linalg.vector_norm(
        codes, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    generator = torch.Generator().manual_seed(experiment.COARSEN_SEED)
    indices = torch.randperm(n, generator=generator)[:capacity]
    centroids = unit[indices].clone()
    assignment = torch.zeros(n, dtype=torch.long)
    for _ in range(50):
        assignment = torch.argmax(unit @ centroids.T, dim=1)
        updated = centroids.clone()
        for k in range(capacity):
            members = assignment == k
            if members.any():
                updated[k] = unit[members].mean(dim=0)
        updated = updated / torch.linalg.vector_norm(
            updated, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        if torch.allclose(updated, centroids, atol=1e-6):
            centroids = updated
            break
        centroids = updated
    # An empty cluster borrows the nearest used one's centroid, so the effective
    # capacity matches what was asked for rather than silently shrinking.
    used = torch.unique(assignment)
    if len(used) < capacity:
        spare = [int(u) for u in used]
        for k in range(capacity):
            if k not in spare:
                assignment[assignment == k] = spare[k % len(spare)]
    merged = torch.zeros(capacity, codes.shape[1], dtype=codes.dtype)
    for k in range(capacity):
        members = assignment == k
        if members.any():
            merged[k] = codes[members].mean(dim=0)
    return merged.to(codes.device)


@torch.no_grad()
def corpus_pass(
    model: RQVAE,
    embeddings: torch.Tensor,
    merged_l2: torch.Tensor,
    level: int,
    batch_size: int = 4096,
) -> np.ndarray:
    """Assign codes for the whole catalogue with L2 replaced by ``merged_l2``.

    L2's assignment is computed against the merged centroids directly, so the
    geometry and the Sinkhorn balancing both see the reduced codebook rather
    than an original assignment being remapped afterwards.
    """
    layers = model.rq.vq_layers
    out = np.empty((embeddings.shape[0], len(layers)), dtype=np.int64)
    for start in range(0, embeddings.shape[0], batch_size):
        batch = embeddings[start : start + batch_size]
        residual = model.encoder(batch)
        previous_codes = None
        for lv, layer in enumerate(layers):
            c = float(layer.get_curvature())
            target_norm = model.rq._radius_for_level(lv, c, residual)
            pinned = model.rq._pin_to_radius(residual, target_norm)
            codebook = merged_l2 if lv == level else layer.get_code_embs()
            distances = _pairwise_poincare_distance_tangents(pinned, codebook, c)
            chosen = layer._indices(
                distances, infer_use_sk=True, bucket=previous_codes
            )
            out[start : start + batch.shape[0], lv] = chosen.cpu().numpy()
            residual = model.rq._restore_norm(
                _hyperbolic_residual(pinned, codebook[chosen], c),
                residual,
                target_norm,
            )
            previous_codes = chosen
    return out


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
    partners: list[set[int]] = [set() for _ in range(n_items)]
    for a, b in zip(source_ids, successor_ids):
        partners[a].add(int(b))
        partners[b].add(int(a))

    level = experiment.COARSEN_LEVEL
    lines: list[str] = []
    result: dict = {
        "parent_checkpoint": str(experiment.PARENT_CKPT),
        "coarsen_level": level,
        "pairs_total": int(len(source_ids)),
        "capacities": {},
    }
    lines.append(
        f"virtual coarsening of L{level + 1}: the parent's trained codewords are "
        f"merged by tangent-frame similarity and nothing is retrained"
    )
    lines.append("")

    for capacity in experiment.COARSEN_CAPACITIES:
        merged = merge_codewords(model.rq.vq_layers[level].get_code_embs(), capacity)
        codes = corpus_pass(model, embeddings, merged, level)

        # Behaviour enrichment against negatives matched to the anchor's L1
        # bucket, which is the control iter116 established as the only valid one.
        bucket = codes[:, 0]
        members_by_bucket = {
            int(b): np.flatnonzero(bucket == b) for b in np.unique(bucket)
        }
        eligible = np.flatnonzero(bucket[source_ids] == bucket[successor_ids])
        pos_idx: list[int] = []
        neg_idx: list[int] = []
        for index in eligible:
            anchor = int(source_ids[index])
            members = members_by_bucket[int(bucket[anchor])]
            forbidden = partners[anchor] | {anchor}
            candidates = np.array(
                [int(m) for m in members if int(m) not in forbidden],
                dtype=np.int64,
            )
            if len(candidates) == 0:
                continue
            take = min(experiment.MATCHED_NEGATIVES, len(candidates))
            for c in generator.choice(candidates, size=take, replace=False):
                pos_idx.append(index)
                neg_idx.append(int(c))
        pos_arr = np.asarray(pos_idx)
        neg_arr = np.asarray(neg_idx)
        if len(neg_arr) > 0:
            anchor_l2 = codes[source_ids[pos_arr], level]
            kept_pos = anchor_l2 == codes[successor_ids[pos_arr], level]
            kept_neg = anchor_l2 == codes[neg_arr, level]
            rate_pos = float(kept_pos.mean())
            rate_neg = float(kept_neg.mean())
            enrichment = rate_pos / rate_neg if rate_neg > 0 else float("inf")
            n = len(kept_pos)
            p_bar = (kept_pos.sum() + kept_neg.sum()) / (2 * n)
            z = (
                float(
                    (rate_pos - rate_neg)
                    / np.sqrt(2.0 * p_bar * (1.0 - p_bar) / n)
                )
                if 0.0 < p_bar < 1.0
                else float("nan")
            )
        else:
            rate_pos = rate_neg = enrichment = z = float("nan")

        keys = codes[:, : level + 1]
        _, counts = np.unique(keys, axis=0, return_counts=True)
        singletons = float((counts == 1).mean())
        median_size = float(np.median(counts))
        raw_unique = int(len(np.unique(codes, axis=0)))
        collision = 1.0 - raw_unique / n_items

        entry = {
            "capacity": capacity,
            "behaviour_enrichment": enrichment,
            "behaviour_positive": rate_pos,
            "behaviour_negative": rate_neg,
            "z_score": z,
            "prefix_singleton_rate": singletons,
            "prefix_median_size": median_size,
            "raw_unique": raw_unique,
            "collision": collision,
            "eligible_positives": int(len(eligible)),
            "matched_pairs": int(len(neg_arr)),
        }
        result["capacities"][str(capacity)] = entry
        lines.append(
            f"L{level + 1} capacity {capacity:>4}:  "
            f"E={enrichment:>6.2f}x  P+={rate_pos * 100:6.3f}%  "
            f"P-={rate_neg * 100:6.3f}%  z={z:>+6.2f}"
        )
        lines.append(
            f"{'':>16}prefix singletons={singletons * 100:6.2f}%  "
            f"median group={median_size:>7.2f}  raw_unique={raw_unique:>6}  "
            f"collision={collision * 100:7.4f}%"
        )

    lines.append("")
    lines.append("trade-off read against the parent's own L2")
    parent = result["capacities"][str(experiment.COARSEN_CAPACITIES[0])]
    for capacity in experiment.COARSEN_CAPACITIES[1:]:
        entry = result["capacities"][str(capacity)]
        lines.append(
            f"   {capacity:>4}: E {parent['behaviour_enrichment']:.2f}x -> "
            f"{entry['behaviour_enrichment']:.2f}x    "
            f"collision {parent['collision'] * 100:.4f}% -> "
            f"{entry['collision'] * 100:.4f}%    "
            f"singletons {parent['prefix_singleton_rate'] * 100:.1f}% -> "
            f"{entry['prefix_singleton_rate'] * 100:.1f}%"
        )

    report = "\n".join(lines)
    print(report, flush=True)
    experiment.REPORT_JSON.write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    experiment.REPORT_LOG.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()