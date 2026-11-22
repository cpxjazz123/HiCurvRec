"""Does the quantiser learn the community model's relation strengths?

This is the audit the weight-mapping hypothesis needs, in the order the
hypothesis can fail:

1. **the target** — does the mapping put the community model's strengths into
   the target with a real spread, or does normalisation compress them away?
   Measured on the training graph, before any model is involved.
2. **the quantised representation** — for an anchor's two neighbours, does the
   representation order them the way the community model does, and does that
   ordering beat the same measurement with the affinities shuffled? Measured on
   the CHECKPOINT: q_i is the sum of the three codewords at the assignment the
   export uses, so this is the representation the identifiers are built from.
3. **the real behaviour** — on the held-out validation interactions, which the
   arm never trained on, are the genuinely related pairs further apart from the
   unrelated ones? And do the unrelated pairs stay apart (the collapse check)?

Every number is reported for each checkpoint handed in, so the comparison is
between weight mappings rather than between a mechanism and nothing.

Run:
    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        tools/research/science/community_curvature/quantised_relation_audit.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[4]
TREE = REPO / "stage2_RQ-VAE" / "TIGER_community_graph_recon"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
EMBEDDINGS = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
COMMUNITY = (REPO / "results" / "research" / "issue121" / "community_curvature"
             / "item_community_representation.npz")
VALID_PAIRS = (REPO / "results" / "stage3_T5Train" / "monotonicity"
               / "behavior_pairs_valid_s42.npz")
OUT = REPO / "results" / "research" / "issue121" / "community_curvature"
SEED = 42

CHECKPOINTS = {
    "frozen baseline": REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "rqvae_best.pth",
    "baseline re-run": (REPO / "results" / "stage2_RQ-VAE"
                        / "_reference_baseline_deterministic" / "rqvae_best.pth"),
    "weights 1+s": (REPO / "results" / "stage2_RQ-VAE"
                    / "TIGER_community_graph_recon" / "rqvae_best.pth"),
    "weights s": (REPO / "results" / "stage2_RQ-VAE"
                  / "_ablation_affinity_weights" / "rqvae_best.pth"),
}


def load_tree_modules():
    sys.path.insert(0, str(TREE))
    from model import RQVAE  # noqa: E402
    from model.community_graph import affinity_weights, behaviour_edges  # noqa: E402
    return RQVAE, affinity_weights, behaviour_edges


def build_graph(behaviour_edges) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(TRAIN, columns=["history", "target"])
    sequences = []
    for history, target in zip(frame["history"].tolist(), frame["target"].tolist()):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        sequences.append(values)
    edges = behaviour_edges(sequences, 24587).numpy()
    return edges


def edge_weights(edges: np.ndarray, affinity_weights, mode: str) -> torch.Tensor:
    """The weight the loss puts on each edge under one mapping."""
    payload = np.load(COMMUNITY)
    free = torch.from_numpy(payload["free"])
    ball = torch.from_numpy(payload["ball"])
    curvature = float(payload["curvature"])
    return affinity_weights(free, ball, torch.from_numpy(edges), curvature, mode=mode)


def affinities(edges: np.ndarray, affinity_weights) -> torch.Tensor:
    """The community affinity itself, independent of how it is mapped."""
    return edge_weights(edges, affinity_weights, "affinity")


class Config:
    hidden_sizes = (512, 256, 128)
    codebook_num = 3
    codebook_size = (256, 256, 256)
    codebook_dim = 32
    vq_type = "vq"
    beta = 0.25
    sk_epsilon = 0.003
    sk_iters = 50
    ema_decay = 0.99
    fix_code_embs = False
    dropout = 0.0


def representation(RQVAE, checkpoint: Path, embeddings: torch.Tensor) -> torch.Tensor:
    """The quantised representation of every item, under the export assignment."""
    model = RQVAE(Config(), in_dim=int(embeddings.shape[1]))
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state,
                          strict=True)
    model.eval()
    views = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], 4096):
            straight_through, codewords, _ = model.quantised_views(
                embeddings[start:start + 4096])
            views.append(codewords)
    return torch.cat(views).float()


def normalised(rows: torch.Tensor) -> torch.Tensor:
    return rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def pairwise_concordance(similarity, edges, affinity, limit: int = 4000) -> float:
    """Over each anchor's neighbour pairs, how often the closer one is the
    one the community model scores higher."""
    from collections import defaultdict

    by_anchor = defaultdict(list)
    left, right = edges[0], edges[1]
    for position in range(left.shape[0]):
        by_anchor[int(left[position])].append(position)
        by_anchor[int(right[position])].append(position)
    agree = total = 0
    visited = 0
    for anchor, positions in sorted(by_anchor.items()):
        if len(positions) < 2:
            continue
        visited += 1
        if visited > limit:
            break
        partners = np.asarray([
            int(right[p]) if int(left[p]) == anchor else int(left[p]) for p in positions])
        scores = np.asarray([float(affinity[p]) for p in positions])
        sims = similarity[anchor, partners].numpy()
        for a in range(len(positions)):
            for b in range(a + 1, len(positions)):
                if scores[a] == scores[b]:
                    continue
                total += 1
                agree += int((scores[a] > scores[b]) == (sims[a] > sims[b]))
    return agree / total if total else float("nan"), total


def main() -> int:
    RQVAE, affinity_weights, behaviour_edges = load_tree_modules()
    edges = build_graph(behaviour_edges)
    affinity = affinities(edges, affinity_weights)
    print("behaviour edges:", edges.shape[1])

    # ---- question 1: the target, before any model ----
    from collections import defaultdict
    by_anchor = defaultdict(list)
    for position in range(edges.shape[1]):
        by_anchor[int(edges[0, position])].append(position)
        by_anchor[int(edges[1, position])].append(position)
    targets = {}
    for mode in ("one_plus_s", "affinity"):
        weight = edge_weights(edges, affinity_weights, mode=mode)
        entropies, uniform_entropies, supports, ratios = [], [], [], []
        for anchor, positions in by_anchor.items():
            weights = np.asarray([float(weight[p]) for p in positions])
            if weights.sum() <= 0 or len(weights) < 2:
                continue
            target = weights / weights.sum()
            entropies.append(float(-(target * np.log(np.clip(target, 1e-12, None))).sum()))
            uniform_entropies.append(float(np.log(len(target))))
            supports.append(float(1.0 / (target ** 2).sum()))
            positive = target[target > 1e-9]
            if positive.size > 1:
                ratios.append(float(positive.max() / positive.min()))
        targets[mode] = {
            "anchors": len(entropies),
            "mean_target_entropy": float(np.mean(entropies)),
            "mean_uniform_entropy": float(np.mean(uniform_entropies)),
            "kl_from_uniform": float(np.mean(uniform_entropies) - np.mean(entropies)),
            "mean_effective_support": float(np.mean(supports)),
            "mean_neighbours": float(np.mean([np.log(1) + 0 for _ in [0]] + [0])) if False else None,
            "mean_max_over_min_target": float(np.mean(ratios)),
        }
    print("\n=== 1. the target the loss is trained against")
    for mode, row in targets.items():
        print("%-12s KL %7.4f nats | entropy %6.4f vs uniform %6.4f | support %6.2f | max/min %8.2f"
              % (mode, row["kl_from_uniform"], row["mean_target_entropy"],
                 row["mean_uniform_entropy"], row["mean_effective_support"],
                 row["mean_max_over_min_target"]))

    # ---- the teacher's own quality: does its affinity separate a held-out
    # behaviour pair from an unrelated one? This is the supervision's own
    # signal, independent of any quantiser.
    payload = np.load(COMMUNITY)
    free = torch.from_numpy(payload["free"])
    ball = torch.from_numpy(payload["ball"])
    curvature = float(payload["curvature"])
    pairs = np.load(VALID_PAIRS)
    valid_anchor = torch.from_numpy(pairs["anchors"])
    valid_positive = torch.from_numpy(pairs["positives"])
    rng_local = np.random.default_rng(SEED)
    random_negative_pairs = torch.from_numpy(
        rng_local.integers(0, free.shape[0], size=valid_anchor.shape[0]))

    # the scales must be the ones the arm itself used: the median squared
    # distance of each branch over the TRAINING edges
    from model.community_graph import affinity_of_edge, edge_weight_ball, edge_weight_free
    edge_tensor = torch.from_numpy(edges)
    free_scale = edge_weight_free(edge_tensor, free).double().median().clamp_min(1e-12)
    ball_scale = edge_weight_ball(edge_tensor, ball, curvature).double().median().clamp_min(1e-12)

    def pair_affinity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return affinity_of_edge(free[left], free[right], ball[left], ball[right],
                                curvature, free_scale.float(), ball_scale.float())

    positive_affinity = pair_affinity(valid_anchor, valid_positive).numpy()
    negative_affinity = pair_affinity(valid_anchor, random_negative_pairs).numpy()

    labels = np.concatenate([np.ones_like(positive_affinity),
                             np.zeros_like(negative_affinity)])
    scores = np.concatenate([positive_affinity, negative_affinity])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    positives = labels.sum()
    teacher_auc = float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                        / (positives * (labels.size - positives)))
    teacher = {
        "validation_pair_mean_affinity": float(positive_affinity.mean()),
        "random_pair_mean_affinity": float(negative_affinity.mean()),
        "auc": teacher_auc,
    }
    print("\n=== 0. the teacher itself: does its affinity separate a held-out pair from a random one?")
    print("validation pairs %+0.4f | random pairs %+0.4f | AUC %6.4f"
          % (teacher["validation_pair_mean_affinity"],
             teacher["random_pair_mean_affinity"], teacher["auc"]))

    embeddings = torch.from_numpy(np.load(EMBEDDINGS).astype(np.float32))
    rng = np.random.default_rng(SEED)

    results = {"teacher": teacher, "target": targets, "checkpoints": {}}
    print("\n=== 2 and 3, per checkpoint")
    for name, checkpoint in CHECKPOINTS.items():
        if not checkpoint.is_file():
            print("%-18s missing: %s" % (name, checkpoint))
            continue
        q = representation(RQVAE, checkpoint, embeddings)
        unit = normalised(q)
        similarity = unit @ unit.t()

        concordance, pairs_seen = pairwise_concordance(similarity, edges, affinity)
        shuffled = affinity[torch.from_numpy(rng.permutation(affinity.shape[0]))]
        concordance_shuffled, _ = pairwise_concordance(similarity, edges, shuffled)

        positive_similarity = similarity[valid_anchor, valid_positive].numpy()
        random_negative = rng.integers(0, q.shape[0], size=valid_anchor.shape[0])
        negative_similarity = similarity[valid_anchor, random_negative].numpy()
        gap = float(positive_similarity.mean() - negative_similarity.mean())
        labels = np.concatenate([np.ones_like(positive_similarity),
                                 np.zeros_like(negative_similarity)])
        scores = np.concatenate([positive_similarity, negative_similarity])
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, scores.size + 1)
        positives = labels.sum()
        auc = float((ranks[labels == 1].sum() - positives * (positives + 1) / 2)
                    / (positives * (labels.size - positives)))

        random_left = rng.integers(0, q.shape[0], size=20000)
        random_right = rng.integers(0, q.shape[0], size=20000)
        unrelated = float(similarity[random_left, random_right].mean())

        results["checkpoints"][name] = {
            "concordance_with_affinity": concordance,
            "concordance_with_shuffled_affinity": concordance_shuffled,
            "pairs_compared": pairs_seen,
            "validation_positive_mean_cos": float(positive_similarity.mean()),
            "validation_negative_mean_cos": float(negative_similarity.mean()),
            "validation_gap": gap,
            "validation_auc": auc,
            "unrelated_pair_mean_cos": unrelated,
        }
        print("%-18s concordance %6.4f (shuffled %6.4f, n=%d) | valid gap %+0.4f AUC %6.4f | unrelated cos %+0.4f"
              % (name, concordance, concordance_shuffled, pairs_seen, gap, auc, unrelated))

    payload = json.dumps(results, indent=2, default=float) + "\n"
    path = OUT / "quantised_relation_audit.json"
    path.write_text(payload, encoding="utf-8")
    print("\nwritten:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
