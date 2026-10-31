"""Does a hyperbolic code space preserve behaviour distance better than a Euclidean one?

This is a measurement, not a training run. For each frozen RQ-VAE it compares two
distance functions on the SAME code vectors -- the Euclidean chord and the
Poincare distance of the same vector pushed into the unit ball -- against a
behaviour distance read from the TRAINING split only:

    d_beh(i,j) = -log( (n_ij + 1) / (sqrt(pop_i pop_j) + 1) )

with n_ij the symmetrised consecutive-pair count and pop the number of training
users. Two views are reported per model: the Spearman correlation over a pair
sample, and the behaviour-neighbour retrieval overlap (how much of an item's
top-10 successor set is recovered by its top-10 nearest codes).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

REPO = Path("/fs04/ar57/wenyu/GeneRec")
PARQUET = REPO / "results" / "stage0_build_parquet"
ARMS = ["TIGER_RQ-VAE", "TIGER_BRad-RQ", "TIGER_CURV-RESID-CONTRAST", "TIGER_BEHDIR-L4"]
PAIR_SAMPLE = 60_000
RANDOM_PAIRS = 60_000
TOP_K = 10
BLOCK = 512


def poincare(points: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(points, axis=1, keepdims=True)
    alpha = 1.0 / float(norm.max())
    return np.tanh(alpha * norm) * points / np.maximum(norm, 1e-12)


def hyperbolic_matrix(points: np.ndarray, others: np.ndarray) -> np.ndarray:
    difference = ((points[:, None, :] - others[None, :, :]) ** 2).sum(-1)
    p_sq = (points ** 2).sum(-1)[:, None]
    c_sq = (others ** 2).sum(-1)[None, :]
    argument = 1.0 + 2.0 * difference / np.maximum((1.0 - p_sq) * (1.0 - c_sq), 1e-12)
    return np.arccosh(np.maximum(argument, 1.0 + 1e-8))


def top_k(matrix: np.ndarray, k: int) -> np.ndarray:
    return np.argpartition(matrix, k, axis=1)[:, :k]


def main() -> int:
    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    counts: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    users: dict[int, set] = defaultdict(set)
    for user, (history, target) in enumerate(zip(train["history"].tolist(),
                                                 train["target"].tolist())):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for item in set(values):
            users[item].add(user)
        for first, second in zip(values[:-1], values[1:]):
            counts[first][second] += 1.0
            counts[second][first] += 1.0
    pop = {item: float(len(users[item])) for item in users}

    rows = []
    for arm in ARMS:
        directory = REPO / "results" / "stage2_RQ-VAE" / arm
        checkpoint = directory / "rqvae_best.pth"
        sids = directory / "item_sids.json"
        if not checkpoint.exists() or not sids.exists():
            rows.append({"arm": arm, "status": "missing"})
            continue
        state = torch.load(checkpoint, map_location="cpu")["state_dict"]
        codebooks = [state["rq.vq_layers.%d.embed.weight" % level].double().numpy()
                     for level in range(3)]
        baseline = json.loads(sids.read_text())
        items = sorted(int(key) for key in baseline)
        codes = np.asarray([baseline[str(item)][:3] for item in items], dtype=np.int64)
        vectors = np.zeros((len(items), codebooks[0].shape[1]), dtype=np.float64)
        for level, book in enumerate(codebooks):
            vectors += book[codes[:, level]]
        ball = poincare(vectors)
        latent = torch.load(checkpoint, map_location="cpu")["state_dict"]
        del latent

        # behaviour pairs, restricted to items present in the model
        index = {item: row for row, item in enumerate(items)}
        observed = [(index[first], index[second], count)
                    for first, table in counts.items() if first in index
                    for second, count in table.items() if second in index]
        if len(observed) > PAIR_SAMPLE:
            step = len(observed) // PAIR_SAMPLE
            observed = observed[::step]
        rng = np.random.default_rng(42)
        random_pairs = rng.integers(0, len(items), size=(RANDOM_PAIRS, 2))
        pairs = np.asarray([(first, second) for first, second, _ in observed], dtype=np.int64)
        counts_ij = np.asarray([count for _, _, count in observed], dtype=np.float64)
        first_row = pairs[:, 0]
        second_row = pairs[:, 1]
        pop_i = np.asarray([pop.get(items[row], 1.0) for row in pairs[:, 0]])
        pop_j = np.asarray([pop.get(items[row], 1.0) for row in pairs[:, 1]])
        behaviour = -np.log((counts_ij + 1.0) / (np.sqrt(pop_i * pop_j) + 1.0))

        euclid = np.linalg.norm(vectors[first_row] - vectors[second_row], axis=1)
        curved = np.asarray([hyperbolic_matrix(ball[first_row[a]:first_row[a] + 1],
                                               ball[second_row[a]:second_row[a] + 1])[0, 0]
                             for a in range(len(first_row))])

        rho_e = spearmanr(behaviour, euclid).correlation
        rho_h = spearmanr(behaviour, curved).correlation

        # behaviour-neighbour retrieval: the successor set versus the nearest codes
        overlap_e, overlap_h = 0.0, 0.0
        asked = 0
        for row in range(0, len(items), max(1, len(items) // 2000)):
            item = items[row]
            table = counts.get(item)
            if not table:
                continue
            successors = sorted(table, key=lambda target: -table[target])[:TOP_K]
            successors = [index[target] for target in successors if target in index]
            if len(successors) < TOP_K:
                continue
            nearest = np.array([
                np.argpartition(hyperbolic_matrix(ball[row:row + 1], ball)[0], TOP_K)[:TOP_K],
            ])[0] if False else None
            del nearest
            euclid_row = np.linalg.norm(vectors - vectors[row], axis=1)
            curved_row = hyperbolic_matrix(ball[row:row + 1], ball)[0]
            near_e = set(np.argpartition(euclid_row, TOP_K)[:TOP_K].tolist())
            near_h = set(np.argpartition(curved_row, TOP_K)[:TOP_K].tolist())
            overlap_e += len(near_e & set(successors)) / len(successors)
            overlap_h += len(near_h & set(successors)) / len(successors)
            asked += 1

        rows.append({
            "arm": arm,
            "status": "measured",
            "spearman_behaviour_vs_euclidean": round(float(rho_e), 4),
            "spearman_behaviour_vs_poincare": round(float(rho_h), 4),
            "retrieval_top10_euclidean": round(overlap_e / max(asked, 1), 4),
            "retrieval_top10_poincare": round(overlap_h / max(asked, 1), 4),
            "items_asked": asked,
            "pairs": int(len(pairs)),
        })
        print(json.dumps(rows[-1]), flush=True)

    out = REPO / "results" / "research" / "121" / "rounds" / "round_20261031_115851_915" / "behaviour_distance_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print("written", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
