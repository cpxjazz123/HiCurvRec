"""Precompute the behaviour constants of GDELTA-SUCC-PULL-RQ from the TRAINING split.

For every item:

* ``N_i``  -- the four successors with ``n_ij > 0`` and the smallest behaviour
  distance ``d_beh(i,j) = -log((n_ij+1)/(sqrt(pop_i pop_j)+1))``;
* ``w_ij`` -- ``softmax_j(-d_beh(i,j))`` over ``N_i``;
* ``S_i``  -- ``{i}`` union ``N_i`` union the four predecessors with the largest
  ``n_ji`` (at most nine items), with ``l_ij = (d_beh(i,j)+d_beh(j,i))/2`` and the
  shortest-path closure on the induced subgraph;
* ``delta_i`` -- the local Gromov delta, the maximum over quadruples of the
  difference between the largest and the middle of the three pair sums;
* ``K_i``  -- ``clip(delta_ref/(delta_ref+delta_i), 0.1, 2)`` with
  ``delta_ref = median_i delta_i``;
* ``g[l]`` -- the successor pull target at level ``l``: the ``w``-weighted mean of
  the successors' frozen code vectors (read from the frozen Stage-2 SID, which is
  read-only), in the tangent space the quantizer scores.

Written next to the candidate as ``behaviour_constants.npz``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _repository_root(start: Path) -> Path:
    """The repository root: the nearest ancestor that holds both trees.

    The candidate is copied to ``results/stage2_RQ-VAE/<ARM>/`` before it runs,
    so a fixed parents[N] would resolve to the filesystem root there. The root is
    the directory that contains the two markers, found by walking upwards.
    """
    for candidate in (start, *start.parents):
        if (candidate / "stage2_RQ-VAE").is_dir() and (candidate / "results").is_dir():
            return candidate
    raise RuntimeError("cannot locate the repository root above " + str(start))


REPO = _repository_root(HERE)
PARQUET = REPO / "results" / "stage0_build_parquet"
BASELINE = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
OUTPUT = HERE / "behaviour_constants.npz"
NEIGHBOURS = 4
PREDECESSORS = 4
DELTA_REF_FALLBACK = 0.05


def build() -> dict:
    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    forward: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    backward: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    users: dict[int, set] = defaultdict(set)
    for user, (history, target) in enumerate(zip(train["history"].tolist(),
                                                 train["target"].tolist())):
        values = [int(item) for item in history]
        if target is not None:
            values.append(int(target))
        for item in set(values):
            users[item].add(user)
        for first, second in zip(values[:-1], values[1:]):
            forward[first][second] += 1.0
            backward[second][first] += 1.0

    items = sorted({item for sequence in forward.values() for item in sequence} |
                   {item for item in forward})
    index = {item: row for row, item in enumerate(items)}
    population = {item: float(len(users[item])) for item in items}

    def behaviour(first: int, second: int) -> float:
        count = forward.get(first, {}).get(second, 0.0)
        return float(-np.log((count + 1.0) /
                             (np.sqrt(population.get(first, 1.0)
                                      * population.get(second, 1.0)) + 1.0)))

    def symmetric(first: int, second: int) -> float:
        return 0.5 * (behaviour(first, second) + behaviour(second, first))

    deltas = []
    for item in items:
        successors = sorted((target for target in forward.get(item, {})
                             if forward[item][target] > 0),
                            key=lambda target: behaviour(item, target))[:NEIGHBOURS]
        predecessors = sorted(backward.get(item, {}),
                              key=lambda source: -backward[item][source])[:PREDECESSORS]
        members = list(dict.fromkeys([item] + successors + predecessors))
        size = len(members)
        if size < 4:
            deltas.append((item, DELTA_REF_FALLBACK))
            continue
        distance = np.full((size, size), np.inf)
        for a, first in enumerate(members):
            for b, second in enumerate(members):
                if a != b:
                    distance[a, b] = symmetric(first, second)
        # shortest-path closure (Floyd-Warshall on at most nine nodes)
        for k in range(size):
            distance = np.minimum(distance, distance[:, k][:, None] + distance[k][None, :])
        if not np.isfinite(distance).all():
            deltas.append((item, DELTA_REF_FALLBACK))
            continue
        best = 0.0
        for a, b, c, p in combinations(range(size), 4):
            sums = sorted([distance[a, b] + distance[c, p],
                           distance[a, c] + distance[b, p],
                           distance[a, p] + distance[b, c]])
            best = max(best, 0.5 * (sums[2] - sums[1]))
        deltas.append((item, best))

    delta_of = dict(deltas)
    reference = float(np.median(list(delta_of.values())))
    curvature = {item: float(np.clip(reference / (reference + delta_of[item]), 0.1, 2.0))
                 for item in items}

    baseline = json.loads((BASELINE / "item_sids.json").read_text())
    import torch
    checkpoint = torch.load(BASELINE / "rqvae_best.pth", map_location="cpu")["state_dict"]
    books = [checkpoint["rq.vq_layers.%d.embed.weight" % level].double().numpy()
             for level in range(3)]

    tables = [np.zeros((len(items), books[0].shape[1])) for _ in range(3)]
    weights = np.zeros((len(items), NEIGHBOURS))
    for item in items:
        successors = sorted((target for target in forward.get(item, {})
                             if forward[item][target] > 0),
                            key=lambda target: behaviour(item, target))[:NEIGHBOURS]
        if not successors:
            continue
        scores = np.asarray([behaviour(item, target) for target in successors])
        mass = np.exp(-scores - (-scores).max())
        mass = mass / mass.sum()
        row = index[item]
        for position, target in enumerate(successors):
            if target not in index or str(target) not in baseline:
                continue
            codes = baseline[str(target)][:3]
            for level in range(3):
                tables[level][row] += mass[position] * books[level][int(codes[level])]
        weights[row, :len(mass)] = mass

    payload = {
        "items": np.asarray(items, dtype=np.int64),
        "curvature": np.asarray([curvature[item] for item in items], dtype=np.float32),
        "delta": np.asarray([delta_of[item] for item in items], dtype=np.float32),
        "g0": tables[0].astype(np.float32), "g1": tables[1].astype(np.float32),
        "g2": tables[2].astype(np.float32),
        "weights": weights.astype(np.float32),
        "delta_ref": np.float32(reference),
    }
    if __name__ != "__main__":
        return payload
    np.savez(OUTPUT, **payload)
    print(json.dumps({
        "items": len(items),
        "delta_ref": reference,
        "curvature_median": float(np.median([curvature[item] for item in items])),
        "curvature_min": float(np.min([curvature[item] for item in items])),
        "curvature_max": float(np.max([curvature[item] for item in items])),
        "output": str(OUTPUT),
    }, indent=1))
    return payload


if __name__ == "__main__":
    build()
    raise SystemExit(0)
