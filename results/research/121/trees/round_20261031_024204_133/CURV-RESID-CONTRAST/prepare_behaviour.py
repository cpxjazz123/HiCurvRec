"""The behaviour constants: one successor-contrast direction and gain per item.

Exactly the proposal's recipe, from the TRAINING interactions only:

* ``S_i``: the multiset of next items observed after item i inside a training
  session, capped at the 32 highest-count successors;
* ``N_i``: 32 items drawn with probability proportional to training popularity
  from the SAME popularity decile as the items of ``S_i``, excluding i, its 1-hop
  neighbours and its successors, with a seed keyed deterministically on the item
  id;
* ``u_i = mean_{j in S_i} log_{x_i}(x_j) - mean_{j in N_i} log_{x_i}(x_j)`` in the
  ball of curvature 1, where ``x_i = exp_0(s z_i)`` with ``s = 0.9 / max||z||``;
* ``u_hat_i = u_i / ||u_i||_{x_i}`` and
  ``alpha_i = clip(||u_i||_{x_i} / median_j ||u_j||_{x_j}, 0.5, 2.0)``;
* items with fewer than two successors or a norm below 1e-6 fall back to
  ``alpha = 1`` and ``u_hat = 0`` (the isotropic arm).

The result is a constant file consumed by the trainer; nothing here looks at
validation or test interactions, at any SID, or at any recall.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model.curved_chain import conformal, exp_map, log_at  # noqa: E402

REPO = HERE.parents[5]
PARQUET = REPO / "results" / "stage0_build_parquet"
TEXT = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUTPUT = HERE / "dataset" / "behaviour_constants.npz"
MANIFEST = HERE / "dataset" / "behaviour_constants.json"

N_ITEMS = 24587
SUCCESSORS = 32
NULLS = 32
RADIAL_USAGE = 0.9
SEED = 42


def main() -> int:
    embeddings = np.load(TEXT).astype(np.float64)
    order = np.argsort(-np.linalg.norm(embeddings, axis=1))
    del order
    scale = RADIAL_USAGE / float(np.linalg.norm(embeddings, axis=1).max())
    points = exp_map(torch.from_numpy(embeddings * scale))

    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    sequences = []
    for history, target in zip(train["history"].tolist(), train["target"].tolist()):
        sequence = [int(item) for item in history]
        if target is not None:
            sequence.append(int(target))
        if len(sequence) > 1:
            sequences.append(sequence)

    successors: dict[int, dict[int, int]] = {}
    popularity = np.zeros(N_ITEMS, dtype=np.float64)
    for sequence in sequences:
        for item in sequence:
            popularity[item] += 1.0
        for first, second in zip(sequence[:-1], sequence[1:]):
            successors.setdefault(first, {})
            successors[first][second] = successors[first].get(second, 0) + 1

    # the popularity deciles, from the training counts only
    ranks = np.argsort(np.argsort(-popularity))
    decile = (ranks * 10 // max(N_ITEMS, 1)).clip(0, 9)
    members = {value: np.flatnonzero(decile == value) for value in range(10)}

    adjacency = {}
    for item, table in successors.items():
        adjacency[item] = np.fromiter(table.keys(), dtype=np.int64)

    direction = np.zeros((N_ITEMS, embeddings.shape[1]), dtype=np.float64)
    norm = np.zeros(N_ITEMS, dtype=np.float64)
    for item in range(N_ITEMS):
        table = successors.get(item)
        if not table:
            continue
        ranked = sorted(table.items(), key=lambda pair: -pair[1])[:SUCCESSORS]
        positive = np.asarray([pair[0] for pair in ranked], dtype=np.int64)
        if len(positive) < 2:
            continue
        rng = np.random.default_rng(SEED * 1_000_003 + item)
        pool = members[int(decile[item])]
        excluded = set(positive.tolist()) | {item}
        excluded |= set(adjacency.get(item, np.zeros(0, dtype=np.int64)).tolist())
        candidates = np.asarray([value for value in pool if value not in excluded],
                                dtype=np.int64)
        if len(candidates) < NULLS:
            candidates = np.asarray([value for value in range(N_ITEMS)
                                     if value not in excluded], dtype=np.int64)
        if len(candidates) < NULLS:
            continue
        probabilities = popularity[candidates] / max(popularity[candidates].sum(), 1e-12)
        null = rng.choice(candidates, size=NULLS, replace=False, p=probabilities)
        here = points[item]
        positive_tangent = log_at(here.expand(len(positive), -1),
                                  points[torch.from_numpy(positive)], 1.0)
        null_tangent = log_at(here.expand(len(null), -1), points[torch.from_numpy(null)], 1.0)
        difference = positive_tangent.mean(dim=0) - null_tangent.mean(dim=0)
        direction[item] = difference.numpy()
        norm[item] = float((conformal(here, 1.0) * difference.norm()).item())

    # The codebook lives in CODEBOOK_DIM dimensions while the text geometry is
    # 768: the direction is projected onto its own leading 32 components (a
    # deterministic, training-only projection; no recall, no search), and the
    # per-item unit direction is renormalised there. The gain stays the one
    # measured in the text geometry.
    from sklearn.decomposition import TruncatedSVD

    components = min(32, direction.shape[0] - 1, direction.shape[1])
    projection = TruncatedSVD(n_components=components, random_state=SEED)
    projected = projection.fit_transform(direction)
    direction = np.asarray(projected, dtype=np.float64)
    # the frozen BALL POINTS live in the same 32-dimensional subspace, so the
    # direction and the geometry the chain quantizes share one basis
    points_projected = embeddings @ projection.components_.T
    points_frozen = exp_map(torch.from_numpy(points_projected * scale)).numpy()

    valid = norm > 1e-6
    median = float(np.median(norm[valid])) if valid.any() else 1.0
    gain = np.ones(N_ITEMS, dtype=np.float64)
    gain[valid] = np.clip(norm[valid] / max(median, 1e-12), 0.5, 2.0)
    unit = np.zeros_like(direction)
    lengths = np.linalg.norm(direction, axis=1, keepdims=True)
    safe = np.maximum(lengths, 1e-12)
    unit[valid] = direction[valid] / safe[valid]
    unit[~valid] = 0.0
    gain[~valid] = 1.0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, direction=unit.astype(np.float32),
                        gain=gain.astype(np.float32), norm=norm.astype(np.float32),
                        valid=valid, points=points_frozen.astype(np.float32))
    MANIFEST.write_text(json.dumps({
        "text_embeddings": str(TEXT.relative_to(REPO)),
        "train_parquet": str((PARQUET / "train.parquet").relative_to(REPO)),
        "radial_scale": scale,
        "successors_cap": SUCCESSORS,
        "nulls": NULLS,
        "seed": SEED,
        "items_with_direction": int(valid.sum()),
        "items_fallback_isotropic": int((~valid).sum()),
        "norm_median": median,
        "gain_min": float(gain.min()), "gain_max": float(gain.max()),
        "codebook_dim": int(direction.shape[1]),
        "projection_explained_variance": float(projection.explained_variance_ratio_.sum()),
        "rule": "successor mean log map minus popularity-decile-matched null mean "
                "log map, in the curvature-1 ball, projected onto its own leading "
                "components with a TruncatedSVD; training split only",
    }, indent=2) + "\n")
    print(json.dumps({"items_with_direction": int(valid.sum()),
                      "median_norm": median,
                      "gain_range": [float(gain.min()), float(gain.max())]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
