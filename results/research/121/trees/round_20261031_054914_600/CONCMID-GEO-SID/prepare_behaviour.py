"""CONCMID-GEO-SID: move every item toward its behaviour successors.

The proposal, exactly as specified, up to the two switches:

    C[i_t, i_{t+1}] += 1.0 ;  C[i_t, i_{t+2}] += 0.5      (training sequences only)
    p_ij = C_ij / row_sum ;  a_ij = p_ij / sqrt(p_j) ;  w_ij = a_ij / sum a
    eta_i = sum_j w_ij^2                                   (Herfindahl concentration)
    z_i = exp_0(kappa t_i)                                 (kappa = 0.9 / max ||t||)
    v_i = sum_j w_ij log_{z_i}(z_j)
    b_i = exp_{z_i}(eta_i v_i)

``BEHAV_SMOOTH=False`` sets ``eta_i = 0`` for every item, so ``b_i = z_i`` and the
run is the previous round's frozen-geometry chain exactly. ``CURV_C = 0`` replaces
the ball by flat space: ``z_i = kappa t_i``, ``log_z(w) = w - z`` and
``b_i = z_i + eta_i sum_j w_ij (z_j - z_i)``, i.e. the same weights and the same
concentration applied as an arithmetic blend.

The points are written in the same 32-dimensional subspace the codebooks live in:
the frozen text latents are projected onto their own leading components once,
deterministically, before any geometry happens.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model.curved_chain import exp_at, log_at, exp_map  # noqa: E402

REPO = HERE.parents[5]
PARQUET = REPO / "results" / "stage0_build_parquet"
TEXT = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
OUTPUT = HERE / "dataset" / "behaviour_constants.npz"
MANIFEST = HERE / "dataset" / "behaviour_constants.json"

N_ITEMS = 24587
CODEBOOK_DIM = 32
RADIAL_USAGE = 0.9
SEED = 42

#: the two switches of the round; only the both-on arm is trained
BEHAV_SMOOTH = True
CURV_C = 1.0


def main() -> int:
    embeddings = np.load(TEXT).astype(np.float64)
    projection = TruncatedSVD(n_components=CODEBOOK_DIM, random_state=SEED)
    latents = projection.fit_transform(embeddings)
    scale = RADIAL_USAGE / float(np.linalg.norm(latents, axis=1).max())
    points = exp_map(torch.from_numpy(latents * scale), 1.0) if CURV_C > 0 else \
        torch.from_numpy(latents * scale)

    train = pd.read_parquet(PARQUET / "train.parquet", columns=["history", "target"])
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for history, target in zip(train["history"].tolist(), train["target"].tolist()):
        sequence = [int(item) for item in history]
        if target is not None:
            sequence.append(int(target))
        for position in range(len(sequence) - 1):
            first, second = sequence[position], sequence[position + 1]
            if first != second:
                rows.append(first)
                columns.append(second)
                values.append(1.0)
        for position in range(len(sequence) - 2):
            first, third = sequence[position], sequence[position + 2]
            if first != third:
                rows.append(first)
                columns.append(third)
                values.append(0.5)
    counts = csr_matrix((np.asarray(values, dtype=np.float64),
                          (np.asarray(rows), np.asarray(columns))),
                         shape=(N_ITEMS, N_ITEMS))
    counts.setdiag(0.0)
    counts.eliminate_zeros()

    row_sum = np.asarray(counts.sum(axis=1)).ravel()
    frequency = np.asarray(counts.sum(axis=0)).ravel()
    popularity = frequency / max(frequency.sum(), 1e-12)
    inverse = 1.0 / np.sqrt(np.maximum(popularity, 1e-12))

    step = np.zeros((N_ITEMS, CODEBOOK_DIM), dtype=np.float64)
    eta = np.zeros(N_ITEMS, dtype=np.float64)
    counts_flat = counts.tocsr()
    for item in range(N_ITEMS):
        start, stop = counts_flat.indptr[item], counts_flat.indptr[item + 1]
        if stop <= start:
            continue
        targets = counts_flat.indices[start:stop]
        mass = counts_flat.data[start:stop]
        probabilities = mass / max(mass.sum(), 1e-12)
        adjusted = probabilities * inverse[targets]
        weights = adjusted / max(adjusted.sum(), 1e-12)
        eta[item] = float((weights * weights).sum())
        anchor = points[item]
        tangent = log_at(anchor.expand(len(targets), -1),
                         points[torch.from_numpy(targets)], CURV_C if CURV_C > 0 else 0.0)
        step[item] = (torch.from_numpy(weights[:, None]) * tangent).sum(dim=0).numpy()

    if not BEHAV_SMOOTH:
        eta[:] = 0.0
    moved = np.zeros_like(step)
    for item in range(N_ITEMS):
        if eta[item] <= 0.0:
            moved[item] = points[item].numpy()
            continue
        anchor = points[item]
        pushed = exp_at(anchor.expand(1, -1), torch.from_numpy(step[item:item + 1] * eta[item]),
                        CURV_C if CURV_C > 0 else 0.0)
        vector = pushed[0].numpy()
        radius = float(np.linalg.norm(vector))
        if radius > 0.995:
            vector = vector * (0.995 / radius)
        moved[item] = vector

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT,
                        direction=np.zeros((N_ITEMS, CODEBOOK_DIM), dtype=np.float32),
                        gain=np.ones(N_ITEMS, dtype=np.float32),
                        norm=np.zeros(N_ITEMS, dtype=np.float32),
                        valid=eta > 0,
                        points=moved.astype(np.float32),
                        eta=eta.astype(np.float32))
    displacement = np.linalg.norm(moved - points.numpy(), axis=1)
    MANIFEST.write_text(json.dumps({
        "mechanism": "CONCMID-GEO-SID",
        "behav_smooth": BEHAV_SMOOTH,
        "curv_c": CURV_C,
        "text_embeddings": str(TEXT.relative_to(REPO)),
        "train_parquet": str((PARQUET / "train.parquet").relative_to(REPO)),
        "codebook_dim": CODEBOOK_DIM,
        "radial_scale": scale,
        "projection_explained_variance": float(projection.explained_variance_ratio_.sum()),
        "items_with_transitions": int((eta > 0).sum()),
        "items_without_transitions": int((eta <= 0).sum()),
        "eta_median": float(np.median(eta)),
        "eta_p90": float(np.percentile(eta, 90)),
        "displacement_median": float(np.median(displacement)),
        "displacement_p90": float(np.percentile(displacement, 90)),
        "share_moved_above_1e-3": float(np.mean(displacement > 1e-3)),
        "rule": "lag-1 (weight 1) and lag-2 (weight 0.5) training transitions, "
                "row-normalised, popularity-adjusted by p_j^(-1/2), geodesic step "
                "scaled by the Herfindahl concentration; training split only",
    }, indent=2) + "\n")
    print(json.dumps({"items_with_transitions": int((eta > 0).sum()),
                      "eta_median": float(np.median(eta)),
                      "displacement_median": float(np.median(displacement)),
                      "share_moved": float(np.mean(displacement > 1e-3))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
