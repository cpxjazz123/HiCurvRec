"""Prefix co-occurrence coherence objective.

The behaviour signal is the train-only co-consumption matrix. For every user the
longest training sequence is reconstructed (the train split repeats a user's
history at several prefix lengths, so only the longest prefix plus its target is
the complete available sequence) and every pair of items inside that sequence is
counted once. Each row is L2-normalised, so the inner product of two rows is the
cosine similarity of the two items' co-consumption profiles.

The objective multiplies that fixed similarity with the model's own probability
that two items share all three prefix codes and is maximised:

    L_coh = - mean_{i != j}  S_ij * prod_{l=1..3} (p_l,i . p_l,j)

``S_ij`` is detached, so the gradient reaches the encoder and the three
codebooks only through the soft assignment ``p_l``. ``p_l`` is the softmax over
the level-l codebook of the negative curved (Poincare) distance between the
level-l residual and the codewords, which is where the curvature enters the
coupling path.

Two ablations are wired as constants in ``train_rqvae.py``:

* ``behaviour_off`` permutes the item-to-row assignment of the co-occurrence
  matrix, so the signal is destroyed while the machinery is untouched;
* ``geometry_off`` replaces the curved distance with the Euclidean one inside
  the soft assignment.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def recover_train_sequences(frame) -> list[tuple[int, ...]]:
    """Longest training prefix per user, so no event is counted twice."""
    latest: dict[int, tuple[int, ...]] = {}
    for user, history, target in frame[["user", "history", "target"]].itertuples(
        index=False, name=None
    ):
        sequence = tuple(int(x) for x in history) + (int(target),)
        current = latest.get(int(user))
        if current is None or len(sequence) > len(current):
            latest[int(user)] = sequence
    return [latest[key] for key in sorted(latest)]


def build_cooccurrence(
    sequences: Iterable[tuple[int, ...]],
    item_count: int,
    *,
    permute_rows: bool = False,
    seed: int = 42,
):
    """Row-normalised co-occurrence matrix over the training sequences.

    Returns a ``scipy.sparse`` CSR matrix with unit-L2 rows (or all-zero rows for
    items that never co-occur with anything). ``permute_rows`` shuffles the
    item-to-row assignment, which is the behaviour-off ablation.
    """
    import scipy.sparse as sp

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for sequence in sequences:
        unique = sorted({int(item) for item in sequence})
        for left_index, left in enumerate(unique):
            if not 0 <= left < item_count:
                continue
            for right in unique[left_index + 1:]:
                if not 0 <= right < item_count:
                    continue
                rows.append(left)
                cols.append(right)
                data.append(1.0)
                rows.append(right)
                cols.append(left)
                data.append(1.0)
    matrix = sp.csr_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(item_count, item_count),
    )
    matrix.sum_duplicates()
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
    norms[norms == 0.0] = 1.0
    scale = sp.diags(1.0 / norms)
    matrix = scale @ matrix
    matrix = matrix.tocsr()
    if permute_rows:
        generator = np.random.default_rng(seed)
        permutation = generator.permutation(item_count)
        matrix = matrix[permutation]
    return matrix


def batch_similarity(matrix, item_ids: np.ndarray) -> np.ndarray:
    """Cosine similarity of the co-occurrence rows of the given items."""
    block = matrix[item_ids]
    product = (block @ block.T).toarray().astype(np.float32)
    np.fill_diagonal(product, 0.0)
    return product


def soft_prefix_assignment(
    residuals: list[torch.Tensor],
    codebooks: list[torch.Tensor],
    *,
    curvature: float,
    temperature: float,
    euclidean: bool,
    encode_points,
) -> list[torch.Tensor]:
    """Per-level soft assignment of the residuals over the level codebooks.

    With ``euclidean=False`` the distance is the Poincare geodesic distance in
    the ball, so the curvature participates in the assignment; with
    ``euclidean=True`` it is the plain Euclidean distance (geometry-off).

    The distances of one row are min-max normalised before the softmax. The
    geodesic distances inside a ball of radius ``1/sqrt(c)`` span a very small
    range, so an unnormalised softmax would be uniform over the codebook: the
    prefix agreement would then be ``1/K`` at every level and the objective
    would carry no gradient at all.
    """
    from hyperbolic import pair_poincare_distance

    assignments: list[torch.Tensor] = []
    for level, (residual, codebook) in enumerate(zip(residuals, codebooks)):
        if euclidean:
            distances = torch.cdist(residual, codebook, p=2)
        else:
            points = encode_points(residual, curvature)
            codes = encode_points(codebook, curvature)
            distances = pair_poincare_distance(points, codes, curvature)
        lowest = distances.min(dim=-1, keepdim=True).values
        highest = distances.max(dim=-1, keepdim=True).values
        spread = (highest - lowest).clamp_min(1e-6)
        normalised = (distances - lowest) / spread
        assignments.append(torch.softmax(-normalised / max(temperature, 1e-6), dim=-1))
    return assignments


def prefix_joint_probability(assignments: list[torch.Tensor]) -> torch.Tensor:
    """Probability that two items share every level's code.

    ``J_ij = prod_l (p_l,i . p_l,j)``; the diagonal is the item's own
    probability of being assigned to its own codes.
    """
    if not assignments:
        raise ValueError("no level assignment was provided")
    joint = assignments[0] @ assignments[0].T
    for assignment in assignments[1:]:
        joint = joint * (assignment @ assignment.T)
    return joint


def coherence_term(
    similarity: np.ndarray,
    assignments: list[torch.Tensor],
    codebook_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Negative mean co-occurrence-weighted prefix agreement.

    The raw agreement of two random items is ``1/K**L`` (one code per level out
    of ``K``), so the raw value of the term is ~1e-7 and cannot be weighed
    against a loss of order one. The term is therefore divided by that uniform
    baseline, which turns it into the *lift* of the prefix agreement over
    random: it is 1 when the prefixes carry no information and grows when the
    model puts behaviourally similar items together.
    """
    joint = prefix_joint_probability(assignments)
    weights = torch.as_tensor(similarity, dtype=joint.dtype, device=joint.device)
    diagonal = torch.diagonal(joint)
    off_diagonal = joint - torch.diag_embed(diagonal)
    count = max(int(off_diagonal.numel()) - int(off_diagonal.shape[0]), 1)
    uniform = 1.0 / float(codebook_size ** max(len(assignments), 1))
    agreement = (weights * off_diagonal).sum() / count
    loss = -agreement / uniform
    stats = {
        "joint_mean": float(joint.detach().mean()),
        "similarity_mean": float(similarity.mean()),
        "agreement": float(agreement.detach()),
        "agreement_lift": float((agreement / uniform).detach()),
    }
    return loss, stats
