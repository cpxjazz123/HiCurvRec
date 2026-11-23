"""Quantized geometry fidelity against the frozen graph targets (issue #249, CONTRACT.md §2, §8).

Two inputs are kept strictly apart:

* the **continuous** vectors ``z`` (what an encoder produces), and
* the **hard SID decoded** vectors ``q`` (what the discrete three-layer code actually decodes to).

They are reported in separate blocks; continuous performance never substitutes for the discrete
SID. The evaluator never trains a probe and never lets the training side read a held-out target.
"""
from __future__ import annotations
import numpy as np
import torch

from tools.graph_infra.diff_sectional import build_metric


def prefix_collision(sid: np.ndarray, length: int = 3) -> dict:
    """Prefix collision and duplicate share under the frozen two calibers."""
    rows = np.asarray(sid)
    prefix = rows[:, :length]
    distinct = len(np.unique(prefix, axis=0))
    _, counts = np.unique(prefix, axis=0, return_counts=True)
    duplicated = int(counts[counts > 1].sum())
    return {'items': int(len(rows)), 'distinct_prefix': distinct,
            'collision_prefix': 1.0 - distinct / len(rows),
            'duplicate_prefix_share': duplicated / len(rows),
            'length': int(length)}


def prefix_sharing(sid: np.ndarray) -> dict:
    """Share of items sharing the first, second and third code with another item."""
    rows = np.asarray(sid)
    shares = {}
    for level in range(1, 4):
        _, counts = np.unique(rows[:, :level], axis=0, return_counts=True)
        duplicated = int(counts[counts > 1].sum())
        shares[f'L{level}'] = duplicated / len(rows)
    return shares


def codebook_occupancy(sid: np.ndarray) -> dict:
    """Per-level usage, entropy in bits and the largest bucket share."""
    rows = np.asarray(sid)
    report = []
    for level in range(3):
        values, counts = np.unique(rows[:, level], return_counts=True)
        probabilities = counts / counts.sum()
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        report.append({'level': level, 'used': int(len(values)),
                       'entropy': entropy, 'entropy_unit': 'bits',
                       'max_share': float(probabilities.max())})
    return report


def distance_distortion(d_real: np.ndarray, d_quantized: np.ndarray) -> dict:
    """How far the quantized geometry is from the real one."""
    real = np.asarray(d_real, dtype=np.float64)
    quantized = np.asarray(d_quantized, dtype=np.float64)
    difference = quantized - real
    return {'mean_abs_error': float(np.abs(difference).mean()),
            'mean_signed_error': float(difference.mean()),
            'rms_error': float(np.sqrt((difference ** 2).mean())),
            'max_abs_error': float(np.abs(difference).max()),
            'relative_mean_abs_error': float(np.abs(difference).mean() / max(np.abs(real).mean(), 1e-12))}


def scale_factor(kappa_graph: np.ndarray, kappa_quantized: np.ndarray) -> float:
    """The least-squares ratio that puts the two sides on one scale."""
    graph = np.asarray(kappa_graph, dtype=np.float64)
    quantized = np.asarray(kappa_quantized, dtype=np.float64)
    denominator = float((quantized ** 2).sum())
    if denominator <= 0:
        return float('nan')
    return float((graph * quantized).sum() / denominator)


def sign_agreement(kappa_graph: np.ndarray, kappa_quantized: np.ndarray,
                    tolerance: float = 1e-12) -> dict:
    """Scale-free agreement of the curvature sign."""
    graph = np.asarray(kappa_graph, dtype=np.float64)
    quantized = np.asarray(kappa_quantized, dtype=np.float64)
    graph_sign = np.sign(np.where(np.abs(graph) <= tolerance, 0.0, graph))
    quantized_sign = np.sign(np.where(np.abs(quantized) <= tolerance, 0.0, quantized))
    same = (graph_sign == quantized_sign)
    both_nonzero = (graph_sign != 0) & (quantized_sign != 0)
    positive = graph_sign > 0
    negative = graph_sign < 0
    return {
        'sign_agreement': float(same.mean()),
        'sign_agreement_nonzero_only': float(same[both_nonzero].mean()) if both_nonzero.any() else float('nan'),
        'positive_count': int(positive.sum()),
        'negative_count': int(negative.sum()),
        'zero_count': int((graph_sign == 0).sum()),
        'positive_sign_agreement': float(same[positive].mean()) if positive.any() else float('nan'),
        'negative_sign_agreement': float(same[negative].mean()) if negative.any() else float('nan'),
        'tolerance': tolerance,
    }


def distribution_difference(kappa_graph: np.ndarray, kappa_quantized: np.ndarray) -> dict:
    """Macro and micro differences between the two curvature distributions."""
    graph = np.asarray(kappa_graph, dtype=np.float64)
    quantized = np.asarray(kappa_quantized, dtype=np.float64)
    difference = quantized - graph
    return {'macro_mean_abs_difference': float(np.abs(difference).mean()),
            'micro_mean_abs_difference': float(np.abs(difference).mean()),
            'micro_mean_signed_difference': float(difference.mean()),
            'micro_rms_difference': float(np.sqrt((difference ** 2).mean())),
            'note': 'the macro figure is over the evaluated triples; the micro figure is the same '
                    'quantity over the same set and is reported for symmetry with the per-item view'}


def per_item_distribution(kappa_graph: np.ndarray, kappa_quantized: np.ndarray,
                          item_of_triple: np.ndarray) -> dict:
    """Per-item distribution differences of the curvature."""
    graph = np.asarray(kappa_graph, dtype=np.float64)
    quantized = np.asarray(kappa_quantized, dtype=np.float64)
    items = np.asarray(item_of_triple)
    unique = np.unique(items)
    mean_difference = np.zeros(len(unique))
    mean_abs_error = np.zeros(len(unique))
    for index, item in enumerate(unique.tolist()):
        mask = items == item
        mean_difference[index] = float((quantized[mask] - graph[mask]).mean())
        mean_abs_error[index] = float(np.abs(quantized[mask] - graph[mask]).mean())
    return {'items': unique.tolist(), 'mean_difference': mean_difference.tolist(),
            'mean_abs_difference': float(np.abs(mean_difference).mean()),
            'macro_mean_abs_error': float(mean_abs_error.mean()),
            'micro_mean_abs_error': float(np.abs(quantized - graph).mean())}


def evaluate_triples(vectors: torch.Tensor, graph_vectors: torch.Tensor, triples: np.ndarray,
                      references: np.ndarray, metric_family: str = 'euclidean',
                      kappa: float = 1.0, scale: float = 1.0,
                      weights: torch.Tensor | None = None) -> dict:
    """Evaluate one block of triples in the quantized space and compare with the graph space.

    ``references`` holds the reference node indices per triple; the same indices are used on both
    sides, so the comparison is apples to apples.
    """
    metric = build_metric(metric_family, kappa, scale)
    triples = np.asarray(triples, dtype=np.int64)
    references = np.asarray(references, dtype=np.int64)
    reference_vectors = vectors[references]
    centre = vectors[triples[:, 0]]
    b = vectors[triples[:, 1]]
    c = vectors[triples[:, 2]]
    reference_graph = graph_vectors[references]
    graph_centre = graph_vectors[triples[:, 0]]
    graph_b = graph_vectors[triples[:, 1]]
    graph_c = graph_vectors[triples[:, 2]]
    quantized_distances = {
        'd_am': metric(reference_vectors, centre.unsqueeze(-2)),
        'd_ab': metric(reference_vectors, b.unsqueeze(-2)),
        'd_ac': metric(reference_vectors, c.unsqueeze(-2)),
        'd_bc': metric(b, c),
    }
    graph_distances = {
        'd_am': metric(reference_graph, graph_centre.unsqueeze(-2)),
        'd_ab': metric(reference_graph, graph_b.unsqueeze(-2)),
        'd_ac': metric(reference_graph, graph_c.unsqueeze(-2)),
        'd_bc': metric(graph_b, graph_c),
    }
    from tools.graph_infra.diff_sectional import sectional_statistic
    kappa_quantized = sectional_statistic(quantized_distances, weights=weights).detach().numpy()
    kappa_graph = sectional_statistic(graph_distances, weights=weights).detach().numpy()
    return {'kappa_graph': kappa_graph, 'kappa_quantized': kappa_quantized,
            'distance_distortion': distance_distortion(
                graph_distances['d_am'].detach().numpy().ravel(),
                quantized_distances['d_am'].detach().numpy().ravel())}


def decode_hard_sid(sid: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """Decode a hard SID into ``q`` by summing the three level codewords.

    ``codebooks`` holds one (entries, dim) array per level. The SID's fourth column is the
    disambiguation token and is not part of the semantic code.
    """
    rows = np.asarray(sid)
    if len(codebooks) != 3:
        raise ValueError('three level codebooks are required')
    dimension = codebooks[0].shape[1]
    decoded = np.zeros((len(rows), dimension), dtype=np.float64)
    for level in range(3):
        decoded += codebooks[level][rows[:, level]]
    return decoded
