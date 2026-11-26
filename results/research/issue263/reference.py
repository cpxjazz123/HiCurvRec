"""Independent NumPy reference for the Issue #263 label and distribution math.

This module deliberately does not import the production pretraining script or
the candidate GNN. It provides a second implementation of the histogram,
normalisation and KL objective used to audit the pretraining labels.
"""
from __future__ import annotations

import math

import numpy as np


def fixed_edges():
    return np.asarray([-np.inf, -4., -3., -2., -1., 0., 1., 2., 3., 4., 8., np.inf])


def histogram(values, edges=None):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    edges = fixed_edges() if edges is None else np.asarray(edges, dtype=np.float64)
    if not len(values):
        return np.zeros(len(edges) - 1, dtype=np.float64)
    # NumPy's histogram is an independent implementation path from the
    # production one; the caller supplies the same frozen edges.
    counts, _ = np.histogram(values, bins=edges)
    return counts.astype(np.float64) / max(float(counts.sum()), 1.0)


def normalise(counts, mask):
    counts = np.asarray(counts, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if counts.shape[0] != mask.shape[0]:
        raise ValueError("mask must align with rows")
    out = np.zeros_like(counts)
    for row in np.flatnonzero(mask):
        total = counts[row].sum()
        if total > 0:
            out[row] = counts[row] / total
    return out


def cross_entropy(logits, target, mask):
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if logits.shape != target.shape or mask.shape != logits.shape[:1]:
        raise ValueError("aligned logits/target/mask required")
    values = []
    for row in np.flatnonzero(mask & (target.sum(1) > 0)):
        shifted = logits[row] - logits[row].max()
        log_z = math.log(np.exp(shifted).sum())
        log_prob = shifted - log_z
        positive = target[row] > 0
        values.append(float((target[row][positive] *
                             (-log_prob[positive])).sum()))
    return float(np.mean(values)) if values else None


def kl(logits, target, mask):
    ce = cross_entropy(logits, target, mask)
    if ce is None:
        return None
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    entropy = []
    for row in np.flatnonzero(mask & (target.sum(1) > 0)):
        positive = target[row] > 0
        entropy.append(float((target[row][positive] *
                              (-np.log(target[row][positive]))).sum()))
    return float(ce - np.mean(entropy))


def scalar_sign_accuracy(target, predicted):
    target = np.sign(np.asarray(target, dtype=np.float64))
    predicted = np.sign(np.asarray(predicted, dtype=np.float64))
    return float((target == predicted).mean())
