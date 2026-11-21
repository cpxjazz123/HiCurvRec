#!/usr/bin/env python3
"""Independently re-derive the negative-side scores that the first audit got wrong.

The first version read ``positive_scores[negative_id]`` instead of scoring
``cos(h[anchor], h[negative])``.  This module rebuilds the negative scores with a
different implementation path (gather + einsum, no shared helper) and compares them
elementwise with what the corrected audit produces, so an agreement is evidence
rather than a repeated mistake.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audit_relation as ar  # noqa: E402


def reference_scores(states: np.ndarray, anchor: np.ndarray,
                     partner: np.ndarray) -> np.ndarray:
    """cos of two explicit index arrays, computed without the audit's helpers."""
    left = np.take(states, anchor, axis=0)
    right = np.take(states, partner, axis=0)
    return np.einsum("ij,ij->i", left, right)


def compare(states: np.ndarray, pairs: np.ndarray, negatives: np.ndarray) -> dict:
    pairs = np.asarray(pairs, dtype=np.int64)
    negatives = np.asarray(negatives, dtype=np.int64)
    pos, neg = ar.triplet_scores(states, pairs, negatives)
    ref_pos = reference_scores(states, pairs[:, 0], pairs[:, 1])
    ref_neg = reference_scores(states, pairs[:, 0], negatives)
    wrong = pos[negatives]           # the superseded reading, kept only to size the error
    return {
        "triplets": int(len(pairs)),
        "positive_max_abs_diff": float(np.abs(pos - ref_pos).max()),
        "negative_max_abs_diff": float(np.abs(neg - ref_neg).max()),
        "negative_mean_audit": float(neg.mean()),
        "negative_mean_reference": float(ref_neg.mean()),
        "superseded_reading_mean": float(wrong.mean()),
        "superseded_reading_max_abs_diff": float(np.abs(wrong - ref_neg).max()),
    }
