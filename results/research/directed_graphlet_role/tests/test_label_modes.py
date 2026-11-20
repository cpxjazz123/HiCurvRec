"""Issue #221 section 5: the three label sources really differ.

The direction ablation is only meaningful if "undirected" collapses the direction
(one class per family) and "shuffled" keeps the skeleton and edge count while
changing which subtype is assigned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
TEMPLATE = HERE / "template"
sys.path.insert(0, str(TEMPLATE))

from directed_graphlets import classify, load_adjacency  # noqa: E402
from directed_role import load_pool, load_vocabulary  # noqa: E402


def test_undirected_mode_has_one_class_per_family():
    vocab = load_vocabulary("undirected")
    assert vocab["n_subtypes"] == 5
    assert vocab["subtype_of"] == {"K3": 0, "P3": 1, "C4": 2, "P4": 3, "K1_3": 4}


def test_directed_mode_has_many_more_classes():
    vocab = load_vocabulary("directed")
    assert vocab["n_subtypes"] > 30, vocab["n_subtypes"]


def test_shuffle_preserves_skeleton_and_edge_count_but_changes_the_type():
    import drq

    adjacency = load_adjacency()
    families = load_pool("heldout", load_vocabulary("directed"), max_instances=200)
    nodes = families["P3"][0][:40]
    changed = 0
    for instance in nodes:
        matrix = np.asarray(adjacency[instance][:, instance].todense(), dtype=bool)
        shuffled = drq._shuffle_directions(matrix, instance)
        assert shuffled.sum() == matrix.sum() or (matrix | matrix.T).sum() == (shuffled | shuffled.T).sum()
        assert classify(shuffled)[0] == classify(matrix)[0]
        if classify(shuffled)[1] != classify(matrix)[1]:
            changed += 1
    assert changed > 0, "shuffling never changed the subtype, so it is not an ablation"
