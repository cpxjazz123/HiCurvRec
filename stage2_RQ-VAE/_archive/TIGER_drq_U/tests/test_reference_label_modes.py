"""The undirected and shuffled reference arms must produce usable labels.

Both modes shipped without ever being run: the undirected vocabulary had no
orbit mask (joint_loss raised KeyError: 'allowed'), and the undirected targets
sliced the first N rows of the whole pool instead of the drawn instances.  These
tests pin the contract the two reference arms need.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import drq  # noqa: E402
from directed_role import FAMILIES, load_pool, load_vocabulary  # noqa: E402


def tiny_draws(mode: str, per_family: int = 8):
    vocab = load_vocabulary(mode)
    families = load_pool("train", vocab, max_instances=per_family)
    return {family: (torch.from_numpy(nodes), torch.from_numpy(orbits), torch.from_numpy(subtypes))
            for family, (nodes, orbits, subtypes) in families.items()}, vocab


def test_undirected_vocabulary_carries_the_orbit_mask():
    vocab = load_vocabulary("undirected")
    assert "allowed" in vocab
    assert tuple(vocab["allowed"].shape) == (vocab["n_subtypes"], vocab["n_orbits"])
    assert bool(vocab["allowed"].all())


def test_undirected_targets_are_computed_from_the_drawn_instances():
    draws, _ = tiny_draws("directed")
    vocab = load_vocabulary("undirected")
    targets = drq.targets_for(draws, vocab)
    for family, row in targets.items():
        assert row["subtype"].shape == (len(draws[family][0]),)
        assert row["orbit"].shape == (len(draws[family][0]), draws[family][0].shape[1])
        assert row["subtype"].min() >= 0
        assert row["orbit"].max() < vocab["n_orbits"]


def test_undirected_orbit_matches_the_symmetrised_pattern():
    """A directed path A->B->C collapses to a path: B is the centre (orbit 0)."""
    from directed_graphlets import canonical, load_adjacency

    adjacency = load_adjacency()
    draws, _ = tiny_draws("directed", per_family=64)
    vocab = load_vocabulary("undirected")
    targets = drq.targets_for(draws, vocab)
    for family, (nodes, _orbits, _subtypes) in draws.items():
        for row, (orbit_ids, expected) in enumerate(zip(targets[family]["orbit"],
                                                        _expected_orbits(adjacency, nodes.numpy()))):
            assert list(orbit_ids) == list(expected), (family, row)


def _expected_orbits(adjacency, nodes: np.ndarray):
    from directed_graphlets import canonical

    out = []
    for instance in nodes:
        matrix = np.asarray(adjacency[instance][:, instance].todense(), dtype=bool)
        _key, orbit_ids, _matrix = canonical(matrix | matrix.T)
        out.append(orbit_ids)
    return out


def test_shuffle_keeps_skeleton_and_reciprocity():
    matrix = np.array([[False, True, True],
                       [False, False, True],
                       [True, False, False]])       # 0->1, 0->2, 1->2, 2->0
    shuffled = drq._shuffle_directions(matrix, instance=(0, 1, 2))
    original = matrix | matrix.T
    kept = shuffled | shuffled.T
    assert np.array_equal(original, kept), "the undirected skeleton must not change"
    mutual = (matrix & matrix.T)
    assert np.array_equal(mutual, shuffled & shuffled.T), "reciprocity must be preserved"
    assert not np.array_equal(matrix, shuffled)


def test_joint_loss_survives_unknown_subtypes():
    """An unknown subtype (-1) must be dropped from the role term, not NaN it."""
    from directed_role import DirectedTypeRoleHead

    draws, _ = tiny_draws("directed", per_family=8)
    vocab = drq.vocabulary()
    targets = drq.targets_for(draws, vocab)
    first = next(iter(targets))
    targets[first]["subtype"][0] = -1                     # force the invalid row
    head = DirectedTypeRoleHead(dim=32, n_subtypes=vocab["n_subtypes"],
                                n_orbits=vocab["n_orbits"])
    features = {family: torch.randn(len(row[0]), row[0].shape[1], 32)
                for family, row in draws.items()}
    loss, _diagnostics = drq.joint_loss(head, features, draws, vocab, targets)
    assert torch.isfinite(loss)


def test_heldout_scores_runs_in_the_arms_own_label_space():
    """The post-training evaluation must not crash on the undirected vocabulary."""
    from directed_role import DirectedTypeRoleHead

    # heldout_scores takes the numpy pool (the trainer passes load_pool output),
    # not the tensor draws used during training.
    vocab = load_vocabulary("undirected" if drq.LABEL_MODE == "undirected" else "directed")
    families = load_pool("train", vocab, max_instances=8)
    head = DirectedTypeRoleHead(dim=32, n_subtypes=vocab["n_subtypes"],
                                n_orbits=vocab["n_orbits"])

    def encoder(x):
        return torch.zeros(len(x), 32)

    embeddings = torch.zeros(30000, 8)   # room for every real item id
    scores = drq.heldout_scores(head, encoder, embeddings, "cpu", families, vocab)
    assert set(scores) == {"subtype", "role"}
    assert scores["role"]["macro_over_families"] == scores["role"]["macro_over_families"]
