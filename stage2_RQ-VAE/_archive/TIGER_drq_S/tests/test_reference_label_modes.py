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


def test_shuffled_targets_are_global_and_in_range():
    """The shuffled arm predicts GLOBAL orbit ids; local 0..3 made the loss NaN."""
    vocab = load_vocabulary("directed")
    families = load_pool("train", vocab, max_instances=16)
    draws = {family: (torch.from_numpy(nodes), torch.from_numpy(orbits),
                      torch.from_numpy(subtypes))
             for family, (nodes, orbits, subtypes) in families.items()}
    targets = drq.targets_for(draws, vocab)
    for family, row in targets.items():
        assert row["subtype"].min() >= 0, family
        assert row["orbit"].max() < vocab["n_orbits"], family


def test_shuffle_keeps_skeleton_and_reciprocity():
    matrix = np.array([[False, True, True],
                       [False, False, True],
                       [True, False, False]])
    shuffled = drq._shuffle_directions(matrix, instance=(0, 1, 2))
    assert np.array_equal(matrix | matrix.T, shuffled | shuffled.T)
    assert np.array_equal(matrix & matrix.T, shuffled & shuffled.T)
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
