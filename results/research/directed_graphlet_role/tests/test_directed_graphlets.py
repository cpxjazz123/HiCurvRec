"""Issue #221 CGV 2: directed subtypes and directed orbits, not undirected guesses."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from directed_graphlets import canonical, classify, skeleton_of, stable_split  # noqa: E402


def wedge(converging: bool) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=bool)
    if converging:
        matrix[0, 1] = matrix[2, 1] = True      # A -> B <- C
    else:
        matrix[1, 0] = matrix[1, 2] = True      # A <- B -> C
    return matrix


def test_converging_and_diverging_wedges_are_different_subtypes():
    converging = classify(wedge(True))
    diverging = classify(wedge(False))
    assert converging[0] == diverging[0] == "P3"
    assert converging[1] != diverging[1], "the two wedge directions collapsed into one subtype"


def test_orbits_follow_the_directed_roles():
    converging = classify(wedge(True))[2]
    assert converging[0] == converging[2] != converging[1], "the two sources must share an orbit"
    path = np.zeros((4, 4), dtype=bool)
    path[0, 1] = path[1, 2] = path[2, 3] = True
    assert len(set(classify(path)[2])) == 4, "a directed path has four distinct roles"
    star = np.zeros((4, 4), dtype=bool)
    star[0, 1] = star[0, 2] = star[0, 3] = True
    orbits = classify(star)[2]
    assert orbits[0] != orbits[1] and orbits[1] == orbits[2] == orbits[3]


def test_mutual_edges_are_not_plain_edges():
    mutual = np.zeros((2, 2), dtype=bool)   # too small to classify, but the key must differ
    one_way = np.zeros((2, 2), dtype=bool)
    one_way[0, 1] = True
    mutual[0, 1] = mutual[1, 0] = True
    assert canonical(one_way)[0] != canonical(mutual)[0]


def test_classification_is_permutation_invariant():
    matrix = wedge(True)
    base = classify(matrix)
    for order in ([1, 0, 2], [2, 1, 0], [0, 2, 1]):
        permuted = matrix[np.ix_(order, order)]
        result = classify(permuted)
        assert result[0] == base[0] and result[1] == base[1]
        # the orbit *partition* travels with the nodes (the ids are renumbered)
        moved = [base[2][order.index(node)] for node in range(3)]
        for left in range(3):
            for right in range(3):
                assert (result[2][left] == result[2][right]) == (moved[left] == moved[right])


def test_skeleton_of_reads_the_undirected_union():
    assert skeleton_of(wedge(True)) == "P3"
    triangle = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(triangle, False)
    assert skeleton_of(triangle) == "K3"
    disconnected = np.zeros((3, 3), dtype=bool)
    disconnected[0, 1] = True
    assert skeleton_of(disconnected) is None


def test_split_depends_on_the_node_set_only():
    assert stable_split([5, 1, 3]) == stable_split([3, 5, 1])
    assert stable_split([5, 1, 3]) == stable_split([1, 3, 5])
    values = {stable_split(list(range(start, start + 3))) for start in range(0, 60, 3)}
    assert values == {0, 1}


def test_the_pool_file_carries_directed_labels():
    payload = np.load(HERE / "instances.npz", allow_pickle=False)
    for name in ("K3", "P3", "C4", "P4", "K1_3"):
        nodes = payload[f"{name}_nodes"]
        roles = payload[f"{name}_roles"]
        subtypes = payload[f"{name}_subtype"]
        split = payload[f"{name}_split"]
        assert len(nodes) == len(roles) == len(subtypes) == len(split)
        assert nodes.shape[1] == roles.shape[1]
        assert set(np.unique(split).tolist()) <= {0, 1}
        # role ids are orbit classes inside the subtype, never random
        assert (roles >= 0).all()
        assert (subtypes >= 0).all()


def test_orbit_ids_do_not_depend_on_the_input_node_order():
    """The same structural role gets the same id whatever order the caller used.

    This is the defect the review found: labeling by input position gave the
    source of a directed path a different id depending on the node order, so the
    head was trained on inconsistent supervision.
    """
    path = np.zeros((4, 4), dtype=bool)
    path[0, 1] = path[1, 2] = path[2, 3] = True
    base = classify(path)
    for perm in ([3, 2, 1, 0], [1, 0, 3, 2], [2, 0, 3, 1], [0, 2, 3, 1]):
        permuted = path[np.ix_(perm, perm)]
        got = classify(permuted)
        assert got[0] == base[0]
        mapped = [0] * 4
        for position, original in enumerate(perm):
            mapped[original] = got[2][position]
        assert tuple(mapped) == base[2], (perm, mapped, base[2])


def test_source_and_sink_roles_are_stable_under_reversal():
    path = np.zeros((4, 4), dtype=bool)
    path[0, 1] = path[1, 2] = path[2, 3] = True
    forward = classify(path)[2]
    backward = classify(path[::-1, ::-1])[2]
    assert forward[0] == backward[3], "the source role must stay the source role"
    assert forward[3] == backward[0], "the sink role must stay the sink role"


def test_the_pool_never_uses_the_other_id_for_a_kept_subtype():
    import json
    coverage = json.loads((HERE / "coverage.json").read_text())
    payload = np.load(HERE / "instances.npz", allow_pickle=False)
    for family in ("K3", "P3", "C4", "P4", "K1_3"):
        row = coverage["families"][family]
        kept = row["subtypes_kept"]
        other = row["other_id"]
        assert other == kept, (family, kept, other)
        subtypes = payload[f"{family}_subtype"]
        assert subtypes.min() >= 0
        # the `other` id only ever holds instances whose subtype was merged
        merged = int((subtypes == other).sum())
        assert merged == row["other_instances"], (family, merged, row["other_instances"])
