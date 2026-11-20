"""Issue #221 phase 2: strict induced DIRECTED graphlets, canonical subtypes, directed orbits.

A node set S of size 3 or 4 induces the boolean adjacency A[S, S].  Its *skeleton*
is the undirected pattern that #217 used (K3, P3, C4, P4, K1_3); its *subtype* is
the canonical directed pattern, i.e. the lexicographically smallest bit string over
all node permutations, so A->B<-C and A<-B->C are different subtypes even though
both are wedges.  A node's *directed orbit* is its equivalence class under the
automorphism group of the directed pattern: every permutation that maps the
pattern onto its canonical form is an automorphism, and two nodes are in the same
orbit when some automorphism exchanges them.

Nothing here reads the undirected graph; the skeleton is only used to name the
family and to keep the first pool comparable with #217.
"""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

def _repository_root() -> Path:
    """The repository root by marker: the module is copied into every arm tree."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE").is_dir() and (parent / "results").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


REPO = _repository_root()
GRAPH_DIR = REPO / "results" / "research" / "behavior_graph_directed"
OUT = REPO / "results" / "research" / "directed_graphlet_role"

TYPES = ("K3", "P3", "C4", "P4", "K1_3")


def load_adjacency():
    """The directed adjacency as a boolean CSR matrix (weights collapsed)."""
    from scipy import sparse

    payload = np.load(GRAPH_DIR / "graph.npz", allow_pickle=False)
    n = int(payload["nodes"])
    matrix = sparse.csr_matrix((np.ones(len(payload["rows"]), dtype=bool),
                                (payload["rows"], payload["cols"])), shape=(n, n))
    matrix.data[:] = True
    return matrix


def skeleton_of(matrix: np.ndarray) -> str | None:
    """The undirected skeleton family of a small symmetric-union adjacency."""
    adjacency = matrix | matrix.T
    nodes = len(matrix)
    edges = np.argwhere(np.triu(adjacency, 1))
    degree = adjacency.sum(axis=1)
    if nodes == 3:
        if len(edges) == 3:
            return "K3"
        if len(edges) == 2 and (degree == 2).any():
            return "P3"
        return None
    if nodes == 4:
        if len(edges) == 4 and (degree == 2).all():
            return "C4"
        if len(edges) == 3 and sorted(degree.tolist()) == [1, 1, 2, 2]:
            return "P4"
        if len(edges) == 3 and sorted(degree.tolist()) == [1, 1, 1, 3]:
            return "K1_3"
        return None
    raise ValueError("only 3- and 4-node graphlets are enumerated")


@dataclass
class Pattern:
    """One canonical directed pattern and its automorphism orbits."""
    key: str
    size: int
    matrix: np.ndarray
    skeleton: str
    orbits: tuple[int, ...]
    support: int = 0
    split_support: dict = field(default_factory=lambda: {"train": 0, "heldout": 0})


_PERMS: dict[int, tuple] = {}


def _permutations(n: int):
    """All node permutations plus their index arrays and bit weights, cached."""
    if n not in _PERMS:
        order = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
        weights = (1 << np.arange(n * n - 1, -1, -1)).astype(object)
        _PERMS[n] = (order, weights)
    return _PERMS[n]


def canonical(matrix: np.ndarray) -> tuple[str, tuple[int, ...], np.ndarray]:
    """(canonical bit key, per-node orbit id, canonical matrix) for one pattern.

    The key is the lexicographically smallest row-major bit string over all node
    permutations, so it does not depend on the order the caller listed the nodes in.

    The orbit id is the node's **canonical position** class, not its position in the
    caller's ordering: two nodes share an id exactly when an automorphism of the
    pattern exchanges them, and the ids are the smallest canonical position in each
    class.  Labelling by input position (the earlier version) gave the same
    structural role different ids for different orderings, which is inconsistent
    supervision for the head.
    """
    n = matrix.shape[0]
    order, weights = _permutations(n)
    permuted = matrix[order[:, :, None], order[:, None, :]]          # (P, n, n)
    bits = permuted.reshape(len(order), -1)
    value = bits.astype(object) @ weights                             # (P,)
    best = int(np.argmin(value))
    best_matrix = permuted[best]
    best_order = order[best]
    best_key = "".join("1" if best_matrix[i, j] else "0" for i in range(n) for j in range(n))

    # automorphisms of the canonical form: permutations that leave it unchanged
    canonical_permuted = best_matrix[order[:, :, None], order[:, None, :]]
    automorphisms = order[np.all(canonical_permuted == best_matrix, axis=(1, 2))]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for mapping in automorphisms:
        for canonical_position, image in enumerate(mapping.tolist()):
            a, b = find(canonical_position), find(image)
            if a != b:
                parent[max(a, b)] = min(a, b)
    canonical_orbit = [find(position) for position in range(n)]
    # original node i sits at canonical position pos[i]
    pos = [0] * n
    for canonical_position, original in enumerate(best_order.tolist()):
        pos[original] = canonical_position
    orbit_ids = tuple(canonical_orbit[pos[node]] for node in range(n))
    return best_key, orbit_ids, best_matrix


def stable_split(nodes) -> int:
    """0 = train, 1 = heldout, from the sorted node set only (permutation free)."""
    key = ",".join(str(value) for value in sorted(int(v) for v in nodes))
    digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
    return int(digest[0]) & 1


def classify(matrix: np.ndarray) -> tuple[str, str, tuple[int, ...]] | None:
    """(skeleton, canonical key, orbit ids) or None when the skeleton is not ours."""
    skeleton = skeleton_of(matrix)
    if skeleton is None:
        return None
    key, orbits, _canonical_matrix = canonical(matrix)
    return skeleton, key, orbits
