#!/usr/bin/env python3
"""The Issue #230 evaluation contract: strictly induced directed instances and the Δ table.

This module is the single source of truth the issue asks for (§1, §2). It exists because three
consumers disagreed before:

* the trainer sampled ``merge`` as ``(B, C, A)`` while the scorer read ``A`` as the centre;
* the trainer's chain pattern never looked at the reverse edge ``B->A``;
* the scorer compared "both adjacent pairs beat the endpoint" for a chain, which the prefix
  equivalence relation makes unsatisfiable in general.

Every instance is now defined by a role tuple whose position means the same thing in the sampler,
in the scorer, and in the tests:

    chain   (A, B, C)    A->B->C, edges A->B and B->C present, the other four directed
                         positions among {A,B,C} absent (B->A, C->B, A->C, C->A).
    fork    (A, B, C)    A is the centre, A->B and A->C present, the other four
                         positions absent; B and C are interchangeable.
    merge   (A, B, C)    A is the centre, B->A and C->A present, the other four
                         positions absent; B and C are interchangeable.

Direction is not recoverable from a three-code prefix, and this module never claims it: fork and
merge share one Δ formula and differ only in which edges must exist.

Δ uses S(X, Y) = length of the shared leading run of the first three codes, i.e. 0..3, and never
the disambiguation column:

    Δ_chain = (S(A,B) + S(B,C)) / 2 - S(A,C)
    Δ_fork  = S(B,C) - (S(A,B) + S(A,C)) / 2
    Δ_merge = S(B,C) - (S(A,B) + S(A,C)) / 2

Run it directly to print the audit of one directory::

    python3 -m analysis_sid_recall.graphlet_contract --audit results/research/behavior_graph_directed
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CODES = 3
#: The four directed positions that must be absent for a strictly induced triangle.
_ORDER = ("AB", "BC", "AC", "BA", "CB", "CA")
_REQUIRED = {"chain": ("AB", "BC"), "fork": ("AB", "AC"), "merge": ("BA", "CA")}
#: Role-slot indices for each directed position: AB = slot 0 -> 1, BA = 1 -> 0, ...
_ORDER_SRC = {"AB": 0, "BC": 1, "AC": 0, "BA": 1, "CB": 2, "CA": 2}
_ORDER_DST = {"AB": 1, "BC": 2, "AC": 2, "BA": 0, "CB": 1, "CA": 0}


def _position(u: int, v: int) -> str:
    return "AB" if (u, v) == (0, 1) else "BC" if (u, v) == (1, 2) else \
           "AC" if (u, v) == (0, 2) else "BA" if (u, v) == (1, 0) else \
           "CB" if (u, v) == (2, 1) else "CA"


def edges_of(instances: np.ndarray, kind: str) -> dict:
    """The six boolean positions of every instance, as a ``{position: array}`` map."""
    if kind not in _REQUIRED:
        raise ValueError("unknown kind: " + str(kind))
    out = {}
    for i in range(3):
        for j in range(3):
            if i != j:
                out[_position(i, j)] = (instances[:, i], instances[:, j])
    return out


def induced_mask(out_edges, instances: np.ndarray, kind: str) -> np.ndarray:
    """Which instances are strictly induced: required positions present, others absent.

    ``out_edges`` is any :class:`EdgeIndex`, or a plain set of ``(src, dst)`` pairs for the small
    hand-written cases in the tests.
    """
    if len(instances) == 0:
        return np.zeros(0, dtype=bool)
    if not isinstance(out_edges, EdgeIndex):
        out_edges = EdgeIndex.from_pairs(out_edges)
    triples = instances[:, :3]
    required = _REQUIRED[kind]
    mask = np.ones(len(triples), dtype=bool)
    for pos in _ORDER:
        present = out_edges.has(triples[:, _ORDER_SRC[pos]], triples[:, _ORDER_DST[pos]])
        mask &= present if pos in required else ~present
    # A role tuple must name three distinct items; a self-loop can never satisfy the pattern.
    mask &= (triples[:, 0] != triples[:, 1]) & (triples[:, 1] != triples[:, 2]) & \
            (triples[:, 0] != triples[:, 2])
    return mask


class EdgeIndex:
    """Sorted directed adjacency: vectorised membership plus per-node neighbour sets."""

    def __init__(self, rows: np.ndarray, cols: np.ndarray, nodes: int):
        self.nodes = max(int(nodes), 1)
        self.base = self.nodes
        order = np.lexsort((cols, rows))
        self.keys = rows.astype(np.int64)[order] * self.base + cols.astype(np.int64)[order]
        self.out = {}
        self.inn = {}
        for src, dst in zip(rows.tolist(), cols.tolist()):
            self.out.setdefault(int(src), set()).add(int(dst))
            self.inn.setdefault(int(dst), set()).add(int(src))

    @classmethod
    def from_pairs(cls, pairs) -> "EdgeIndex":
        pairs = list(pairs)
        if not pairs:
            return cls(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 1)
        rows = np.asarray([p[0] for p in pairs], dtype=np.int64)
        cols = np.asarray([p[1] for p in pairs], dtype=np.int64)
        return cls(rows, cols, int(max(rows.max(), cols.max())) + 1)

    def has(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        if len(self.keys) == 0:
            return np.zeros(len(src), dtype=bool)
        key = src.astype(np.int64) * self.base + dst.astype(np.int64)
        pos = np.clip(np.searchsorted(self.keys, key), 0, len(self.keys) - 1)
        return self.keys[pos] == key

    def centres(self, mode: str) -> list:
        source = self.out if mode == "fork" else self.inn
        return sorted(k for k, v in source.items() if len(v) >= 2)


def sample(kind: str, index: EdgeIndex, cap: int, rng: np.random.Generator,
           neighbours: int = 32) -> np.ndarray:
    """Uniformly walk the role slots and keep only strictly induced instances.

    The sampler and the scorer share one role definition, which is the defect C3 asked to fix:
    every returned row is ``(A, B, C)`` with ``A`` at position zero for all three kinds.
    """
    found = []
    centres = index.centres("fork" if kind == "fork" else "merge") if kind != "chain" else []
    if kind == "chain":
        tails = sorted(index.out)
        for a in (tails[i] for i in rng.permutation(len(tails))):
            for b in list(index.out.get(a, ()))[:neighbours]:
                for c in list(index.out.get(b, ()))[:neighbours]:
                    found.append((a, b, c))
                    if len(found) >= cap * 4:
                        break
                if len(found) >= cap * 4:
                    break
            if len(found) >= cap * 4:
                break
    else:
        source = index.out if kind == "fork" else index.inn
        for centre in (centres[i] for i in rng.permutation(len(centres))):
            pool = sorted(source[centre])[:neighbours]
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    found.append((centre, pool[i], pool[j]))
                    if len(found) >= cap * 4:
                        break
                if len(found) >= cap * 4:
                    break
            if len(found) >= cap * 4:
                break
    if not found:
        return np.zeros((0, 3), dtype=np.int64)
    triples = np.asarray(found, dtype=np.int64)
    order = rng.permutation(len(triples))
    triples = triples[order]
    mask = induced_mask(index, triples, kind)
    return triples[mask][:cap]


def shared_prefix(codes: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """S(X, Y): the number of leading codes two items agree on, 0..3, extra columns ignored."""
    codes = codes[:, :CODES]
    equal = codes[u] == codes[v]
    return np.cumprod(equal, axis=1).sum(axis=1).astype(np.float64)


def delta_of(kind: str, s_ab, s_bc, s_ac):
    """The issue's contrast. Chain uses the mean of the adjacent pairs, never both strictly."""
    if kind == "chain":
        return (s_ab + s_bc) / 2.0 - s_ac
    if kind in ("fork", "merge"):
        return s_bc - (s_ab + s_ac) / 2.0
    raise ValueError("unknown kind: " + str(kind))


def measure(codes: np.ndarray, kind: str, instances: np.ndarray) -> dict:
    """Δ, the strict-satisfy / tie / violate split, and the absolute near and far shares."""
    if len(instances) == 0:
        return {"kind": kind, "instances": 0}
    a, b, c = instances[:, 0], instances[:, 1], instances[:, 2]
    s_ab = shared_prefix(codes, a, b)
    s_bc = shared_prefix(codes, b, c)
    s_ac = shared_prefix(codes, a, c)
    delta = delta_of(kind, s_ab, s_bc, s_ac)
    near = (s_ab + s_bc) / 2.0 if kind == "chain" else s_bc
    far = s_ac if kind == "chain" else (s_ab + s_ac) / 2.0
    return {
        "kind": kind,
        "instances": int(len(delta)),
        "delta": float(delta.mean()),
        "satisfy_rate": float((delta > 0).mean()),
        "tie_rate": float((delta == 0).mean()),
        "violate_rate": float((delta < 0).mean()),
        "near_share": float(near.mean()),
        "far_share": float(far.mean()),
        "pairs": {"AB": float(s_ab.mean()), "BC": float(s_bc.mean()),
                  "AC": float(s_ac.mean())},
        "by_level": {str(level): float(delta_of(kind, s_ab, s_bc, s_ac).mean() / 1.0)
                     for level in (1,)},
    }


def level_measure(codes: np.ndarray, kind: str, instances: np.ndarray, levels=(1, 2, 3)) -> dict:
    """The same contrast restricted to the first `levels` codes, one entry per depth."""
    out = {}
    for depth in levels:
        sliced = codes[:, :depth]
        a, b, c = instances[:, 0], instances[:, 1], instances[:, 2]
        # S at this depth: the number of matching codes among the first `depth`.
        def s_at(u, v):
            return (sliced[u] == sliced[v]).all(axis=1).astype(np.float64) * depth

        s_ab, s_bc, s_ac = s_at(a, b), s_at(b, c), s_at(a, c)
        out[str(depth)] = float(delta_of(kind, s_ab, s_bc, s_ac).mean())
    return out


def load_directed(directory: Path) -> EdgeIndex:
    """The directed graph stored by ``results/research/behavior_graph_directed``."""
    payload = np.load(Path(directory) / "graph.npz")
    return EdgeIndex(payload["rows"].astype(np.int64), payload["cols"].astype(np.int64),
                     int(payload["nodes"]))


def audit(directory: Path, cap: int = 20000, seed: int = 42) -> dict:
    """Induction rates and a bank sample per kind: the §1 self-check on the real graph."""
    index = load_directed(directory)
    rng = np.random.default_rng(seed)
    report = {"directory": str(directory), "nodes": index.nodes, "edges": int(len(index.keys)),
              "cap": cap, "kinds": {}}
    for kind in ("chain", "fork", "merge"):
        bank = sample(kind, index, cap, rng)
        report["kinds"][kind] = {"banked": int(len(bank)),
                                 "share_of_cap": float(len(bank) / cap),
                                 "sample": bank[:3].tolist()}
    return report


def digest(paths) -> str:
    """A stable sha256 over a set of files, used for the bank hash the issue asks for."""
    h = hashlib.sha256()
    for path in sorted(str(p) for p in paths):
        h.update(path.encode())
        h.update(Path(path).read_bytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.audit)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
