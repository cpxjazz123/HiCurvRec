#!/usr/bin/env python3
"""Ordered alignment hypotheses: does a chain / fork / merge organise its codes differently?

The question is no longer "do same-orbit nodes share a prefix".  Each local structure carries
its own expected *ordering* of the pairwise prefix-sharing matrix, and the four structures
must be told apart by the code organisation, not just by which nodes are nearby:

* chain  A->B->C   : S(A,B) > S(A,C)          adjacent pair closer than the 2-hop pair;
* fork   A->{B,C}  : S(B,C) > S(A,B)          siblings share more than the parent does;
* merge  {B,C}->A  : S(B,C) > S(B,A)          the same statement, computed on in-neighbours.

``S(u, v)`` is the number of leading codes two items agree on (0..3), so it is graded rather
than binary, and the same statistic is used for every structure.

The null is a **permutation of the item-to-SID assignment**: it keeps the code frequencies and
the SID's own collision rate, and destroys only which item holds which code.  A structure that
organises the codes must beat that null; a structure whose "sharing" is just frequent codes
will not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
GRAPH = REPO / "results" / "research" / "behavior_graph" / "graph.npz"
SIDS = Path(__file__).resolve().parent / "sids"
BASELINE = REPO / "results" / "stage2_RQ-VAE" / "TIGER_drq_D1" / "item_sids.json"
ROLE_ARMS = ("E", "T", "R", "Q", "QR")
OUT = Path(__file__).resolve().parent / "sid_alignment.json"
CAP = 120000
PERMUTATIONS = 200
SEED = 42


def load_sid(path: Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    return np.array([payload[str(index)][:3] for index in range(len(payload))], dtype=np.int64)


def shared_prefix_length(codes: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    left, right = codes[pairs[:, 0]], codes[pairs[:, 1]]
    equal = left == right
    return np.cumprod(equal, axis=1).sum(axis=1)


def adjacency(rows: np.ndarray, cols: np.ndarray):
    out, inn = {}, {}
    for a, b in zip(rows.tolist(), cols.tolist()):
        out.setdefault(a, set()).add(b)
        inn.setdefault(b, set()).add(a)
    return out, inn


def chain_instances(out: dict, edges: np.ndarray, rng, cap: int) -> np.ndarray:
    """A -> B -> C with A and C not adjacent: pair1 = (A,B), pair2 = (A,C)."""
    instances = []
    order = rng.permutation(len(edges))
    for index in order:
        a, b = edges[index]
        for c in out.get(int(b), ()):
            if c == a:
                continue
            if c in out.get(int(a), set()) or int(a) in out.get(c, set()):
                continue
            instances.append((a, b, a, c))
            if len(instances) >= cap:
                return np.array(instances, dtype=np.int64)
    return np.array(instances, dtype=np.int64)


def fan_instances(adjacency_map: dict, rng, cap: int, mode: str) -> np.ndarray:
    """Pairs of neighbours of one node that are not adjacent to each other.

    ``mode`` decides which pair is the "should share more" one:
    fork compares the siblings against the parent, merge compares the siblings against
    the shared child.
    """
    instances = []
    centres = np.array([k for k, v in adjacency_map.items() if len(v) >= 2], dtype=np.int64)
    order = rng.permutation(len(centres))
    for index in order:
        centre = int(centres[index])
        neighbours = sorted(adjacency_map[centre])[:20]  # a hub's pair space is quadratic
        for first in range(len(neighbours)):
            for second in range(first + 1, len(neighbours)):
                b, c = neighbours[first], neighbours[second]
                if c in adjacency_map.get(b, set()) or b in adjacency_map.get(c, set()):
                    continue
                instances.append((b, c, centre, b) if mode == "fork"
                                 else (b, c, b, centre))
                if len(instances) >= cap:
                    return np.array(instances, dtype=np.int64)
    return np.array(instances, dtype=np.int64)


def ordered_statistic(codes: np.ndarray, instances: np.ndarray):
    """mean S(first pair) - mean S(second pair), plus the share of instances ordered."""
    if len(instances) < 100:
        return None
    first = shared_prefix_length(codes, instances[:, :2])
    second = shared_prefix_length(codes, instances[:, 2:])
    delta = float(first.mean() - second.mean())
    ordered = float(((first > second).sum() + 0.5 * (first == second).sum()) / len(first))
    return {"delta": delta, "ordered_share": ordered,
            "first_mean": float(first.mean()), "second_mean": float(second.mean()),
            "instances": int(len(first))}


def permutation_null(codes: np.ndarray, instances: np.ndarray, rng, rounds: int):
    deltas = []
    for _ in range(rounds):
        shuffled = codes[rng.permutation(len(codes))]
        first = shared_prefix_length(shuffled, instances[:, :2])
        second = shared_prefix_length(shuffled, instances[:, 2:])
        deltas.append(float(first.mean() - second.mean()))
    deltas = np.asarray(deltas)
    return {"mean": float(deltas.mean()),
            "percentile_of_observed": None,
            "low": float(np.percentile(deltas, 2.5)),
            "high": float(np.percentile(deltas, 97.5))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    graph = np.load(GRAPH)
    rows, cols = graph["rows"].astype(np.int64), graph["cols"].astype(np.int64)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "augment_gnn"))
    import augment_gnn as base
    (msg_rows, msg_cols), _, _, _ = base.split_edges(rows, cols)
    edges = np.stack([msg_rows, msg_cols], axis=1)
    out_map, in_map = adjacency(msg_rows, msg_cols)
    structures = {
        "chain": (chain_instances(out_map, edges, rng, CAP), "S(A,B) > S(A,C)"),
        "fork": (fan_instances(out_map, rng, CAP, "fork"), "S(B,C) > S(A,B)"),
        "merge": (fan_instances(in_map, rng, CAP, "merge"), "S(B,C) > S(B,A)"),
    }
    arms = {"baseline": BASELINE}
    arms.update({name: SIDS / f"{name}.json" for name in
                 ("A_none", "B_undirected", "C_random", "D_directed")})
    for role_arm in ROLE_ARMS:
        candidate = (REPO / "results" / "stage2_RQ-VAE" / f"TIGER_graphlet_role_{role_arm}"
                     / "item_sids.json")
        if candidate.is_file():
            arms[f"role_{role_arm}"] = candidate
    report = {"protocol": {
        "S": "number of leading codes two items agree on (0..3)",
        "instances": {name: int(len(value[0])) for name, value in structures.items()},
        "hypotheses": {name: value[1] for name, value in structures.items()},
        "null": f"item-to-SID permutation, {PERMUTATIONS} rounds: keeps code frequencies and "
                f"the collision rate, destroys which item holds which code",
        "graph": "directed train edges of results/research/behavior_graph/graph.npz"},
        "arms": {}}
    for name, path in arms.items():
        if not Path(path).is_file():
            report["arms"][name] = {"missing": str(path)}
            continue
        codes = load_sid(Path(path))
        entry = {"sid": str(Path(path).relative_to(REPO))}
        for structure, (instances, hypothesis) in structures.items():
            observed = ordered_statistic(codes, instances)
            if observed is None:
                entry[structure] = {"hypothesis": hypothesis, "insufficient": True}
                continue
            null = permutation_null(codes, instances, rng, PERMUTATIONS)
            entry[structure] = {"hypothesis": hypothesis, **observed, "null": null,
                                "beats_null": observed["delta"] > null["high"]}
        report["arms"][name] = entry
        cells = []
        for structure in structures:
            value = entry[structure]
            if value.get("insufficient"):
                cells.append(f"{structure} n/a")
            else:
                cells.append(f"{structure} d={value['delta']:+.3f} "
                             f"({value['ordered_share']:.2f} vs null "
                             f"{value['null']['low']:+.3f}..{value['null']['high']:+.3f})")
        print(f"{name}: " + " | ".join(cells), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
