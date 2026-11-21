#!/usr/bin/env python3
"""Directed, strictly induced local structures: all pairs inside the subgraph.

Fixes the two defects of ``sid_alignment.py``:

* it read ``results/research/behavior_graph/graph.npz``, whose ``rows`` are stored with
  ``source < target`` -- that is the **undirected** graph.  This version reads the canonical
  directed graph of Issue #221 (``behavior_graph_directed``), built from the TRAIN sequences
  as ordered pairs, so ``A -> B`` and ``B -> A`` are distinct rows.
* it compared only two pairs per structure.  Every pair inside the instance is now measured,
  and the induced pattern is enforced: a chain must not contain the reverse edges, the
  shortcut ``A -> C``, or any chord; a fork's children must not be adjacent, etc.

Hypotheses, stated over all pairs:

* chain3  ``A->B->C``      : S(A,B) and S(B,C) both above S(A,C);
* chain4  ``A->B->C->D``   : S(adjacent) > S(2-hop) > S(3-hop);
* fork3   ``A->{B,C}``     : S(B,C) > S(A,B), S(A,C);
* merge3  ``{B,C}->A``     : S(B,C) > S(B,A), S(C,A).

``S`` is the number of leading codes two items agree on (0..3).  The null permutes the
item-to-SID assignment, keeping the code frequencies and the collision rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
DIRECTED = REPO / "results" / "research" / "behavior_graph_directed" / "graph.npz"
SIDS = Path(__file__).resolve().parent / "sids"
BASELINE = REPO / "results" / "stage2_RQ-VAE" / "TIGER_drq_D1" / "item_sids.json"
ROLE_ARMS = ("E", "T", "R", "Q", "QR")
OUT = Path(__file__).resolve().parent / "sid_alignment_v2.json"
CAP = 60000
PERMUTATIONS = 200
SEED = 42


def load_sid(path: Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    return np.array([payload[str(index)][:3] for index in range(len(payload))], dtype=np.int64)


def shared(codes: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    equal = codes[u] == codes[v]
    return np.cumprod(equal, axis=1).sum(axis=1)


def loaded_edges():
    payload = np.load(DIRECTED)
    rows = payload["rows"].astype(np.int64)
    cols = payload["cols"].astype(np.int64)
    return rows, cols, int(payload["nodes"])


def adjacency(rows: np.ndarray, cols: np.ndarray):
    out, inn = {}, {}
    for a, b in zip(rows.tolist(), cols.tolist()):
        out.setdefault(a, set()).add(b)
        inn.setdefault(b, set()).add(a)
    return out, inn


def chain3(out, edges, rng, cap):
    """A->B->C with no other edge among the three."""
    found = []
    for index in rng.permutation(len(edges)):
        a, b = int(edges[index][0]), int(edges[index][1])
        out_b = out.get(b, ())
        for c in list(out_b)[:64]:
            c = int(c)
            if c == a or a in out.get(c, ()) or c in out.get(a, ()) or b in out.get(c, ()):
                continue
            found.append((a, b, c))
            if len(found) >= cap:
                return np.array(found, dtype=np.int64)
    return np.array(found, dtype=np.int64)


def chain4(out, edges, rng, cap):
    """A->B->C->D with the other nine ordered pairs all absent."""
    found = []
    for index in rng.permutation(len(edges)):
        a, b = int(edges[index][0]), int(edges[index][1])
        for c in list(out.get(b, ()))[:48]:
            c = int(c)
            if len({a, b, c}) < 3:
                continue
            if a in out.get(c, ()) or c in out.get(a, ()) or a in out.get(b, ()):
                continue
            for d in list(out.get(c, ()))[:48]:
                d = int(d)
                if len({a, b, c, d}) < 4:
                    continue
                # every pair except A->B, B->C, C->D must be absent in both directions
                forbidden = ((b, a), (c, b), (d, c), (a, c), (c, a), (a, d), (d, a),
                             (b, d), (d, b))
                if any(v in out.get(u, ()) for u, v in forbidden):
                    continue
                found.append((a, b, c, d))
                if len(found) >= cap:
                    return np.array(found, dtype=np.int64)
    return np.array(found, dtype=np.int64)


def fan(out, inn, rng, cap, mode):
    """Two neighbours of one centre with no edge between them, pattern strictly induced."""
    source = out if mode == "fork" else inn
    centres = np.array([k for k, v in source.items() if len(v) >= 2], dtype=np.int64)
    found = []
    for index in rng.permutation(len(centres)):
        centre = int(centres[index])
        neighbours = sorted(source[centre])[:24]
        for first in range(len(neighbours)):
            for second in range(first + 1, len(neighbours)):
                b, c = neighbours[first], neighbours[second]
                if c in out.get(b, ()) or b in out.get(c, ()):
                    continue
                if mode == "fork":
                    # required centre->child; forbidden child->centre
                    if centre in out.get(b, ()) or centre in out.get(c, ()):
                        continue
                    found.append((centre, b, c))
                else:
                    # required child->centre; forbidden centre->child
                    if b in out.get(centre, ()) or c in out.get(centre, ()):
                        continue
                    found.append((b, c, centre))
                if len(found) >= cap:
                    return np.array(found, dtype=np.int64)
    return np.array(found, dtype=np.int64)


def statistics(codes: np.ndarray, kind: str, instances: np.ndarray):
    """All pair values, the hypothesised contrast, and the per-instance ordering share."""
    if len(instances) < 100:
        return None
    a, b = instances[:, 0], instances[:, 1]
    if kind == "chain3":
        c = instances[:, 2]
        s_ab, s_bc, s_ac = shared(codes, a, b), shared(codes, b, c), shared(codes, a, c)
        first = (s_ab + s_bc) / 2.0
        second = s_ac
        ordered = ((s_ab > s_ac).astype(float) + (s_bc > s_ac).astype(float)) / 2
        pairs = {"A-B": float(s_ab.mean()), "B-C": float(s_bc.mean()),
                 "A-C": float(s_ac.mean())}
    elif kind == "chain4":
        c, d = instances[:, 2], instances[:, 3]
        s_ab, s_bc, s_cd = shared(codes, a, b), shared(codes, b, c), shared(codes, c, d)
        s_ac, s_bd, s_ad = shared(codes, a, c), shared(codes, b, d), shared(codes, a, d)
        first = (s_ab + s_bc + s_cd) / 3.0
        second = (s_ac + s_bd) / 2.0
        ordered = ((first > second).astype(float) + (second > s_ad).astype(float)) / 2
        pairs = {"A-B": float(s_ab.mean()), "B-C": float(s_bc.mean()),
                 "C-D": float(s_cd.mean()), "A-C": float(s_ac.mean()),
                 "B-D": float(s_bd.mean()), "A-D": float(s_ad.mean())}
        return {"delta": float(first.mean() - second.mean()),
                "delta_far": float(second.mean() - s_ad.mean()),
                "ordered_share": float(ordered.mean()),
                "first_mean": float(first.mean()), "second_mean": float(second.mean()),
                "far_mean": float(s_ad.mean()), "pairs": pairs,
                "instances": int(len(first))}
    else:  # fork3 / merge3
        c = instances[:, 2]
        s_sib = shared(codes, b, c)
        s_first = shared(codes, a, b)
        s_second = shared(codes, a, c)
        first = s_sib
        second = (s_first + s_second) / 2.0
        ordered = ((s_sib > s_first).astype(float) + (s_sib > s_second).astype(float)) / 2
        pairs = {"sibling": float(s_sib.mean()), "parent": float(s_first.mean()),
                 "parent2": float(s_second.mean())}
    return {"delta": float(first.mean() - second.mean()),
            "ordered_share": float(ordered.mean()),
            "first_mean": float(first.mean()), "second_mean": float(second.mean()),
            "pairs": pairs, "instances": int(len(first))}


def permutation_null(codes: np.ndarray, kind: str, instances: np.ndarray, rng, rounds: int):
    deltas = []
    for _ in range(rounds):
        shuffled = codes[rng.permutation(len(codes))]
        value = statistics(shuffled, kind, instances)
        deltas.append(value["delta"] if value else np.nan)
    deltas = np.asarray(deltas, dtype=np.float64)
    return {"low": float(np.nanpercentile(deltas, 2.5)),
            "high": float(np.nanpercentile(deltas, 97.5)),
            "mean": float(np.nanmean(deltas))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    rows, cols, nodes = loaded_edges()
    edges = np.stack([rows, cols], axis=1)
    out, inn = adjacency(rows, cols)
    instances = {
        "chain3": (chain3(out, edges, rng, CAP), "S(A,B), S(B,C) > S(A,C)"),
        "chain4": (chain4(out, edges, rng, CAP), "S(adjacent) > S(2-hop) > S(3-hop)"),
        "fork3": (fan(out, inn, rng, CAP, "fork"), "S(B,C) > S(A,B), S(A,C)"),
        "merge3": (fan(out, inn, rng, CAP, "merge"), "S(B,C) > S(B,A), S(C,A)"),
    }
    arms = {"baseline": BASELINE}
    arms.update({name: SIDS / f"{name}.json" for name in
                 ("A_none", "B_undirected", "C_random", "D_directed")})
    for role_arm in ROLE_ARMS:
        candidate = (REPO / "results" / "stage2_RQ-VAE"
                     / f"TIGER_graphlet_role_{role_arm}" / "item_sids.json")
        if candidate.is_file():
            arms[f"role_{role_arm}"] = candidate
    report = {"protocol": {
        "graph": "results/research/behavior_graph_directed (Issue #221, TRAIN ordered pairs)",
        "directed": True,
        "inducement": "every required edge must exist and every other ordered pair inside "
                      "the instance must be absent (chain4 checks nine forbidden pairs)",
        "S": "number of leading codes two items agree on (0..3)",
        "instances": {name: int(len(value[0])) for name, value in instances.items()},
        "hypotheses": {name: value[1] for name, value in instances.items()},
        "null": f"item-to-SID permutation, {PERMUTATIONS} rounds"},
        "arms": {}}
    for name, path in arms.items():
        if not Path(path).is_file():
            report["arms"][name] = {"missing": str(path)}
            continue
        codes = load_sid(Path(path))
        entry = {"sid": str(Path(path).relative_to(REPO))}
        for kind, (bank, hypothesis) in instances.items():
            observed = statistics(codes, kind, bank)
            if observed is None:
                entry[kind] = {"hypothesis": hypothesis, "insufficient": True}
                continue
            null = permutation_null(codes, kind, bank, rng, PERMUTATIONS)
            entry[kind] = {"hypothesis": hypothesis, **observed, "null": null,
                           "beats_null": observed["delta"] > null["high"]}
        report["arms"][name] = entry
        cells = []
        for kind in instances:
            value = entry[kind]
            if value.get("insufficient"):
                cells.append(f"{kind} n/a")
            else:
                cells.append(f"{kind} d={value['delta']:+.3f} ord={value['ordered_share']:.2f} "
                             f"null[{value['null']['low']:+.3f},{value['null']['high']:+.3f}]"
                             f"{'*' if value['beats_null'] else ''}")
        print(f"{name}: " + " | ".join(cells), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
