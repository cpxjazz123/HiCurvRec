#!/usr/bin/env python3
"""The frozen banks and the pair-disjoint audit the issue asks for (§1, C6–C11).

Three claims have to be true before any arm trains:

* supervision reads the training split only;
* the unordered pairs of a bank are split across supervision / validation / test with **no**
  pair appearing in two of them, conflicting instances are dropped and counted;
* the sampler fixes degrees and hashes, so a later reader can prove it saw the same banks.

This module builds the banks once, writes them with their hashes, and offers
:func:`assert_loss_isolated` which the trainer calls on every batch: the pairs that actually
enter the loss may not intersect the evaluation banks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis_sid_recall import graphlet_contract as contract

KINDS = ("chain", "fork", "merge")
#: Unordered pairs of a single instance, by kind. Chain and fork share the shape; merge is the
#: reverse orientation of fork but its pair set is the same three items, so the audit is
#: per-kind rather than shared: an item pair may be supervision for one structure and
#: evaluation for another, and the issue only forbids overlap inside one structure.
PAIRS = ((0, 1), (1, 2), (0, 2))


def unordered(a: int, b: int) -> tuple:
    return (a, b) if a <= b else (b, a)


def pair_set(bank: np.ndarray) -> set:
    out = set()
    for row in bank.tolist():
        for i, j in PAIRS:
            out.add(unordered(row[i], row[j]))
    return out


def _nbrs(bank: np.ndarray, groups: dict) -> dict:
    """Instance indices per pair, so the selector can see which instances compete."""
    out = {}
    for index, row in enumerate(bank.tolist()):
        for i, j in PAIRS:
            out.setdefault(unordered(row[i], row[j]), set()).add(index)
    return out


def split(bank: np.ndarray, seed: int = 42, train: float = 0.6, valid: float = 0.2):
    """Choose the split for each *item pair*, then place every instance from one of its pairs.

    Per-pair ownership alone keeps the evaluation sets tiny, because only a quarter of the
    twenty thousand sampled instances happens to have all three pairs in the same split. The
    pass below picks a pair that has not been placed yet, gives it a split by ``train`` /
    ``valid`` shares, and places into that split every instance of the bank for which the pair
    is still free. An instance is placed at most once; the first pair that claims it decides,
    and an instance whose pairs end up owned by two splits is a conflict: dropped and counted.

    The result is pair-disjoint across supervision, validation and test by construction.
    """
    rng = np.random.default_rng(seed)
    members = _nbrs(bank, None)
    shares = (train, valid, 1.0 - train - valid)
    names = ("supervision", "validation", "test")
    owner: dict = {}
    placed: list = [False] * len(bank)
    parts = {name: [] for name in names}
    dropped = []
    pairs = sorted(members)
    for pair in rng.permutation(len(pairs)):
        pair = pairs[pair]
        if pair in owner:
            continue
        name = names[int(rng.choice(len(names), p=shares))]
        owner[pair] = name
        for index in members[pair]:
            if placed[index]:
                continue
            row = bank[index]
            row_pairs = [unordered(row[i], row[j]) for i, j in PAIRS]
            if any(p in owner and owner[p] != name for p in row_pairs):
                continue
            placed[index] = True
            parts[name].append(row.tolist())
            for p in row_pairs:
                owner.setdefault(p, name)
    for index, row in enumerate(bank.tolist()):
        if placed[index]:
            continue
        owners = sorted({owner[unordered(row[i], row[j])] for i, j in PAIRS
                         if unordered(row[i], row[j]) in owner})
        dropped.append({"instance": row,
                        "reason": ("pairs split across " + ",".join(owners)) if len(owners) > 1
                        else "no free pair left when its turn came"})
    out = {name: np.asarray(rows, dtype=np.int64).reshape(-1, 3) for name, rows in parts.items()}
    return out, dropped


def audit_partition(parts: dict, kind: str) -> dict:
    sets = {name: pair_set(bank) for name, bank in parts.items()}
    overlap = {
        "supervision∩validation": len(sets["supervision"] & sets["validation"]),
        "supervision∩test": len(sets["supervision"] & sets["test"]),
        "validation∩test": len(sets["validation"] & sets["test"]),
    }
    return {"kind": kind,
            "instances": {name: int(len(bank)) for name, bank in parts.items()},
            "pairs": {name: len(values) for name, values in sets.items()},
            "overlap": overlap,
            "isolated": all(value == 0 for value in overlap.values())}


def degree_report(bank: np.ndarray, out_degree: np.ndarray, in_degree: np.ndarray) -> dict:
    """Degree profile of the sampled instances, so "top hubs only" cannot happen silently."""
    nodes = np.unique(bank)
    return {"nodes": int(len(nodes)),
            "out_degree_mean": float(out_degree[nodes].mean()),
            "out_degree_median": float(np.median(out_degree[nodes])),
            "out_degree_max": int(out_degree[nodes].max()),
            "in_degree_mean": float(in_degree[nodes].mean()),
            "in_degree_median": float(np.median(in_degree[nodes])),
            "graph_out_degree_mean": float(out_degree.mean()),
            "graph_in_degree_mean": float(in_degree.mean())}


def build(directory: Path, cap: int = 20000, seed: int = 42) -> dict:
    index = contract.load_directed(directory)
    payload = np.load(Path(directory) / "graph.npz")
    out_degree = payload["out_degree"]
    in_degree = payload["in_degree"]
    rng = np.random.default_rng(seed)
    report = {"source": str(directory), "cap": cap, "seed": seed, "kinds": {}}
    for kind in KINDS:
        bank = contract.sample(kind, index, cap, np.random.default_rng(seed), neighbours=64)
        parts, dropped = split(bank, seed=seed)
        report["kinds"][kind] = {
            "bank": int(len(bank)),
            "partition": audit_partition(parts, kind),
            "dropped": len(dropped),
            "dropped_sample": dropped[:3],
            "degree": degree_report(bank, out_degree, in_degree),
            "hash": hashlib.sha256(np.ascontiguousarray(bank).tobytes()).hexdigest(),
            "partitions": {name: hashlib.sha256(np.ascontiguousarray(part).tobytes()).hexdigest()
                           for name, part in parts.items()},
        }
    return report


def save(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = contract.load_directed(Path(report["source"]))
    seed = report["seed"]
    for kind in KINDS:
        bank = contract.sample(kind, index, report["cap"], np.random.default_rng(seed),
                               neighbours=64)
        parts, _ = split(bank, seed=seed)
        np.save(out_dir / f"bank_{kind}.npy", bank)
        np.save(out_dir / f"{kind}_supervision.npy", parts["supervision"])
        np.save(out_dir / f"{kind}_validation.npy", parts["validation"])
        np.save(out_dir / f"{kind}_test.npy", parts["test"])
    (out_dir / "banks.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                                        encoding="utf-8")


def assert_loss_isolated(kind: str, triples, evaluation: set[tuple], strict: bool = True) -> int:
    """Run inside the trainer: no pair that enters the loss may sit in an evaluation bank."""
    pairs = {unordered(int(row[i]), int(row[j])) for row in triples for i, j in PAIRS}
    collision = len(pairs & evaluation)
    if strict and collision:
        raise RuntimeError(f"{kind}: {collision} loss pairs also sit in an evaluation bank")
    return collision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=20000)
    args = parser.parse_args()
    report = build(args.graph, cap=args.cap)
    save(report, args.out)
    print(json.dumps({kind: report["kinds"][kind]["partition"] for kind in KINDS},
                     indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
