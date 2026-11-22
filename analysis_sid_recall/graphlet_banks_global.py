"""Issue #231 C32/C33: one global pair partition across all three structures.

#230 built the banks per structure, which left 261 item pairs shared across structures and 279
pairs that appeared in one structure's supervision and another's evaluation -- measured on the
#230 banks. #231 requires the merged set to be disjoint: "三种结构合并后的全局无序商品对交集
必须为零（含实际进入 loss 的负样本）".

The split is therefore assigned per **unordered pair over the union of all structures**, so a
pair can be supervision for chain only if it is evaluation for nothing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from analysis_sid_recall import graphlet_banks as per_kind

KINDS = ("chain", "fork", "merge")
PAIRS = ((0, 1), (1, 2), (0, 2))
SPLITS = ("supervision", "validation", "test")
SHARES = (0.6, 0.2, 0.2)


def unordered(a: int, b: int) -> tuple:
    return (a, b) if a <= b else (b, a)


def instance_pairs(row) -> list:
    return [unordered(int(row[i]), int(row[j])) for i, j in PAIRS]


def global_split(banks: dict, seed: int = 42) -> tuple:
    """Assign every pair in the union of the banks to exactly one split.

    An instance is placed only when **all three** of its pairs landed in the same split; an
    instance whose pairs straddle splits is dropped and counted. That drop is the price of
    global isolation and is reported, never hidden.
    """
    union = sorted({pair for bank in banks.values() for row in bank.tolist()
                    for pair in instance_pairs(row)})
    rng = np.random.default_rng(seed)
    owner = {pair: SPLITS[int(rng.choice(len(SPLITS), p=SHARES))] for pair in union}
    out, dropped = {}, {kind: [] for kind in banks}
    for kind, bank in banks.items():
        parts = {name: [] for name in SPLITS}
        for row in bank.tolist():
            split = {owner[pair] for pair in instance_pairs(row)}
            if len(split) == 1:
                parts[split.pop()].append(row)
            else:
                dropped[kind].append(row)
        out[kind] = {name: np.asarray(rows, dtype=np.int64).reshape(-1, 3)
                     for name, rows in parts.items()}
    return out, dropped


def claim_pairs(parts: dict) -> dict:
    """Drop, from each structure's supervision, any pair another structure owns elsewhere.

    C33 forbids a pair from being supervision in one structure and evaluation in another --
    and, because the merged set must be disjoint, from being supervision in two structures.
    Placement alone cannot guarantee that (one pair can sit inside two kept instances of
    different structures), so the claim is applied explicitly: a pair belongs to the first
    structure that owns it in a supervision split; later structures lose those rows.
    """
    claimed = {}
    out, removed = {}, {}
    for kind in KINDS:
        kept = {name: [] for name in SPLITS}
        removed[kind] = 0
        for name in SPLITS:
            for row in parts[kind][name].tolist():
                pairs = instance_pairs(row)
                taken = [pair for pair in pairs if pair in claimed]
                if name == "supervision":
                    if taken:
                        removed[kind] += 1
                        continue
                    for pair in pairs:
                        claimed[pair] = kind
                elif any(claimed.get(pair, kind) != kind for pair in pairs):
                    removed[kind] += 1
                    continue
                kept[name].append(row)
        out[kind] = {name: np.asarray(rows, dtype=np.int64).reshape(-1, 3)
                     for name, rows in kept.items()}
    return out, removed


def audit(parts: dict) -> dict:
    """C33: zero overlap between any supervision pair and any evaluation pair."""
    sets = {(kind, name): {pair for row in bank.tolist() for pair in instance_pairs(row)}
            for kind, split in parts.items() for name, bank in split.items()}
    eval_pairs = set().union(*[sets[(kind, name)] for kind in KINDS
                               for name in ("validation", "test")])
    report = {"cross_structure": {}, "supervision_vs_eval": {}, "within_split": {}}
    for left, right in (("chain", "fork"), ("chain", "merge"), ("fork", "merge")):
        report["cross_structure"][f"{left}∩{right}"] = len(
            sets[(left, "supervision")] & sets[(right, "supervision")])
    for kind in KINDS:
        report["supervision_vs_eval"][kind] = len(sets[(kind, "supervision")] & eval_pairs)
    for kind in KINDS:
        report["within_split"][kind] = {
            "validation∩test": len(sets[(kind, "validation")] & sets[(kind, "test")])}
    report["isolated"] = (all(v == 0 for v in report["cross_structure"].values())
                          and all(v == 0 for v in report["supervision_vs_eval"].values())
                          and all(v["validation∩test"] == 0
                                  for v in report["within_split"].values()))
    return report


def build(directory: Path, seed: int = 42, cap: int = 20000) -> dict:
    report = json.loads((Path(per_kind.__file__).parent.parent / "results" / "research"
                         / "issue231" / "banks" / "source.json").read_text()) if False else {}
    index = per_kind.contract.load_directed(directory)
    banks = {kind: per_kind.contract.sample(kind, index, cap,
                                            np.random.default_rng(seed), neighbours=64)
             for kind in KINDS}
    parts, dropped = global_split(banks, seed=seed)
    parts, removed = claim_pairs(parts)
    return {"kinds": banks, "parts": parts, "dropped": dropped, "removed": removed,
            "audit": audit(parts), "report": report}


def save(result: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for kind in KINDS:
        for name, bank in result["parts"][kind].items():
            path = out_dir / f"{kind}_{name}.npy"
            np.save(path, bank)
            hashes[f"{kind}_{name}"] = hashlib.sha256(
                np.ascontiguousarray(bank).tobytes()).hexdigest()
    manifest = {"seed": 42, "shares": SHARES, "hashes": hashes,
                "audit": result["audit"],
                "counts": {kind: {name: int(len(bank))
                                  for name, bank in split.items()}
                           for kind, split in result["parts"].items()},
                "dropped": {kind: len(rows) for kind, rows in result["dropped"].items()},
                "removed_by_claim": result["removed"]}
    (out_dir / "banks_global.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
