#!/usr/bin/env python3
"""The report issue #127 asks for: one arm's Stage 2 evidence in one file.

Reads the training log of one HGS arm and its exported SID, and writes a JSON
report with the numbers the issue lists:

* the final-epoch reconstruction, the raw and scaled value of every auxiliary
  term, and the applied gradient scale of each;
* the per-epoch code usage trajectory, so a late collapse is visible;
* the sampling coverage per hierarchy level;
* the radius per prefix level, the boundary saturation and the within-/between-
  branch angular gap;
* the SID structure (L1 codes, L2 prefixes, L3 triples, collision, fourth-token
  vocabulary, cell sizes) against the frozen baseline.

    python3 tools/research/science/hgs_arm_report.py --arm TIGER_HGS-TREE
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

#: The frozen baseline the issue compares every arm against.
BASELINE = {"l1_codes": 247, "l2_prefixes": 7414, "l3_triples": 20223,
            "collision": 1 - 20223 / 24587}

LINE_TERMS = re.compile(r"^\[hgs\] epoch=(?P<epoch>\d+) tree=(?P<rest>.*)$")
LINE_RADII = re.compile(r"^\[hgs\] epoch=(?P<epoch>\d+) radii=(?P<rest>.*)$")
LINE_TRAIN = re.compile(r"epoch=(?P<epoch>\d+) loss=(?P<loss>[0-9.e+-]+) "
                        r"recon=(?P<recon>[0-9.e+-]+) lr=(?P<lr>[0-9.e+-]+) "
                        r"(?:raw_unique=(?P<raw>[0-9]+)/(?P<total>\d+) )?"
                        r"collision=(?P<collision>[0-9.]+)")


def _parse_dict(text: str) -> dict:
    text = text.strip()
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {}


def _parse_list(text: str) -> list:
    text = text.strip()
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []


def read_log(path: Path) -> dict:
    """The current run's lines only: a restarted run appends to the same file."""
    lines = path.read_text(errors="ignore").splitlines()
    starts = [index for index, line in enumerate(lines)
              if re.match(r"^\[RQ-VAE\] epoch=1 ", line)]
    if starts:
        lines = lines[starts[-1]:]
    terms: list[dict] = []
    radii: list[dict] = []
    train: list[dict] = []
    for line in lines:
        match = LINE_TERMS.match(line)
        if match:
            rest = match.group("rest")
            parts = dict(part.split("=", 1) for part in rest.split(" ") if "=" in part)
            terms.append({"epoch": int(match.group("epoch")),
                          "tree": float(parts.get("tree", "nan")),
                          "radius": float(parts.get("radius", "nan")),
                          "direction": float(parts.get("direction", "nan")),
                          "prefix": float(parts.get("prefix", "nan")),
                          "scale": _parse_dict(parts.get("scale", "{}")),
                          "coverage": _parse_dict(parts.get("coverage", "{}"))})
            continue
        match = LINE_RADII.match(line)
        if match:
            rest = match.group("rest")
            parts = dict(part.split("=", 1) for part in rest.split(" ") if "=" in part)
            radii.append({"epoch": int(match.group("epoch")),
                          "radii": _parse_list(parts.get("radii", "[]")),
                          "saturated": _parse_list(parts.get("saturated", "[]")),
                          "angular_gap": _parse_dict(parts.get("angular_gap", "{}"))})
            continue
        match = LINE_TRAIN.search(line)
        if match:
            train.append({"epoch": int(match.group("epoch")),
                          "loss": float(match.group("loss")),
                          "recon": float(match.group("recon")),
                          "lr": float(match.group("lr")),
                          "raw_unique": int(match.group("raw")) if match.group("raw") else None,
                          "collision": float(match.group("collision"))})
    return {"terms": terms, "radii": radii, "train": train}


def read_sid(path: Path) -> dict:
    import collections

    payload = json.loads(path.read_text())
    sequences = [tuple(value) for value in payload.values()]
    total = len(sequences)
    out: dict = {"items": total, "width": len(sequences[0])}
    for name, width in (("l1_codes", 1), ("l2_prefixes", 2), ("l3_triples", 3)):
        counter = collections.Counter(sequence[:width] for sequence in sequences)
        sizes = sorted(counter.values())
        out[name] = len(counter)
        out[name + "_max_cell"] = sizes[-1]
        out[name + "_p99_cell"] = sizes[int(0.99 * len(sizes)) - 1]
    out["collision"] = 1 - out["l3_triples"] / total
    fourth = collections.Counter(sequence[3] for sequence in sequences)
    out["fourth_token_vocabulary"] = len(fourth)
    out["fourth_token_constant_share"] = max(fourth.values()) / total
    shared = sum(1 for sequence in sequences
                 if collections.Counter(sequence[:3] for sequence in sequences)[sequence[:3]] > 1)
    out["items_in_shared_l3"] = shared / total
    residual = 0.0
    counter = collections.Counter(sequence[:3] for sequence in sequences)
    for size in counter.values():
        residual += size * math.log2(size)
    out["residual_identity_bits_per_item"] = residual / total
    return out


def build(arm: str) -> dict:
    tree_log = REPO / "stage2_RQ-VAE" / arm / "logs" / "train_migrated.log"
    result_log = REPO / "results" / "stage2_RQ-VAE" / arm / "logs" / "train_migrated.log"
    log = tree_log if tree_log.exists() else result_log
    sid_path = REPO / "results" / "stage2_RQ-VAE" / arm / "item_sids.json"
    parsed = read_log(log)
    report = {
        "schema": 1,
        "kind": "hgs_arm_report",
        "arm": arm,
        "log": str(log.relative_to(REPO)),
        "baseline": BASELINE,
        "final_epoch": parsed["train"][-1] if parsed["train"] else None,
        "trajectory": [{"epoch": row["epoch"], "raw_unique": row["raw_unique"],
                        "collision": row["collision"], "loss": row["loss"],
                        "recon": row["recon"]}
                       for row in parsed["train"]],
        "terms": parsed["terms"],
        "radii": parsed["radii"],
    }
    if sid_path.exists():
        report["sid"] = read_sid(sid_path)
        report["sid"]["sha256"] = __import__("hashlib").sha256(sid_path.read_bytes()).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    report = build(args.arm)
    out = REPO / "results" / "stage2_RQ-VAE" / args.arm / "hgs_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(out.relative_to(REPO)),
                      "final_epoch": report["final_epoch"],
                      "sid": report.get("sid")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
