#!/usr/bin/env python3
"""Structural alignment of a finished Stage2 arm, measured on its exported SID.

Reads ``item_sids.json`` plus the checkpoint's codebooks, rebuilds each item's quantised
vector as the sum of its three codewords, and reports -- against the pair-disjoint evaluation
banks -- the chain, fork and merge alignment, the three-code collision rate and a semantic
probe.  Nothing is trained here; this is the measurement the 3000-epoch run is judged by.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "augment_gnn"))
import sid_alignment_v2 as chains  # noqa: E402
import prototype_chain_long as proto  # noqa: E402

OUT = HERE / "arm_alignment.json"
BASELINE = Path(__file__).resolve().parents[4] / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"


def load_codes(directory: Path) -> np.ndarray:
    payload = json.loads((directory / "item_sids.json").read_text())
    return np.array([payload[str(index)][:3] for index in range(len(payload))], dtype=np.int64)


def quantised_vectors(directory: Path, codes: np.ndarray, width: int) -> np.ndarray:
    state = torch.load(directory / "rqvae_best.pth", map_location="cpu")["state_dict"]
    tables = []
    for level in range(3):
        key = f"rq.vq_layers.{level}.embed.weight"
        table = state[key].float()
        projection = state.get(f"rq.vq_layers.{level}.embed_proj.weight")
        if projection is not None:
            table = table @ projection.float().t()
        tables.append(table)
    vectors = np.zeros((len(codes), width), dtype=np.float32)
    for level, table in enumerate(tables):
        vectors += table[codes[:, level]].numpy()
    return vectors


def semantic_probe(vectors: np.ndarray, semantics: np.ndarray) -> float:
    rng = np.random.default_rng(42)
    order = rng.permutation(len(vectors))
    cut = int(0.8 * len(order))
    train, evaluate = order[:cut], order[cut:]
    x = vectors[train].astype(np.float64)
    xc = x - x.mean(0, keepdims=True)
    lam = 1e-3 * float(np.trace(xc.T @ xc)) / xc.shape[1]
    weights = np.linalg.solve(xc.T @ xc + lam * np.eye(xc.shape[1]),
                              xc.T @ (semantics[train] - semantics[train].mean(0)))
    predicted = (vectors[evaluate] - x.mean(0)) @ weights + semantics[train].mean(0)
    return float(1.0 - ((predicted - semantics[evaluate]) ** 2).mean()
                 / ((semantics[evaluate] - semantics[evaluate].mean(0)) ** 2).mean())


def structure_delta(codes: np.ndarray, bank: np.ndarray, kind: str) -> dict:
    """Same definition as the survey: cumulative three-code sharing, adjacent minus far."""
    a, b, c = bank[:, 0], bank[:, 1], bank[:, 2]
    cumulative = lambda left, right: np.cumprod(codes[left] == codes[right], axis=1).sum(1)
    if kind == "chain":
        near = (cumulative(a, b) + cumulative(b, c)) / 2.0
        far = cumulative(a, c)
    elif kind == "fork":
        near = cumulative(b, c)
        far = (cumulative(a, b) + cumulative(a, c)) / 2.0
    else:
        near = cumulative(b, c)
        far = (cumulative(b, a) + cumulative(c, a)) / 2.0
    return {"delta": float(near.mean() - far.mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, help="results/stage2_RQ-VAE/<arm>")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    rng = np.random.default_rng(42)
    rows, cols, _ = chains.loaded_edges()
    out_map, in_map = chains.adjacency(rows, cols)
    edges = np.stack([rows, cols], axis=1)
    banks = {
        "chain": chains.chain3(out_map, edges, rng, chains.CAP),
        "fork": chains.fan(out_map, in_map, rng, chains.CAP, "fork"),
        "merge": chains.fan(out_map, in_map, rng, chains.CAP, "merge"),
    }
    evaluation = {}
    for kind, bank in banks.items():
        _, ev = proto.split_pair_disjoint(bank)
        evaluation[kind] = ev
    semantics = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy").astype(
        np.float32)
    report = {}
    for name, directory in (("chain_align", root / "results" / "stage2_RQ-VAE" / args.arm),
                            ("baseline", BASELINE)):
        if not (directory / "item_sids.json").is_file():
            report[name] = {"missing": str(directory)}
            continue
        codes = load_codes(directory)
        entry = {"sid": str(directory.relative_to(root)),
                 "collision": float(1.0 - len(np.unique(codes, axis=0)) / len(codes))}
        for kind, bank in evaluation.items():
            entry[kind] = structure_delta(codes, bank, kind)
        checkpoint = directory / "rqvae_best.pth"
        if checkpoint.is_file():
            vectors = quantised_vectors(directory, codes, 32)
            entry["semantic_r2"] = semantic_probe(vectors, semantics)
        report[name] = entry
        print(f"{name}: collision {entry['collision']:.4f} " + " ".join(
            f"{kind} {entry[kind]['delta']:+.4f}" for kind in evaluation)
              + (f" semR2 {entry['semantic_r2']:.3f}" if "semantic_r2" in entry else ""),
              flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
