"""Measure the frozen TIGER baseline on the same footing as the five arms.

The five arms were compared against each other, and the best of them (A) is only
meaningful next to the tree it was derived from. That tree never had its
structural distortion measured: it exports `sids_for_hgrec.npy` and a checkpoint,
not an `item_sids.json`.

Its readout is the same shape the arms use -- 32 numbers per item -- with the
Euclidean sum of the three codewords in place of the mixed readout, so the same
shared calibration, the same support and the same Huber scale apply without a
change. This script reconstructs both sides of that readout from the baseline's
own artifacts and measures them.

Usage:
    python3 results/research/issue139/measure_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASELINE_TREE = REPO / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
BASELINE_RESULT = REPO / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"

sys.path.insert(0, str(OUT))
sys.path.insert(0, str(BASELINE_TREE))

import measure_structure as M  # noqa: E402  (the arms' own measurement, reused as is)


def baseline_phi() -> tuple[torch.Tensor, torch.Tensor, dict]:
    """The baseline's continuous and discrete readout, over every item."""
    from model.model import RQVAE  # the Euclidean tree, not the mixed one
    import train_rqvae as trainer

    state = torch.load(BASELINE_RESULT / "rqvae_best.pth", map_location="cpu")
    state = state.get("state_dict", state)
    config = trainer._tokenizer_config()
    model = RQVAE(config, in_dim=768)
    model.load_state_dict(state, strict=True)
    model.eval()

    tokens = np.load(BASELINE_RESULT / "sids_for_hgrec.npy")[:, :3]
    tokens = torch.as_tensor(tokens, dtype=torch.long)
    embeddings = torch.from_numpy(np.load(M.EMBEDDINGS).astype(np.float32))
    with torch.no_grad():
        continuous = model.encoder(embeddings)
        codebooks = [layer.get_code_embs() for layer in model.rq.vq_layers]
        discrete = torch.zeros(embeddings.shape[0], codebooks[0].shape[1])
        for level, book in enumerate(codebooks):
            discrete = discrete + book[tokens[:, level]]
    return continuous, discrete, {"tokens": int(tokens.shape[0]),
                                  "codebooks": [int(book.shape[0]) for book in codebooks]}


def main() -> int:
    graph = M.structure.load_graph(M.AF3_CACHE)
    kappa = M.structure.target_kappa(graph)
    import pandas as pd
    frame = pd.read_parquet(REPO / "results" / "stage0_build_parquet" / "train.parquet")
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    embeddings = np.load(M.EMBEDDINGS)[train_ids]
    support = M.support_for(graph, kappa, embeddings)
    targets = M.per_pair_targets(kappa, support["pairs"])
    readout = json.loads(M.READOUT.read_text())
    print("support %d pairs | tau %.6f intercept %.6f"
          % (len(support["pairs"]), readout["tau"], readout["intercept"]), flush=True)

    continuous, discrete, info = baseline_phi()
    index = torch.as_tensor(train_ids, dtype=torch.long)
    continuous, discrete = continuous[index], discrete[index]
    cont = M.distortion(continuous, support, kappa, targets, readout["tau"],
                        readout["intercept"], graph["n_nodes"])
    disc = M.distortion(discrete, support, kappa, targets, readout["tau"],
                        readout["intercept"], graph["n_nodes"])
    retention = M.neighbour_retention(discrete, graph, 16)
    print("TIGER  D_cont %.6f  D_disc %.6f  cost %+.6f  Brier %.4f  retention %.4f"
          % (cont["distortion"], disc["distortion"],
             disc["distortion"] - cont["distortion"], disc["brier"],
             retention["retention"] or 0.0))

    # Fold it into the shared record next to the arms, under its own name.
    path = OUT / "structural_fidelity.json"
    payload = json.loads(path.read_text())
    payload.setdefault("arms", {})["TIGER"] = {
        "items": info["tokens"],
        "D_continuous": cont["distortion"], "D_discrete": disc["distortion"],
        "quantization_cost": disc["distortion"] - cont["distortion"],
        "edge_half_continuous": cont["edge_half"], "edge_half_discrete": disc["edge_half"],
        "unobserved_half_continuous": cont["unobserved_half"],
        "unobserved_half_discrete": disc["unobserved_half"],
        "brier_continuous": cont["brier"], "brier_discrete": disc["brier"],
        "neighbour_retention": retention["retention"],
        "neighbour_pairs": retention["pairs"], "scale": disc["scale"],
        "note": "the frozen baseline, reconstructed from sids_for_hgrec.npy and its "
                "checkpoint; its readout is the Euclidean sum of the three codewords",
    }
    arms = payload["arms"]
    if "A" in arms:
        payload["baseline_comparison"] = {
            "statement": "A versus the frozen baseline it was derived from",
            "A_minus_TIGER": arms["A"]["D_discrete"] - arms["TIGER"]["D_discrete"],
            "A_better": arms["A"]["D_discrete"] < arms["TIGER"]["D_discrete"],
        }
        print("A - TIGER = %+.6f -> %s"
              % (payload["baseline_comparison"]["A_minus_TIGER"],
                 "A is better" if payload["baseline_comparison"]["A_better"] else "the baseline is better"))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("written:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
