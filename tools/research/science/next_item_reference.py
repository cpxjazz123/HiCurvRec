#!/usr/bin/env python3
"""A NEXT-ITEM reference: the user's history predicts the behaviour of the next item.

Issue #126's frozen reference is a PAIRWISE model: it learns which item follows
which. Its source tower therefore says "this item looks like that one", which is
item similarity, not intent.

This reference is the other kind: it reads the user's history and predicts the
TARGET representation of the item the user is about to buy. The prediction is the
query point a metric can use, and it is trained on the TRAINING interactions only
-- no validation, no test, no Stage 2, no Stage 3.

    python3 tools/research/science/next_item_reference.py --epochs 30

writes `results/research/issue126/reference_next_item/` with the frozen per
interaction predictions and a record that carries the digest and the spec.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EMBEDDING_FILE = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_PAIRS = REPO / "results" / "stage3_T5Train" / "monotonicity" / "behavior_pairs_s42.npz"
REFERENCE_NPZ = REPO / "results" / "research" / "issue126" / "reference" / "reference.npz"
OUTPUT_DIR = REPO / "results" / "research" / "issue126" / "reference_next_item"

SPEC = {
    "version": "issue126-next-item-reference-v1",
    "hidden": 256,
    "dim": 32,
    "epochs": 30,
    "batch_size": 4096,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "seed": 42,
    "history_length": 8,
    "input": "the mean of the last HISTORY_LENGTH Stage1 embeddings of the history",
    "target": "the frozen Issue #126 target representation of the item to be bought",
    "geometry": "the prediction is mapped into the unit ball with exp_0",
    "loss": "the squared Euclidean distance to the frozen target representation",
    "fitting_data": "the training interactions only",
}


def training_interactions() -> list:
    import pandas as pd

    frame = pd.read_parquet(REPO / "results" / "stage0_build_parquet" / "train.parquet",
                            columns=["user", "history", "target"])
    frame = frame.sort_values("user", kind="stable")
    last = frame.groupby("user", sort=False).tail(1)
    interactions = []
    for history, target in zip(last["history"].to_numpy(), last["target"].to_numpy()):
        sequence = [int(item) for item in list(history)]
        if sequence:
            interactions.append((sequence, int(target)))
    return interactions


def fit(device: str = "cuda:0", epochs: int = None) -> dict:
    import torch

    epochs = int(epochs or SPEC["epochs"])
    features = np.load(EMBEDDING_FILE).astype(np.float32)
    reference = np.load(REFERENCE_NPZ)
    target_representation = reference["target"].astype(np.float32)

    interactions = training_interactions()
    length = int(SPEC["history_length"])
    inputs = np.zeros((len(interactions), features.shape[1]), dtype=np.float32)
    targets = np.zeros((len(interactions), target_representation.shape[1]), dtype=np.float32)
    keep = []
    for index, (history, target) in enumerate(interactions):
        recent = history[-length:]
        inputs[index] = features[recent].mean(axis=0)
        targets[index] = target_representation[target]
        keep.append(index)

    torch.manual_seed(int(SPEC["seed"]))
    torch.cuda.manual_seed_all(int(SPEC["seed"]))
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(inputs).to(dev)
    y = torch.from_numpy(targets).to(dev)

    class NextItem(torch.nn.Module):
        """The history in, the behaviour of the next item out."""

        def __init__(self):
            super().__init__()
            self.body = torch.nn.Sequential(
                torch.nn.Linear(features.shape[1], int(SPEC["hidden"])), torch.nn.GELU(),
                torch.nn.Linear(int(SPEC["hidden"]), int(SPEC["dim"])))

        def forward(self, batch):
            return self.body(batch)

    model = NextItem().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(SPEC["learning_rate"]),
                                  weight_decay=float(SPEC["weight_decay"]))
    generator = torch.Generator().manual_seed(int(SPEC["seed"]))
    order = torch.randperm(x.shape[0], generator=generator).to(dev)
    batch = int(SPEC["batch_size"])
    steps = max(1, x.shape[0] // batch)
    for epoch in range(epochs):
        model.train()
        shuffled = order[torch.randperm(order.shape[0], generator=generator).to(dev)]
        total = 0.0
        for step in range(steps):
            index = shuffled[step * batch:(step + 1) * batch]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x[index]), y[index])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch} loss {total / steps:.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        predictions = model(x).cpu().numpy().astype(np.float32)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    array_path = OUTPUT_DIR / "predictions.npz"
    np.savez_compressed(array_path, predictions=predictions)
    record = {
        "spec": {**SPEC, "epochs": epochs},
        "interactions": int(predictions.shape[0]),
        "arrays": {"path": str(array_path.relative_to(REPO)),
                   "sha256": hashlib.sha256(array_path.read_bytes()).hexdigest()},
        "target_reference_sha256": hashlib.sha256(REFERENCE_NPZ.read_bytes()).hexdigest(),
        "fitted_at": time.time(),
    }
    (OUTPUT_DIR / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    print(json.dumps({"interactions": record["interactions"],
                      "sha256": record["arrays"]["sha256"][:12]}, indent=1))
    return record


def load() -> dict:
    payload = np.load(OUTPUT_DIR / "predictions.npz")
    record = json.loads((OUTPUT_DIR / "record.json").read_text())
    return {"predictions": np.asarray(payload["predictions"], dtype=np.float64),
            "record": record}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    fit(device=args.device, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
