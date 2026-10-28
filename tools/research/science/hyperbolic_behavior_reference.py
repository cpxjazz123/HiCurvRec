"""The frozen hyperbolic behaviour reference of Issue #126, built once.

The reference turns the frozen Stage 1 item features into two Poincaré-ball
representations -- one for the item that conditions the query, one for the item
being ranked -- and trains them on the legitimate directed next-item relations of
the TRAINING split only. It is then frozen: the same vectors, curvature,
dimension, scale and reading rule score every hard SID, so a difference between
two SIDs cannot come from a reference that was retrained for one of them.

Everything here is a source constant plus one recorded rule. There is no flag
that changes the fit, no early stopping and no selection against validation or
test: the budget is the fixed number of epochs below, and the checkpoint is the
one the last step leaves.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]

#: The frozen inputs. Stage 1 features are shared with Stage 2 and never rebuilt.
EMBEDDING_FILE = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_PAIRS = REPO / "results" / "stage3_T5Train" / "monotonicity" / "behavior_pairs_s42.npz"
VALID_PAIRS = REPO / "results" / "stage3_T5Train" / "monotonicity" / "behavior_pairs_valid_s42.npz"

#: Where the frozen reference and its record live. Nothing else writes here.
OUTPUT_DIR = REPO / "results" / "research" / "issue126" / "reference"

#: The frozen reference specification. Every value is fixed before the fit and
#: is part of the record, so the same inputs give the same vectors.
SPEC = {
    "version": "issue126-reference-v1",
    "curvature": 1.0,
    "dim": 32,
    "hidden": 256,
    "epochs": 20,
    "batch_size": 4096,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "seed": 42,
    "temperature_rule": "median |delta| over the first 20000 train pairs at the end of the fit",
    "ball_eps": 1e-4,
    "checkpoint": "last step (no early stopping, no selection)",
}


def _tokens(text_path: Path) -> dict:
    return {"path": str(text_path.relative_to(REPO)),
            "sha256": hashlib.sha256(text_path.read_bytes()).hexdigest()}


def frozen_spec() -> dict:
    """The specification with the digests of its inputs."""
    spec = dict(SPEC)
    spec["inputs"] = {
        "embeddings": _tokens(EMBEDDING_FILE),
        "train_pairs": _tokens(TRAIN_PAIRS),
        "valid_pairs": _tokens(VALID_PAIRS),
    }
    spec["digest"] = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
    return spec


def _geometry():
    """The canonical hyperbolic helpers of the frozen baseline.

    The reference is built on the same projection and distance functions the
    Stage 2 trees use, loaded from the frozen baseline's own module rather than
    reimplemented here: a second copy of the formulas would be a second geometry.
    """
    import importlib.util
    import sys

    path = REPO / "stage2_RQ-VAE" / "TIGER_RQ-VAE" / "model" / "utils.py"
    if "model" in sys.modules:
        del sys.modules["model"]
    spec = importlib.util.spec_from_file_location("baseline_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit(device: str = "cuda:0") -> dict:
    """Train the two projections once and write the frozen reference."""
    import torch

    geometry = _geometry()

    torch.manual_seed(SPEC["seed"])
    torch.cuda.manual_seed_all(SPEC["seed"])
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    features = torch.from_numpy(np.load(EMBEDDING_FILE).astype(np.float32)).to(dev)
    n_items, in_dim = features.shape
    hidden, dim = int(SPEC["hidden"]), int(SPEC["dim"])
    c = float(SPEC["curvature"])

    class Reference(torch.nn.Module):
        """Two projections into the ball: the source and the target of a step."""

        def __init__(self):
            super().__init__()
            self.source = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, dim))
            self.target = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, dim))

        def forward(self):
            source = geometry.proj_to_ball(geometry.expmap0(self.source(features), c), c)
            target = geometry.proj_to_ball(geometry.expmap0(self.target(features), c), c)
            return source, target

    model = Reference().to(dev)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(SPEC["learning_rate"]),
                                  weight_decay=float(SPEC["weight_decay"]))
    pairs = np.load(TRAIN_PAIRS)
    anchors = torch.from_numpy(pairs["anchors"].astype(np.int64)).to(dev)
    positives = torch.from_numpy(pairs["positives"].astype(np.int64)).to(dev)
    negatives = torch.from_numpy(pairs["negatives"].astype(np.int64)).to(dev)
    count = int(anchors.shape[0])
    batch = int(SPEC["batch_size"])

    # The temperature is the scale of the frozen distance differences themselves,
    # read from the TRAIN pairs only, at the end of the fit: a fixed rule, not a
    # value chosen by looking at any downstream result.
    def deltas(source, target, index):
        u = source[anchors[index]]
        d_pos = geometry.poincare_distance(u, target[positives[index]], c) ** 2
        d_neg = geometry.poincare_distance(
            u.unsqueeze(1).expand(-1, negatives.shape[1], -1),
            target[negatives[index]], c) ** 2
        return d_neg - d_pos.unsqueeze(1)

    for epoch in range(int(SPEC["epochs"])):
        order = torch.randperm(count, device=dev)
        epoch_loss = 0.0
        for start in range(0, count, batch):
            index = order[start:start + batch]
            source, target = model()
            # The train-side loss uses the raw difference (scale 1); the frozen
            # temperature only enters the recorded score.
            per_pair = torch.nn.functional.softplus(-deltas(source, target, index))
            loss = per_pair.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * int(index.numel())
        print("reference epoch %d/%d loss %.6f" % (epoch + 1, SPEC["epochs"],
                                                   epoch_loss / count), flush=True)

    with torch.no_grad():
        source, target = model()
        sample = torch.arange(min(20000, count), device=dev)
        delta = deltas(source, target, sample)
        temperature = float(delta.abs().median())
        # The reference's own signal is read on the fixed VALIDATION probe, never
        # on the pairs it was trained on: a teacher that only orders its training
        # set is not a teacher.
        held_arrays = np.load(VALID_PAIRS)
        held = _heldout_signal(
            geometry, source, target,
            torch.from_numpy(held_arrays["anchors"].astype(np.int64)).to(dev),
            torch.from_numpy(held_arrays["positives"].astype(np.int64)).to(dev),
            torch.from_numpy(held_arrays["negatives"].astype(np.int64)).to(dev), c)
    reference = {
        "spec": frozen_spec(),
        "temperature": temperature,
        "source": source.double().cpu().numpy().astype(np.float32),
        "target": target.double().cpu().numpy().astype(np.float32),
        "items": int(n_items),
        "heldout": held,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    array_path = OUTPUT_DIR / "reference.npz"
    np.savez_compressed(array_path, source=reference["source"],
                        target=reference["target"])
    record = {key: reference[key] for key in ("spec", "temperature", "items", "heldout")}
    record["arrays"] = {** _tokens(array_path), "source_shape": list(source.shape),
                        "target_shape": list(target.shape)}
    record["cost"] = {"epochs": SPEC["epochs"], "pairs": count,
                      "batch_size": batch, "device": str(dev)}
    record_path = OUTPUT_DIR / "reference.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("temperature", "heldout")}, indent=2)[:800],
          flush=True)
    return record


def _heldout_signal(geometry, source, target, anchors, positives, negatives, c) -> dict:
    """The reference's own ranking signal on the fixed VALIDATION probe.

    This is a diagnostic of the reference, reported before any SID is scored: a
    reference that cannot order a held-out positive above its negatives cannot be
    a teacher, and saying so is the honest outcome.
    """
    import torch

    with torch.no_grad():
        u = source[anchors]
        d_pos = geometry.poincare_distance(u, target[positives], c) ** 2
        d_neg = geometry.poincare_distance(
            u.unsqueeze(1).expand(-1, negatives.shape[1], -1), target[negatives], c) ** 2
        delta = d_neg - d_pos.unsqueeze(1)
        win = (delta > 0).float()
        return {
            "pairs": int(anchors.shape[0]),
            "negatives_per_pair": int(negatives.shape[1]),
            "pair_accuracy": float(win.mean()),
            "all_negatives_ranked_below": float((win.min(dim=1).values).mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(delta.median()),
        }


def load() -> dict:
    """The frozen reference, read back with its record."""
    record = json.loads((OUTPUT_DIR / "reference.json").read_text(encoding="utf-8"))
    arrays = np.load(OUTPUT_DIR / "reference.npz")
    return {"record": record, "source": arrays["source"], "target": arrays["target"]}


if __name__ == "__main__":
    print(json.dumps(fit(os.environ.get("GENREC_REFERENCE_DEVICE", "cuda:0")),
                     indent=2, sort_keys=True)[:400])
