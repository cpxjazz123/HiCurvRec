"""Does the #154 supervision reach the encoder at all? One batch, one backward, per term."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import prefix_geometry as pg  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402


def _root() -> Path:
    for parent in TREE.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("root not found")


def main() -> int:
    root = _root()
    table = json.loads((root / "results" / "research" / "issue154" / "af3_edges.json")
                       .read_text())
    raw = np.load(root / "results" / "research" / "issue150" / "structure_prior.npz",
                  allow_pickle=True)
    item_of_node = {index: int(item) for index, item in enumerate(raw["item_ids"])}
    train_ids = sorted(item_of_node.values())
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    import train_rqvae as trainer
    device = torch.device("cpu")
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1],
                       curvature=1.0)
    trainer.initialize_tiger_weights(model)
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    model.init_codebook(rows[torch.from_numpy(np.asarray(train_ids[:4096]))])
    generator = torch.Generator().manual_seed(154)
    pick = torch.randperm(len(train_ids), generator=generator)[:2048].numpy()
    batch_ids = np.asarray(train_ids)[pick]
    batch = rows[torch.from_numpy(batch_ids)]
    position = {int(item): index for index, item in enumerate(batch_ids)}

    high, low = [], []
    for tail, head in table["high_edges"]:
        left, right = position.get(item_of_node.get(int(tail), -1)), position.get(
            item_of_node.get(int(head), -1))
        if left is not None and right is not None:
            high.append((left, right))
    for tail, head in table["low_edges"]:
        left, right = position.get(item_of_node.get(int(tail), -1)), position.get(
            item_of_node.get(int(head), -1))
        if left is not None and right is not None:
            low.append((left, right))
    triples = []
    for triangle in ():
        pass
    model.train()
    out = model(batch)
    prefixes = model.rq.last_prefixes
    losses = pg.structure_losses(prefixes[-1][0], prefixes[-1][1],
                                 torch.tensor(high, dtype=torch.long),
                                 torch.tensor(low, dtype=torch.long),
                                 torch.zeros(0, 3, dtype=torch.long),
                                 delta=torch.tensor(0.0), curvature=1.0)
    report = {"batch": 2048, "high_edges_in_batch": len(high), "low_edges_in_batch": len(low),
              "gap": float(losses["gap"]), "tri": float(losses["tri"]),
              "d_bar": float(losses["delta_bar"]), "tau": float(losses["tau"])}
    for name in ("gap", "tri"):
        model.zero_grad(set_to_none=True)
        losses[name].backward(retain_graph=True)
        encoder_grad = sum(float(parameter.grad.abs().sum())
                           for parameter in model.encoder.parameters()
                           if parameter.grad is not None)
        report[f"{name}_encoder_grad"] = encoder_grad
    out_path = root / "results" / "research" / "issue154" / "gradient_probe.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
