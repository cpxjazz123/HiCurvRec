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
    pick = torch.randperm(len(train_ids), generator=generator)[:4096].numpy()
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
    # the closed high triangles of the frozen graph: keep those whose three nodes are all
    # in this batch, which is what the batch-restricted L_tri can actually see
    edge_key = {}
    for index, (tail, head) in enumerate(raw["edges"].tolist()):
        edge_key[(min(tail, head), max(tail, head))] = index
    high_mask = np.zeros(raw["edges"].shape[0], dtype=bool)
    for tail, head in table["high_edges"]:
        index = edge_key.get((min(int(tail), int(head)), max(int(tail), int(head))))
        if index is not None:
            high_mask[index] = True
    triples = []
    for triangle in raw["triangles"].tolist():
        nodes = [int(triangle[0]), int(triangle[1]), int(triangle[2])]
        items = [item_of_node.get(node) for node in nodes]
        if any(item is None or item not in position for item in items):
            continue
        members = [(min(nodes[0], nodes[1]), max(nodes[0], nodes[1])),
                   (min(nodes[1], nodes[2]), max(nodes[1], nodes[2])),
                   (min(nodes[0], nodes[2]), max(nodes[0], nodes[2]))]
        indices = [edge_key.get(member) for member in members]
        if all(index is not None and high_mask[index] for index in indices):
            triples.append([position[item] for item in items])
    model.train()
    out = model(batch)
    prefixes = model.rq.last_prefixes
    losses = pg.structure_losses(prefixes[-1][0], prefixes[-1][1],
                                 torch.tensor(high, dtype=torch.long),
                                 torch.tensor(low, dtype=torch.long),
                                 torch.tensor(triples, dtype=torch.long)
                                 if triples else torch.zeros(0, 3, dtype=torch.long),
                                 delta=torch.tensor(0.0), curvature=1.0)
    report = {"batch": 4096, "high_edges_in_batch": len(high), "low_edges_in_batch": len(low),
              "closed_triangles_in_batch": len(triples),
              "gap": float(losses["gap"]), "tri": float(losses["tri"]),
              "d_bar": float(losses["delta_bar"]), "tau": float(losses["tau"])}
    report["prefix_grad_flags"] = [bool(value.requires_grad) for value in prefixes[-1]]
    report["loss_grad_flags"] = {name: bool(losses[name].requires_grad)
                                 for name in ("gap", "tri")}
    for name in ("gap", "tri"):
        if not losses[name].requires_grad:
            continue
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
