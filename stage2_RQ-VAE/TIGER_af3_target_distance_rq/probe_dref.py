"""d_ref on the frozen probe (issue #155 item 3) plus gates 3 and 4, all on one batch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import target_distance as td  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402

SIZE = 1024


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    table = json.loads((root / "results" / "research" / "issue155" / "af3_edges.json").read_text())
    raw = np.load(root / "results" / "research" / "issue150" / "structure_prior.npz",
                  allow_pickle=True)
    item_of_node = {index: int(item) for index, item in enumerate(raw["item_ids"])}
    all_ids = sorted(item_of_node.values())
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
    trainer.initialize_tiger_weights(model)
    model.init_codebook(rows[torch.from_numpy(np.asarray(all_ids[:4096]))])
    generator = torch.Generator().manual_seed(1550)
    pick = torch.randperm(len(all_ids), generator=generator)[:SIZE].numpy()
    batch_ids = np.asarray(all_ids)[pick]
    position = {int(item): index for index, item in enumerate(batch_ids)}
    edges, targets = [], []
    for (tail, head), u in zip(table["edges"], table["u"]):
        left = position.get(item_of_node.get(int(tail), -1))
        right = position.get(item_of_node.get(int(head), -1))
        if left is not None and right is not None:
            edges.append((left, right))
            targets.append(u)
    edge_tensor = torch.tensor(edges, dtype=torch.long) if edges else torch.zeros(0, 2, dtype=torch.long)
    u_tensor = torch.tensor(targets, dtype=torch.float64) if targets else torch.zeros(0, dtype=torch.float64)
    model.train()
    # no no_grad here: the prefix must reach the encoder
    model(rows[torch.from_numpy(batch_ids)])
    prefixes = model.rq.last_prefixes
    distances = [td.edge_distance(pe, ph, edge_tensor, 1.0) for pe, ph in prefixes]
    mean_distance = [float(d.mean()) if d.numel() else 0.0 for d in distances]
    d_ref = float(np.mean(mean_distance)) if any(mean_distance) else 1.0
    target = td.target_distance(u_tensor, d_ref)
    baseline = td.structure_losses(prefixes, edge_tensor, target, torch.zeros(0, 3, dtype=torch.long),
                                   torch.tensor(d_ref), 1.0)
    gradients = {}
    for name, curvature in (("curved", 1.0), ("flat", 0.0)):
        model.zero_grad(set_to_none=True)
        loss = td.structure_losses(prefixes, edge_tensor, target,
                                   torch.zeros(0, 3, dtype=torch.long),
                                   torch.tensor(d_ref), curvature)["struct"]
        loss.backward(retain_graph=True)
        gradients[name] = torch.cat([p.grad.reshape(-1) for p in model.encoder.parameters()
                                     if p.grad is not None])
    generator2 = torch.Generator().manual_seed(1551)
    random_edges = torch.randint(0, SIZE, (edge_tensor.shape[0], 2), generator=generator2)
    model.zero_grad(set_to_none=True)
    random_loss = td.structure_losses(prefixes, random_edges, target,
                                      torch.zeros(0, 3, dtype=torch.long),
                                      torch.tensor(d_ref), 1.0)["struct"]
    random_loss.backward(retain_graph=True)
    random_grad = torch.cat([p.grad.reshape(-1) for p in model.encoder.parameters()
                             if p.grad is not None])
    cosine = float(torch.nn.functional.cosine_similarity(gradients["curved"], random_grad, dim=0))
    report = {
        "batch": SIZE, "edges_in_batch": int(edge_tensor.shape[0]),
        "mean_prefix_distance_per_level": mean_distance, "d_ref": d_ref,
        "struct_loss": float(baseline["struct"]), "tri_loss": float(baseline["tri"]),
        "grad_norm_curved": float(gradients["curved"].norm()),
        "grad_norm_flat": float(gradients["flat"].norm()),
        "gate4_curved_vs_flat_loss": float(abs(float(td.structure_losses(
            prefixes, edge_tensor, target, torch.zeros(0, 3, dtype=torch.long),
            torch.tensor(d_ref), 1.0)["struct"]) - float(td.structure_losses(
            prefixes, edge_tensor, target, torch.zeros(0, 3, dtype=torch.long),
            torch.tensor(d_ref), 0.0)["struct"]))),
        "gate3_cosine_real_vs_random": cosine,
    }
    report["gate3_pass"] = bool(abs(cosine) < 0.5)
    report["gate4_pass"] = bool(report["gate4_curved_vs_flat_loss"] > 1e-6)
    out = root / "results" / "research" / "issue155" / "gate_dref.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
