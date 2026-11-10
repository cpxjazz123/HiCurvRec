"""Gates 2, 3 and 5 of issue #154 on the frozen TRAIN stream (gate 1 needs the fit)."""
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

BATCHES = 8
SIZE = 4096


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    table = json.loads((root / "results" / "research" / "issue154" / "af3_edges.json").read_text())
    raw = np.load(root / "results" / "research" / "issue150" / "structure_prior.npz",
                  allow_pickle=True)
    item_of_node = {index: int(item) for index, item in enumerate(raw["item_ids"])}
    edges = raw["edges"].tolist()
    edge_key = {(min(int(u), int(v)), max(int(u), int(v))): i for i, (u, v) in enumerate(edges)}
    high_mask = torch.zeros(len(edges), dtype=torch.bool)
    for tail, head in table["high_edges"]:
        index = edge_key.get((min(int(tail), int(head)), max(int(tail), int(head))))
        if index is not None:
            high_mask[index] = True
    high_pairs = [(item_of_node[int(u)], item_of_node[int(v)])
                  for (u, v), flag in zip(edges, high_mask.tolist()) if flag and int(u) in item_of_node and int(v) in item_of_node]
    low_pairs = [(item_of_node[int(u)], item_of_node[int(v)])
                 for (u, v), flag in zip(edges, high_mask.tolist()) if not flag and int(u) in item_of_node and int(v) in item_of_node]
    all_ids = sorted(item_of_node.values())
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
    trainer.initialize_tiger_weights(model)
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    model.init_codebook(rows[torch.from_numpy(np.asarray(all_ids[:4096]))])
    model.train()

    generator = torch.Generator().manual_seed(154)
    census = {"batches": BATCHES, "with_both": 0, "high": [], "low": [], "tri": []}
    losses_curved, losses_flat = [], []
    for step in range(BATCHES):
        pick = torch.randperm(len(all_ids), generator=generator)[:SIZE].numpy()
        batch_ids = np.asarray(all_ids)[pick]
        position = {int(item): index for index, item in enumerate(batch_ids)}
        high = [(position[a], position[b]) for a, b in high_pairs
                if a in position and b in position]
        low = [(position[a], position[b]) for a, b in low_pairs
               if a in position and b in position]
        census["high"].append(len(high))
        census["low"].append(len(low))
        if high and low:
            census["with_both"] += 1
        with torch.no_grad():
            model(rows[torch.from_numpy(batch_ids)])
        prefix_e, prefix_h = model.rq.last_prefixes[-1]
        pairs = torch.tensor(high + low, dtype=torch.long) if (high or low) else torch.zeros(0, 2, dtype=torch.long)
        split = torch.tensor(len(high)) if pairs.numel() else torch.tensor(0)
        curved = pg.structure_losses(prefix_e, prefix_h, pairs[:split], pairs[split:],
                                     torch.zeros(0, 3, dtype=torch.long),
                                     delta=torch.tensor(0.0), curvature=1.0)
        flat = pg.structure_losses(prefix_e, prefix_h, pairs[:split], pairs[split:],
                                   torch.zeros(0, 3, dtype=torch.long),
                                   delta=torch.tensor(0.0), curvature=0.0)
        losses_curved.append(float(curved["gap"]))
        losses_flat.append(float(flat["gap"]))
    report = {
        "census": census,
        "coverage_fraction": census["with_both"] / BATCHES,
        "gate5_pass": census["with_both"] / BATCHES >= 0.95,
        "gate3_curved_mean": float(np.mean(losses_curved)),
        "gate3_flat_mean": float(np.mean(losses_flat)),
        "gate3_difference": float(abs(np.mean(losses_curved) - np.mean(losses_flat))),
        "gate3_pass": bool(abs(np.mean(losses_curved) - np.mean(losses_flat)) > 1e-6),
    }
    pg.STRUCTURE_OFF = True
    off = pg.structure_losses(torch.zeros(4, 4), torch.zeros(4, 4), torch.tensor([]), torch.tensor([]),
                              torch.tensor([]), delta=torch.tensor(1.0), curvature=1.0)
    pg.STRUCTURE_OFF = False
    report["gate4_structure_off_zero"] = bool(float(off["gap"]) == 0.0 and float(off["tri"]) == 0.0)
    out = root / "results" / "research" / "issue154" / "gate_partial.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
