"""Gate 1 of issue #154: a fixed 200-step fit, baseline against the supervised arm."""
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

STEPS = 200
SIZE = 1024          # the frozen per-rank batch
EVAL = 2048


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    table = json.loads((root / "results" / "research" / "issue154" / "af3_edges.json").read_text())
    raw = np.load(root / "results" / "research" / "issue150" / "structure_prior.npz",
                  allow_pickle=True)
    item_of_node = {index: int(item) for index, item in enumerate(raw["item_ids"])}
    high_pairs, low_pairs = [], []
    for key, bucket in (("high_edges", high_pairs), ("low_edges", low_pairs)):
        for tail, head in table[key]:
            left, right = item_of_node.get(int(tail)), item_of_node.get(int(head))
            if left is not None and right is not None:
                bucket.append((left, right))
    all_ids = sorted(item_of_node.values())
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))

    def fresh() -> MixedRQVAE:
        torch.manual_seed(trainer.SEED)
        model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1],
                           curvature=1.0)
        trainer.initialize_tiger_weights(model)
        model.init_codebook(rows[torch.from_numpy(np.asarray(all_ids[:4096]))])
        return model

    def fit(supervised: bool) -> dict:
        model = fresh()
        optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.LR,
                                      weight_decay=trainer.WEIGHT_DECAY)
        model.train()
        generator = torch.Generator().manual_seed(1540)
        history = []
        for step in range(STEPS):
            pick = torch.randperm(len(all_ids), generator=generator)[:SIZE].numpy()
            batch_ids = np.asarray(all_ids)[pick]
            batch = rows[torch.from_numpy(batch_ids)]
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch)
            loss, _ = model.compute_loss(batch, reconstructed, quant_loss)
            parts = {"recon": float(loss.detach())}
            if supervised:
                position = {int(item): index for index, item in enumerate(batch_ids)}
                high = [(position[a], position[b]) for a, b in high_pairs
                        if a in position and b in position]
                low = [(position[a], position[b]) for a, b in low_pairs
                       if a in position and b in position]
                prefix_e, prefix_h = model.rq.last_prefixes[-1]
                terms = pg.structure_losses(
                    prefix_e, prefix_h,
                    torch.tensor(high, dtype=torch.long) if high else torch.zeros(0, 2, dtype=torch.long),
                    torch.tensor(low, dtype=torch.long) if low else torch.zeros(0, 2, dtype=torch.long),
                    torch.zeros(0, 3, dtype=torch.long), torch.tensor(0.0), 1.0)
                loss = loss + terms["gap"] + terms["tri"]
                parts.update({"gap": float(terms["gap"].detach()),
                              "tri": float(terms["tri"].detach())})
            loss.backward()
            optimizer.step()
            history.append(parts)
        tail = history[-20:]
        head = history[:20]
        summary = {"recon_head": float(np.mean([h["recon"] for h in head])),
                   "recon_tail": float(np.mean([h["recon"] for h in tail]))}
        if supervised:
            summary["gap_head"] = float(np.mean([h["gap"] for h in head]))
            summary["gap_tail"] = float(np.mean([h["gap"] for h in tail]))
            summary["tri_head"] = float(np.mean([h["tri"] for h in head]))
            summary["tri_tail"] = float(np.mean([h["tri"] for h in tail]))
            summary["steps_with_triangle"] = int(sum(1 for h in history if h["tri"] > 0.0))
        with torch.no_grad():
            model.eval()
            probe = rows[torch.from_numpy(np.asarray(all_ids[:EVAL]))]
            model(probe)
            summary["codes"] = model.rq.last_codes[-1].tolist()
        return summary

    pg.STRUCTURE_OFF = True
    baseline = fit(False)
    pg.STRUCTURE_OFF = False
    supervised = fit(True)
    base_codes = torch.tensor(baseline.pop("codes"))
    sup_codes = torch.tensor(supervised.pop("codes"))
    report = {
        "steps": STEPS, "batch": SIZE,
        "baseline": baseline, "supervised": supervised,
        "changed_last_level": float((base_codes != sup_codes).float().mean()),
        "gate1_gap_down": bool(supervised["gap_tail"] < supervised["gap_head"]),
        "gate1_tri_down": bool(supervised["tri_tail"] < supervised["tri_head"]),
    }
    report["gate1_changed"] = bool(report["changed_last_level"] >= 0.10)
    report["gate1_pass"] = bool(report["gate1_changed"] and report["gate1_gap_down"]
                                and report["gate1_tri_down"])
    out = root / "results" / "research" / "issue154" / "gate_fit.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
