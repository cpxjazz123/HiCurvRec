"""Gate 2 of issue #154: the same loss with random edge sets must change by >= 2x."""
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

SIZE = 1024


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    table = json.loads((root / "results" / "research" / "issue154" / "af3_edges.json").read_text())
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
    generator = torch.Generator().manual_seed(1542)
    pick = torch.randperm(len(all_ids), generator=generator)[:SIZE].numpy()
    batch_ids = np.asarray(all_ids)[pick]
    position = {int(item): index for index, item in enumerate(batch_ids)}
    model.train()
    with torch.no_grad():
        model(rows[torch.from_numpy(batch_ids)])
    prefix_e, prefix_h = model.rq.last_prefixes[-1]

    def sets(high_key: str, low_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        high, low = [], []
        for key, bucket in ((high_key, high), (low_key, low)):
            for tail, head in table[key]:
                left = position.get(item_of_node.get(int(tail), -1))
                right = position.get(item_of_node.get(int(head), -1))
                if left is not None and right is not None:
                    bucket.append((left, right))
        return (torch.tensor(high, dtype=torch.long) if high else torch.zeros(0, 2, dtype=torch.long),
                torch.tensor(low, dtype=torch.long) if low else torch.zeros(0, 2, dtype=torch.long))

    real_high, real_low = sets("high_edges", "low_edges")

    def loss_of(high: torch.Tensor, low: torch.Tensor) -> float:
        terms = pg.structure_losses(prefix_e, prefix_h, high, low,
                                    torch.zeros(0, 3, dtype=torch.long),
                                    torch.tensor(0.0), 1.0)
        return float(terms["gap"])

    real = loss_of(real_high, real_low)
    ratios = []
    for trial in range(3):
        generator = torch.Generator().manual_seed(9000 + trial)
        noise_high = torch.randint(0, SIZE, (real_high.shape[0], 2), generator=generator)
        noise_low = torch.randint(0, SIZE, (real_low.shape[0], 2), generator=generator)
        shuffled = torch.randperm(prefix_h.shape[0], generator=generator)[:real_high.shape[0]]
        shuffled_high = torch.stack([shuffled, torch.roll(shuffled, 1)], dim=1)
        ratios.append({"trial": trial, "random_high_low": loss_of(noise_high, noise_low),
                       "shuffled_high": loss_of(shuffled_high, real_low)})
    report = {"batch": SIZE, "real_loss": real, "real_high": int(real_high.shape[0]),
              "real_low": int(real_low.shape[0]), "controls": ratios,
              "ratios": [max(abs(row["random_high_low"] - real), abs(row["shuffled_high"] - real))
                         / max(real, 1e-12) for row in ratios]}
    report["gate2_pass"] = bool(min([
        1.0 / max(report["ratios"][0], 1e-12) >= 2.0 if report["ratios"][0] else False]))
    out = root / "results" / "research" / "issue154" / "gate_random.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
