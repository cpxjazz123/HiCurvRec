"""Issue #168 gate 3: the 200-step pilot must not let the occupancy drift."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

STEPS, SIZE, EVAL = 200, 512, 2048


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    from model import item_index as ii
    from model import rank_objective as ro
    from model.model import MixedRQVAE
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    kappa = json.loads((root / "results" / "research" / "issue167" / "kappa.json").read_text())["kappa"]
    item_ids = sorted(int(key) for key in kappa)
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
    trainer.initialize_tiger_weights(model)
    model.init_codebook(rows[:4096])
    model.attach_kappa(kappa, ii.build(embeddings))
    eval_rows = rows[torch.from_numpy(np.asarray(item_ids)[:EVAL])]

    def profile() -> dict:
        model.train()
        model(eval_rows)
        codes = model.rq.last_codes[-1]
        counts = torch.bincount(codes.long(), minlength=256).double()
        # "batch mean prefix distance": average over many pairs, not over one arbitrary pair (which is
        # what the earlier version did -- it made the check meaningless)
        generator_pairs = torch.Generator().manual_seed(1681)
        n = eval_rows.shape[0]
        left = torch.randint(0, n, (4096,), generator=generator_pairs)
        right = torch.randint(0, n, (4096,), generator=generator_pairs)
        pairs = torch.stack([left, right], dim=1)
        distance = float(ro.edge_distance(model.rq.last_prefixes[-1][0],
                                          model.rq.last_prefixes[-1][1], pairs, 1.0).mean())
        return {"codes_used": int((counts > 0).sum()),
                "max_share": float(counts.max() / counts.sum()),
                "mean_distance": distance}

    before = profile()
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.LR,
                                  weight_decay=trainer.WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(1680)
    model.train()
    for _ in range(STEPS):
        pick = torch.randperm(len(item_ids), generator=generator)[:SIZE].numpy()
        batch = rows[torch.from_numpy(np.asarray(item_ids)[pick])]
        optimizer.zero_grad(set_to_none=True)
        reconstructed, quant_loss, _, _ = model(batch)
        loss, _ = model.compute_loss(batch, reconstructed, quant_loss)
        loss.backward()
        optimizer.step()
    after = profile()
    report = {
        "steps": STEPS, "batch": SIZE,
        "before": before, "after": after,
        "gate3_codes_used_ok": after["codes_used"] >= 240,
        "gate3_max_share_ok": after["max_share"] <= 0.05,
        "gate3_distance_ok": after["mean_distance"] >= 0.5 * before["mean_distance"],
    }
    report["gate3_pass"] = bool(report["gate3_codes_used_ok"] and report["gate3_max_share_ok"]
                                and report["gate3_distance_ok"])
    out = root / "results" / "research" / "issue168" / "gate_pilot.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
