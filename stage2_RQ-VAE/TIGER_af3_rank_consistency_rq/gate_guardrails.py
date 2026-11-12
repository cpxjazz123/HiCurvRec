"""Gate 3 (run guardrails) plus gates 4-6 of issue #157, on a 200-step pair of arms."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import rank_objective as ro  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402
from model.structure_runtime import StructureSupervision  # noqa: E402

STEPS, SIZE, EVAL = 200, 512, 2048
SIGMA_MIN = 0.48247611144034613


def main() -> int:
    root = TREE.parents[1]
    import train_rqvae as trainer
    raw = np.load(root / "results" / "research" / "issue150" / "structure_prior.npz",
                  allow_pickle=True)
    item_ids = sorted(int(x) for x in raw["item_ids"])
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    supervision = StructureSupervision.from_repo(root, SIGMA_MIN)
    eval_ids = np.asarray(item_ids)[:EVAL]
    eval_rows = rows[torch.from_numpy(eval_ids)]

    def fresh() -> MixedRQVAE:
        torch.manual_seed(trainer.SEED)
        model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
        trainer.initialize_tiger_weights(model)
        model.init_codebook(rows[torch.from_numpy(np.asarray(item_ids[:4096]))])
        return model

    def fit(mode: str) -> MixedRQVAE:
        model = fresh()
        optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.LR,
                                      weight_decay=trainer.WEIGHT_DECAY)
        generator = torch.Generator().manual_seed(15720)
        model.train()
        for _ in range(STEPS):
            pick = torch.randperm(len(item_ids), generator=generator)[:SIZE].numpy()
            batch_ids = np.asarray(item_ids)[pick]
            batch = rows[torch.from_numpy(batch_ids)]
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _, _ = model(batch)
            loss, _ = model.compute_loss(batch, reconstructed, quant_loss)
            if mode == "R":
                terms = supervision.terms(model, batch, 1.0)
                if terms is not None:
                    loss = loss + terms
            loss.backward()
            optimizer.step()
        return model

    def profile(model: MixedRQVAE) -> dict:
        model.eval()
        model(eval_rows)
        codes = model.rq.last_codes[-1]
        counts = torch.bincount(codes.long(), minlength=256).double()
        pairs = torch.tensor([(0, 1)], dtype=torch.long)
        distances = []
        for prefix_e, prefix_h in model.rq.last_prefixes:
            distances.append(float(ro.edge_distance(prefix_e, prefix_h, pairs, 1.0).mean()))
        return {"codes_used": int((counts > 0).sum()),
                "max_share": float(counts.max() / counts.sum()),
                "mean_distance": distances[-1]}

    baseline = fit("U")
    initial = profile(baseline)
    ranked = fit("R")
    after = profile(ranked)
    losses_curved = supervision.terms(ranked, eval_rows, 1.0)
    losses_flat = supervision.terms(ranked, eval_rows, 0.0)
    ro.STRUCTURE_OFF = True
    ranked(eval_rows)
    off_codes = torch.stack(ranked.rq.last_codes, dim=1).clone()
    ro.STRUCTURE_OFF = False
    ranked(eval_rows)
    on_codes = torch.stack(ranked.rq.last_codes, dim=1).clone()
    report = {
        "steps": STEPS, "batch": SIZE,
        "baseline_profile": initial, "arm_profile": after,
        "gate3_codes_used_ok": after["codes_used"] >= 0.8 * initial["codes_used"],
        "gate3_max_share_ok": after["max_share"] <= 2.0 * initial["max_share"],
        "gate3_spread_ok": after["mean_distance"] >= 0.5 * initial["mean_distance"],
        "gate4_curved_vs_flat": float(abs(float(losses_curved) - float(losses_flat))),
        "gate5_structure_off_bitwise": bool(torch.equal(off_codes, on_codes)),
    }
    report["gate3_pass"] = bool(report["gate3_codes_used_ok"] and report["gate3_max_share_ok"]
                                and report["gate3_spread_ok"])
    report["gate4_pass"] = bool(report["gate4_curved_vs_flat"] > 1e-6)
    target = root / "results" / "research" / "issue157" / "gate_guardrails.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
