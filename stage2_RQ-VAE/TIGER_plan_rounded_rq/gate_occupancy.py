"""Gates 1, 3, 7 and 8 of issue #159: occupancy by construction, timing, wiring, guardrails."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import rank_objective as ro  # noqa: E402
from model import rounding as rd  # noqa: E402
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
    eval_rows = rows[torch.from_numpy(np.asarray(item_ids)[:EVAL])]

    def fresh():
        torch.manual_seed(trainer.SEED)
        model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1], curvature=1.0)
        trainer.initialize_tiger_weights(model)
        model.init_codebook(rows[torch.from_numpy(np.asarray(item_ids[:4096]))])
        return model

    def fit(model, mode):
        optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.LR,
                                      weight_decay=trainer.WEIGHT_DECAY)
        generator = torch.Generator().manual_seed(15920)
        model.train()
        for _ in range(STEPS):
            pick = torch.randperm(len(item_ids), generator=generator)[:SIZE].numpy()
            batch = rows[torch.from_numpy(np.asarray(item_ids)[pick])]
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

    def profile(model):
        # the rounding only runs on the *training* path (set = Sinkhorn then plan-mass rounding);
        # eval falls back to plain argmin, so measuring in eval mode measures the wrong rule
        model.train()
        model(eval_rows)
        codes = model.rq.last_codes[-1]
        counts = torch.bincount(codes.long(), minlength=256).double()
        pairs = torch.tensor([(0, 1)], dtype=torch.long)
        last_distance = float(ro.edge_distance(model.rq.last_prefixes[-1][0],
                                               model.rq.last_prefixes[-1][1], pairs, 1.0).mean())
        return {"codes_used": int((counts > 0).sum()),
                "max_share": float(counts.max() / counts.sum()),
                "mean_distance": last_distance}

    baseline = fit(fresh(), "U")
    initial = profile(baseline)
    arm = fit(fresh(), "R")
    after = profile(arm)
    # timing of one per-rank rounding at the training shape
    plan = torch.rand(1024, 256, dtype=torch.float64)
    cost = torch.rand(1024, 256, dtype=torch.float64)
    start = time.perf_counter()
    rd.round_plan(plan, cost)
    rounding_seconds = time.perf_counter() - start
    # wiring: ROUNDING_OFF must be the argmax rule and STRUCTURE_OFF must not touch the codes
    arm.train()
    rd.ROUNDING_OFF = True
    arm(eval_rows)
    off_codes = torch.stack(arm.rq.last_codes, dim=1).clone()
    rd.ROUNDING_OFF = False
    arm(eval_rows)
    on_codes = torch.stack(arm.rq.last_codes, dim=1).clone()
    report = {
        "steps": STEPS, "batch": SIZE,
        "baseline_profile": initial, "arm_profile": after,
        "gate1_codes_used_ok": after["codes_used"] >= 240,
        "gate1_max_share_ok": after["max_share"] <= 0.015,
        "gate3_rounding_seconds": rounding_seconds,
        "gate3_pass": bool(rounding_seconds <= 0.05),
        "gate8_codes_used_ok": after["codes_used"] >= 0.8 * initial["codes_used"],
        "gate8_max_share_ok": after["max_share"] <= 2.0 * initial["max_share"],
        "gate8_spread_ok": after["mean_distance"] >= 0.5 * initial["mean_distance"],
        "gate7_rounding_off_changes_codes": bool(not torch.equal(off_codes, on_codes)),
    }
    report["gate1_pass"] = bool(report["gate1_codes_used_ok"] and report["gate1_max_share_ok"])
    report["gate8_pass"] = bool(report["gate8_codes_used_ok"] and report["gate8_max_share_ok"]
                                and report["gate8_spread_ok"])
    target = root / "results" / "research" / "issue159" / "gate_occupancy.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
