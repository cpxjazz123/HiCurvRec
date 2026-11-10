"""Pre-dispatch gates of issue #152: is pi realisable once the layer is a feasibility mask?

Read-only, frozen 8192-row TRAIN probe. Measures (1) the masked plan's relative error
against pi, (2) the hard per-bucket mass against rho, (3) per-bucket occupancy guardrails,
(4) MASKS_OFF against the unbiased baseline histogram. Writes
results/research/issue152/gate.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

from model import frozen_prior as prior  # noqa: E402
from model.mixed_rq import split  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402
from model.poincare_contract import expmap0  # noqa: E402

ROWS = 8192
GUARD_CODES = 48          # per bucket, out of 64
GUARD_SHARE = 0.10


def _root() -> Path:
    for parent in TREE.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root")


def main() -> int:
    import pandas as pd
    import train_rqvae as trainer

    root = _root()
    frozen = prior.load_partition(root)
    layers = prior.load_item_layers(root)
    pi = torch.as_tensor(frozen["marginal"], dtype=torch.float64)
    pi_cpu = pi
    rho = torch.tensor([sum(float(pi[k]) for k in bucket) for bucket in frozen["partition"]],
                       dtype=torch.float64)
    embeddings = np.load(root / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy")
    frame = pd.read_parquet(root / "results" / "stage0_build_parquet" / "train.parquet",
                            columns=["target"])
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(trainer.SEED)
    model = MixedRQVAE(trainer._tokenizer_config(), in_dim=embeddings.shape[1],
                       curvature=1.0).to(device)
    trainer.initialize_tiger_weights(model)
    rows = torch.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32)))
    model.init_codebook(rows[torch.from_numpy(train_ids)].to(device))
    generator = torch.Generator().manual_seed(152)
    pick = torch.randperm(len(train_ids), generator=generator)[:ROWS].numpy()
    picked_ids = train_ids[pick]
    groups = torch.tensor([layers.get(int(item), 0) for item in picked_ids], dtype=torch.long,
                          device=device)
    batch = rows[torch.from_numpy(picked_ids)].to(device)
    model.attach_frozen_prior(frozen)

    def measure(masks_off: bool) -> dict:
        prior.MASKS_OFF = bool(masks_off)
        model.train()
        with torch.no_grad():
            encoded = model.encoder(batch)
            z_e, z_t = split(encoded)
            residual_e, residual_h = z_e, expmap0(z_t, model.rq.curvature)
            for level, layer in enumerate(model.rq.levels, start=1):
                alpha = model.rq.alpha_table.lookups(level, []).to(device)
                record: dict = {}
                last = level == len(model.rq.levels)
                (residual_e, residual_h), _, _, index, _, _ = layer(
                    residual_e, residual_h, alpha, infer_use_sk=True, record=record,
                    marginal=model.rq.frozen_marginal if last else None,
                    groups=groups if last else None)
            plan = record["plan"]
        prior.MASKS_OFF = False
        return {"plan": plan, "histogram": prior.hard_histogram(plan, int(pi.numel()))}

    masked = measure(masks_off=False)
    plain = measure(masks_off=True)
    table = model.rq.levels[-1].bucket_partition
    model.rq.levels[-1].bucket_partition = None
    unbucketed = measure(masks_off=False)
    model.rq.levels[-1].bucket_partition = table
    plan = masked["plan"]
    mean = plan.double().mean(0).cpu()
    rel_pi = float(((mean - pi).abs() / pi.clamp_min(1e-12)).max())
    floor = float((0.5 / ROWS) / float(pi.min()))
    hard = plan.argmax(-1)
    per_bucket = {}
    for index, bucket in enumerate(frozen["partition"]):
        in_bucket = torch.as_tensor(bucket, device=device)
        picked = torch.isin(hard, in_bucket)
        rows_here = groups == index
        # the issue's metric is the *global* mass of the bucket against rho; the share
        # inside the layer's own rows is 1.0 by construction and is not that quantity
        per_bucket[str(index)] = {
            "global_mass": float(picked.float().mean()),
            "mass_error": float(abs(float(picked.float().mean()) - float(rho[index]))),
            "codes_used": int(torch.unique(hard[rows_here]).numel()),
            "max_share": float(torch.bincount(
                torch.searchsorted(in_bucket, hard[rows_here]), minlength=len(bucket)).max()
                / max(int(rows_here.sum()), 1)),
            "rows": int(rows_here.sum()),
        }
    checks = {
        "rows": ROWS,
        "pi_min": float(pi.min()),
        "floor_percent": 100.0 * floor,
        "gate1_plan_relative_error": rel_pi,
        "gate1_pass": bool(rel_pi <= 0.05),
        "gate2_mass_pass": bool(all(row["mass_error"] <= 1e-6 for row in per_bucket.values())),
        "gate3_pass": bool(all(row["codes_used"] >= GUARD_CODES and row["max_share"] <= GUARD_SHARE
                               for row in per_bucket.values())),
        # the wiring identity: MASKS_OFF must be exactly the run with no bucket table
        "gate4_masks_off_is_unmasked": bool(torch.equal(
            plain["plan"].argmax(-1), unbucketed["plan"].argmax(-1))),
        "per_bucket": per_bucket,
        "rho": [float(value) for value in rho],
        "masked_histogram": masked["histogram"],
        "plain_histogram": plain["histogram"],
    }
    checks["passed"] = bool(checks["gate1_pass"] and checks["gate2_mass_pass"]
                            and checks["gate3_pass"] and checks["gate4_masks_off_is_unmasked"])
    out = root / "results" / "research" / "issue152" / "gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: checks[key] for key in
                      ("rows", "floor_percent", "gate1_plan_relative_error", "gate1_pass",
                       "gate2_mass_pass", "gate3_pass", "passed")}, indent=2))
    for name, row in checks["per_bucket"].items():
        print("bucket", name, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
