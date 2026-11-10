"""Pre-dispatch gates of issue #151: the hard-object check and its direction.

Read-only. On a frozen >=8192-row TRAIN probe, the last level's hard codes (the plan's
argmax at the frozen 50 sweeps) are measured against pi and against the uniform target,
with T = 0 as the wiring control. Writes results/research/issue151/gate.json.
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
from model.poincare_contract import expmap0, mobius_add, project  # noqa: E402

ROWS = 8192


def _root() -> Path:
    for parent in TREE.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root")


def main() -> int:
    import pandas as pd
    import train_rqvae as trainer

    root = _root()
    structure = prior.load_structure(root)
    frozen = prior.load_partition(root)
    pi = torch.as_tensor(frozen["marginal"], dtype=torch.float64)
    uniform = torch.full_like(pi, 1.0 / float(pi.numel()))
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
    generator = torch.Generator().manual_seed(151)
    pick = torch.randperm(len(train_ids), generator=generator)[:ROWS].numpy()
    batch = rows[torch.from_numpy(train_ids[pick])].to(device)

    def measure(scale: float) -> dict:
        model.attach_frozen_prior(frozen)
        last = model.rq.levels[-1]
        last.prior_scale = float(scale)
        model.train()
        with torch.no_grad():
            encoded = model.encoder(batch)
            z_e, z_t = split(encoded)
            residual_e, residual_h = z_e, expmap0(z_t, model.rq.curvature)
            for level, layer in enumerate(model.rq.levels, start=1):
                alpha = model.rq.alpha_table.lookups(level, []).to(device)
                record: dict = {}
                (residual_e, residual_h), _, _, index, picked_e, picked_h = layer(
                    residual_e, residual_h, alpha, infer_use_sk=True, record=record)
            plan = record["plan"]
        return {"vs_pi": prior.hard_deviation(plan, pi.to(device)),
                "vs_uniform": prior.hard_deviation(plan, uniform.to(device)),
                "histogram": prior.hard_histogram(plan, int(pi.numel()))}

    base = measure(0.0)
    biased = measure(prior.PRIOR_T)
    checks = {
        "rows": ROWS,
        "prior_T": prior.PRIOR_T,
        "gate1_abs_rows": biased["vs_pi"]["max_row_deviation"] <= 1.5,
        "gate2_direction": biased["vs_pi"]["max_relative_error"]
                           < biased["vs_uniform"]["max_relative_error"],
        "direction_ratio": biased["vs_uniform"]["max_relative_error"]
                           / max(biased["vs_pi"]["max_relative_error"], 1e-12),
        "wire_T0_matches_unbiased": abs(base["vs_uniform"]["max_relative_error"]
                                        - biased["vs_uniform"]["max_relative_error"]) >= 0.0,
        "base": base, "biased": biased,
    }
    checks["passed"] = bool(checks["gate1_abs_rows"] and checks["gate2_direction"])
    out = root / "results" / "research" / "issue151" / "gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: checks[key] for key in
                      ("rows", "prior_T", "gate1_abs_rows", "gate2_direction",
                       "direction_ratio", "passed")}, indent=2))
    print("pi   ", checks["biased"]["vs_pi"])
    print("unif ", checks["biased"]["vs_uniform"])
    print("hist ", checks["biased"]["histogram"], "base", base["histogram"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
