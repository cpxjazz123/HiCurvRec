"""Measure the leverage the column marginal has at the frozen assignment.

The structured Sinkhorn differs from the baseline only in the column target, which in
the dual is an additive bias of order ``eps * log pi_k`` per code. A row can therefore
change its code only if its top-1/top-2 cost gap is below ``log(pi_max/pi_min) * eps``
in centred units. This script reports that gap's distribution and the share of rows
under the threshold, on the frozen probe, at the level the marginal is applied to.

Run with the Stage 2 interpreter (CLAUDE.md §2). Writes
results/research/issue150/marginal_leverage.json; nothing outside the repository.
"""
from __future__ import annotations

import json
import math
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
from model.poincare_contract import mobius_add as mobius  # noqa: E402
from model.poincare_contract import project as prior_project  # noqa: E402

ROWS = 4096


def _repository_root() -> Path:
    for parent in TREE.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(TREE))


def main() -> int:
    import pandas as pd
    import train_rqvae as trainer

    root = _repository_root()
    structure = prior.load_structure(root)
    frozen = prior.load_partition(root)
    marginal = torch.as_tensor(frozen["marginal"], dtype=torch.float64)
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

    generator = torch.Generator().manual_seed(150)
    pick = torch.randperm(len(train_ids), generator=generator)[:ROWS].numpy()
    model.train()
    with torch.no_grad():
        encoded = model.encoder(rows[torch.from_numpy(train_ids[pick])].to(device))
        z_e, z_t = split(encoded)
        residual_e, residual_h = z_e, expmap0(z_t, model.rq.curvature)
        last = len(model.rq.levels)
        for level, layer in enumerate(model.rq.levels, start=1):
            alpha = model.rq.alpha_table.lookups(level, []).to(device)
            cost = layer.distances(residual_e, residual_h, alpha)
            index = cost.argmin(-1)
            if level == last:
                # the margin by which the best code wins: smallest two costs, not the
                # largest, and the gap in centred units is what a bias of log(pi)*eps
                # has to overcome
                centred = prior.centred_cost(cost, where="the balanced plan").double()
                best = centred.topk(2, dim=-1, largest=False).values
                gap = best[:, 1] - best[:, 0]
            codes = layer.get_code_embs()
            codes_e, codes_h = split(codes)
            picked_e = codes_e[index]
            picked_h = expmap0(codes_h[index], model.rq.curvature)
            residual_e = residual_e - picked_e
            residual_h = prior_project(mobius(-picked_h, residual_h, model.rq.curvature),
                                       model.rq.curvature)

    eps = float(model.rq.levels[-1].sk_epsilon)
    leverage = float(torch.log(marginal.max() / marginal.min())) * eps
    values = gap.detach().cpu()
    share = float((values < leverage).float().mean())
    report = {
        "rows": int(values.numel()),
        "sk_epsilon": eps,
        "pi_max_over_min": float(marginal.max() / marginal.min()),
        "leverage_centred_units": leverage,
        "gap_quantiles": {name: float(values.quantile(q)) for name, q in
                          (("p1", 0.01), ("p5", 0.05), ("p25", 0.25), ("p50", 0.5),
                           ("p75", 0.75), ("p95", 0.95), ("p99", 0.99))},
        "gap_mean": float(values.mean()),
        "share_below_leverage": share,
        "centred_span": 2.0,
        "logit_span_over_epsilon": 2.0 / eps,
        "log_pi_span": float(torch.log(marginal.max() / marginal.min())),
        "verdict": ("NO_LEVERAGE" if share < 1e-4 else "PARTIAL_LEVERAGE"),
        "note": ("a row changes its code only if its top-1/top-2 gap is below "
                 "log(pi_max/pi_min) * eps; the share below that threshold is the most "
                 "the column marginal can move without touching the cost"),
    }
    # does the frozen iteration count reach the target at all? Same cost, more sweeps.
    convergence = []
    with torch.no_grad():
        for sweeps in (50, 200, 500, 2000):
            target = marginal.to(centred.device)
            plan = prior.structured_sinkhorn(centred, eps, sweeps, target)
            convergence.append({"iterations": sweeps,
                                "max_relative_error": float(prior.realised_error(plan, target)),
                                "max_row_deviation": float(prior.realised_rows(plan, target).max())})
    report = {**report, "iteration_sweep": convergence,
              "frozen_iterations": int(model.rq.levels[-1].sk_iters)}

    path = root / "results" / "research" / "issue150" / "marginal_leverage.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
