"""Issue #198 part B: where the curvature enters, and what it does to the gradient.

Three things, all read-only on the shipped checkpoint: the exact lines the curvature passes through (and the single place that keeps it out of levels two
and three), the per-block gradient norms of one real batch with the two-regime curvature against the same batch with curvature 1 everywhere, and the curvature
quantiles of the twenty-eight enabled items together with how often they land in a batch.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TREE = REPO_ROOT / "stage2_RQ-VAE" / "TIGER_sign_rq"
RESULT = REPO_ROOT / "results" / "stage2_RQ-VAE" / "TIGER_sign_rq"
BATCH = 1024
BATCHES = 20
SEED = 42


def load():
    sys.path.insert(0, str(TREE))
    spec = importlib.util.spec_from_file_location("arm_sign", TREE / "train_rqvae.py")
    arm = importlib.util.module_from_spec(spec)
    sys.modules["arm_sign"] = arm
    spec.loader.exec_module(arm)
    from model import RQVAE, sign_metric
    return arm, RQVAE, sign_metric


def source_lines() -> dict:
    """The hook's own line numbers, quoted rather than paraphrased."""
    out = {}
    model = (TREE / "model" / "model.py").read_text().splitlines()
    out["model.py kappa from ids"] = [f"{i+1}: {l.strip()}" for i, l in enumerate(model)
                                       if "regime_kappa" in l or "self.rq(encoded, kappa=kappa)" in l]
    mixed = (TREE / "model" / "mixed_rq.py").read_text().splitlines()
    out["mixed_rq.py level routing"] = [f"{i+1}: {l.strip()}" for i, l in enumerate(mixed)
                                        if "kappa if level == 1 else None" in l]
    out["mixed_rq.py cost use"] = [f"{i+1}: {l.strip()}" for i, l in enumerate(mixed)
                                   if "curved_squared_paired" in l or "mixed_cost_with_kappa(residual_e" in l]
    sign = (TREE / "model" / "sign_metric.py").read_text().splitlines()
    out["sign_metric.py regime"] = [f"{i+1}: {l.strip()}" for i, l in enumerate(sign)
                                    if "def regime_kappa" in l or "enabled = bool" in l or "clamp_min(KAPPA_FLOOR)" in l]
    return out


def block_norms(model, batch, ids, kappa) -> dict:
    model.zero_grad(set_to_none=True)
    reconstructed, quant_loss, _, _ = model(batch, item_ids=ids)
    loss, recon = model.compute_loss(batch, reconstructed, quant_loss)
    loss.backward()
    groups = {"encoder": [], "decoder": [], "codebook": [], "other": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.norm())
        if name.startswith("encoder."):
            groups["encoder"].append(norm)
        elif name.startswith("decoder."):
            groups["decoder"].append(norm)
        elif ".embed" in name:
            groups["codebook"].append(norm)
        else:
            groups["other"].append(norm)
    return {"loss": float(loss.detach()), "recon": float(recon.detach()),
            **{f"{k}_norm": float(np.linalg.norm(v)) if v else 0.0 for k, v in groups.items()},
            **{f"{k}_tensors": len(v) for k, v in groups.items()}}


def main() -> int:
    arm, RQVAE, sign = load()
    embeddings = arm.load_embeddings(arm.EMBEDDING_FILE)
    frame = arm.pd.read_parquet(arm.TRAIN_FILE, columns=["seen_history", "target"])
    train_ids = np.unique(frame["target"].to_numpy(dtype=np.int64))
    edges = sorted({(int(h[-1]), int(t)) for h, t in
                    zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64))
                    if h is not None and len(h) > 0})
    groups = sign.components(sign.closed_subgraph(edges))
    delta = sign.delta_of(edges, sign.af3_weights(edges), groups)
    enabled = sorted(i for i, v in delta.items() if v is not None and v > 0.0)

    table = sign.load_table()
    kappa_enabled = [float(table["kappa"].get(str(i), table["mean_frozen"])) for i in enabled]

    model = RQVAE(arm._tokenizer_config(), in_dim=embeddings.shape[1])
    model.load_state_dict(torch.load(RESULT / "rqvae_best.pth", map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    model.delta = delta

    all_embeddings = torch.from_numpy(np.ascontiguousarray(embeddings)).float()
    rng = np.random.default_rng(SEED)
    batch_ids = torch.from_numpy(np.sort(rng.choice(train_ids, size=BATCH, replace=False)))
    batch = all_embeddings[batch_ids]

    with_gate = block_norms(model, batch, batch_ids, None)          # delta is set, so the gate is live
    saved = model.delta
    model.delta = None                                              # the same batch with curvature 1 everywhere
    baseline = block_norms(model, batch, batch_ids, None)
    model.delta = saved

    hits = []
    for _ in range(BATCHES):
        sample = set(rng.choice(train_ids, size=BATCH, replace=False).tolist())
        hits.append(len(sample & set(enabled)))

    kappa_all = sign.regime_kappa(torch.from_numpy(train_ids), delta, table)
    report = {
        "enabled_items": len(enabled),
        "enabled_item_ids_first": enabled[:10],
        "kappa_of_enabled": {"min": float(min(kappa_enabled)), "q25": float(np.quantile(kappa_enabled, 0.25)),
                             "median": float(np.median(kappa_enabled)), "q75": float(np.quantile(kappa_enabled, 0.75)),
                             "max": float(max(kappa_enabled)), "mean": float(np.mean(kappa_enabled))},
        "kappa_rows_uniform_one_share": float((kappa_all == 1.0).float().mean()),
        "batch_size": BATCH,
        "enabled_hits_per_batch": {"samples": hits, "mean": float(np.mean(hits)), "max": int(max(hits)),
                                   "batches_with_none": int(sum(1 for h in hits if h == 0))},
        "expected_items_per_batch": len(enabled) * BATCH / len(train_ids),
        "gradient_norms_gate_on": with_gate,
        "gradient_norms_gate_off": baseline,
        "source_hooks": source_lines(),
    }
    (HERE / "audit_b.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "source_hooks"}, indent=1))
    print(json.dumps(report["source_hooks"], indent=1)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
