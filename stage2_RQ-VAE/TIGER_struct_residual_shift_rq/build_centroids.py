"""The frozen centroid table of issue #153, plus the baseline denominators of its gates.

Read-only on the frozen baseline: on the frozen 8192-row TRAIN probe, the per-level
pre-quantisation residual (centred units) of every row is collected, and for each
structure layer g and level l in {1,2} the mean centroid, the median residual norm and
the derived alpha are written to results/research/issue153/centroids.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

TREE = Path(__file__).resolve().parent
if str(TREE) not in sys.path:
    sys.path.insert(0, str(TREE))

def load_item_layers(root: Path) -> dict:
    raw = np.load(root / "results" / "research" / "issue150" / "structure_prior.npz",
                  allow_pickle=True)
    return {int(item): int(layer) for item, layer in zip(raw["item_ids"], raw["layer"])}


def load_partition(root: Path) -> dict:
    # frozen_partition.json only points at the binary table; the table itself is the npz
    manifest = json.loads((root / "results" / "research" / "issue150"
                           / "frozen_partition.json").read_text())
    raw = np.load(root / "results" / "research" / "issue150" / "frozen_partition.npz",
                  allow_pickle=True)
    return {"partition": [list(map(int, bucket)) for bucket in raw["partition"]],
            "marginal": raw["marginal"],
            "sha256": manifest["table_sha256"]}
from model.mixed_rq import split  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402
from model.poincare_contract import expmap0  # noqa: E402

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
    layers = load_item_layers(root)
    frozen = load_partition(root)
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
    generator = torch.Generator().manual_seed(153)
    pick = torch.randperm(len(train_ids), generator=generator)[:ROWS].numpy()
    picked = train_ids[pick]
    groups = torch.tensor([layers.get(int(item), 0) for item in picked], dtype=torch.long,
                          device=device)
    batch = rows[torch.from_numpy(picked)].to(device)
    counts = torch.bincount(groups.cpu(), minlength=len(frozen["partition"]))
    print("layer counts on the probe:", counts.tolist(), flush=True)
    if int(counts.min()) == 0:
        raise SystemExit("a structure layer has no probed row: the id->layer map missed")

    model.eval()
    with torch.no_grad():
        encoded = model.encoder(batch)
        z_e, z_t = split(encoded)
        residual_e, residual_h = z_e, expmap0(z_t, model.rq.curvature)
        per_level = []
        tokens = []
        for level, layer in enumerate(model.rq.levels, start=1):
            alpha = model.rq.alpha_table.lookups(level, []).to(device)
            (residual_e, residual_h), _, _, index, _, _ = layer(residual_e, residual_h, alpha)
            per_level.append(residual_h.detach().clone())
            tokens.append(index.detach().clone())

    centroids, medians, alphas = {}, {}, {}
    for level in (1, 2):
        residual = per_level[level - 1].double()
        for layer_id in range(len(frozen["partition"])):
            here = residual[groups == layer_id]
            centroid = here.mean(dim=0)
            norm = float(centroid.norm())
            median = float(here.norm(dim=-1).median())
            centroids[f"{layer_id}_{level}"] = centroid.tolist()
            medians[f"{layer_id}_{level}"] = median
            alphas[f"{layer_id}_{level}"] = norm / (norm + median)
    pairwise = {}
    for level in (1, 2):
        keys = [f"{layer_id}_{level}" for layer_id in range(len(frozen["partition"]))]
        worst = 0.0
        for first in range(len(keys)):
            for second in range(first + 1, len(keys)):
                left = torch.tensor(centroids[keys[first]]).double()
                right = torch.tensor(centroids[keys[second]]).double()
                relative = float((left - right).norm() / max(float(left.norm()), 1e-12))
                worst = min(worst, relative) if worst else relative
        pairwise[f"level_{level}"] = worst
    baseline_codes = tokens[-1].int().cpu()
    baseline_hist = {"codes_used": int(torch.unique(baseline_codes).numel()),
                     "max_share": float(torch.bincount(baseline_codes, minlength=256).max()
                                        / float(baseline_codes.numel()))}
    payload = {
        "issue": 153,
        "rows": ROWS,
        "seed": 153,
        "partition_sha256": frozen["sha256"],
        "structure_prior_sha256": hashlib.sha256(
            (root / "results" / "research" / "issue150" / "structure_prior.npz").read_bytes()
        ).hexdigest(),
        "alpha_rule": "alpha = ||c|| / (||c|| + median||r||), both measured on this probe",
        "centroids": centroids,
        "median_residual_norm": medians,
        "alpha": alphas,
        "layer_pairwise_worst_relative": pairwise,
        "baseline_last_level": baseline_hist,
    }
    out = root / "results" / "research" / "issue153" / "centroids.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8")
    print(json.dumps({"rows": ROWS, "alphas": alphas, "pairwise": pairwise,
                      "baseline": baseline_hist,
                      "norm": [float(torch.tensor(v).norm()) for v in centroids.values()],
                      "sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
