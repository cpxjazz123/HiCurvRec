"""Freeze the bucket partition of issue #150's corrected arm, once, before training.

The partition is decided from the **initial latent space**: with the codebooks
initialised and the weights at their starting point, each structure layer's mean
encoder output over the TRAIN items is its centre, and every code goes to the layer
whose centre is nearest under an equal-quota rule. The result, the marginal it
implies and the dispatch-time realizability measurement are written once with their
SHA-256 and never recomputed from the run's own usage.

Run with the Stage 2 interpreter (CLAUDE.md §2). Nothing is written outside the
repository (§86).
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
from model.frozen_prior import (affinity_partition, equal_partition, realised_error,
                                target_marginal)  # noqa: E402
from model.mixed_rq import split  # noqa: E402
from model.model import MixedRQVAE  # noqa: E402
from model.poincare_contract import expmap0  # noqa: E402


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

    layer_of_item = np.full(int(structure["metadata"]["n_items"]), -1, dtype=np.int64)
    layer_of_item[structure["item_ids"]] = structure["layer"]
    with torch.no_grad():
        latent = model.encoder(rows[torch.from_numpy(train_ids)].to(device))
        layers = torch.from_numpy(layer_of_item[train_ids]).to(device)
        centres = torch.stack([latent[layers == group].mean(0)
                               for group in range(prior.GROUPS)])
        codes = model.rq.levels[-1].embed.weight.detach()
        method = "affinity"
        partition = affinity_partition(codes, centres)
        if len({tuple(bucket) for bucket in partition}) < prior.GROUPS:
            method = "equal_fallback"
            partition = equal_partition(int(codes.shape[0]), prior.GROUPS)

    rho = np.asarray(structure["rho"], dtype=np.float64)
    marginal = target_marginal(partition, rho, codes=int(sum(len(b) for b in partition)))

    # Dispatch-time realizability: one forward on a frozen slice, last level only.
    generator = torch.Generator().manual_seed(150)
    pick = torch.randperm(len(train_ids), generator=generator)[:1024].numpy()
    model.rq.attach_frozen_prior({"partition": partition,
                                  "marginal": marginal.numpy(),
                                  "sha256": "preview"})
    # train mode: the balanced assignment (and therefore the plan) only runs while
    # training, which is the regime the marginal is for
    model.train()
    with torch.no_grad():
        encoded = model.encoder(rows[torch.from_numpy(train_ids[pick])].to(device))
        z_e, z_t = split(encoded)
        residual_e, residual_h = z_e, expmap0(z_t, model.rq.curvature)
        for level, layer in enumerate(model.rq.levels, start=1):
            alpha = model.rq.alpha_table.lookups(level, []).to(device)
            record: dict = {}
            (residual_e, residual_h), _, _, index, picked_e, picked_h = layer(
                residual_e, residual_h, alpha, infer_use_sk=True,
                marginal=marginal.to(device) if level == len(model.rq.levels) else None,
                record=record)
        plan = model.rq.last_plan if model.rq.last_plan is not None else record["plan"]
        error = float(realised_error(plan.cpu(), marginal))

    path = prior.partition_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "issue": 150,
        "arm": "TIGER_frozen_prior_marginal",
        "method": method,
        "groups": int(len(partition)),
        "bucket_sizes": [int(len(bucket)) for bucket in partition],
        "rho": [float(value) for value in rho],
        "marginal_min": float(marginal.min()),
        "marginal_max": float(marginal.max()),
        "realised_error": error,
        "realised_tolerance": prior.REALISED_TOLERANCE,
        "structure_table_sha256": structure["sha256"],
        "frozen_before_training": True,
        "note": ("bucket members never change after this file is written; the run's own "
                 "usage is recorded but never used to re-partition (that was #150's "
                 "failure root)"),
    }
    np.savez_compressed(path, partition=np.asarray(partition, dtype=np.int32),
                        marginal=marginal.numpy().astype(np.float64),
                        centres=centres.cpu().numpy().astype(np.float32),
                        metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    manifest = {"table": str(path), "table_sha256": prior.sha256_of(path),
                "metadata": metadata}
    prior.manifest_path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                         encoding="utf-8")
    print(json.dumps({"method": method, "realised_error": error,
                      "tolerance": prior.REALISED_TOLERANCE,
                      "partition_sha256": manifest["table_sha256"]}, indent=2))
    if error > prior.REALISED_TOLERANCE:
        print("REALISABILITY_FAILED: the frozen marginal cannot be realised at the last level")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
