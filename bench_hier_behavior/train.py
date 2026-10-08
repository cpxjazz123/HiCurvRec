"""Train and score one (arm, dimension, tree, mix, data arm) cell.

Every run writes its native artefacts under
``results/bench_hier_behavior/<run key>/``: a training curve, the item SIDs and
a metrics file. Nothing here reads another run's output.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import bench_config as cfg
from . import metrics as M
from .models import RQVAE, cone_margin_loss
from .synth_data import build_dataset, cone_edges


def run_key(arm: str, dim: int, tree_shape: str, mix_name: str,
            data_arm: str) -> str:
    return f"{arm}_d{dim}_{tree_shape}_{mix_name}_{data_arm}"


def _item_level_edges(tree, pairs: np.ndarray) -> np.ndarray:
    """Supervision edges whose child is an item (the interest -> item level).

    Returns ``(interest index, item id)``. Node ids in the tree are global and
    level-offset, so both columns are rebased to their own index space here.
    """
    item_offset = int(tree.level_offsets[tree.interest_level + 1])
    pairs = pairs[pairs[:, 1] >= item_offset].copy()
    pairs[:, 0] -= int(tree.level_offsets[tree.interest_level])
    pairs[:, 1] -= item_offset
    return pairs


def run_one(
    arm: str,
    dim: int,
    tree_shape: str,
    mix_name: str = cfg.DEFAULT_MIX,
    data_arm: str = "tree",
    seed: int = cfg.SEED,
    device: str = "cuda",
    epochs: int | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    started = time.time()

    dataset = build_dataset(tree_shape, mix_name, data_arm, seed)
    tree = dataset.tree
    features = torch.as_tensor(dataset.features, dtype=torch.float32, device=device)
    train_pairs, test_pairs = cone_edges(tree, seed)
    item_train_pairs = _item_level_edges(tree, train_pairs)

    interest_items = torch.as_tensor(dataset.interest_items, dtype=torch.long,
                                     device=device)
    edge_parents = torch.as_tensor(item_train_pairs[:, 0], dtype=torch.long,
                                   device=device)
    edge_children = torch.as_tensor(item_train_pairs[:, 1], dtype=torch.long,
                                    device=device)

    model = RQVAE(cfg.ARM_GEOMETRY[arm], cfg.FEATURE_DIM, dim).to(device)
    model.init_codebooks(features, seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    hierarchy = cfg.ARM_HIERARCHY_LOSS[arm]
    total_epochs = cfg.EPOCHS if epochs is None else epochs

    generator = torch.Generator(device=device).manual_seed(seed)
    curve = []
    for epoch in range(total_epochs):
        model.train()
        order = torch.randperm(features.shape[0], device=device,
                               generator=generator)
        running = np.zeros(3, dtype=np.float64)
        steps = 0
        for start in range(0, len(order), cfg.BATCH_SIZE):
            batch = order[start:start + cfg.BATCH_SIZE]
            reconstructed, quant_loss, _, _ = model(features[batch])
            loss = F.mse_loss(reconstructed, features[batch]) + quant_loss
            if hierarchy:
                loss = loss + cfg.EDGE_LOSS_WEIGHT * _hierarchy_step(
                    model, features, interest_items,
                    edge_parents, edge_children, generator,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += (loss.item(), quant_loss.item(), float(len(batch)))
            steps += 1
        curve.append(
            {"epoch": epoch, "loss": running[0] / steps,
             "quant_loss": running[1] / steps, "items": int(running[2])}
        )

    model.eval()
    with torch.no_grad():
        latent = model.encode(features)
        _, _, tokens, _ = model(features)
    latent_np = latent.cpu().numpy()
    tokens_np = tokens.cpu().numpy()

    results = {"arm": arm, "dim": dim, "tree": tree_shape, "mix": mix_name,
               "data_arm": data_arm, "seed": seed,
               "train_seconds": time.time() - started}
    results.update(M.dataset_stats(dataset))
    results.update(M.distortion(tree, model.geometry, latent_np, seed))
    results.update(M.sid_quality(tree, tokens_np, seed))
    results.update(
        M.cone_quality(tree, model.geometry, latent_np, train_pairs, test_pairs)
    )
    results.update(M.behaviour_ranking(dataset, tokens_np, seed))
    results["final_loss"] = curve[-1]["loss"]
    with torch.no_grad():
        results["final_recon_mse"] = float(
            F.mse_loss(model(features)[0], features).item()
        )

    out_dir = Path(cfg.RESULT_ROOT) / run_key(arm, dim, tree_shape, mix_name,
                                              data_arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    with (out_dir / "training_metrics.jsonl").open("w") as handle:
        for row in curve:
            handle.write(json.dumps(row) + "\n")
    np.save(out_dir / "sids.npy", tokens_np.astype(np.int16))
    np.save(out_dir / "latent.npy", latent_np.astype(np.float32))
    return results


def _hierarchy_step(
    model: RQVAE,
    features: torch.Tensor,
    interest_items: torch.Tensor,
    edge_parents: torch.Tensor,
    edge_children: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """One max-margin cone step; adds no parameters beyond the encoder."""
    picks = torch.randint(len(edge_parents), (cfg.EDGE_BATCH_SIZE,),
                          device=features.device, generator=generator)
    parents = edge_parents[picks]
    children = edge_children[picks]

    size = interest_items.shape[1]
    offsets = torch.randint(
        size, (cfg.EDGE_BATCH_SIZE, cfg.EDGE_APEX_SAMPLE),
        device=features.device, generator=generator,
    )
    apex_items = interest_items[parents.unsqueeze(1), offsets]
    apex_tangent = model.encode(features[apex_items]).mean(dim=1)

    n_interest = interest_items.shape[0]
    offset_nodes = torch.randint(
        1, n_interest, (cfg.EDGE_BATCH_SIZE,), device=features.device,
        generator=generator,
    )
    negative_nodes = (parents + offset_nodes) % n_interest
    negative_items = interest_items[
        negative_nodes,
        torch.randint(size, (cfg.EDGE_BATCH_SIZE,), device=features.device,
                      generator=generator),
    ]
    return cone_margin_loss(
        model.geometry_name, model.geometry, apex_tangent,
        model.encode(features[children]),
        model.encode(features[negative_items]),
    )
