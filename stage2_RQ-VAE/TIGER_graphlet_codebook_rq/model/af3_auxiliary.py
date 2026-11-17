"""A-arm adapter for the existing four-term AF3 hierarchy supervision.

The auxiliary soft readout preserves the mixed-cost/tangent-residual adaptation
introduced in 87f2563. It does NOT replace the native mixed hard assignment.
Sample counts, temperatures, four unit loss weights, linear
1500->3000 ramp and 0.10 gradient share are the original protocol. Graph IDs
are explicitly translated to catalog IDs, and negative pools exclude TRAIN
edges/self-pairs instead of silently labelling them as unrelated.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from .af3_hierarchy import GradientBudget, af3_weights, gradient_norm, hierarchical_loss

EDGES_PER_STEP = 192
PAIRS_PER_STEP = 192
START_EPOCH = 1500
END_EPOCH = 3000
GRADIENT_SHARE = 0.10
LOSS_WEIGHTS = {"level1": 1.0, "level2": 1.0, "level3": 1.0, "balance": 1.0}


def prepare_state(graph: dict, all_embeddings: torch.Tensor, device: torch.device) -> dict:
    item_ids = np.asarray(graph["item_ids"], dtype=np.int64)
    if (len(item_ids) != graph["n_nodes"] or len(np.unique(item_ids)) != len(item_ids)
            or item_ids.min() < 0 or item_ids.max() >= len(all_embeddings)):
        raise ValueError("invalid TRAIN graph position-to-item mapping")
    rows = torch.as_tensor(item_ids[graph["rows"]], dtype=torch.long, device=device)
    cols = torch.as_tensor(item_ids[graph["cols"]], dtype=torch.long, device=device)
    weight = af3_weights(torch.as_tensor(graph["weights"], dtype=torch.float32, device=device),
                         graph["af3_median"], graph["af3_p90"])
    dense = weight > 0
    edge_set = {tuple(sorted((int(i), int(j)))) for i, j in zip(rows.cpu(), cols.cpu())}
    return {"rows": rows, "cols": cols, "af3_weight": weight,
            "dense_rows": rows[dense], "dense_cols": cols[dense], "dense_weight": weight[dense],
            "all_embeddings": all_embeddings, "item_ids": item_ids, "edge_set": edge_set,
            "n_items": len(all_embeddings), "n_edges": len(rows), "n_dense": int(dense.sum()),
            "af3_median": graph["af3_median"], "af3_p90": graph["af3_p90"]}


def sample_hierarchical_pairs(state: dict, rng) -> dict:
    device = state["rows"].device
    if not state["n_edges"]:
        raise ValueError("AF3 auxiliary supervision requires TRAIN edges")
    dense_count = state["n_dense"]
    dense_choice = torch.as_tensor(rng.integers(dense_count, size=EDGES_PER_STEP)
                                   if dense_count else [], dtype=torch.long, device=device)
    edge_choice = torch.as_tensor(rng.integers(state["n_edges"], size=PAIRS_PER_STEP * 4),
                                  dtype=torch.long, device=device)
    ids, seen = state["item_ids"], state["edge_set"]
    unrelated = []
    requested = PAIRS_PER_STEP * 4
    attempts = 0
    while len(unrelated) < requested and attempts < 80 * requested:
        left, right = ids[rng.integers(len(ids), size=2)]
        attempts += 1
        if left == right or tuple(sorted((int(left), int(right)))) in seen:
            continue
        unrelated.append((int(left), int(right)))
    if len(unrelated) != requested:
        raise RuntimeError("insufficient TRAIN non-edges for the fixed auxiliary pools")
    random_pairs = torch.as_tensor(unrelated, dtype=torch.long, device=device)
    return {"af3": torch.stack([state["dense_rows"][dense_choice],
                                state["dense_cols"][dense_choice]], dim=1),
            "af3_weight": state["dense_weight"][dense_choice],
            "positive": torch.stack([state["rows"][edge_choice], state["cols"][edge_choice]], dim=1),
            "negative": random_pairs[:PAIRS_PER_STEP * 2],
            "unrelated": random_pairs[PAIRS_PER_STEP * 2:]}


def augment_loss(base_loss: torch.Tensor, raw_module, state: dict | None,
                  epoch: int, rng) -> tuple[torch.Tensor, dict]:
    ramp = min(1.0, max(0.0, (epoch - START_EPOCH) / float(END_EPOCH - START_EPOCH)))
    if state is None or ramp == 0.0:
        return base_loss, {"ramp": ramp, "active": False}
    device = next(raw_module.parameters()).device
    picked = sample_hierarchical_pairs(state, rng)
    endpoints = torch.cat([picked[name] for name in ("af3", "positive", "negative", "unrelated")])
    embeddings = state["all_embeddings"]
    flat = embeddings[endpoints.reshape(-1).to(embeddings.device)].to(device)
    latent = raw_module.encoder(flat)
    codebooks = [layer.get_code_embs() for layer in raw_module.rq.vq_layers]
    batch = {"af3_weight": picked["af3_weight"]}
    offset = 0
    for name in ("af3", "positive", "negative", "unrelated"):
        count = len(picked[name])
        left = torch.arange(offset, offset + count, dtype=torch.long, device=device) * 2
        batch[name + "_left"], batch[name + "_right"] = left, left + 1
        offset += count
    auxiliary, parts = hierarchical_loss(latent, codebooks, batch, LOSS_WEIGHTS,
                                          curvature=raw_module.rq.curvature)
    if not torch.isfinite(auxiliary):
        raise RuntimeError("non-finite A-arm auxiliary loss")
    encoder = list(raw_module.encoder.parameters())
    base_encoder, base_codebook = gradient_norm(base_loss, encoder), gradient_norm(base_loss, codebooks)
    new_encoder, new_codebook = gradient_norm(auxiliary, encoder), gradient_norm(auxiliary, codebooks)
    budget = GradientBudget(share=GRADIENT_SHARE)
    scale = budget.observe(base_encoder, new_encoder, base_codebook, new_codebook)
    scale_tensor = torch.tensor(scale, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(scale_tensor, op=dist.ReduceOp.MIN)
    scale = float(scale_tensor)
    log = {name: parts[name] for name in LOSS_WEIGHTS}
    log.update(ramp=ramp, scale=scale, active=True,
               base_encoder_norm=base_encoder, base_codebook_norm=base_codebook,
               auxiliary_encoder_norm=new_encoder, auxiliary_codebook_norm=new_codebook)
    return base_loss + ramp * scale * auxiliary, log
