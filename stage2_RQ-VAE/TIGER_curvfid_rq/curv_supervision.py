"""Curvature supervision for issue #238: sample triplets, evaluate K_q on hard q.

`L_curv` is defined on the *hard* quantised vector `q = e1 + e2 + e3`, which is a function of the
whole catalogue, not of one minibatch.  Training on the minibatch's own items would almost never
contain a complete `(m, b, c)` triple, so each step samples a fixed-size triplet batch (seeded by
the step index) and runs one extra forward over just the nodes it needs — the references plus the
unique triplet nodes, a few hundred items.  That keeps the supervision global in its target and
local in its cost.

Everything the loss depends on that is not a parameter (triplet construction, reference set, the
Huber delta, the global scale) comes from `calibration.json`, produced before training started.
"""
from __future__ import annotations

import numpy as np
import torch

#: Triplets supervised per optimisation step.  Fixed by the frozen pipeline, not tuned.
TRIPLETS_PER_STEP = 128


def _csr(edges_path, n_items: int):
    import pandas as pd
    import scipy.sparse as sp

    edges = pd.read_parquet(edges_path)
    src = edges.iloc[:, 0].to_numpy()
    dst = edges.iloc[:, 1].to_numpy()
    a = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n_items, n_items))
    return ((a + a.T) > 0).astype(np.float64).tocsr()


def prepare(calibration: dict, graph_dir, n_items: int, seed: int = 42) -> dict:
    """Rebuild the triplets and the graph target, deterministically, from the frozen recipe."""
    import graph_curv

    graph_spec = calibration["graph"]
    graph = _csr(graph_dir / "edges.parquet", n_items)
    triplets = graph_curv.build_triplets(
        graph, max_per_node=int(graph_spec["max_triplets_per_node"]), seed=int(graph_spec["seed"]))
    refs = graph_curv.reference_nodes(n_items, n_refs=int(graph_spec["n_references"]),
                                      seed=int(graph_spec["seed"]))
    dists = graph_curv.graph_distances(graph, refs)
    k_g = graph_curv.sectional_curvature(dists, triplets, refs)
    usable = np.isfinite(k_g)
    triplets = {key: value[usable] for key, value in triplets.items()}
    k_g = k_g[usable]
    if len(k_g) != int(graph_spec["usable"]):
        raise RuntimeError(
            "the rebuilt triplets do not match the calibrated count (%d vs %d): the frozen recipe "
            "was not reproduced" % (len(k_g), int(graph_spec["usable"])))
    return {
        "m": torch.as_tensor(triplets["m"], dtype=torch.long),
        "b": torch.as_tensor(triplets["b"], dtype=torch.long),
        "c": torch.as_tensor(triplets["c"], dtype=torch.long),
        "k_g": torch.as_tensor(k_g, dtype=torch.float32),
        "refs": torch.as_tensor(refs, dtype=torch.long),
    }


def sample_indices(targets: dict, step: int, size: int = TRIPLETS_PER_STEP,
                   seed: int = 42) -> torch.Tensor:
    """A reproducible per-step subsample; the seed depends only on the step index."""
    n = targets["k_g"].numel()
    generator = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(step))
    return torch.randperm(n, generator=generator)[:int(min(size, n))]


def curvature_of_nodes(encode_quantised, targets: dict, indices: torch.Tensor,
                       device) -> torch.Tensor:
    """`K_q` for the sampled triplets, evaluated on the hard quantised vectors.

    `encode_quantised(ids)` must return `q` for those item ids with the straight-through path
    intact; the caller owns whether that is the training model or a frozen copy.
    """
    from distances import mixed

    # the triplet tables live on the host; index them there and move only the rows needed
    host = indices.cpu() if indices.is_cuda else indices
    m = targets["m"][host].to(device)
    b = targets["b"][host].to(device)
    c = targets["c"][host].to(device)
    refs = targets["refs"].to(device)

    needed = torch.unique(torch.cat([m, b, c, refs]))
    q = encode_quantised(needed)
    lookup = {int(node): q[i] for i, node in enumerate(needed.tolist())}

    def rows(ids):
        return torch.stack([lookup[int(i)] for i in ids.tolist()], dim=0)

    q_m, q_b, q_c, q_a = rows(m), rows(b), rows(c), rows(refs)

    # The formula only ever uses squared distances, so they stay squared: taking a square root first
    # gives an infinite derivative whenever two points coincide (items sharing a codeword have
    # exactly the same q), and that turned three parameter gradients into NaN at step 1 of the
    # first attempt.  No sqrt, no singularity.
    d2_am = mixed.distances(q_a, q_m)                              # (R, T)
    d2_ab = mixed.distances(q_a, q_b)
    d2_ac = mixed.distances(q_a, q_c)
    d2_bc = mixed.pairwise_sq(q_b, q_c).clamp_min(0.0)             # (T,)
    k_q = (d2_am + d2_bc.unsqueeze(0) / 4.0
           - (d2_ab + d2_ac) / 2.0).mean(dim=0)
    return k_q


def q_lookup_from_model(model, embeddings: torch.Tensor, device):
    """Return a function mapping item ids to their hard `q` under **argmin** quantisation.

    The residual stack is walked explicitly instead of calling `model.forward`, because at the
    balanced level `forward` runs Sinkhorn, and Sinkhorn's global centring is undefined when the
    cost matrix is constant.  That is not hypothetical: the supervised triplets are built around
    high-degree nodes, whose L3 residuals are nearly identical, and the third smoke attempt died on
    `Cannot center a constant distance matrix` for exactly that subset.

    Argmin is also the right assignment for the target: it is what the exported SID uses
    (`infer_use_sk=False`), i.e. the discrete object Stage3 actually reads.  The encoder still runs
    in training mode, so batch-normalisation statistics match the step that produced the latents,
    and the codeword lookup keeps the straight-through path intact for the gradient check.
    """
    from distances import mixed

    def encode_quantised(ids: torch.Tensor) -> torch.Tensor:
        rows = embeddings[ids.to(embeddings.device)].to(device)
        latent = model.encoder(rows)
        residual = latent
        quantised = torch.zeros_like(latent)
        for layer in model.rq.vq_layers:
            codes = layer.get_code_embs()
            index = mixed.distances(residual, codes).argmin(dim=1)
            code = codes[index]
            quantised = quantised + code
            residual = residual - code
        return quantised

    return encode_quantised
