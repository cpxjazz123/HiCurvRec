"""Pre-training calibration for issue #238.  Run once, before the single Stage2 run.

Everything that could otherwise be chosen by looking at results is measured here and frozen into
`calibration.json`:

1. **mapping scale** of the mixed distance — the pooled RMS norm of the *frozen baseline*
   encoder output on the train inputs.  This sets how far the ball/sphere pushes points, so it must
   come from data, not from a sweep.
2. **global scale s** = mean|K_G| / mean|K_q| with K_q measured on the *frozen baseline's hard q*
   at that mapping scale.  This is what puts prediction and target in comparable units; it is
   computed once and never re-derived during training.
3. **Huber delta** — the median absolute deviation of K_G, i.e. the natural robust scale of the
   target itself.
4. **triplet coverage and strata** — how many triplets exist, how many are usable, and the
   positive / negative / near-zero split.

The baseline checkpoint is the reference quantiser because the issue asks for K_q on the hard q of
the frozen TIGER model before supervision starts.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _repository_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "stage2_RQ-VAE").is_dir():
            return parent
    raise RuntimeError("cannot locate the repository root above " + str(here))


REPO_ROOT = _repository_root()
ARM_NAME = "TIGER_curvloss_rq"
RESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME
BASELINE_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / "TIGER_RQ-VAE"
GRAPH_DIR = REPO_ROOT / "results" / "research" / "behavior_graph"
EMBEDDING_FILE = REPO_ROOT / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN_FILE = REPO_ROOT / "results" / "stage0_build_parquet" / "train.parquet"

MAX_TRIPLETS_PER_NODE = 8
SEED = 42
N_REFS = 64


def _csr(path: Path):
    import scipy.sparse as sp

    payload = np.load(path, allow_pickle=True)
    n = int(payload["triangles_per_node"].shape[0])
    # graph.npz stores CSR arrays; rebuild symmetric binary adjacency from the edge table instead,
    # which is the representation `metadata.json` defines.
    import pandas as pd

    edges = pd.read_parquet(GRAPH_DIR / "edges.parquet")
    src = edges.iloc[:, 0].to_numpy()
    dst = edges.iloc[:, 1].to_numpy()
    a = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    return ((a + a.T) > 0).astype(np.float64).tocsr()


def main() -> int:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import graph_curv
    from distances import build_distance, mixed
    from model.model import RQVAE
    from model.layers import RQLayer
    import train_rqvae as entry

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "calibration.json"
    if out_path.exists() and "--force" not in sys.argv:
        print(json.dumps({"refused": f"calibration already exists at {out_path}; frozen"}))
        return 2

    # --- 1. graph target -------------------------------------------------------
    graph = _csr(GRAPH_DIR / "graph.npz")
    triplets = graph_curv.build_triplets(graph, max_per_node=MAX_TRIPLETS_PER_NODE, seed=SEED)
    refs = graph_curv.reference_nodes(graph.shape[0], n_refs=N_REFS, seed=SEED)
    dists = graph_curv.graph_distances(graph, refs)
    k_g = graph_curv.sectional_curvature(dists, triplets, refs)
    masks = graph_curv.strata(k_g)
    usable = np.isfinite(k_g)
    if usable.sum() < 1000:
        raise SystemExit(f"too few usable triplets: {int(usable.sum())}")
    k_g_use = k_g[usable]

    # --- 2. frozen baseline hard q --------------------------------------------
    payload = torch.load(BASELINE_DIR / "rqvae_best.pth", map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    config = entry._tokenizer_config()
    model = RQVAE(config, in_dim=int(state["encoder.mlp.1.weight"].shape[1]),
                  distance=build_distance(None, name="euclidean"))   # baseline metric for q
    model.load_state_dict(state, strict=False)
    model.eval()

    embeddings = entry.load_embeddings(EMBEDDING_FILE)
    vectors = torch.from_numpy(entry.maybe_apply_pca(embeddings, entry.PCA_DIM, entry.SEED))
    with torch.no_grad():
        latent = model.encoder(vectors)
        indices = model.rq(latent)[3]                     # (N, 3) hard codes
        hard_q = torch.zeros_like(latent)
        residual = latent
        for level, layer in enumerate(model.rq.vq_layers):
            code = layer.get_code_embs()[indices[:, level]]
            hard_q = hard_q + code
            residual = residual - code
        # `residual` is now the L3 residual: exactly what the mixed distance is evaluated on,
        # because the balanced level quantises the residual, not the encoder output.

    # --- 3. mapping scale from the L3 residual the metric actually sees -------
    # Calibrating on the encoder output instead saturates tanh: the L3 residual is orders of
    # magnitude smaller, so every point collapsed to the same ball radius and the assignment
    # matrix became constant (the first three smokes failed on exactly that).
    spans = []
    offset = 0
    for d in mixed.DIMS:
        spans.append(slice(offset, offset + d))
        offset += d
    rms = {name: float(residual[:, span].pow(2).mean().sqrt())
           for name, span in zip(("hyperbolic", "euclidean", "spherical"), spans)}
    mapping_scale = float(np.mean(list(rms.values())))
    if not mapping_scale > 0:
        raise SystemExit("the L3 residual is degenerate; refusing to calibrate on it")
    mixed.set_scale(mapping_scale)

    # --- 4. K_q on the frozen baseline's hard q -------------------------------
    with torch.no_grad():
        def pair_distance(x, y):
            return mixed.pairwise_sq(x, y).clamp_min(0.0).sqrt()

        # K_q uses the same formula and the same reference *set*, with D_q in place of d_G.
        # The reference nodes are real items, so their D_q coordinates are the same hard q rows.
        idx_m = torch.as_tensor(triplets["m"], dtype=torch.long)
        idx_b = torch.as_tensor(triplets["b"], dtype=torch.long)
        idx_c = torch.as_tensor(triplets["c"], dtype=torch.long)
        ref_idx = torch.as_tensor(refs, dtype=torch.long)
        k_q = torch.full((len(idx_m),), float("nan"))
        for r in range(0, len(idx_m), 4096):
            sl = slice(r, r + 4096)
            m, b, c = hard_q[idx_m[sl]], hard_q[idx_b[sl]], hard_q[idx_c[sl]]
            terms = []
            # D_q is a product metric, so it cannot be delegated to cdist; the reference block is
            # (R, T) and the triplet-internal distance is element-wise (T,).
            # squared distances throughout: a sqrt at zero distance has an infinite derivative
            d2_bc = mixed.pairwise_sq(b, c).clamp_min(0.0)
            for a in ref_idx.split(16):
                a_pt = hard_q[a]
                d2_am = mixed.distances(a_pt, m)
                d2_ab = mixed.distances(a_pt, b)
                d2_ac = mixed.distances(a_pt, c)
                terms.append(d2_am + d2_bc.unsqueeze(0) / 4.0 - (d2_ab + d2_ac) / 2.0)
            k_q[sl] = torch.cat(terms, dim=0).mean(dim=0)
        k_q_np = k_q.numpy()

    ok_q = np.isfinite(k_q_np) & usable
    if ok_q.sum() < 1000:
        raise SystemExit(f"too few K_q values: {int(ok_q.sum())}")
    global_scale = float(np.mean(np.abs(k_g[ok_q])) / max(np.mean(np.abs(k_q_np[ok_q])), 1e-12))
    delta = float(np.median(np.abs(k_g_use - np.median(k_g_use))))

    calibration = {
        "schema": "genrec.issue238.calibration/1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_before_training": True,
        "graph": {
            "edges": str(GRAPH_DIR / "edges.parquet"),
            "nodes": int(graph.shape[0]),
            "triplets": int(len(triplets["m"])),
            "usable": int(usable.sum()),
            "unusable": int((~usable).sum()),
            "strata": {k: int(v.sum()) for k, v in masks.items()},
            "max_triplets_per_node": MAX_TRIPLETS_PER_NODE,
            "n_references": int(len(refs)),
            "seed": SEED,
        },
        "distance": {
            "dims": list(mixed.DIMS),
            "weights": list(mixed.WEIGHTS),
            "curvature": list(mixed.CURVATURE),
            "mapping_scale": mapping_scale,
            "block_rms": rms,
            "calibrated_on": "frozen baseline L3 residual (the tensor the balanced level sees)",
        },
        "target": {
            "global_scale_s": global_scale,
            "huber_delta": delta,
            "k_g_mean": float(np.mean(k_g_use)),
            "k_g_sd": float(np.std(k_g_use)),
            "k_g_quantiles": [float(np.quantile(k_g_use, q)) for q in (0.05, 0.5, 0.95)],
        },
        "baseline_reference": {
            "checkpoint": str(BASELINE_DIR / "rqvae_best.pth"),
            "quantiser": "euclidean (frozen baseline assignment)",
        },
    }
    out_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps(calibration, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
