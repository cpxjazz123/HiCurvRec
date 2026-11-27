"""Issue #268 stage A: freeze Z_ref and probe fixed local curvature.

Z_ref is the parent #265 final continuous 32-d encoder output on the frozen
768-d catalogue.  It is read from the immutable run snapshot of the executed
source, never from a new candidate.

The probe evaluates the frozen distortion

    J_t(k) = mean_{pairs in P_t} (d_k(w_i, w_j) / g(i,j) - 1)^2

on a deterministic sample of target rows, where w = z / (2*S) and
S = max_i ||z_i|| so that k=0 equals the Euclidean distance z/S.  It also
verifies the exact node decomposition used by the full fit:

    sum_{r in R} [(d/g - 1)^2] on (m,r),(b,r),(c,r)
      = f(m) + f(b) + f(c) + 3*|R| terms folded into f.

No production code is touched; this is a candidate-specific offline study.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.graph_infra.assets import load_sparse

REPO = Path(__file__).resolve().parents[3]
ISSUE = REPO / "results/research/issue268"
ART = ISSUE / "artifacts"
SRC = (REPO / "results/stage2_RQ-VAE/_runs/"
       "stage2_4891fa8df3e843839de770e52e156892/"
       "TIGER_euclid_local_curve_supervision_rq/src")
CHECKPOINT = (REPO / "results/stage2_RQ-VAE/_runs/"
              "stage2_4891fa8df3e843839de770e52e156892/"
              "TIGER_euclid_local_curve_supervision_rq/result/rqvae_best.pth")
EMBEDDINGS = REPO / "stage1_GeneEmbedding/output/sentence_t5.npy"
GRAPH = REPO / "assets/train/visible_undirected.npz"
TARGETS = REPO / "assets/v8_runtime_layout/targets.npy"
OFFSETS = REPO / "assets/v8_runtime_layout/target_offsets.npy"
COMPONENTS = REPO / "assets/node_moments.npz"

SEED = 268
REFERENCE_SIZE = 256
GRID_STEP = 0.005
SAMPLE_ROWS = 2048
PAIR_CHUNK = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parent():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    for name in list(sys.modules):
        if name in ("model", "distances", "losses") or name.startswith(
                ("model.", "distances.", "losses.")):
            del sys.modules[name]
    import torch
    from model import RQVAE
    from train_config import load_train_config
    config = load_train_config(SRC / "config/train_config.json")
    from types import SimpleNamespace
    sizes = list(config.CODEBOOK_SIZE)
    if len(sizes) == 1:
        sizes *= config.CODEBOOK_NUM
    tokenizer = SimpleNamespace(
        hidden_sizes=config.HIDDEN_SIZES, codebook_num=config.CODEBOOK_NUM,
        codebook_size=tuple(sizes), codebook_dim=config.CODEBOOK_DIM,
        dropout=0.0, beta=config.BETA, vq_type=config.VQ_TYPE,
        ema_decay=config.EMA_DECAY, sk_epsilon=config.SK_EPSILON,
        sk_iters=config.SK_ITERS, kmeans_init=False, kmeans_iters=0)
    model = RQVAE(tokenizer, in_dim=768)
    model.load_checkpoint(str(CHECKPOINT))
    model.eval()
    return torch, model


def build_zref() -> Path:
    ART.mkdir(parents=True, exist_ok=True)
    target = ART / "z_ref.npy"
    torch, model = load_parent()
    embeddings = np.load(EMBEDDINGS)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(embeddings), 2048):
            block = torch.from_numpy(
                np.ascontiguousarray(embeddings[start:start + 2048])).to(device)
            chunks.append(model.encoder(block).cpu().numpy())
    z = np.concatenate(chunks).astype(np.float32)
    np.save(target, z)
    manifest = {
        "schema": "genrec.issue268.z_ref/1",
        "z_ref": str(target.relative_to(REPO)),
        "z_ref_sha256": sha256(target),
        "shape": list(z.shape), "dtype": str(z.dtype),
        "checkpoint": str(CHECKPOINT.relative_to(REPO)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "source_tree": str(SRC.relative_to(REPO)),
        "embeddings_sha256": sha256(EMBEDDINGS),
        "max_norm": float(np.linalg.norm(z, axis=1).max()),
        "rms_norm": float(np.sqrt((z.astype(np.float64) ** 2).sum(1).mean())),
        "identity": "final encoder of the #265 run; no candidate weight, no test data",
    }
    (ART / "z_ref_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=1))
    return target


def stereo_squared_torch(torch, x, y, k):
    """Unified kappa-stereographic squared distance, curvature k per row.

    x: [..., D], y: [..., D], k: [...] broadcastable.  k=0 returns ||x-y||^2.
    Values are the GraphMoRE/geoopt family squared distances (factor-two
    convention), shape-compatible with the frozen `manifold.dist`.
    """
    diff = x - y
    xy = (diff * diff).sum(-1)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    kk = k if torch.is_tensor(k) else torch.as_tensor(k, dtype=x.dtype, device=x.device)
    kk = kk.to(x.dtype)
    magnitude = kk.abs().clamp(min=1e-30)
    denom = (1.0 + kk * x2) * (1.0 + kk * y2)
    t2 = (magnitude * xy / denom).clamp(min=0.0)
    pos = kk > 0
    neg = kk < 0
    sphere = torch.asin(t2.clamp(max=1.0 - 1e-7).sqrt()).square() / magnitude
    hyp = torch.asinh(t2.sqrt()).square() / magnitude
    out = torch.where(pos, sphere, torch.where(neg, hyp, xy))
    return 4.0 * out


def graph_and_references():
    graph = load_sparse(GRAPH).tocsr().astype(np.int8)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    count, labels = connected_components(graph, directed=False)
    rng = np.random.default_rng(SEED)
    references = []
    for component in range(count):
        members = np.flatnonzero(labels == component)
        if len(members) < 2:
            continue
        take = min(REFERENCE_SIZE, len(members))
        references.append(np.sort(rng.choice(members, take, replace=False)))
    references = np.concatenate(references)
    distances = shortest_path(graph, unweighted=True, directed=False, indices=references)
    coo = graph.tocoo()
    upper = coo.row < coo.col
    keys = coo.row[upper].astype(np.int64) * len(labels) + coo.col[upper].astype(np.int64)
    edge_keys = set(keys.tolist())
    return graph, labels, references, distances.T, edge_keys  # [items, references]


def sample_rows(count=SAMPLE_ROWS):
    offsets = np.load(OFFSETS, mmap_mode="r")
    total = int(offsets[-1])
    rng = np.random.default_rng(SEED + 1)
    valid = []
    while len(valid) < count:
        candidate = np.sort(rng.integers(0, total, count))
        rows = np.load(TARGETS, mmap_mode="r")[candidate]
        keep = np.isfinite(rows["target"])
        valid.append(candidate[keep])
    return np.sort(np.concatenate(valid))[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("zref", "probe", "both"))
    parser.add_argument("--scale", choices=("maxnorm", "edge", "value"), default="maxnorm")
    parser.add_argument("--scale-value", type=float, default=None)
    args = parser.parse_args()
    started = time.time()
    if args.stage in ("zref", "both"):
        build_zref()
    if args.stage == "probe":
        pass
    if args.stage not in ("probe", "both"):
        return
    torch, _ = load_parent()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ART.mkdir(parents=True, exist_ok=True)
    z = np.load(ART / "z_ref.npy").astype(np.float64)
    if args.scale == "value":
        if args.scale_value is None or not np.isfinite(args.scale_value) or args.scale_value <= 0:
            raise ValueError("--scale value needs a positive --scale-value")
        scale = float(args.scale_value)
    elif args.scale == "maxnorm":
        scale = float(np.linalg.norm(z, axis=1).max())
    else:
        graph_edges = load_sparse(GRAPH).tocoo()
        upper = graph_edges.row < graph_edges.col
        edge_lengths = np.linalg.norm(
            z[graph_edges.row[upper]] - z[graph_edges.col[upper]], axis=1)
        scale = float(np.median(edge_lengths))
    max_coord = float(np.linalg.norm(z / (2.0 * scale), axis=1).max())
    # Ball domain: every coordinate must satisfy |k| * ||w||^2 < 1 at the box corner.
    k_min = float(np.ceil((-0.99 / (max_coord ** 2)) / GRID_STEP) * GRID_STEP)
    grid = np.round(np.arange(k_min, 1.0 + 1e-9, GRID_STEP), 10)
    if not np.any(grid == 0.0):
        grid = np.unique(np.round(np.append(grid, 0.0), 10))
    w = torch.from_numpy((z / (2.0 * scale)).astype(np.float32)).to(device)
    graph, labels, references, gdist, edge_keys = graph_and_references()
    rows = sample_rows()
    table = np.load(TARGETS, mmap_mode="r")
    offsets = np.load(OFFSETS, mmap_mode="r")
    centres = np.searchsorted(offsets, rows, side="right") - 1
    triples = np.column_stack((centres, table["b"][rows], table["c"][rows]))
    targets = table["target"][rows]

    gi = torch.from_numpy(gdist.astype(np.float32)).to(device)  # [items, R]
    usable = torch.from_numpy(np.isfinite(gdist) & (gdist > 0)).to(device)
    gval = gi.clone()
    gval[~usable] = 1.0
    wr = w[torch.from_numpy(references).to(device)]
    all_nodes = torch.arange(len(w), device=device)

    def node_f(k):
        """f_k(v) = sum_r (d/g - 1)^2 and valid count, for every item."""
        kk = torch.as_tensor(k, dtype=torch.float32, device=device)
        total = torch.zeros(len(w), device=device)
        counts = usable.sum(1).to(torch.float32)
        for start in range(0, len(all_nodes), 512):
            ids = all_nodes[start:start + 512]
            x = w[ids][:, None, :]
            d2 = stereo_squared_torch(torch, x.expand(-1, len(references), -1),
                                      wr[None], kk)
            ratio = d2.clamp_min(0).sqrt() / gval[ids]
            terms = torch.where(usable[ids], (ratio - 1.0).square(), torch.zeros_like(ratio))
            total[ids] = terms.sum(1)
        return total, counts

    def hop_distance(b_ids, c_ids):
        b_host = np.asarray(b_ids, dtype=np.int64)
        c_host = np.asarray(c_ids, dtype=np.int64)
        low = np.minimum(b_host, c_host)
        high = np.maximum(b_host, c_host)
        keys = low * len(labels) + high
        values = np.fromiter((1.0 if key in edge_keys else 2.0 for key in keys.tolist()),
                             dtype=np.float32, count=len(keys))
        return torch.from_numpy(values).to(device)

    def pair_term(k, b, c):
        kk = torch.as_tensor(k, dtype=torch.float32, device=device)
        d2 = stereo_squared_torch(torch, w[b], w[c], kk)
        g = hop_distance(b.cpu().numpy(), c.cpu().numpy())
        return (d2.clamp_min(0).sqrt() / g - 1.0).square()

    b = torch.from_numpy(triples[:, 1]).to(device)
    c = torch.from_numpy(triples[:, 2]).to(device)
    m = torch.from_numpy(triples[:, 0]).to(device)
    best = torch.full((len(rows),), float("inf"), device=device)
    best_k = torch.zeros(len(rows), device=device)
    j0 = None
    curves = np.zeros((len(grid), len(rows)), dtype=np.float64)
    for index, k in enumerate(grid):
        fsum, cnt = node_f(k)
        j = (fsum[m] + fsum[b] + fsum[c] + pair_term(k, b, c)) / (
            cnt[m] + cnt[b] + cnt[c] + 1.0)
        curves[index] = j.cpu().numpy()
        if k == 0.0:
            j0 = j.clone()
        lower = j < best
        best = torch.where(lower, j, best)
        best_k = torch.where(lower, torch.full_like(best_k, float(k)), best_k)
    # Independent direct evaluation on the same rows (no node decomposition).
    direct = np.zeros(len(grid))
    for index, k in enumerate(grid):
        total = torch.zeros(len(rows), device=device)
        kk = torch.as_tensor(k, dtype=torch.float32, device=device)
        for start in range(0, len(rows), PAIR_CHUNK):
            ids = torch.arange(start, min(start + PAIR_CHUNK, len(rows)), device=device)
            for node in (m[ids], b[ids], c[ids]):
                d2 = stereo_squared_torch(torch, w[node][:, None, :].expand(-1, len(references), -1),
                                          wr[None], kk)
                ratio = d2.clamp_min(0).sqrt() / gval[node]
                total[ids] += torch.where(usable[node], (ratio - 1.0).square(),
                                          torch.zeros_like(ratio)).sum(1)
            d2 = stereo_squared_torch(torch, w[b[ids]], w[c[ids]], kk)
            g = hop_distance(b[ids].cpu().numpy(), c[ids].cpu().numpy())
            total[ids] += (d2.clamp_min(0).sqrt() / g - 1.0).square()
        direct[index] = float((total / (cnt[m] + cnt[b] + cnt[c] + 1.0)).mean())
    decomposition_error = float(np.max(np.abs(direct - curves.mean(1))))
    summary = {
        "schema": "genrec.issue268.probe/1",
        "rows": len(rows),
        "references": int(len(references)),
        "graph": str(GRAPH.relative_to(REPO)),
        "graph_sha256": sha256(GRAPH),
        "targets_sha256": sha256(TARGETS),
        "z_ref_sha256": sha256(ART / "z_ref.npy"),
        "scale": scale, "scale_rule": args.scale,
        "grid": grid.tolist(),
        "grid_step": GRID_STEP,
        "k_min": float(grid[0]), "k_max": float(grid[-1]),
        "max_coord_norm": max_coord,
        "mean_J_by_k": curves.mean(1).tolist(),
        "mean_J0": float(j0.mean()),
        "mean_Jfit": float(best.mean()),
        "improved_fraction": float((best < j0 - 1e-12).float().mean()),
        "improvement_median": float(torch.median((j0 - best) / j0)),
        "best_k_histogram": {str(k): int((best_k == k).sum()) for k in np.unique(
            best_k.cpu().numpy())},
        "boundary_fraction": float(((best_k <= grid[0] + 1e-9) | (best_k >= grid[-1] - 1e-9)).float().mean()),
        "decomposition_max_abs_error": decomposition_error,
        "rows_compared": len(rows),
        "runtime_s": time.time() - started,
    }
    (ISSUE / "analysis").mkdir(parents=True, exist_ok=True)
    name = {"maxnorm": "probe_J.json", "edge": "probe_J_edge.json"}.get(
        args.scale, "probe_J_scale_%s.json" % ("%.6f" % scale).replace(".", "p"))
    (ISSUE / "analysis" / name).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("grid", "mean_J_by_k")},
                     indent=1))


if __name__ == "__main__":
    main()
