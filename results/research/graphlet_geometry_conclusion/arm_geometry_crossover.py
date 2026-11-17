"""Do the two matched arms trade one graphlet class against another?

The readout probe moved the geometry on a fixed SID.  This one asks the
counterfactual the matched arms actually ran: F0 (E16 x H16, alpha = 1/2) and E
(same recipe, curvature 0, so both halves are Euclidean) differ only in the
geometry, and their SIDs are almost disjoint (same L3 token: 0.0000).  If
different graphlets need different geometric biases, the per-graphlet-class
behaviour-edge recovery must cross over: each geometry preserves some classes
better and loses others.

For every arm the script rebuilds the quantised readout from the *exported* SID
tokens and the arm's own codebooks, builds the k=10 graph under that arm's own
metric, and measures recovery per triangle-multiplicity class.  The fit/evaluate
split over behaviour edges is the same for every arm.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/graphlet_geometry_conclusion/arm_geometry_crossover.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
F0_TREE = REPO / "analysis_sid_recall" / "r2" / "19" / "TIGER_mixed_rq_F0"
E_TREE = REPO / "analysis_sid_recall" / "r2" / "16" / "TIGER_mixed_rq_E"
F0_VECTORS = REPO / "results" / "research" / "tiger_cluster_graph" / "mixed_vectors.npz"
BASE_VECTORS = REPO / "results" / "research" / "tiger_cluster_graph" / "tiger_vectors.npz"

K = 10
CHUNK = 256
BOOTSTRAP = 2000
SEED = 42
CLASSES = {"triangles_0": (0, 0), "triangles_1_2": (1, 2),
           "triangles_3_5": (3, 5), "triangles_6plus": (6, 10 ** 9)}
ARMS = {
    "F0_mixed": {"vectors": F0_VECTORS, "metric": "mixed_0.5"},
    "E_euclidean": {"tree": E_TREE, "checkpoint": REPO / "analysis_sid_recall" / "r2" / "16" / "rqvae_best.pth",
                    "sid": REPO / "analysis_sid_recall" / "r2" / "16" / "item_sids.json",
                    "metric": "euclidean"},
    "baseline_e32": {"vectors": BASE_VECTORS, "metric": "euclidean"},
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_package(name: str, tree: Path):
    spec = importlib.util.spec_from_file_location(
        name, tree / "model" / "__init__.py",
        submodule_search_locations=[str(tree / "model")],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package


def rebuild_euclidean_q():
    """The E arm's quantised readout, rebuilt from its exported SID and codebooks."""
    package = load_package("e_model", E_TREE)
    config = SimpleNamespace(
        hidden_sizes=(512, 256, 128), codebook_num=3, codebook_size=(256, 256, 256),
        codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
        fix_code_embs=False, sk_epsilon=0.003, sk_iters=50,
    )
    model = package.MixedRQVAE(config, in_dim=768, curvature=0.0)
    payload = torch.load(ARMS["E_euclidean"]["checkpoint"], map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"E checkpoint mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
    model.eval()
    exported = json.loads(ARMS["E_euclidean"]["sid"].read_text())
    tokens = np.array([[int(v) for v in exported[str(i)][:3]] for i in range(len(exported))])
    embeddings = torch.from_numpy(np.load(EMBEDDING).astype(np.float32))
    from e_model.mixed_rq import quantized_readout
    levels = list(range(3))
    with torch.no_grad():
        z = model.encoder(embeddings)
        # The tokens are the exported ones; the code rows come from this arm's codebooks.
        codebooks = [model.rq.levels[level].embed.weight.detach() for level in levels]
        picked_e, picked_h = [], []
        for level in levels:
            rows = codebooks[level][torch.from_numpy(tokens[:, level])]
            e_half, h_half = rows[:, :16], rows[:, 16:]
            picked_e.append(e_half)
            picked_h.append(h_half)
        q = quantized_readout(picked_e, picked_h, 0.0)
    return q.numpy().astype(np.float32), tokens


def load_vectors(path: Path):
    payload = np.load(path)
    q = payload["q"].astype(np.float32)
    tokens = payload["tokens"]
    return q, tokens


def build_graph(q: np.ndarray, metric: str, hyp_sq=None, expmap0=None) -> set:
    """The symmetrised k-NN graph under the arm's own metric."""
    tensor = torch.from_numpy(q).cuda()
    n_items = tensor.shape[0]
    rows = torch.arange(n_items)
    if metric == "euclidean":
        for start in range(0, n_items, CHUNK):
            stop = min(start + CHUNK, n_items)
            distance = torch.cdist(tensor[start:stop], tensor).pow(2)
            distance[rows[:stop - start], rows[start:stop]] = float("inf")
            yield_index = torch.topk(distance, K, dim=1, largest=False).indices.cpu().tolist()
            for offset, row in enumerate(yield_index):
                left = start + offset
                for right in row:
                    yield (left, right) if left < right else (right, left)
    else:
        euclidean = tensor[:, :16]
        ball = expmap0(tensor[:, 16:], 1.0)
        norm2 = ball.pow(2).sum(-1)
        denominator = (1.0 - norm2).clamp_min(1e-12)
        for start in range(0, n_items, CHUNK):
            stop = min(start + CHUNK, n_items)
            d_e = torch.cdist(euclidean[start:stop], euclidean).pow(2)
            difference = torch.cdist(ball[start:stop], ball).pow(2)
            argument = 1.0 + 2.0 * difference / (denominator[start:stop, None] * denominator[None, :])
            d_h = torch.acosh(argument.clamp_min(1.0 + 1e-9)).pow(2) / 4.0
            distance = 0.5 * d_e + 0.5 * d_h
            distance[rows[:stop - start], rows[start:stop]] = float("inf")
            index = torch.topk(distance, K, dim=1, largest=False).indices.cpu().tolist()
            for offset, row in enumerate(index):
                left = start + offset
                for right in row:
                    yield (left, right) if left < right else (right, left)


def load_graph():
    train = pd.read_parquet(TRAIN)
    pairs = set()
    for history, target in zip(train["history"].to_numpy(), train["target"].to_numpy()):
        history = list(history)
        for left, right in zip(history, history[1:]):
            pairs.add((left, right) if left < right else (right, left))
        if history:
            tail = history[-1]
            pairs.add((tail, target) if tail < target else (target, tail))
    edges = np.array(sorted(pairs), dtype=np.int64)
    if edges.shape[0] != 315391:
        raise SystemExit(f"behaviour graph has {edges.shape[0]} edges, expected 315391")
    return edges


def main():
    package = load_package("f0_model", F0_TREE)
    from f0_model.poincare_contract import expmap0, squared_distance as hyp_sq
    q_e, tokens_e = rebuild_euclidean_q()
    print(f"E q rebuilt: {q_e.shape}")

    edges = load_graph()
    n_items = q_e.shape[0]
    adjacency = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_items, n_items)
    ).tocsr()
    adjacency = (adjacency + adjacency.T).tocsr()
    triangle_count = np.rint(np.asarray(
        (adjacency @ adjacency)[edges[:, 0], edges[:, 1]]).ravel()).astype(np.int64)
    split = np.array([int(hashlib.sha256(f"{int(a)}:{int(b)}".encode()).hexdigest()[:8], 16) % 2
                      for a, b in edges], dtype=bool)
    print(f"edges={len(edges)} fit={split.sum()} eval={(~split).sum()}")

    recovery = {}
    graphs = {}
    for name, spec in ARMS.items():
        if "vectors" in spec:
            q, tokens = load_vectors(spec["vectors"])
        else:
            q, tokens = q_e, tokens_e
        graph = set(build_graph(q, spec["metric"], hyp_sq, expmap0))
        graphs[name] = graph
        row = {}
        for label, (low, high) in CLASSES.items():
            mask = (triangle_count >= low) & (triangle_count <= high)
            hit = np.array([(int(edges[i][0]), int(edges[i][1])) in graph
                            for i in np.flatnonzero(mask)], dtype=np.float64)
            row[label] = {"all": float(hit.mean()), "fit": float(hit[split[mask]].mean()),
                          "eval": float(hit[~split[mask]].mean()), "n": int(hit.size)}
        used = {(int(a), int(b)) for a, b in edges}
        overall = np.array([(a, b) in graph for a, b in used], dtype=np.float64)
        row["all_edges"] = {"all": float(overall.mean()), "n": int(overall.size),
                            "fit": float(overall[split].mean()),
                            "eval": float(overall[~split].mean())}
        recovery[name] = row
        print(f"{name:14s} overall={overall.mean():.5f}  "
              + "  ".join(f"{label}={row[label]['all']:.5f}" for label in CLASSES), flush=True)

    # Crossover: F0 minus E, per class, with a paired bootstrap over edges.
    rng = np.random.default_rng(SEED)
    f0_graph = graphs["F0_mixed"]
    e_graph = graphs["E_euclidean"]
    crossover = {}
    for label, (low, high) in CLASSES.items():
        mask = (triangle_count >= low) & (triangle_count <= high)
        indices = np.flatnonzero(mask)
        hits_f0 = np.array([(int(edges[i][0]), int(edges[i][1])) in f0_graph for i in indices],
                           dtype=np.float64)
        hits_e = np.array([(int(edges[i][0]), int(edges[i][1])) in e_graph for i in indices],
                          dtype=np.float64)
        delta = hits_f0 - hits_e
        weights = rng.poisson(1.0, size=(BOOTSTRAP, delta.size))
        boot = (weights @ delta) / delta.size
        crossover[label] = {
            "F0": float(hits_f0.mean()), "E": float(hits_e.mean()),
            "delta": float(delta.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "share_positive": float((boot > 0).mean()),
            "edges": int(delta.size),
        }
        print(f"crossover {label:16s} F0={hits_f0.mean():.5f} E={hits_e.mean():.5f} "
              f"delta={delta.mean():+.5f} CI=[{np.quantile(boot,0.025):+.5f},"
              f"{np.quantile(boot,0.975):+.5f}]")

    report = {
        "schema": "genrec.arm_geometry_crossover.v1",
        "n_items": int(n_items),
        "n_edges": int(len(edges)),
        "k": K,
        "classes": list(CLASSES),
        "arms": {name: recovery[name] for name in ARMS},
        "crossover_f0_minus_e": crossover,
        "provenance": {
            "f0_vectors_sha256": sha256_of(F0_VECTORS),
            "e_checkpoint_sha256": sha256_of(ARMS["E_euclidean"]["checkpoint"]),
            "e_sid_sha256": sha256_of(ARMS["E_euclidean"]["sid"]),
            "baseline_vectors_sha256": sha256_of(BASE_VECTORS),
            "train_parquet_sha256": sha256_of(TRAIN),
            "metrics": {"F0_mixed": "0.5*d_E^2 + 0.5*d_H^2/4",
                        "E_euclidean": "squared Euclidean on the 32-dim readout",
                        "baseline_e32": "squared Euclidean on the 32-dim readout"},
        },
    }
    out = HERE / "arm_geometry_crossover.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
