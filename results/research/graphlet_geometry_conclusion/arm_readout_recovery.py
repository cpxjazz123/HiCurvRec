"""Per-class behaviour-edge recovery for the codeword arms, from their own SIDs.

The Stage3 comparison answers "does the allocation help recall"; this answers the
structural question independently of Stage3: with the exported SID and the arm's
own codebooks, rebuild the mixed readout and ask which graphlet classes survive
in the k=10 graph.  The readout metric is held at alpha = 1/2 for every arm, so
the comparison isolates the partition the mechanism produced, not a different
distance.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 \
        results/research/graphlet_geometry_conclusion/arm_readout_recovery.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
STAGE2 = REPO / "results" / "stage2_RQ-VAE"

K = 10
CHUNK = 256
ALPHA = 0.5
CLASSES = {"triangles_0": (0, 0), "triangles_1_2": (1, 2),
           "triangles_3_5": (3, 5), "triangles_6plus": (6, 10 ** 9)}
ARMS = {
    "F0": {"tree": REPO / "analysis_sid_recall" / "r2" / "19" / "TIGER_mixed_rq_F0",
           "checkpoint": STAGE2 / "TIGER_mixed_rq_F0" / "rqvae_best.pth",
           "sid": STAGE2 / "TIGER_mixed_rq_F0" / "item_sids.json"},
    "codeword": {"tree": REPO / "stage2_RQ-VAE" / "TIGER_graphlet_codebook_rq",
                 "checkpoint": STAGE2 / "TIGER_graphlet_codebook_rq" / "rqvae_best.pth",
                 "sid": STAGE2 / "TIGER_graphlet_codebook_rq" / "item_sids.json"},
    "global": {"tree": REPO / "stage2_RQ-VAE" / "TIGER_graphlet_codebook_global_rq",
               "checkpoint": STAGE2 / "TIGER_graphlet_codebook_global_rq" / "rqvae_best.pth",
               "sid": STAGE2 / "TIGER_graphlet_codebook_global_rq" / "item_sids.json"},
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


def rebuild_readout(spec: dict) -> np.ndarray:
    """q from the exported tokens and the arm's own codebooks (curvature 1)."""
    package = load_package("arm_model", spec["tree"])
    from types import SimpleNamespace
    config = SimpleNamespace(
        hidden_sizes=(512, 256, 128), codebook_num=3, codebook_size=(256, 256, 256),
        codebook_dim=32, dropout=0.0, beta=0.25, vq_type="vq", ema_decay=0.99,
        fix_code_embs=False, sk_epsilon=0.003, sk_iters=50,
    )
    try:
        # The codeword arms carry a per-codeword table; the archived F0 recipe does not.
        model = package.MixedRQVAE(config, in_dim=768, curvature=1.0, code_alpha_mode=True)
    except TypeError:
        model = package.MixedRQVAE(config, in_dim=768, curvature=1.0)
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch for {spec['tree'].name}: "
                         f"missing={missing[:3]} unexpected={unexpected[:3]}")
    model.eval()
    exported = json.loads(spec["sid"].read_text())
    tokens = np.array([[int(v) for v in exported[str(i)][:3]] for i in range(len(exported))])
    from arm_model.mixed_rq import quantized_readout
    picked_e, picked_h = [], []
    for level in range(3):
        rows = model.rq.levels[level].embed.weight.detach()[torch.from_numpy(tokens[:, level])]
        picked_e.append(rows[:, :16])
        picked_h.append(rows[:, 16:])
    q = quantized_readout(picked_e, picked_h, 1.0)
    return q.numpy().astype(np.float32)


def build_graph(q: np.ndarray, expmap0, hyp_sq) -> set:
    tensor = torch.from_numpy(q).cuda()
    n_items = tensor.shape[0]
    rows = torch.arange(n_items)
    euclidean = tensor[:, :16]
    ball = expmap0(tensor[:, 16:], 1.0)
    norm2 = ball.pow(2).sum(-1)
    denominator = (1.0 - norm2).clamp_min(1e-12)
    graph = set()
    for start in range(0, n_items, CHUNK):
        stop = min(start + CHUNK, n_items)
        d_e = torch.cdist(euclidean[start:stop], euclidean).pow(2)
        difference = torch.cdist(ball[start:stop], ball).pow(2)
        argument = 1.0 + 2.0 * difference / (denominator[start:stop, None] * denominator[None, :])
        d_h = torch.acosh(argument.clamp_min(1.0 + 1e-9)).pow(2) / 4.0
        distance = ALPHA * d_e + (1.0 - ALPHA) * d_h
        distance[rows[:stop - start], rows[start:stop]] = float("inf")
        for offset, row in enumerate(torch.topk(distance, K, dim=1, largest=False).indices.cpu().tolist()):
            left = start + offset
            for right in row:
                graph.add((left, right) if left < right else (right, left))
    return graph


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


def main() -> None:
    package = load_package("f0_contract", ARMS["F0"]["tree"])
    from f0_contract.poincare_contract import expmap0, squared_distance as hyp_sq
    edges = load_graph()
    n_items = len(json.loads(ARMS["F0"]["sid"].read_text()))
    adjacency = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_items, n_items)
    ).tocsr()
    adjacency = (adjacency + adjacency.T).tocsr()
    triangle = np.rint(np.asarray(
        (adjacency @ adjacency)[edges[:, 0], edges[:, 1]]).ravel()).astype(np.int64)
    split = np.array([int(hashlib.sha256(f"{int(a)}:{int(b)}".encode()).hexdigest()[:8], 16) % 2
                      for a, b in edges], dtype=bool)

    recovery = {}
    for name, spec in ARMS.items():
        if not spec["sid"].is_file():
            print(f"{name}: no SID yet, skipped")
            continue
        q = rebuild_readout(spec)
        graph = build_graph(q, expmap0, hyp_sq)
        row = {}
        for label, (low, high) in CLASSES.items():
            mask = (triangle >= low) & (triangle <= high)
            hit = np.array([(int(edges[i][0]), int(edges[i][1])) in graph
                            for i in np.flatnonzero(mask)], dtype=np.float64)
            row[label] = {"all": float(hit.mean()), "eval": float(hit[~split[mask]].mean()),
                          "fit": float(hit[split[mask]].mean()), "n": int(hit.size)}
        used = {(int(a), int(b)) for a, b in edges}
        overall = np.array([(a, b) in graph for a, b in used], dtype=np.float64)
        row["all_edges"] = {"all": float(overall.mean()), "n": int(overall.size),
                            "eval": float(overall[~split].mean()),
                            "fit": float(overall[split].mean())}
        recovery[name] = row
        print(f"{name:10s} overall={overall.mean():.5f}  "
              + "  ".join(f"{label}={row[label]['all']:.5f}" for label in CLASSES), flush=True)

    report = {
        "schema": "genrec.arm_readout_recovery.v1",
        "n_items": int(n_items), "n_edges": int(len(edges)), "k": K, "alpha": ALPHA,
        "classes": list(CLASSES), "arms": recovery,
        "provenance": {
            "train_parquet_sha256": sha256_of(TRAIN),
            "sid_sha256": {name: sha256_of(spec["sid"]) for name, spec in ARMS.items()
                           if spec["sid"].is_file()},
            "checkpoint_sha256": {name: sha256_of(spec["checkpoint"]) for name, spec in ARMS.items()
                                  if spec["checkpoint"].is_file()},
        },
    }
    out = HERE / "arm_readout_recovery.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
