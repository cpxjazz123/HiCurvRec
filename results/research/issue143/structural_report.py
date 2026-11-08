"""The shared structural evaluation every one of #140-#143 asks to report.

The issues require the same numbers for every arm, with the same support, the same
calibration and the same code as the frozen tree -- an arm must not be scored with
a metric it defined itself. So this script imports the frozen `af3_structure`
module from a frozen tree, rebuilds the frozen support and calibration from the
TRAIN graph, and reports for one exported SID:

  * the hard-SID ``D_AF3`` (class-balanced Huber of the expectation surrogate),
  * neighbour retention at 16,
  * the adjacency Brier score,
  * the reconstruction MSE of the embedding the SID was built from,
  * code usage per level, level-two prefixes, level-three triples and the two
    collision counts,

with the SID's SHA-256 and the support's identity written next to them. A value
that cannot be computed says so; nothing is filled in with a zero.

    python3 results/research/issue143/structural_report.py TIGER_mixed_rq_F0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULT = REPO / "results" / "stage2_RQ-VAE"
FROZEN = REPO / "stage2_RQ-VAE" / "TIGER_mixed_rq"
for path in (FROZEN, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from model import af3_structure as structure  # noqa: E402
from model.mixed_rq import encode_tokens  # noqa: E402
from model.af3_selector import _readout_of  # noqa: E402

AF3_CACHE = (REPO / "results" / "research" / "issue121" / "community_curvature"
             / "train_edge_af3.npz")
EMBEDDING_FILE = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"


def frozen_graph():
    from tools.research.science.community_curvature import data as graph_data
    graph = structure.load_graph(AF3_CACHE)
    canonical = graph_data.build_graph("train")
    if (graph["n_nodes"] != canonical.n_nodes
            or not np.array_equal(graph["rows"], canonical.edges[0])
            or not np.array_equal(graph["cols"], canonical.edges[1])):
        raise ValueError("the AF3 cache does not match the canonical TRAIN graph")
    graph["item_ids"] = np.asarray(canonical.nodes, dtype=np.int64)
    kappa = structure.target_kappa(graph)
    embeddings = np.load(EMBEDDING_FILE)
    support = structure.build_support(graph, kappa, embeddings[graph["item_ids"]],
                                      k=16, seed=42)
    return graph, kappa, support, embeddings


def neighbour_retention(phi: torch.Tensor, pairs: np.ndarray, *, n_nodes: int,
                        k: int = 16, sample: int = 2048, seed: int = 42) -> dict:
    """``NeighbourRetention@k``: are the k nearest readouts TRAIN edges?

    The full 24587 x 24587 distance matrix would be 2.4 GB, so the retention is
    measured on a fixed random sample of nodes and the sample size is reported with
    the number. A pair is an edge when it appears in the support's edge array.
    """
    edges = {(int(i), int(j)) for i, j in pairs}
    edges |= {(j, i) for i, j in edges}
    rng = np.random.default_rng(seed)
    anchors = rng.choice(n_nodes, size=min(sample, n_nodes), replace=False)
    values = torch.as_tensor(phi, dtype=torch.float64)
    index = torch.as_tensor(anchors, dtype=torch.long)
    distance = torch.cdist(values[index], values)
    distance[torch.arange(len(index)), index] = float("inf")
    nearest = torch.topk(distance, k=k, largest=False).indices.cpu().numpy()
    retained = [sum((int(anchor), int(other)) in edges for other in row)
                for anchor, row in zip(anchors, nearest)]
    return {"k": int(k), "sample": int(len(anchors)),
            "mean_edges_among_k": float(np.mean(retained)),
            "retention": float(np.mean(retained) / max(k, 1))}


def sid_table(arm: str) -> dict:
    path = RESULT / arm / "item_sids.json"
    if not path.is_file():
        raise SystemExit("%s has not exported an identifier" % arm)
    return path, json.loads(path.read_text())


def collision(codes: np.ndarray) -> dict:
    """The frozen item-weighted collision: share of items whose code repeats.

    This is the definition the mandatory layer record uses; counting
    ``1 - distinct/items`` instead would treat one item of every repeated bucket
    as non-colliding and understate the collapse.
    """
    from collections import Counter
    counts = Counter(map(tuple, np.asarray(codes).tolist()))
    total = len(codes)
    shared = sum(count for count in counts.values() if count > 1)
    return {"items": int(total), "distinct": int(len(counts)),
            "colliding_items": int(shared), "share": float(shared / total) if total else 0.0,
            "max_bucket": int(max(counts.values())) if counts else 0}


def decoder_reconstruction(state: dict, phi: torch.Tensor, embeddings: np.ndarray) -> dict:
    """MSE of the SID reconstruction: decoder(phi) against the T5 embedding.

    The decoder is the arm's own trained MLP (32 -> 128 -> 256 -> 512 -> 768,
    ReLU between layers); at eval the dropout layers are the identity. The target
    is the embedding the SID was built from, in the export's item order.
    """
    import torch.nn as nn
    weights = sorted((key for key in state if key.startswith("decoder.mlp.")
                      and key.endswith(".weight")),
                     key=lambda name: int(name.split(".")[2]))
    layers = []
    for position, key in enumerate(weights):
        weight = state[key]
        bias = state[key.replace(".weight", ".bias")]
        linear = nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if position != len(weights) - 1:
            layers.append(nn.ReLU())
    decoder = nn.Sequential(*layers).float().eval()
    with torch.no_grad():
        reconstructed = decoder(torch.as_tensor(phi, dtype=torch.float32))
    target = torch.as_tensor(embeddings, dtype=torch.float32)
    if reconstructed.shape != target.shape:
        raise ValueError("decoder output %r does not match the embedding %r"
                         % (tuple(reconstructed.shape), tuple(target.shape)))
    return {"mse": float(torch.mean((reconstructed - target) ** 2)),
            "items": int(target.shape[0]), "dim": int(target.shape[1])}


def sid_change(tokens: np.ndarray, levels: int) -> dict:
    """How the exported semantic SID differs from the frozen baseline's.

    Only the ``levels`` semantic tokens are compared: the fourth token is the
    deterministic collision-extension, not a level of the identifier.
    """
    path = RESULT / "TIGER_RQ-VAE" / "item_sids.json"
    if not path.is_file():
        return {"available": False, "reason": "the frozen baseline SID is missing"}
    base = json.loads(path.read_text())
    if len(base) != len(tokens):
        return {"available": False,
                "reason": "baseline has %d items, candidate %d" % (len(base), len(tokens))}
    baseline = np.asarray([base[str(index)] for index in range(len(base))], dtype=np.int64)
    baseline = baseline[:, :levels]
    candidate = tokens[:, :levels]
    changed = np.any(baseline != candidate, axis=1)
    return {"available": True, "baseline": str(path),
            "baseline_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "levels": int(levels),
            "changed_items": int(changed.sum()),
            "changed_share": float(changed.mean()),
            "per_level_changed": [float(np.mean(baseline[:, level] != candidate[:, level]))
                                  for level in range(levels)]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    args = parser.parse_args()
    path, payload = sid_table(args.arm)
    tokens = np.asarray([payload[str(index)] for index in range(len(payload))], dtype=np.int64)
    graph, kappa, support, embeddings = frozen_graph()

    # The readout the metric is defined on, composed from the EXPORTED tokens and
    # the checkpoint's codebooks. The frozen TIGER baseline is a plain Euclidean
    # 32-d residual quantizer, so its readout is the sum of the full 32-d codes;
    # the mixed-geometry arms split every code into E16 x H16 and use the frozen
    # mixed readout. Using the mixed readout on the Euclidean baseline would put
    # its last 16 dims through a Poincare log map and change every distance.
    from model.mixed_rq import quantized_readout
    checkpoint = RESULT / args.arm / "rqvae_best.pth"
    if not checkpoint.is_file():
        raise SystemExit("%s has no checkpoint to reconstruct the readout" % args.arm)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    codebooks = [state[key] for key in sorted(state) if key.startswith("rq.")
                 and key.endswith("embed.weight")]
    if not codebooks:
        raise SystemExit("the checkpoint carries no codebooks")
    rows = len(codebooks)
    with torch.no_grad():
        if args.arm == "TIGER_RQ-VAE":
            phi = sum(book[torch.as_tensor(tokens[:, level])]
                      for level, book in enumerate(codebooks))
        else:
            codes_e = [book[:, :16][torch.as_tensor(tokens[:, level])]
                       for level, book in enumerate(codebooks)]
            codes_h = [book[:, 16:][torch.as_tensor(tokens[:, level])]
                       for level, book in enumerate(codebooks)]
            phi = quantized_readout(codes_e, codes_h, curvature=1.0)
    calibration = structure.calibrate(phi, support, device=phi.device)
    result = structure.structural_distortion(
        phi, support, kappa, tau=calibration["tau"], intercept=calibration["intercept"],
        n_nodes=graph["n_nodes"])

    # Neighbour retention at 16 and the adjacency Brier on the same support.
    distance = result["probability"].detach().cpu().numpy() if torch.is_tensor(
        result["probability"]) else np.asarray(result["probability"])
    probability = torch.as_tensor(distance)
    pairs = support["pairs"]
    is_edge = np.asarray(support["is_edge"], dtype=bool)
    retention = neighbour_retention(phi, pairs, n_nodes=graph["n_nodes"], k=16, seed=42)
    brier = float(np.mean((probability.numpy() - is_edge.astype(np.float64)) ** 2))

    codes = tokens[:, :rows]
    reconstruction = decoder_reconstruction(state, phi, embeddings)
    change = sid_change(tokens, rows)
    full = collision(codes)
    first = collision(codes[:, :2]) if rows >= 2 else None
    report = {
        "arm": args.arm,
        "sid": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "items": int(len(tokens)), "tokens": int(rows)},
        "support": {"pairs": int(len(pairs)), "edges": int(is_edge.sum())},
        "distortion": float(result["distortion"]),
        "edge_half": float(result["edge_half"]),
        "unobserved_half": float(result["unobserved_half"]),
        "scale": float(result["scale"]),
        "calibration": {"tau": float(calibration["tau"]),
                        "intercept": float(calibration["intercept"])},
        "neighbour_retention@16": retention,
        "adjacency_brier": brier,
        "usage": [int(len(np.unique(codes[:, level]))) for level in range(rows)],
        "level2_prefixes": int(len(np.unique(codes[:, :2], axis=0))),
        "level3_triples": full["distinct"],
        "collision_full": full["share"],
        "collision_full_items": full["colliding_items"],
        "collision_first": first["share"] if first else None,
        "max_triple_bucket": full["max_bucket"],
        "max_level_code_usage": [int(np.bincount(codes[:, level]).max())
                                 for level in range(rows)],
        "reconstruction": reconstruction,
        "sid_change": change,
        "degree_triangle": {
            "computed": False,
            "reason": "a full 24587x24587 SID graph would be rebuilt from hard "
                      "distances; the issue's D_AF3/retention/Brier are measured on "
                      "the frozen stratified support, and no unbiased degree/triangle "
                      "estimator exists on that support, so none is reported as if "
                      "it were the full-graph number"},
    }
    target = REPO / "results" / "research" / "issue143" / ("structure_%s.json" % args.arm)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
