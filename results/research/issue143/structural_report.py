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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    args = parser.parse_args()
    path, payload = sid_table(args.arm)
    tokens = np.asarray([payload[str(index)] for index in range(len(payload))], dtype=np.int64)
    graph, kappa, support, embeddings = frozen_graph()

    # The readout the metric is defined on, composed from the EXPORTED tokens and
    # the checkpoint's codebooks. The encoder is not needed and is not rebuilt: its
    # width belongs to the arm, while the codebooks are 32-wide in every arm.
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
    codes_e = [book[:, :16][torch.as_tensor(tokens[:, level])] for level, book in enumerate(codebooks)]
    codes_h = [book[:, 16:][torch.as_tensor(tokens[:, level])] for level, book in enumerate(codebooks)]
    with torch.no_grad():
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
        "level3_triples": int(len(np.unique(codes[:, :3], axis=0))) if rows >= 3 else None,
        "collision_full": float(1.0 - len(np.unique(codes, axis=0)) / len(codes)),
        "collision_first": float(1.0 - len(np.unique(codes[:, :2], axis=0)) / len(codes))
        if rows >= 2 else None,
    }
    target = REPO / "results" / "research" / "issue143" / ("structure_%s.json" % args.arm)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
