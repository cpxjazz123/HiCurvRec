"""Export the community representation of every item, for the Stage-2 arm.

The standalone community experiment trains a small model on the behaviour
graph's AF3 curvature and ends with a soft community distribution per item.
The Stage-2 arm needs more than the distribution: it needs the item's point in
the community model's Poincare ball, because the term it adds to the level-1
cost is the contract's distance between that point and the community profile
of a code.

The export is deterministic -- the community model is trained with seed 42 and
the same data -- so this script reproduces the arm's input exactly.

Run:

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        tools/research/science/community_curvature/export_representation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import os
import numpy as np
import torch

# The cuBLAS GEMMs are not deterministic on CUDA without this, and the arms with
# a sparse propagation drift by several percent between runs -- more than the
# differences between the arms, which hides every real effect.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tools.research.science.community_curvature import data, model  # noqa: E402
from tools.research.science.community_curvature.run_experiment import (  # noqa: E402
    FREE_DIM_EXPERIMENT, HYPER_DIM_EXPERIMENT, N_COMMUNITIES, SEED, CURVATURE,
)

OUT = data.REPO / "results" / "research" / "issue121" / "community_curvature"
ITEM_REPRESENTATION = OUT / "item_community_representation.npz"


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    embeddings = data.load_embeddings()

    graph = data.build_graph("train")
    curvature = data.curvature_of(graph)
    conv = _curvature_graph(graph, curvature, embeddings.shape[0])
    net = model.CommunityModel(input_dim=embeddings.shape[1], free_dim=FREE_DIM_EXPERIMENT,
                                hyper_dim=HYPER_DIM_EXPERIMENT, n_communities=N_COMMUNITIES,
                                curvature=CURVATURE, seed=SEED, graph=conv).to(device)
    # the weights are the ones the experiment's report used; the model is
    # retrained here with the same seed and the same data, so the export is
    # the same object the report measured
    from tools.research.science.community_curvature.run_experiment import train
    train(net, embeddings, *_ranking_pairs(graph, curvature), device, seed=SEED)

    every_item = torch.from_numpy(embeddings).to(device)
    with torch.no_grad():
        free, ball = net.represent(every_item)
        # The hard community of every item. A Stage-2 arm that needs a
        # candidate set of "same community" and "other community" items reads
        # it from here: deriving it anywhere else would be a second copy of the
        # teacher's own partition, and the two copies could disagree.
        plan = net.assignment(free, ball)
        community = plan.argmax(dim=-1).to(torch.int64)
    payload = {
        "free": free.cpu().numpy().astype(np.float32),
        "ball": ball.cpu().numpy().astype(np.float32),
        "curvature": float(CURVATURE),
        "community": community.cpu().numpy().astype(np.int64),
        "n_communities": int(plan.shape[1]),
        "seed": SEED,
        "arm": "E_free_64_hyper_64_curvgcn",
        "n_items": int(embeddings.shape[0]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(ITEM_REPRESENTATION, **payload)
    print(json.dumps({"written": str(ITEM_REPRESENTATION),
                      "free_shape": list(payload["free"].shape),
                      "ball_shape": list(payload["ball"].shape),
                      "community_shape": list(payload["community"].shape),
                      "communities_used": int(len(np.unique(payload["community"]))),
                      "ball_norm_max": float(np.linalg.norm(payload["ball"], axis=1).max())},
                     indent=2))
    return 0


def _ranking_pairs(graph, curvature):
    pairs = data.ranking_pairs(graph, curvature)
    nodes = np.asarray(graph.nodes, dtype=np.int64)
    return (nodes[pairs.anchor], nodes[pairs.high], nodes[pairs.low])


def _curvature_graph(graph, curvature, n_items):
    from tools.research.science.community_curvature.run_experiment import curvature_graph
    return curvature_graph(graph, curvature, n_items, curvature_aware=True)


if __name__ == "__main__":
    raise SystemExit(main())
