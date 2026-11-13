"""Three graphs per arm: behaviour, pre-quantization, post-quantization.

Issue #215 asks whether quantization preserves the behaviour graph's *structure* -- not just whether
two items share a token.  This script builds the comparison the owner described:

    G_b   the TRAIN behaviour graph (real user interactions)
    G_z   a k-NN graph over the encoder's pre-quantization vector z
    G_q   a k-NN graph over the model's own quantized readout z_q

G_z and G_q use exactly the same rule and the same number of neighbours, so a difference between
them cannot come from density.  Each graph is then handed to ORCA (the same tool the audit uses) so
every item gets its orbit counts in all three graphs, and the structures can be compared node by
node.

    python3 results/research/issue215/structure_graphs.py

Writes vectors under results/research/issue215/vectors/ and graph files under
results/research/issue215/graphs/, and a manifest with the edge counts.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "issue215"
GRAPHS = OUT / "graphs"
VECTORS = OUT / "vectors"

ARMS = {
    # the historical E32 baseline, kept for the same three-graph treatment: z and q are its
    # encoder output and its Euclidean residual-quantizer readout, both 32-dimensional
    "TIGER": {"tree": REPO / "analysis_sid_recall" / "r2" / "24" / "TIGER_RQ-VAE",
              "checkpoint": REPO / "analysis_sid_recall" / "r2" / "24" / "rqvae_best.pth",
              "mode": None},
    "F0": {"tree": REPO / "analysis_sid_recall" / "r2" / "19" / "TIGER_mixed_rq_F0",
           "checkpoint": REPO / "analysis_sid_recall" / "r2" / "19" / "rqvae_best.pth",
           "mode": None},
    "B": {"tree": REPO / "stage2_RQ-VAE" / "TIGER_role_geom_global_rq",
          "checkpoint": REPO / "results" / "stage2_RQ-VAE" / "TIGER_role_geom_global_rq" / "rqvae_best.pth",
          "mode": "GLOBAL"},
    "C": {"tree": REPO / "stage2_RQ-VAE" / "TIGER_role_geom_rq",
          "checkpoint": REPO / "results" / "stage2_RQ-VAE" / "TIGER_role_geom_rq" / "rqvae_best.pth",
          "mode": "ROLE"},
    "D": {"tree": REPO / "stage2_RQ-VAE" / "TIGER_role_geom_shuffled_rq",
          "checkpoint": REPO / "results" / "stage2_RQ-VAE" / "TIGER_role_geom_shuffled_rq" / "rqvae_best.pth",
          "mode": "SHUFFLED"},
}

ROLE_TABLE = OUT / "orca_roles_frozen.csv"
ROLE_SHA = "c7a29e6559aed1385c5ce514e506be22d88bf6e85eb61120b6dbf69375ea251c"
EMBEDDING = REPO / "stage1_GeneEmbedding" / "output" / "sentence_t5.npy"
TRAIN = REPO / "results" / "stage0_build_parquet" / "train.parquet"
K = 26                       # neighbours per item, so the mean degree matches the behaviour graph


def load_tree_model(tree: Path, checkpoint: Path, mode: str | None):
    """Import one arm's own model package and load its checkpoint."""
    spec = importlib.util.spec_from_file_location(
        f"arm_{tree.name}", tree / "model" / "__init__.py",
        submodule_search_locations=[str(tree / "model")])
    package = importlib.util.module_from_spec(spec)
    sys.modules[f"arm_{tree.name}"] = package
    spec.loader.exec_module(package)
    from types import SimpleNamespace
    config = SimpleNamespace(hidden_sizes=(512, 256, 128), codebook_num=3,
                             codebook_size=(256, 256, 256), codebook_dim=32, dropout=0.0,
                             beta=0.25, vq_type="vq", ema_decay=0.99, fix_code_embs=False,
                             sk_epsilon=0.003, sk_iters=50)
    # the mixed family exposes MixedRQVAE; the historical E32 arm exposes the plain RQVAE
    model_cls = getattr(package, "MixedRQVAE", None) or getattr(package, "RQVAE")
    if model_cls.__name__ == "MixedRQVAE":
        model = model_cls(config, in_dim=768, curvature=1.0)
    else:
        model = model_cls(config, in_dim=768)          # the E32 arm has no curvature argument
    state = torch.load(checkpoint, map_location="cpu")
    # the checkpoints wrap the tensors: {"state_dict": ..., "epoch": ..., "collision": ...}.
    # Loading the wrapper as if it were the state silently matched nothing, which is how the first
    # version of this analysis ended up describing randomly initialised encoders.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    ignored = {"role_weights"}          # passed explicitly by the router instead of loaded
    missing = [key for key in missing if not key.endswith("role_weights")]
    unexpected = [key for key in unexpected if key not in ignored]
    if missing or unexpected:
        raise SystemExit(f"{tree.name}: checkpoint does not match the model; "
                         f"missing {missing[:4]}, unexpected {unexpected[:4]}")
    print(f"{tree.name}: checkpoint loaded ({len(state)} tensors)", flush=True)
    model.eval()
    return model, package


def main() -> int:
    GRAPHS.mkdir(parents=True, exist_ok=True)
    VECTORS.mkdir(parents=True, exist_ok=True)
    embeddings = torch.from_numpy(np.load(EMBEDDING))
    n_items = len(embeddings)

    manifest = {"n_items": n_items, "k": K, "arms": {}}
    for name, spec_arm in ARMS.items():
        tree, checkpoint, mode = spec_arm["tree"], spec_arm["checkpoint"], spec_arm["mode"]
        if not checkpoint.is_file():
            print(f"{name}: no checkpoint yet ({checkpoint})", flush=True)
            continue
        model, package = load_tree_model(tree, checkpoint, mode)
        with torch.no_grad():
            z = model.encoder(embeddings)
            if mode is None:
                readout, _, _, tokens = model.rq(z)
            else:
                # the arm's own router, reading the same frozen table the training used. The
                # package's __init__ does not re-export the router, so it is imported by name.
                import importlib
                router = importlib.import_module(f"arm_{tree.name}.role_geometry")
                role_table = router.load_role_table(OUT / "orca_roles_frozen.csv", ROLE_SHA)
                weights, fallbacks = router.build_weights(mode, n_items, role_table)
                readout, _, _, tokens = model.rq(
                    z, weights=weights, scales=getattr(model, "geometry_scales", (1.0, 1.0, 1.0)))
        np.save(VECTORS / f"{name}_z.npy", z.cpu().numpy())
        np.save(VECTORS / f"{name}_q.npy", readout.cpu().numpy())
        np.save(VECTORS / f"{name}_tokens.npy", tokens.cpu().numpy())
        manifest["arms"][name] = {"z": f"vectors/{name}_z.npy", "q": f"vectors/{name}_q.npy",
                                  "checkpoint": str(checkpoint.relative_to(REPO))}
        print(f"{name}: encoded {n_items} items -> z {tuple(z.shape)} q {tuple(readout.shape)}", flush=True)

    # the behaviour graph
    frame = pd.read_parquet(TRAIN, columns=["seen_history", "target"])
    pairs = set()
    for history, target in zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy(np.int64)):
        if history is None or len(history) == 0:
            continue
        pairs.add((int(history[-1]), int(target)))
    edges = np.asarray(sorted(pairs), dtype=np.int64)
    # the same item set as the k-NN graphs: item ids stay as they are, so a node index means the
    # same item in all three graphs and the orbit counts can be compared row by row
    behaviour = sorted({(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges})
    write_graph(GRAPHS / "behaviour.in", n_items, behaviour)
    manifest["graphs"] = {"behaviour": {"nodes": n_items, "edges": len(behaviour),
                                        "file": "graphs/behaviour.in"}}

    # the two k-NN graphs per arm, same k, same rule
    from sklearn.neighbors import NearestNeighbors
    # The comparison has to be density-free: every k-NN graph is cut to the same number of edges as
    # the behaviour graph, keeping the closest ones, so a difference between G_z and G_q cannot come
    # from one of them being denser.
    budget = len(behaviour)
    for name in manifest["arms"]:
        for kind in ("z", "q"):
            vectors = np.load(VECTORS / f"{name}_{kind}.npy").astype(np.float32)
            index = NearestNeighbors(n_neighbors=K + 1, algorithm="auto").fit(vectors)
            distances, neighbours = index.kneighbors(vectors)
            candidates: dict[tuple[int, int], float] = {}
            for source, (row, row_distances) in enumerate(zip(neighbours, distances)):
                for target, distance in zip(row[1:], row_distances[1:]):
                    if int(target) == source:
                        continue          # identical vectors can put the point itself first
                    a, b = (source, int(target)) if source < target else (int(target), source)
                    key = (a, b)
                    if key not in candidates or distance < candidates[key]:
                        candidates[key] = float(distance)
            kept = sorted(candidates.items(), key=lambda item: item[1])[:budget]
            pairs = sorted(key for key, _ in kept)
            write_graph(GRAPHS / f"{name}_{kind}.in", n_items, pairs)
            manifest["graphs"][f"{name}_{kind}"] = {"nodes": n_items, "edges": len(pairs),
                                                   "k": K, "budget": budget, "candidates": len(candidates),
                                                   "file": f"graphs/{name}_{kind}.in"}
            print(f"{name}_{kind}: {len(candidates)} candidate pairs -> kept {len(pairs)} "
                  f"(same edge budget as the behaviour graph)", flush=True)

    (OUT / "structure_graphs.json").write_text(json.dumps(manifest, indent=1))
    print(f"written: {(OUT / 'structure_graphs.json').relative_to(REPO)}")
    return 0


def write_graph(path: Path, n: int, pairs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{n} {len(pairs)}\n")
        for a, b in pairs:
            handle.write(f"{a} {b}\n")


if __name__ == "__main__":
    raise SystemExit(main())
