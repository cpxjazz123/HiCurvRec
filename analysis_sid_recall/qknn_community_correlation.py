"""q-kNN community recovery vs Stage3 recall across archived TIGER arms.

For every archived arm whose checkpoint has the standard three VQ codebooks
(`rq.vq_layers.{0,1,2}.embed.weight`, each 256 x 32):

1. rebuild the 32-d quantised readout q from the arm's codebooks and exported SID tokens;
2. build a global k=10 Euclidean q-kNN graph;
3. measure recovery of real TRAIN behaviour edges inside and between the fixed Louvain
   communities of the behaviour graph;
4. correlate inside/between recovery and their gap with test_recall@10, controlling q-kNN
   edge count and L3 collision.

Mixed/hyperbolic checkpoints (`rq.levels.*.e_embed/h_embed`, non-standard codebook layouts)
are excluded because their quantised readout needs an architecture-specific geodesic decoder.

    /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/qknn_community_correlation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "results" / "research" / "tiger_cluster_graph"))
import analysis  # noqa: E402
from sid_structural_fidelity import load_behavior_graph  # noqa: E402

R2 = HERE / "r2"
OUT_JSON = HERE / "results" / "qknn_community_correlation.json"
OUT_MD = HERE / "results" / "qknn_community_correlation.md"
K = 10
PERMUTATIONS = 5000
N_ITEMS = 24587


def checkpoint_vq(path: Path) -> tuple[np.ndarray, dict] | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    keys = sorted(
        (key for key in state if key.startswith("rq.vq_layers.") and key.endswith(".embed.weight")),
        key=lambda key: int(key.split(".")[2]),
    )
    if len(keys) != 3:
        return None
    codebooks = []
    for key in keys:
        value = state[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (256, 32):
            return None
        codebooks.append(value.detach().cpu().numpy().astype(np.float32))
    return np.stack(codebooks), payload


def q_vector(codebooks: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    if (tokens < 0).any() or (tokens > 255).any():
        raise ValueError("SID tokens outside 0..255")
    return sum(codebooks[level][tokens[:, level]] for level in range(3)).astype(np.float32)


def qknn_edges(q: np.ndarray, k: int = K) -> np.ndarray:
    n = len(q)
    if torch.cuda.is_available():
        tensor = torch.from_numpy(q).to("cuda", dtype=torch.float32)
        squared = (tensor * tensor).sum(dim=1)
        indices = []
        chunk = 2048
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            rows = torch.arange(start, end, device="cuda")
            distance = squared[start:end, None] + squared[None, :] - 2.0 * tensor[start:end] @ tensor.T
            distance[torch.arange(end - start, device="cuda"), rows] = torch.inf
            nearest = torch.topk(distance, k + 1, dim=1, largest=False).indices[:, :k]
            indices.append(nearest.cpu().numpy())
        indices = np.concatenate(indices, axis=0)
        del tensor
        torch.cuda.empty_cache()
    else:
        model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean", n_jobs=-1)
        model.fit(q)
        indices = model.kneighbors(q, return_distance=False)[:, 1:]
    left = np.repeat(np.arange(n, dtype=np.int64), k)
    right = indices.reshape(-1).astype(np.int64)
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    unique = np.unique(low * n + high)
    return np.stack([unique // n, unique % n], axis=1)


def partial_multiple(first: np.ndarray, second: np.ndarray, controls: np.ndarray, rank: bool = True) -> float:
    if rank:
        first = analysis.average_ranks(first)
        second = analysis.average_ranks(second)
        controls = np.column_stack([analysis.average_ranks(controls[:, i]) for i in range(controls.shape[1])])
    design = np.column_stack([np.ones(len(first)), controls])
    residual_first = first - design @ np.linalg.lstsq(design, first, rcond=None)[0]
    residual_second = second - design @ np.linalg.lstsq(design, second, rcond=None)[0]
    return analysis.pearson(residual_first, residual_second)


def correlation(values: np.ndarray, recall: np.ndarray, controls: np.ndarray) -> dict:
    usable = np.isfinite(values) & np.isfinite(recall) & np.isfinite(controls).all(axis=1)
    first = values[usable]
    second = recall[usable]
    control = controls[usable]
    observed = {"spearman": analysis.spearman(first, second), "pearson": analysis.pearson(first, second)}
    rng = np.random.default_rng(42)
    beats_s = beats_r = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(second)
        beats_s += abs(analysis.spearman(first, shuffled)) >= abs(observed["spearman"])
        beats_r += abs(analysis.pearson(first, shuffled)) >= abs(observed["pearson"])
    return {
        "n": int(len(first)),
        "spearman": observed["spearman"],
        "pearson": observed["pearson"],
        "p_spearman": (beats_s + 1) / (PERMUTATIONS + 1),
        "p_pearson": (beats_r + 1) / (PERMUTATIONS + 1),
        "partial_spearman": partial_multiple(first, second, control, rank=True),
        "partial_pearson": partial_multiple(first, second, control, rank=False),
    }


def main() -> int:
    nodes, behavior_edges, _ = load_behavior_graph()
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes))
    graph.add_edges_from(map(tuple, behavior_edges.tolist()))
    communities = nx.community.louvain_communities(graph, seed=42, resolution=1.0)
    comm = np.full(nodes, -1, dtype=np.int64)
    for index, members in enumerate(communities):
        comm[list(members)] = index
    within = comm[behavior_edges[:, 0]] == comm[behavior_edges[:, 1]]
    inside_total = int(within.sum())
    cross_total = int((~within).sum())
    active_communities = int(len(np.unique(comm[np.bincount(behavior_edges.ravel(), minlength=nodes) > 0])))
    print(f"behaviour edges {len(behavior_edges)}; inside {inside_total}; cross {cross_total}; "
          f"active communities {len(np.unique(comm[np.bincount(behavior_edges.ravel(), minlength=nodes) > 0]))}",
          flush=True)

    rows = []
    skipped = []
    for directory in sorted((path for path in R2.iterdir() if path.is_dir()), key=lambda path: path.name):
        checkpoint = directory / "rqvae_best.pth"
        sid_path = directory / "item_sids.json"
        recall_path = directory / "recall.json"
        if not (checkpoint.is_file() and sid_path.is_file() and recall_path.is_file()):
            continue
        packed = checkpoint_vq(checkpoint)
        if packed is None:
            skipped.append({"number": directory.name, "reason": "not standard 3x(256,32) vq_layers"})
            continue
        codebooks, payload = packed
        tokens = analysis.tokens_of(sid_path, nodes)
        if (tokens < 0).any():
            skipped.append({"number": directory.name, "reason": "incomplete SID table"})
            continue
        q = q_vector(codebooks, tokens)
        edges = qknn_edges(q)
        edge_set = {(int(left), int(right)) for left, right in edges.tolist()}
        recovered = np.array([(int(left), int(right)) in edge_set for left, right in behavior_edges], dtype=bool)
        inside_rate = float(recovered[within].mean()) if inside_total else float("nan")
        cross_rate = float(recovered[~within].mean()) if cross_total else float("nan")
        recall = json.loads(recall_path.read_text()).get("test_recall@10")
        if recall is None:
            skipped.append({"number": directory.name, "reason": "no test_recall@10"})
            continue
        collision = analysis.collision_of(tokens)
        rows.append({
            "number": directory.name,
            "variant": payload.get("arm_variant"),
            "recall": float(recall),
            "q_edges": int(len(edges)),
            "collision": float(collision),
            "inside_recovered": int(recovered[within].sum()),
            "cross_recovered": int(recovered[~within].sum()),
            "inside_rate": inside_rate,
            "cross_rate": cross_rate,
            "gap": inside_rate - cross_rate,
        })
        print(f"{directory.name} recall {recall:.5f} q_edges {len(edges)} "
              f"inside {inside_rate:.5f} cross {cross_rate:.5f}", flush=True)

    recalls = np.array([row["recall"] for row in rows])
    q_edges = np.array([row["q_edges"] for row in rows], dtype=float)
    collision = np.array([row["collision"] for row in rows], dtype=float)
    controls = np.column_stack([q_edges, collision])
    result = {
        "nodes": int(nodes),
        "behavior_edges": int(len(behavior_edges)),
        "inside_edges": inside_total,
        "cross_edges": cross_total,
        "active_communities": active_communities,
        "k": K,
        "arms": len(rows),
        "skipped": skipped,
        "recall_min": float(np.min(recalls)) if len(recalls) else None,
        "recall_max": float(np.max(recalls)) if len(recalls) else None,
        "q_edges_min": int(np.min(q_edges)) if len(q_edges) else None,
        "q_edges_max": int(np.max(q_edges)) if len(q_edges) else None,
        "rows": rows,
        "correlations": {
            "inside_rate": correlation(np.array([row["inside_rate"] for row in rows]), recalls, controls),
            "cross_rate": correlation(np.array([row["cross_rate"] for row in rows]), recalls, controls),
            "gap": correlation(np.array([row["gap"] for row in rows]), recalls, controls),
        },
        "sensitivity_collision_lt_0_9": {
            name: correlation(
                np.array([row[key] for row in rows])[collision < 0.9],
                recalls[collision < 0.9],
                controls[collision < 0.9],
            )
            for name, key in (("inside_rate", "inside_rate"), ("cross_rate", "cross_rate"), ("gap", "gap"))
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)

    lines = [
        "# q-kNN community recovery vs Stage3 recall",
        "",
        "For every archived arm with the standard 3 x (256, 32) VQ codebooks: rebuild q from "
        "the arm's codebooks and exported SID tokens, build a global k=10 Euclidean q-kNN graph, "
        "and measure recovery of TRAIN behaviour edges inside/between fixed Louvain communities.",
        "",
        f"Behaviour graph: **{nodes}** nodes, **{len(behavior_edges)}** edges; "
        f"inside **{inside_total}**, cross **{cross_total}**. "
        f"Communities: **{len(communities)}** Louvain labels, **{active_communities}** active communities "
        f"(the rest are isolated singleton nodes).",
        "",
        f"Standard-vq arms: **{len(rows)}**. q-kNN edges range **{int(np.min(q_edges))}–{int(np.max(q_edges))}**; "
        f"recall range **{np.min(recalls):.6f}–{np.max(recalls):.6f}**.",
        "",
        "## Per-arm recovery",
        "",
        "| arm | variant | recall | q edges | inside rate | cross rate | gap | collision |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['number']} | {row['variant'] or ''} | {row['recall']:.6f} | {row['q_edges']} | "
            f"{row['inside_rate']:.5f} | {row['cross_rate']:.5f} | {row['gap']:+.5f} | {row['collision']:.4f} |"
        )
    lines += [
        "",
        "## Correlations with test_recall@10",
        "",
        "Partial correlations control **q-kNN edge count** and **L3 collision**.",
        "",
        "| quantity | n | Spearman | p | Pearson | p | partial Spearman | partial Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in result["correlations"].items():
        lines.append(
            f"| {name} | {value['n']} | {value['spearman']:+.4f} | {value['p_spearman']:.4f} | "
            f"{value['pearson']:+.4f} | {value['p_pearson']:.4f} | {value['partial_spearman']:+.4f} | "
            f"{value['partial_pearson']:+.4f} |"
        )
    lines += [
        "",
        "Excluding arms with L3 collision >= 0.90 (sensitivity):",
        "",
        "| quantity | n | Spearman | p | Pearson | p | partial Spearman | partial Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in result["sensitivity_collision_lt_0_9"].items():
        lines.append(
            f"| {name} | {value['n']} | {value['spearman']:+.4f} | {value['p_spearman']:.4f} | "
            f"{value['pearson']:+.4f} | {value['p_pearson']:.4f} | {value['partial_spearman']:+.4f} | "
            f"{value['partial_pearson']:+.4f} |"
        )
    lines += [
        "",
        "## Skipped archives",
        "",
        "| arm | reason |",
        "|---|---|",
    ]
    for item in skipped:
        lines.append(f"| {item['number']} | {item['reason']} |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
