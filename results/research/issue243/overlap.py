"""Issue #244 §4: does one item's frozen representation serve several local subgraphs?

For a centre item that appears in several sampled local subgraphs, the frozen representation
must keep every one of them faithful at once -- that is the main hypothesis of this issue.
The measurement:

    E_t = mean over ordered pairs (i != j) inside subgraph t with finite g of |(D/g)^2 - 1|

per centre: mean E, worst-subgraph E, p90, and the fraction of its subgraphs with E <= the
pre-frozen threshold.  Every subgraph of a centre is scored with the SAME frozen embeddings
and gate weights; nothing is re-encoded per subgraph.

Diagnostics use the same checkpoint and never retrain: each single expert alone, and the
gate frozen to uniform weights.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csgraph

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manifold  # noqa: E402
import metrics  # noqa: E402
import topology  # noqa: E402
import teacher as T  # noqa: E402

OUT = HERE / "teacher"
THRESHOLD = 0.5           # frozen before training: "simultaneous fidelity" means E_t <= 0.5
CENTRES = 400
SUBGRAPHS_PER_CENTRE = 3
SEED = 243301


def close_pairs(nodes, out_csr, undirected):
    """Undirected shortest paths restricted to one subgraph's own node set."""
    index = {int(node): k for k, node in enumerate(nodes)}
    rows, cols, data = [], [], []
    for node in nodes:
        start, end = out_csr.indptr[node], out_csr.indptr[node + 1]
        for target in out_csr.indices[start:end]:
            target = int(target)
            if target in index:
                # symmetric: the distance we compare against is on the undirected projection
                rows.append(index[int(node)]); cols.append(index[target]); data.append(1.0)
                rows.append(index[target]); cols.append(source); data.append(1.0)
    if not rows:
        return {}
    import scipy.sparse as sp
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    distance = csgraph.shortest_path(matrix, method="D", unweighted=True)
    return distance


def main() -> int:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    out_csr, in_csr, embeddings, structural = T.load_all()
    n = embeddings.shape[0]
    blob = torch.load(OUT / "teacher.pt", map_location="cpu")
    model = T.MixedGeometryTeacher(embeddings.shape[1], structural.shape[1]).to(T.DEVICE)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    full_x = torch.cat([torch.tensor(embeddings, device=T.DEVICE),
                        torch.tensor(structural, device=T.DEVICE)], dim=1)
    edges = T.edge_index(out_csr)
    with torch.no_grad():
        expert_emb = model.embeddings(full_x, edges)
    undirected = (out_csr + out_csr.T).tocsr()
    undirected.data[:] = 1

    degree = np.diff(out_csr.indptr)
    active = np.where(degree >= 2)[0]
    centres = np.sort(rng.choice(active, size=min(CENTRES, len(active)), replace=False))

    records = []
    for centre in centres:
        samples = []
        for scale in range(SUBGRAPHS_PER_CENTRE):
            radius = 1 if scale == 0 else 2
            sample = topology.ego_induced(int(centre), out_csr, in_csr, rng, radius=radius)
            if len(sample["nodes"]) < 3:
                continue
            samples.append({**sample, "radius": radius})
        if len(samples) < 2:
            continue

        nodes_all = sorted({int(x) for s in samples for x in s["nodes"].tolist()})
        position = {node: k for k, node in enumerate(nodes_all)}
        with torch.no_grad():
            descriptors = model.descriptor.encode_two_scales(
                nodes_all, out_csr, in_csr, rng, embeddings, structural)
            gate = model.weights_for(descriptors)
            gate_all = gate  # descriptor rows follow nodes_all order

        per_subgraph = []
        for sample in samples:
            nodes = [int(x) for x in sample["nodes"].tolist()]
            distance = close_pairs(nodes, out_csr, undirected)
            if not distance:
                continue
            local_index = [position[x] for x in nodes]
            ii, jj, gaps = [], [], []
            for a in range(len(nodes)):
                for b in range(len(nodes)):
                    if a == b:
                        continue
                    gap = distance[a, b]
                    if np.isfinite(gap) and gap > 0:
                        ii.append(local_index[a]); jj.append(local_index[b]); gaps.append(float(gap))
            if len(ii) < 6:
                continue
            # model.distance indexes the FULL embedding matrix, so these must be item ids,
            # while the gate rows are indexed by the descriptor order
            index_i = torch.tensor([nodes[a] for a in ii], device=T.DEVICE)
            index_j = torch.tensor([nodes[b] for b in jj], device=T.DEVICE)
            weight_i = gate_all[torch.tensor([position[nodes[a]] for a in ii], device=T.DEVICE)]
            weight_j = gate_all[torch.tensor([position[nodes[b]] for b in jj], device=T.DEVICE)]
            gap = torch.tensor(gaps, device=T.DEVICE)
            with torch.no_grad():
                d_geo, per_expert, _ = model.distance(
                    expert_emb, weight_i, weight_j, index_i, index_j)
                uniform = torch.full_like(per_expert, 1.0 / per_expert.shape[1])
                uniform_d = (uniform * per_expert.pow(2)).sum(dim=1).clamp(min=0).sqrt()
            distortion = ((d_geo / gap) ** 2 - 1.0).abs()
            per_subgraph.append({
                "radius": sample["radius"],
                "nodes": len(nodes),
                "pairs": len(ii),
                "E": float(distortion.mean()),
                "E_uniform_gate": float(((uniform_d / gap) ** 2 - 1).abs().mean()),
                "E_single_expert": [float(((per_expert[:, e] / gap) ** 2 - 1).abs().mean())
                                    for e in range(per_expert.shape[1])],
            })
        if len(per_subgraph) < 2:
            continue
        errors = [s["E"] for s in per_subgraph]
        records.append({
            "centre": int(centre),
            "degree": int(degree[centre]),
            "subgraphs": len(per_subgraph),
            "mean_E": float(np.mean(errors)),
            "worst_E": float(np.max(errors)),
            "p90_E": float(np.quantile(errors, 0.9)),
            "fraction_within_threshold": float(np.mean([e <= THRESHOLD for e in errors])),
            "mean_E_uniform_gate": float(np.mean([s["E_uniform_gate"] for s in per_subgraph])),
            "mean_E_single_expert": np.mean([s["E_single_expert"] for s in per_subgraph], axis=0).tolist(),
        })

    if not records:
        raise SystemExit("no centre with two or more usable subgraphs")
    records.sort(key=lambda r: r["degree"])
    half = len(records) // 2
    low_degree = records[:half]
    high_degree = records[half:]

    def summarise(group):
        return {
            "centres": len(group),
            "mean_E": float(np.mean([r["mean_E"] for r in group])),
            "worst_E_mean": float(np.mean([r["worst_E"] for r in group])),
            "p90_E_mean": float(np.mean([r["p90_E"] for r in group])),
            "simultaneous_fidelity": float(np.mean([r["fraction_within_threshold"] for r in group])),
            "uniform_gate_mean_E": float(np.mean([r["mean_E_uniform_gate"] for r in group])),
            "single_expert_mean_E": np.mean([r["mean_E_single_expert"] for r in group], axis=0).tolist(),
        }

    report = {
        "threshold": THRESHOLD,
        "centres": len(records),
        "subgraphs_per_centre": SUBGRAPHS_PER_CENTRE,
        "overall": summarise(records),
        "low_degree_half": summarise(low_degree),
        "high_degree_half": summarise(high_degree),
        "per_centre": records[:50],
    }
    (OUT / "overlap.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("centres", "overall", "low_degree_half", "high_degree_half")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
