"""Issue #241 stage 1 gate (C5/C6/C12): does the teacher beat plain structure on unseen edges?

Three edge sets, never mixed:

    target_edges   held out from the ego graph, used for supervision   -> in-sample fit
    eval_unseen    eval users' transitions that the visible graph does  -> the real test
                   not contain at all
    structural     a non-trained baseline: rank candidates by the number of
                   two-hop paths i -> x -> j in the ego graph, tie-broken by
                   degree.  Any teacher that cannot beat this on unseen edges
                   has learned nothing beyond local structure.

C6 fails closed: if the teacher does not beat the structural baseline on the unseen
edges, the issue's own instruction is to report and stop before formal Stage2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from teacher import DEVICE, Teacher, ego_sample, load_graph  # noqa: E402

GRAPH = HERE / "graph"
OUT = HERE / "teacher"
SAMPLE_ROOTS = 1500
SEED = 241004
TOP_K = (10, 50)


def rank_of(scores: np.ndarray, truth: int) -> int:
    return int((scores > scores[truth]).sum()) + 1


def main() -> int:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    out_csr, in_csr = load_graph()
    n_items = int(out_csr.shape[0])
    checkpoint = torch.load(OUT / "teacher.pt", map_location="cpu")
    model = Teacher(checkpoint["n_items"]).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    target = np.load(GRAPH / "target_edges.npy")
    eval_edges = np.load(GRAPH / "eval_edges.npy")
    # truly unseen: eval edges absent from the visible transition set
    visible_pairs = set()
    coo = out_csr.tocoo()
    for src, dst in zip(coo.row, coo.col):
        visible_pairs.add((int(src), int(dst)))
    unseen_mask = np.array([(int(s), int(d)) not in visible_pairs for s, d in eval_edges])
    eval_unseen = eval_edges[unseen_mask]

    # two-hop structural baseline matrix: A^2 counts over the ego graph
    two_hop = (out_csr @ out_csr).tocsr()

    results = {}
    for name, edges in (("target", target), ("eval_unseen", eval_unseen)):
        roots = np.unique(edges[:, 0])
        if len(roots) > SAMPLE_ROOTS:
            roots = rng.choice(roots, size=SAMPLE_ROOTS, replace=False)
        by_root: dict[int, list[int]] = {}
        for src, dst in edges:
            if int(src) in set(int(r) for r in roots):
                by_root.setdefault(int(src), []).append(int(dst))
        teacher_ranks, struct_ranks = [], []
        for root in by_root:
            sample = ego_sample(int(root), out_csr, in_csr, rng)
            with torch.no_grad():
                vector = model.encode([sample], out_csr, in_csr)
                logits = model.head_out(vector)[0].detach().cpu().numpy()
            row = two_hop.getrow(int(root))
            struct = np.zeros(n_items, dtype=np.float64)
            struct[row.indices] = row.data
            struct += 1e-9 * np.log1p(np.diff(out_csr.indptr))
            for truth in by_root[root]:
                teacher_ranks.append(rank_of(logits, truth))
                struct_ranks.append(rank_of(struct, truth))
        teacher_ranks = np.array(teacher_ranks)
        struct_ranks = np.array(struct_ranks)
        entry = {
            "pairs": int(len(teacher_ranks)),
            "roots": int(len(by_root)),
            "teacher_mrr@10": float(np.mean(1.0 / np.minimum(teacher_ranks, 10))),
            "teacher_mrr": float(np.mean(1.0 / teacher_ranks)),
            "teacher_median_rank": float(np.median(teacher_ranks)),
            "structural_mrr@10": float(np.mean(1.0 / np.minimum(struct_ranks, 10))),
            "structural_mrr": float(np.mean(1.0 / struct_ranks)),
            "structural_median_rank": float(np.median(struct_ranks)),
        }
        for k in TOP_K:
            entry[f"teacher_recall@{k}"] = float((teacher_ranks <= k).mean())
            entry[f"structural_recall@{k}"] = float((struct_ranks <= k).mean())
        results[name] = entry
        print(name, json.dumps(entry, indent=1), flush=True)

    verdict = {
        "beats_structural_on_unseen": (
            results["eval_unseen"]["teacher_recall@10"] > results["eval_unseen"]["structural_recall@10"]
            and results["eval_unseen"]["teacher_mrr"] > results["eval_unseen"]["structural_mrr"]
        ),
        "eval_unseen_edges": int(len(eval_unseen)),
        "eval_edges_total": int(len(eval_edges)),
        "visible_pairs": int(len(visible_pairs)),
    }
    results["verdict"] = verdict
    (OUT / "teacher_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("VERDICT", json.dumps(verdict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
