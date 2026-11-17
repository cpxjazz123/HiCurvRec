"""Does each five-node graphlet type prefer its own geometric bias?

The edge probes grouped behaviour edges by how many triangles they sit in.  This
probe uses the graphlet taxonomies themselves: nine strict five-node shape
classes sampled from the TRAIN behaviour graph (`graphlet_instances.npz`), each
with 300 instances.  For a fixed SID (F0's exported tokens) it builds the k=10
graph under the mixed distance

    d_alpha^2 = alpha d_E^2 + (1 - alpha) d_H^2 / 4

and asks two questions per shape:

* **edge recovery** -- the share of that shape's distinct edges that survive;
* **instance recovery** -- the share of instances whose *all ten* edges survive,
  which is the literal "this graphlet is preserved" event.

Instances are split 50/50 by a hash of their node tuple: alpha is chosen on the
fit half and scored on the other.  The bootstrap resamples *instances*, not
edges, because edges inside one instance are not independent.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INSTANCES = REPO / "results" / "research" / "tiger_cluster_graph" / "graphlet_instances.npz"

MODULE = importlib.util.spec_from_file_location(
    "readout_knn_geometry", HERE / "readout_knn_geometry.py")
knn = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(knn)

BOOTSTRAP = 2000
SEED = 42
MIN_INSTANCES = 80


def key(left: int, right: int):
    return (left, right) if left < right else (right, left)


def instance_edges(instance) -> set:
    nodes = [int(node) for node in instance]
    return {key(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]}


def main():
    expmap0, hyp_sq, _, _ = knn.load_contracts()
    payload = np.load(knn.VECTORS)
    q = torch.from_numpy(payload["q"].astype(np.float32)).cuda()
    euclidean = q[:, :knn.EUCLIDEAN_DIM]
    ball = expmap0(q[:, knn.EUCLIDEAN_DIM:], 1.0)

    edges = knn.load_graph()
    print(f"behaviour edges={len(edges)}; building {len(knn.ALPHAS)} k-NN graphs", flush=True)
    graphs = knn.build_graphs(euclidean, ball, hyp_sq, knn.ALPHAS)

    edge_set = {key(int(a), int(b)) for a, b in edges}
    overall = {f"{alpha:.1f}": sum(1 for edge in edge_set if edge in graphs[alpha]) / len(edge_set)
               for alpha in knn.ALPHAS}
    overall_best = max(overall, key=lambda name: overall[name])
    print("overall recovery:", {k: round(v, 5) for k, v in overall.items()},
          "best", overall_best)

    archive = np.load(INSTANCES)
    report = {
        "schema": "genrec.motif_geometry_preference.v1",
        "alphas": list(knn.ALPHAS),
        "overall_recovery": overall,
        "overall_best_alpha": float(overall_best),
        "bootstrap": BOOTSTRAP,
        "motifs": {},
        "provenance": {
            "instances": str(INSTANCES.relative_to(REPO)),
            "instances_sha256": knn.sha256_of(INSTANCES),
            "vectors_sha256": knn.sha256_of(knn.VECTORS),
            "sid_sha256": knn.sha256_of(knn.SID_JSON),
            "train_parquet_sha256": knn.sha256_of(knn.TRAIN),
            "split": "hash(node tuple) % 2; alpha fitted on half 0, scored on half 1",
        },
    }
    rng = np.random.default_rng(SEED)
    for name in sorted(archive.files):
        instances = archive[name]
        if instances.shape[0] < MIN_INSTANCES:
            continue
        per_instance = [instance_edges(row) for row in instances]
        halves = np.array([
            int(hashlib.sha256(":".join(str(int(node)) for node in sorted(row)).encode())
                .hexdigest()[:8], 16) % 2
            for row in instances])
        fit_index = np.flatnonzero(halves == 0)
        eval_index = np.flatnonzero(halves == 1)

        def curves(index):
            edge_rate = {}
            full_rate = {}
            for alpha in knn.ALPHAS:
                graph = graphs[alpha]
                rates = [len(edgeset & graph) / len(edgeset) for edgeset in
                         (per_instance[i] for i in index)]
                edge_rate[f"{alpha:.1f}"] = float(np.mean(rates))
                full_rate[f"{alpha:.1f}"] = float(np.mean(
                    [1.0 if edgeset <= graph else 0.0 for edgeset in
                     (per_instance[i] for i in index)]))
            return edge_rate, full_rate

        fit_edge, fit_full = curves(fit_index)
        eval_edge, eval_full = curves(eval_index)
        best_edge = max(fit_edge, key=lambda k: fit_edge[k])
        best_full = max(fit_full, key=lambda k: fit_full[k])

        def score(edgeset, graph, metric):
            if metric == "full":
                return 1.0 if edgeset <= graph else 0.0
            return len(edgeset & graph) / len(edgeset)

        def cluster_bootstrap(metric, best_key, index):
            base = np.array([score(per_instance[i], graphs[float(overall_best)], metric) for i in index])
            chosen = np.array([score(per_instance[i], graphs[float(best_key)], metric) for i in index])
            draws = rng.integers(0, base.size, size=(BOOTSTRAP, base.size))
            return (chosen[draws] - base[draws]).mean(axis=1)

        boot_edge = cluster_bootstrap("edge", best_edge, eval_index)
        boot_full = cluster_bootstrap("full", best_full, eval_index)
        report["motifs"][name] = {
            "instances": int(instances.shape[0]),
            "fit_instances": int(fit_index.size), "eval_instances": int(eval_index.size),
            "fit_edge_curve": fit_edge, "eval_edge_curve": eval_edge,
            "fit_full_curve": fit_full, "eval_full_curve": eval_full,
            "fit_best_alpha_edge": float(best_edge),
            "fit_best_alpha_full": float(best_full),
            "eval_edge_gain": float(eval_edge[best_edge] - eval_edge[overall_best]),
            "eval_edge_gain_ci95": [float(np.quantile(boot_edge, 0.025)),
                                    float(np.quantile(boot_edge, 0.975))],
            "eval_full_gain": float(eval_full[best_full] - eval_full[overall_best]),
            "eval_full_gain_ci95": [float(np.quantile(boot_full, 0.025)),
                                    float(np.quantile(boot_full, 0.975))],
        }
        entry = report["motifs"][name]
        print(f"{name:16s} n={entry['instances']:3d} "
              f"fit_alpha_edge={best_edge} gain={entry['eval_edge_gain']:+.4f} "
              f"CI=[{entry['eval_edge_gain_ci95'][0]:+.4f},{entry['eval_edge_gain_ci95'][1]:+.4f}] | "
              f"fit_alpha_full={best_full} gain={entry['eval_full_gain']:+.4f} "
              f"CI=[{entry['eval_full_gain_ci95'][0]:+.4f},{entry['eval_full_gain_ci95'][1]:+.4f}]")

    # How many motif types pick a bias that differs from the shared one *and* keeps
    # its gain out of sample?  That count is the claim's support.
    supported = [
        name for name, entry in report["motifs"].items()
        if float(entry["fit_best_alpha_edge"]) != float(overall_best)
        and entry["eval_edge_gain_ci95"][0] > 0]
    report["supported_motifs"] = supported
    report["supported_count"] = len(supported)
    report["motifs_total"] = len(report["motifs"])
    print(f"motifs with a different bias and a positive out-of-sample gain: "
          f"{len(supported)}/{len(report['motifs'])} {supported}")

    out = HERE / "motif_geometry_preference.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
