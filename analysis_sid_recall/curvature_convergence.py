"""Does the three-level curvature estimate converge as the sample grows?

The authoritative sweep (`curvature_analysis.py`) draws a fixed number of edges
per quotient graph. This script answers a different question: with a NESTED
sample — one seeded random order of the graph's edges, reduced to its first
``k`` entries — do the level's curvature estimates, their null, and the
correlation with the recall converge as ``k`` and the number of null replicates
grow?

Per (arm, level):

* one seeded random order of the quotient's undirected edges;
* the curvature of each of the first ``--max-edges`` edges, using the SAME
  per-edge estimator as the sweep (exact hop distance, 1/2-hop transport bound,
  both from `tools/research/science/quotient_curvature.py`);
* the running mean at each ``k`` in the ladder;
* ``--max-nulls`` size-preserving label shuffles, each on its own order, with
  ``--null-edges`` edges, so the null mean/std can be read at 1, 2, 4, ... replicates.

Because the ladder is nested, every ``k`` uses a prefix of the same draw: the
running mean is the estimator the sweep computes, and the differences between
successive ``k`` measure its convergence rather than a different sample.

Outputs (under `analysis_sid_recall/curvature/convergence/`):

* `per_arm_convergence.json` — every arm's ladder and null replicates;
* `convergence_summary.json` — per level: the estimate and correlation at each
  ``k``, the null at each replicate count, and the noise-vs-signal ratio;
* `convergence.md` — the same, for humans.

Run:  `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
        analysis_sid_recall/curvature_convergence.py --max-edges 128 --max-nulls 8`
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis_sid_recall.curvature_analysis import (  # noqa: E402
    build_behaviour_graph,
    bootstrap_ci,
    permutation_p,
    quotient,
    spearman,
)
from tools.research.science.quotient_curvature import (  # noqa: E402
    bounded_transport_cost,
    exact_hop_costs,
)

PAIRS = REPO / "analysis_sid_recall" / "sid_recall.csv"
OUT = REPO / "analysis_sid_recall" / "curvature" / "convergence"
EDGE_LADDER = (8, 16, 32, 64, 128)
NULL_REPLICATES = (1, 2, 3, 4, 6, 8)


def undirected_edges(graph: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """The upper-triangle edge list of a symmetric quotient."""
    coo = graph.tocoo()
    keep = coo.row < coo.col
    return coo.row[keep].astype(np.int64), coo.col[keep].astype(np.int64)


def edge_curvatures(graph: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray,
                    order: np.ndarray, count: int) -> np.ndarray:
    """The curvature of the first ``count`` edges of ``order``.

    ``rows``/``cols`` are the graph's own edges, so the hop distance of an
    endpoint pair is exactly 1 and the curvature is ``1 - W1``. The sweep's
    estimator computes that distance with a full shortest-path search per edge
    and always gets 1; ``--verify-hop`` checks the claim on a sample instead of
    trusting it.
    """
    values = np.empty(min(count, order.size), dtype=np.float64)
    for position, index in enumerate(order[:count]):
        a = int(rows[index])
        b = int(cols[index])
        values[position] = 1.0 - bounded_transport_cost(graph, a, b)
    return values


def verify_hop_claim(graph: sp.csr_matrix, rows: np.ndarray, cols: np.ndarray,
                     samples: int = 20, seed: int = 0) -> float:
    """The largest |d(a, b) - 1| over sampled edges, with the exact search."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(rows.size, size=min(samples, rows.size), replace=False)
    worst = 0.0
    for index in picks.tolist():
        distance = exact_hop_costs(graph, int(rows[index]),
                                   np.asarray([int(cols[index])]))[0]
        worst = max(worst, abs(float(distance) - 1.0))
    return worst


def running_means(values: np.ndarray, ladder: tuple) -> dict:
    """The cumulative mean at each ladder size, ignoring undefined edges."""
    out = {}
    for size in ladder:
        if size > values.size:
            break
        window = values[:size]
        window = window[np.isfinite(window)]
        out[str(size)] = float(window.mean()) if window.size else None
    return out


def table_features(table: dict, adjacency, edges, index, nodes, max_edges: int,
                   max_nulls: int, null_edges: int, seed: int, ladder,
                   verify_hop: bool = False) -> dict:
    """The nested-sample ladder of one SID table."""
    width = len(next(iter(table.values())))
    levels = min(3, width)
    assignments = {}
    sizes_by_level = {}
    for level in range(1, levels + 1):
        groups: dict[tuple, list[int]] = {}
        for item, row in table.items():
            position = index.get(int(item))
            if position is None:
                continue
            groups.setdefault(tuple(row[:level]), []).append(position)
        keys = list(groups)
        assignment = np.full(len(nodes), -1, dtype=np.int64)
        for number, key in enumerate(keys):
            for position in groups[key]:
                assignment[position] = number
        assignments[level] = (keys, assignment)
        sizes_by_level[level] = np.array([int((assignment == number).sum())
                                          for number in range(len(keys))])

    out = {"levels": {}, "codes_used": [], "items": len(table)}
    rows_all = [tuple(row) for row in table.values()]
    out["codes_used"] = [len({row[level] for row in rows_all}) for level in range(levels)]

    for level in range(1, levels + 1):
        keys, assignment = assignments[level]
        sizes = sizes_by_level[level]
        n_groups = len(keys)
        graph = quotient(edges, assignment, n_groups)
        rows, cols = undirected_edges(graph)
        n_edges = rows.size
        if n_edges == 0:
            out["levels"][str(level)] = {"status": "NO_EDGES"}
            continue
        if verify_hop:
            deviation = verify_hop_claim(graph, rows, cols, samples=5, seed=seed + level)
            if deviation > 1e-9:
                raise ValueError(f"an edge pair is not adjacent: |d-1| = {deviation}")
        rng = np.random.default_rng(seed + 13 * level)
        order = rng.permutation(n_edges)
        values = edge_curvatures(graph, rows, cols, order, max_edges)
        entry = {"status": "MEASURED", "prefix_groups": n_groups,
                 "quotient_edges": int(n_edges),
                 "observed": running_means(values, ladder),
                 "per_edge_std": float(np.nanstd(values)) if values.size else None,
                 "per_edge_n": int(np.isfinite(values).sum())}

        nulls = []
        null_rng = np.random.default_rng(seed + 1000 * level)
        for _ in range(max_nulls):
            shuffle_order = null_rng.permutation(len(nodes))
            shuffled = np.empty_like(assignment)
            cursor = 0
            for number in range(n_groups):
                size = int(sizes[number])
                shuffled[shuffle_order[cursor:cursor + size]] = number
                cursor += size
            null_graph = quotient(edges, shuffled, n_groups)
            null_rows, null_cols = undirected_edges(null_graph)
            if null_rows.size == 0:
                continue
            null_order = null_rng.permutation(null_rows.size)
            null_values = edge_curvatures(null_graph, null_rows, null_cols,
                                          null_order, null_edges)
            nulls.append(running_means(null_values, (null_edges,))[str(null_edges)])
        entry["null_samples"] = nulls
        out["levels"][str(level)] = entry
    return out


def summarise(records: list, args, ladder, null_counts) -> dict:
    summary = {"pairs": len(records), "max_edges": args.max_edges,
               "max_nulls": args.max_nulls, "null_edges": args.null_edges,
               "needs_review": [], "levels": {}}
    for level in (1, 2, 3):
        rows = [r for r in records if str(level) in r["features"]["levels"]
                and r["features"]["levels"][str(level)].get("status") == "MEASURED"]
        recalls = [r["recall"] for r in rows]
        block = {"arms": len(rows), "estimate_vs_edges": {}, "null_vs_replicates": {}}
        kmax = str(max(int(k) for k in ladder))
        for size in ladder:
            key = str(size)
            estimates = [r["features"]["levels"][str(level)]["observed"].get(key)
                         for r in rows]
            paired = [(x, y) for x, y in zip(estimates, recalls) if x == x and x is not None]
            if len(paired) < 10:
                continue
            x = [p[0] for p in paired]
            y = [p[1] for p in paired]
            rho = spearman(x, y)
            block["estimate_vs_edges"][key] = {
                "n": len(paired),
                "mean": float(np.mean(x)), "sd": float(np.std(x)),
                "median": float(np.median(x)),
                "spearman": rho,
                "permutation_p": permutation_p(x, y, rho, seed=args.seed),
                "bootstrap_ci95": bootstrap_ci(x, y, seed=args.seed),
                "se_mean": float(np.std(x) / np.sqrt(len(x))),
            }
        for replicates in null_counts:
            null_means = []
            for r in rows:
                samples = [s for s in r["features"]["levels"][str(level)].get("null_samples", [])
                           if s is not None]
                if len(samples) < replicates:
                    continue
                null_means.append(float(np.mean(samples[:replicates])))
            if len(null_means) < 10:
                continue
            block["null_vs_replicates"][str(replicates)] = {
                "n": len(null_means),
                "mean": float(np.mean(null_means)),
                "std": float(np.std(null_means)),
            }
        # convergence: how far the ladder's smaller samples sit from the largest
        anchor = block["estimate_vs_edges"].get(kmax)
        if anchor:
            block["convergence"] = {}
            for key, value in block["estimate_vs_edges"].items():
                block["convergence"][key] = {
                    "delta_mean_vs_max": value["mean"] - anchor["mean"],
                    "delta_rho_vs_max": value["spearman"] - anchor["spearman"],
                    "noise_ratio": (value["se_mean"] / abs(value["mean"])
                                    if value["mean"] else None),
                }
        # the convergence verdict, from the ladder itself
        verdict = {}
        if anchor and len(block["estimate_vs_edges"]) >= 3:
            keys = sorted((int(k) for k in block["estimate_vs_edges"]))
            half = keys[-2]
            small = block["estimate_vs_edges"][str(half)]
            se = small.get("se_mean") or 0.0
            delta_mean = abs(small["mean"] - anchor["mean"])
            verdict["estimate_delta_vs_noise"] = (
                float(delta_mean / se) if se > 0 else None)
            verdict["estimate_halving_converged"] = bool(
                se > 0 and delta_mean <= 2.0 * se)
            delta_rho = abs(small["spearman"] - anchor["spearman"])
            verdict["rho_delta_halving"] = float(delta_rho)
            verdict["rho_halving_converged"] = bool(
                delta_rho <= 0.05
                and (small["spearman"] > 0) == (anchor["spearman"] > 0))
            nulls = {int(k): v for k, v in block["null_vs_replicates"].items()}
            if len(nulls) >= 2:
                top = sorted(nulls)[-1]
                previous = sorted(nulls)[-2]
                spread = nulls[top]["std"]
                verdict["null_std_halving_change"] = (
                    float(abs(nulls[top]["std"] - nulls[previous]["std"]) / spread)
                    if spread > 0 else None)
                verdict["null_replicates_converged"] = bool(
                    spread > 0
                    and abs(nulls[top]["std"] - nulls[previous]["std"]) <= 0.1 * spread)
        block["convergence_verdict"] = verdict
        # does the estimator's own noise dominate the across-arm signal?
        stds = [r["features"]["levels"][str(level)].get("per_edge_std")
                for r in rows if r["features"]["levels"][str(level)].get("per_edge_std")]
        ns = [r["features"]["levels"][str(level)].get("per_edge_n") or 0 for r in rows]
        if stds and anchor:
            se = float(np.mean([s / np.sqrt(max(n, 1)) for s, n in zip(stds, ns)]))
            block["noise_floor"] = {
                "per_edge_std_mean": float(np.mean(stds)),
                "estimator_se_at_max": se,
                "across_arm_sd": anchor["sd"],
                "signal_to_noise": float(anchor["sd"] / se) if se > 0 else None,
            }
        summary["levels"][str(level)] = block
    return summary


def write_markdown(summary: dict, path: Path) -> None:
    lines = ["# Convergence of the level curvature estimates", "",
             f"nested edge ladder {summary['max_edges']} edges, "
             f"null replicates up to {summary['max_nulls']} "
             f"({summary['null_edges']} edges each), {summary['pairs']} arms.", ""]
    for level in ("1", "2", "3"):
        block = summary["levels"].get(level)
        if not block:
            continue
        lines += [f"## Level {level}", "",
                  "| edges | mean curvature | sd across arms | Spearman vs recall | p | 95% CI |",
                  "|---|---|---|---|---|---|"]
        for key, value in block["estimate_vs_edges"].items():
            lo, hi = value["bootstrap_ci95"]
            lines.append(f"| {key} | {value['mean']:+.4f} | {value['sd']:.4f} | "
                         f"{value['spearman']:+.3f} | {value['permutation_p']:.3f} | "
                         f"[{lo:+.3f}, {hi:+.3f}] |")
        converge = block.get("convergence", {})
        if converge:
            lines += ["", "| edges | Δmean vs max | Δrho vs max |", "|---|---|---|"]
            for key, value in converge.items():
                lines.append(f"| {key} | {value['delta_mean_vs_max']:+.4f} | "
                             f"{value['delta_rho_vs_max']:+.3f} |")
        verdict = block.get("convergence_verdict") or {}
        if verdict:
            lines += ["", "| criterion | value | converged |", "|---|---|---|",
                      f"| |mean(k/2) - mean(k)| / SE | "
                      f"{verdict.get('estimate_delta_vs_noise'):.2f} | "
                      f"{verdict.get('estimate_halving_converged')} |"
                      if verdict.get("estimate_delta_vs_noise") is not None else
                      "| estimate | not computable | |",
                      f"| |rho(k/2) - rho(k)| | {verdict.get('rho_delta_halving'):.3f} | "
                      f"{verdict.get('rho_halving_converged')} |"]
            if verdict.get("null_std_halving_change") is not None:
                lines.append(f"| null std change at the last replicate step | "
                             f"{verdict['null_std_halving_change']:.3f} | "
                             f"{verdict.get('null_replicates_converged')} |")
        floor = block.get("noise_floor")
        if floor:
            lines += ["", f"estimator SE at the largest sample: {floor['estimator_se_at_max']:.4f}; "
                          f"across-arm sd: {floor['across_arm_sd']:.4f}; "
                          f"signal-to-noise {floor['signal_to_noise']:.2f}."]
        lines += ["", "| null replicates | null mean | null sd |", "|---|---|---|"]
        for key, value in block["null_vs_replicates"].items():
            lines.append(f"| {key} | {value['mean']:+.4f} | {value['std']:.4f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-edges", type=int, default=128)
    parser.add_argument("--max-nulls", type=int, default=8)
    parser.add_argument("--null-edges", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--verify-hop", action="store_true",
                        help="check d(a,b)=1 on a sample of edges per level")
    parser.add_argument("--from-records", action="store_true",
                        help="re-summarise per_arm_convergence.json without measuring")
    args = parser.parse_args()
    ladder = tuple(size for size in EDGE_LADDER if size <= args.max_edges)
    null_counts = tuple(count for count in NULL_REPLICATES if count <= args.max_nulls)

    if args.from_records:
        records = json.loads((OUT / "per_arm_convergence.json").read_text())
        summary = summarise(records, args, ladder, null_counts)
        (OUT / "convergence_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_markdown(summary, OUT / "convergence.md")
        print(f"re-summarised {len(records)} records; written:", OUT / "convergence.md")
        return 0

    rows = list(csv.DictReader(PAIRS.open()))
    if args.limit:
        rows = rows[:args.limit]
    nodes, index, adjacency, edges = build_behaviour_graph()
    print(f"arms: {len(rows)} | behaviour graph: {len(nodes)} items, "
          f"{int(adjacency.nnz // 2)} edges | ladder {ladder}", flush=True)

    def read_bytes(row):
        sid_path = REPO / row["sid_paths"].split(" | ")[0]
        if not sid_path.is_file():
            sid_path = REPO / "analysis_sid_recall" / "sids" / row["sid_file"].split("/")[-1]
        return (sid_path, sid_path.read_bytes()) if sid_path.is_file() else None

    with ThreadPoolExecutor(max_workers=16) as pool:
        loaded = list(pool.map(read_bytes, rows))
    print(f"tables read: {sum(1 for item in loaded if item)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.time()
    for number, row in enumerate(rows):
        if loaded[number] is None:
            continue
        sid_path, raw = loaded[number]
        table = json.loads(raw)
        features = table_features(table, adjacency, edges, index, nodes,
                                  max_edges=args.max_edges, max_nulls=args.max_nulls,
                                  null_edges=args.null_edges,
                                  seed=args.seed + 97 * number, ladder=ladder,
                                  verify_hop=args.verify_hop)
        records.append({"variant": row["variant"], "recall": float(row["test_recall@10"]),
                        "sid_path": str(sid_path.relative_to(REPO)),
                        "sid_sha256": row["sid_sha256"], "features": features})
        if (number + 1) % 5 == 0:
            elapsed = time.time() - started
            eta = elapsed / (number + 1) * (len(rows) - number - 1)
            (OUT / "progress.txt").write_text(
                f"{number + 1}/{len(rows)} elapsed={elapsed:.0f}s eta={eta:.0f}s\n",
                encoding="utf-8")
            print(f"  {number + 1}/{len(rows)} elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    (OUT / "per_arm_convergence.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8")
    summary = summarise(records, args, ladder, null_counts)
    (OUT / "convergence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, OUT / "convergence.md")
    print("written:", OUT / "convergence.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
