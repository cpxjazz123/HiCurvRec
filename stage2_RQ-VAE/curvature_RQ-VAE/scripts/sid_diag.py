"""Frozen Stage2 SID and geometry diagnostics.

Run sid_diag.py for the CPU SID audit, or sid_diag_geometry.py for D14-D17.
Both hardcoded runners read frozen inputs and avoid GPU contention.
"""
from __future__ import annotations

import csv
import copy
import hashlib
import html
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
import torch

ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
SOURCE_DIR = ROOT / "stage2_RQ-VAE/curvature_RQ-VAE"
OUTPUT_DIR = ROOT / "results/stage2_RQ-VAE/curvature_RQ-VAE/sid_diag"
MAX_BEHAVIOUR_PAIRS = 200_000
N_GEOMETRY_PAIRS = 100_000
SEED = 42
DEVICE = torch.device("cpu")
DIAGNOSTIC_DEVICE = torch.device("cpu")

sys.path.insert(0, str(SOURCE_DIR))
import curvature_config as experiment  # noqa: E402
import train_rqvae as trainer  # noqa: E402
from model import RQVAE  # noqa: E402
from model.layers import (
    _expmap0_tangent,
    _hyperbolic_residual,
    _pairwise_poincare_distance_tangents,
    _poincare_distance_tangent_pairs,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def relative_to_root(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.sum() == 0:
        return 0.0
    ordered = np.sort(values)
    n = len(ordered)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float(np.sum((2.0 * ranks - n - 1.0) * ordered) / (n * ordered.sum()))


def entropy_bits(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def lcp_length(a: np.ndarray, b: np.ndarray) -> int:
    for index, (left, right) in enumerate(zip(a, b)):
        if int(left) != int(right):
            return index
    return min(len(a), len(b))


def contiguous_unique_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.unique(values, axis=0, return_index=True, return_counts=True)


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()) if values.size else 0.0,
        "q10": float(np.quantile(values, 0.10)) if values.size else 0.0,
        "q25": float(np.quantile(values, 0.25)) if values.size else 0.0,
        "median": float(np.median(values)) if values.size else 0.0,
        "mean": float(values.mean()) if values.size else 0.0,
        "q75": float(np.quantile(values, 0.75)) if values.size else 0.0,
        "q90": float(np.quantile(values, 0.90)) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
    }


def write_svg_bar(
    path: Path,
    title: str,
    xlabels: list[str],
    series: list[tuple[str, list[float]]],
    y_label: str,
) -> None:
    width, height = 940, 520
    left, right, top, bottom = 82, 36, 58, 112
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = max((max(values, default=0.0) for _, values in series), default=1.0)
    ymax = max(1.0, max_value * 1.12)
    colors = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="20" font-weight="600" fill="currentColor">{html.escape(title)}</text>',
        f'<text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="13" fill="currentColor">{html.escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="var(--border)"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="var(--border)"/>',
    ]
    for tick in range(6):
        value = ymax * tick / 5
        y = top + plot_h * (1.0 - tick / 5)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="var(--border)" stroke-opacity="0.55"/>')
        lines.append(f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="currentColor">{value:.3g}</text>')
    group_w = plot_w / max(1, len(xlabels))
    bar_group_w = group_w * 0.78
    bar_w = bar_group_w / max(1, len(series))
    for x_index, label in enumerate(xlabels):
        center = left + (x_index + 0.5) * group_w
        lines.append(f'<text x="{center:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="10" fill="currentColor">{html.escape(label)}</text>')
        for series_index, (_, values) in enumerate(series):
            value = values[x_index] if x_index < len(values) else 0.0
            bar_h = plot_h * value / ymax
            x = center - bar_group_w / 2 + series_index * bar_w
            y = top + plot_h - bar_h
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(1.0, bar_w - 1):.2f}" height="{bar_h:.2f}" fill="{colors[series_index % len(colors)]}"/>')
    legend_y = height - 30
    total_legend_w = len(series) * 170
    legend_x = max(left, (width - total_legend_w) / 2)
    for index, (label, _) in enumerate(series):
        x = legend_x + index * 170
        lines.append(f'<rect x="{x:.1f}" y="{legend_y - 10}" width="12" height="12" fill="{colors[index % len(colors)]}"/>')
        lines.append(f'<text x="{x + 18:.1f}" y="{legend_y}" font-size="12" fill="currentColor">{html.escape(label)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_svg_lines(
    path: Path,
    title: str,
    xlabels: list[str],
    series: list[tuple[str, list[float]]],
    y_label: str,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    width, height = 1080, 540
    left, right, top, bottom = 82, 28, 58, 140
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max((max(values, default=y_min) for _, values in series), default=1.0)
    ymin = y_min
    ymax = max(ymin + 1e-9, y_max if y_max is not None else maximum * 1.1)
    colors = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="20" font-weight="600" fill="currentColor">{html.escape(title)}</text>',
        f'<text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="13" fill="currentColor">{html.escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="var(--border)"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="var(--border)"/>',
    ]
    for tick in range(6):
        fraction = tick / 5
        value = ymin + fraction * (ymax - ymin)
        y = top + plot_h * (1.0 - fraction)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="var(--border)" stroke-opacity="0.55"/>')
        lines.append(f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="currentColor">{value:.3g}</text>')
    for x_index, label in enumerate(xlabels):
        x = left + x_index * plot_w / max(1, len(xlabels) - 1)
        lines.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="10" fill="currentColor">{html.escape(label)}</text>')
    for series_index, (label, values) in enumerate(series):
        coords = []
        for x_index, value in enumerate(values):
            x = left + x_index * plot_w / max(1, len(xlabels) - 1)
            y = top + plot_h * (1.0 - (value - ymin) / (ymax - ymin))
            coords.append(f"{x:.2f},{y:.2f}")
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{colors[series_index % len(colors)]}"/>')
        lines.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{colors[series_index % len(colors)]}" stroke-width="2"/>')
    legend_cols = 3
    legend_y = height - 74
    for index, (label, _) in enumerate(series):
        x = left + (index % legend_cols) * 320
        y = legend_y + (index // legend_cols) * 22
        lines.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 20}" y2="{y - 4}" stroke="{colors[index % len(colors)]}" stroke-width="3"/>')
        lines.append(f'<text x="{x + 27}" y="{y}" font-size="11" fill="currentColor">{html.escape(label)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def code_usage(tokens: np.ndarray, n_codes: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for level in range(tokens.shape[1]):
        counts = np.bincount(tokens[:, level], minlength=n_codes)
        total = int(counts.sum())
        level_name = f"L{level + 1}"
        positive = counts[counts > 0]
        top_shares = {
            str(k): float(np.sort(counts)[::-1][:k].sum() / total) if total else 0.0
            for k in (1, 5, 10, 20, 50)
        }
        entropy = entropy_bits(counts)
        summary = {
            "items": total,
            "codewords": int(n_codes),
            "used_codewords": int((counts > 0).sum()),
            "unused_codewords": int((counts == 0).sum()),
            "entropy_bits": entropy,
            "normalized_entropy": entropy / math.log2(n_codes),
            "gini": gini(counts),
            "top_k_share": top_shares,
            "used_frequency": percentile_summary(positive),
        }
        summaries[level_name] = summary
        for code_id, count in enumerate(counts):
            rows.append({"level": level_name, "code_id": code_id, "item_count": int(count), "share": float(count / total) if total else 0.0})
    return rows, summaries


def prefix_audit(tokens: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    node_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    sizes_by_depth: list[np.ndarray] = []
    depth_names = ["L1", "L1-L2", "L1-L2-L3"]
    for depth, name in enumerate(depth_names, start=1):
        prefixes, _, counts = contiguous_unique_rows(tokens[:, :depth])
        sizes_by_depth.append(counts)
        summaries[name] = {
            "unique_prefixes": int(len(prefixes)),
            "items": int(len(tokens)),
            "collision_pairs": int(sum(int(n) * (int(n) - 1) // 2 for n in counts)),
            "clusters": percentile_summary(counts),
            "singleton_clusters": int((counts == 1).sum()),
            "singleton_cluster_fraction": float((counts == 1).mean()) if len(counts) else 0.0,
        }
        for prefix, count in zip(prefixes, counts):
            padded = list(map(int, prefix)) + [None] * (3 - depth)
            node_rows.append({
                "depth": name,
                "code_l1": padded[0],
                "code_l2": padded[1],
                "code_l3": padded[2],
                "cluster_items": int(count),
            })

    for parent_depth, parent_name, child_name in ((1, "L1", "L1-to-L2"), (2, "L1-L2", "L1L2-to-L3")):
        parent_prefixes, parent_inverse = np.unique(tokens[:, :parent_depth], axis=0, return_inverse=True)
        child_counts: list[int] = []
        conditional = 0.0
        for parent_index, parent in enumerate(parent_prefixes):
            members = np.flatnonzero(parent_inverse == parent_index)
            child_ids = tokens[members, parent_depth]
            unique_child, child_n = np.unique(child_ids, return_counts=True)
            child_count = int(len(unique_child))
            child_counts.append(child_count)
            conditional += len(members) / len(tokens) * entropy_bits(child_n)
            key = tuple(map(int, parent))
            branch_rows.append({
                "relation": child_name,
                "parent_l1": key[0],
                "parent_l2": key[1] if parent_depth == 2 else None,
                "items_in_parent": int(len(members)),
                "distinct_children": child_count,
            })
        summaries[child_name] = {
            "parent_nodes": int(len(child_counts)),
            "children_per_parent": percentile_summary(np.asarray(child_counts)),
            "parents_with_one_child": int((np.asarray(child_counts) == 1).sum()),
            "conditional_entropy_bits": float(conditional),
        }
    summaries["conditional_entropy_bits"] = {
        "H(L2|L1)": summaries["L1-to-L2"]["conditional_entropy_bits"],
        "H(L3|L1,L2)": summaries["L1L2-to-L3"]["conditional_entropy_bits"],
    }
    return node_rows, branch_rows, summaries


def popularity_groups(frequencies: np.ndarray) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    n_items = len(frequencies)
    ids = np.arange(n_items, dtype=np.int64)
    order = np.lexsort((ids, -frequencies))
    rank = np.empty(n_items, dtype=np.int64)
    rank[order] = np.arange(n_items)
    first_end = int(math.ceil(0.20 * n_items))
    second_end = int(math.ceil(0.50 * n_items))
    labels = np.full(n_items, "tail", dtype=object)
    labels[rank < second_end] = "mid"
    labels[rank < first_end] = "head"
    return labels, {
        "rule": "Sort items by target interaction frequency descending, item_id ascending for ties; head is first 20% of item ranks, mid next 30%, tail remaining 50%.",
        "tie_handling": "Equal-frequency items may straddle a boundary; item_id is the deterministic tie-break.",
        "head_rank_end_exclusive": first_end,
        "mid_rank_end_exclusive": second_end,
    }


def behavior_pair_audit(
    sources: np.ndarray,
    successors: np.ndarray,
    tokens: np.ndarray,
    popularity_rank: np.ndarray,
    item_popularity_group: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], np.ndarray]:
    edges, counts = np.unique(np.column_stack((sources, successors)), axis=0, return_counts=True)
    rng = np.random.default_rng(SEED)
    take = min(MAX_BEHAVIOUR_PAIRS, len(edges))
    sample_indices = np.sort(rng.choice(len(edges), size=take, replace=False)) if take else np.array([], dtype=np.int64)
    selected = edges[sample_indices]
    selected_counts = counts[sample_indices]
    n_items = len(tokens)
    rank_decile = np.minimum(9, (popularity_rank * 10 // n_items)).astype(np.int8)
    all_items = np.arange(n_items, dtype=np.int64)
    positive_lcps = np.empty(take, dtype=np.int8)
    matched_negatives = np.empty(take, dtype=np.int64)
    random_negatives = np.empty(take, dtype=np.int64)
    for idx, (source, positive) in enumerate(selected):
        positive_lcps[idx] = lcp_length(tokens[source, :3], tokens[positive, :3])
        pool = all_items[rank_decile == rank_decile[positive]]
        candidate = int(rng.choice(pool))
        while candidate == int(source) or candidate == int(positive):
            candidate = int(rng.choice(pool))
        matched_negatives[idx] = candidate
        candidate = int(rng.integers(n_items))
        while candidate == int(source) or candidate == int(positive):
            candidate = int(rng.integers(n_items))
        random_negatives[idx] = candidate
    matched_lcps = np.fromiter((lcp_length(tokens[int(s), :3], tokens[int(n), :3]) for s, n in zip(selected[:, 0], matched_negatives)), dtype=np.int8, count=take)
    random_lcps = np.fromiter((lcp_length(tokens[int(s), :3], tokens[int(n), :3]) for s, n in zip(selected[:, 0], random_negatives)), dtype=np.int8, count=take)

    pair_rows: list[dict[str, Any]] = []
    for idx, ((source, positive), occurrence_count) in enumerate(zip(selected, selected_counts)):
        pair_rows.append({
            "source_item": int(source),
            "positive_successor": int(positive),
            "positive_event_count": int(occurrence_count),
            "positive_successor_group": str(item_popularity_group[positive]),
            "positive_rank_decile": int(rank_decile[positive]),
            "positive_lcp": int(positive_lcps[idx]),
            "matched_negative": int(matched_negatives[idx]),
            "matched_negative_lcp": int(matched_lcps[idx]),
            "random_negative": int(random_negatives[idx]),
            "random_negative_lcp": int(random_lcps[idx]),
        })

    classes = {"positive": positive_lcps, "popularity_matched_negative": matched_lcps, "random_negative": random_lcps}
    summary_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for name, lcps in classes.items():
        summary = {"sampled_pairs": int(len(lcps)), "mean_lcp": float(lcps.mean()) if len(lcps) else 0.0}
        for depth in range(1, 4):
            summary[f"share_lcp_ge_{depth}"] = float((lcps >= depth).mean()) if len(lcps) else 0.0
        for value in range(4):
            summary[f"lcp_{value}_fraction"] = float((lcps == value).mean()) if len(lcps) else 0.0
        summaries[name] = summary
        summary_rows.append({"pair_class": name, **summary})
    summaries.update({
        "transition_events": int(len(sources)),
        "unique_directed_pairs": int(len(edges)),
        "sampled_unique_pairs": int(take),
        "sampling": "Uniform without replacement over unique directed transitions, RNG seed 42; transition multiplicity retained as metadata.",
    })
    return pair_rows, summary_rows, summaries, rank_decile


def distance_audit(
    encoded: torch.Tensor,
    tokens: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    n_items = len(tokens)
    rng = np.random.default_rng(SEED)
    pair_count = N_GEOMETRY_PAIRS
    left = rng.integers(0, n_items, size=pair_count, dtype=np.int64)
    right = rng.integers(0, n_items, size=pair_count, dtype=np.int64)
    same = left == right
    while same.any():
        right[same] = rng.integers(0, n_items, size=int(same.sum()), dtype=np.int64)
        same = left == right
    left_t = torch.from_numpy(left)
    right_t = torch.from_numpy(right)
    tangent = encoded.detach().cpu().contiguous()
    d_h = np.empty(pair_count, dtype=np.float64)
    d_e = np.empty(pair_count, dtype=np.float64)
    batch_size = 20_000
    with torch.no_grad():
        for start in range(0, pair_count, batch_size):
            stop = min(pair_count, start + batch_size)
            x = tangent[left_t[start:stop]]
            y = tangent[right_t[start:stop]]
            d_h[start:stop] = _poincare_distance_tangent_pairs(x, y, 1.0).numpy()
            d_e[start:stop] = torch.linalg.vector_norm(x - y, dim=-1).numpy()
    scale = float(np.median(d_h) / max(np.median(d_e), np.finfo(np.float64).tiny))
    d_e_scaled = d_e * scale
    lcps = np.fromiter((lcp_length(tokens[int(i), :3], tokens[int(j), :3]) for i, j in zip(left, right)), dtype=np.int8, count=pair_count)
    rho, rho_p = scipy.stats.spearmanr(d_h, d_e)
    rho = float(rho) if np.isfinite(rho) else 0.0
    rho_p = float(rho_p) if np.isfinite(rho_p) else 1.0
    sample_rows = [
        {
            "item_i": int(left[i]),
            "item_j": int(right[i]),
            "d_hyperbolic_c1": float(d_h[i]),
            "d_euclidean_tangent": float(d_e[i]),
            "d_euclidean_scale_matched": float(d_e_scaled[i]),
            "lcp": int(lcps[i]),
        }
        for i in range(pair_count)
    ]
    quantile_rows: list[dict[str, Any]] = []
    quantile_summary: dict[str, Any] = {}
    for metric, distances in (("hyperbolic_c1", d_h), ("euclidean_scale_matched", d_e_scaled)):
        quantile_id = pd.qcut(pd.Series(distances), q=10, labels=False, duplicates="drop").to_numpy(dtype=np.int64)
        metric_result = {}
        for quantile in sorted(np.unique(quantile_id)):
            mask = quantile_id == quantile
            row: dict[str, Any] = {
                "metric": metric,
                "distance_decile": int(quantile + 1),
                "pair_count": int(mask.sum()),
                "distance_min": float(distances[mask].min()),
                "distance_median": float(np.median(distances[mask])),
                "distance_max": float(distances[mask].max()),
                "mean_lcp": float(lcps[mask].mean()),
            }
            for depth in range(1, 4):
                row[f"share_lcp_ge_{depth}"] = float((lcps[mask] >= depth).mean())
            quantile_rows.append(row)
            metric_result[str(int(quantile + 1))] = row
        quantile_summary[metric] = metric_result
    return sample_rows, quantile_rows, {
        "pair_count": int(pair_count),
        "sampling": "Uniform random non-self item pairs, RNG seed 42.",
        "curvature": 1.0,
        "euclidean_scale_factor_to_match_hyperbolic_median": scale,
        "median_hyperbolic_distance": float(np.median(d_h)),
        "median_euclidean_distance_before_scale": float(np.median(d_e)),
        "spearman_hyperbolic_vs_tangent_euclidean": rho,
        "spearman_pvalue": rho_p,
        "hyperbolic_distance_summary": percentile_summary(d_h),
        "tangent_euclidean_distance_summary": percentile_summary(d_e),
        "quantile_prefix_sharing": quantile_summary,
    }


def residual_audit(
    model: RQVAE,
    encoded: torch.Tensor,
    tokens: np.ndarray,
    popularity_group: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rq = model.rq
    n_items = len(tokens)
    l1_counts = np.bincount(tokens[:, 0], minlength=rq.codebook_sizes[0])
    # Bucket quartiles are ranked over L1 codewords, with code ID breaking ties.
    code_ids = np.arange(len(l1_counts), dtype=np.int64)
    bucket_order = np.lexsort((code_ids, l1_counts))
    bucket_quartile = np.empty(len(l1_counts), dtype=np.int8)
    bucket_quartile[bucket_order] = np.minimum(np.arange(len(l1_counts)) * 4 // len(l1_counts), 3)
    quartile_names = ["Q1_sparse", "Q2", "Q3", "Q4_dense"]
    item_cluster_size = l1_counts[tokens[:, 0]]
    item_cluster_quartile = np.asarray([quartile_names[int(q)] for q in bucket_quartile[tokens[:, 0]]], dtype=object)

    residual = encoded.detach().clone()
    per_item_rows: list[dict[str, Any]] = []
    raw_errors: dict[int, np.ndarray] = {}
    raw_residual_norms: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for level, layer in enumerate(rq.vq_layers):
            curvature = layer.get_curvature()
            source = residual
            if rq.working_radii[level] == 0.0:
                pinned = source
                target_norm = None
            else:
                target_norm = rq._radius_for_level(level, curvature, source)
                pinned = rq._pin_to_radius(source, target_norm)
            ids = torch.from_numpy(tokens[:, level]).long()
            code = layer.embed_code(ids)
            errors = _poincare_distance_tangent_pairs(pinned, code, curvature).cpu().numpy()
            mapped_residual = _hyperbolic_residual(pinned, code, curvature)
            if target_norm is not None:
                residual = rq._restore_norm(mapped_residual, source, target_norm)
            else:
                residual = mapped_residual
            residual_norm = torch.linalg.vector_norm(residual, dim=-1).cpu().numpy()
            origin_distance = _poincare_distance_tangent_pairs(
                residual, torch.zeros_like(residual), curvature
            ).cpu().numpy()
            raw_errors[level] = errors
            raw_residual_norms[level] = residual_norm
            for item_id in range(n_items):
                per_item_rows.append({
                    "item_id": item_id,
                    "popularity_group": str(popularity_group[item_id]),
                    "l1_prefix_size": int(item_cluster_size[item_id]),
                    "l1_prefix_size_quartile": str(item_cluster_quartile[item_id]),
                    "level": f"L{level + 1}",
                    "quantization_error_hyperbolic": float(errors[item_id]),
                    "residual_tangent_norm_after": float(residual_norm[item_id]),
                    "residual_origin_hyperbolic_distance_after": float(origin_distance[item_id]),
                })

    summary_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "l1_cluster_size_quartile_definition": "Rank 256 L1 codeword cluster sizes ascending, code ID tie-break, split into four equal codeword groups.",
        "l1_cluster_size_quartile_boundaries": {},
        "by_level": {},
    }
    for q, name in enumerate(quartile_names):
        sizes = l1_counts[bucket_quartile == q]
        summary["l1_cluster_size_quartile_boundaries"][name] = percentile_summary(sizes)
    popularity_names = ["head", "mid", "tail"]
    for level in range(3):
        level_summary: dict[str, Any] = {}
        for group in popularity_names:
            for qname in quartile_names:
                mask = (popularity_group == group) & (item_cluster_quartile == qname)
                errors = raw_errors[level][mask]
                norms = raw_residual_norms[level][mask]
                row = {
                    "level": f"L{level + 1}",
                    "popularity_group": group,
                    "l1_prefix_size_quartile": qname,
                    "item_count": int(mask.sum()),
                    "mean_quantization_error_hyperbolic": float(errors.mean()) if len(errors) else 0.0,
                    "median_quantization_error_hyperbolic": float(np.median(errors)) if len(errors) else 0.0,
                    "q90_quantization_error_hyperbolic": float(np.quantile(errors, 0.90)) if len(errors) else 0.0,
                    "mean_residual_tangent_norm_after": float(norms.mean()) if len(norms) else 0.0,
                }
                summary_rows.append(row)
                level_summary[f"{group}/{qname}"] = row
        overall_errors = raw_errors[level]
        level_summary["all_items"] = {
            "mean_quantization_error_hyperbolic": float(overall_errors.mean()),
            "median_quantization_error_hyperbolic": float(np.median(overall_errors)),
            "q90_quantization_error_hyperbolic": float(np.quantile(overall_errors, 0.90)),
            "mean_residual_tangent_norm_after": float(raw_residual_norms[level].mean()),
        }
        summary["by_level"][f"L{level + 1}"] = level_summary
    return per_item_rows, summary_rows, summary


def paired_lower_auc(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return float("nan")
    delta = left[valid] - right[valid]
    return float(((delta < 0).sum() + 0.5 * (delta == 0).sum()) / len(delta))


def paired_geometry_features(
    vectors: torch.Tensor,
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    chunk_size: int = 40_000,
) -> dict[str, np.ndarray]:
    keys = (
        "angle_radians",
        "radial_difference",
        "radial_cosh_term",
        "angular_cosh_term",
        "angular_cosh_fraction",
        "hyperbolic_distance",
        "euclidean_tangent_distance",
        "angle_only_hyperbolic_distance",
        "angle_only_euclidean_scale2",
        "law_distance_abs_error",
    )
    output = {key: np.empty(len(left_ids), dtype=np.float64) for key in keys}
    left = torch.as_tensor(left_ids, device=DIAGNOSTIC_DEVICE, dtype=torch.long)
    right = torch.as_tensor(right_ids, device=DIAGNOSTIC_DEVICE, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(left_ids), chunk_size):
            stop = min(len(left_ids), start + chunk_size)
            # Double precision prevents acosh cancellation for clipped near-boundary points.
            x = vectors[left[start:stop]].to(torch.float64)
            y = vectors[right[start:stop]].to(torch.float64)
            nx = torch.linalg.vector_norm(x, dim=-1).clamp_min(1e-12)
            ny = torch.linalg.vector_norm(y, dim=-1).clamp_min(1e-12)
            cosine = ((x * y).sum(-1) / (nx * ny)).clamp(-1.0, 1.0)
            angle = torch.acos(cosine)
            x_point = _expmap0_tangent(x, 1.0)
            y_point = _expmap0_tangent(y, 1.0)
            point_norm_x = torch.linalg.vector_norm(x_point, dim=-1).clamp(max=1.0 - 1e-6)
            point_norm_y = torch.linalg.vector_norm(y_point, dim=-1).clamp(max=1.0 - 1e-6)
            rho_x, rho_y = 2.0 * torch.atanh(point_norm_x), 2.0 * torch.atanh(point_norm_y)
            radial_term = torch.cosh(rho_x - rho_y)
            angular_term = torch.sinh(rho_x) * torch.sinh(rho_y) * (1.0 - cosine)
            cosh_distance = (radial_term + angular_term).clamp_min(1.0)
            law_distance = torch.acosh(cosh_distance)
            direct_distance = _poincare_distance_tangent_pairs(x, y, 1.0)
            rho_mean = 0.5 * (rho_x + rho_y)
            angle_only_cosh = (
                torch.cosh(rho_mean).square()
                - torch.sinh(rho_mean).square() * cosine
            ).clamp_min(1.0)
            output["angle_radians"][start:stop] = angle.cpu().numpy()
            output["radial_difference"][start:stop] = torch.abs(rho_x - rho_y).cpu().numpy()
            output["radial_cosh_term"][start:stop] = radial_term.cpu().numpy()
            output["angular_cosh_term"][start:stop] = angular_term.cpu().numpy()
            output["angular_cosh_fraction"][start:stop] = (
                angular_term / cosh_distance
            ).cpu().numpy()
            output["hyperbolic_distance"][start:stop] = direct_distance.cpu().numpy()
            output["euclidean_tangent_distance"][start:stop] = (
                torch.linalg.vector_norm(x - y, dim=-1).cpu().numpy()
            )
            output["angle_only_hyperbolic_distance"][start:stop] = (
                torch.acosh(angle_only_cosh).cpu().numpy()
            )
            output["angle_only_euclidean_scale2"][start:stop] = (
                2.0 * rho_mean * torch.sin(angle / 2.0)
            ).cpu().numpy()
            output["law_distance_abs_error"][start:stop] = (
                torch.abs(law_distance - direct_distance).cpu().numpy()
            )
    return output


def catalog_rank_and_neighbors(
    vectors: torch.Tensor,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    metric: str,
    top_k: int = 100,
    batch_size: int = 32,
) -> tuple[dict[str, Any], np.ndarray]:
    sources = torch.as_tensor(source_ids, device=DIAGNOSTIC_DEVICE, dtype=torch.long)
    targets = torch.as_tensor(target_ids, device=DIAGNOSTIC_DEVICE, dtype=torch.long)
    ranks = np.empty(len(source_ids), dtype=np.float64)
    hit_probabilities = {
        k: np.empty(len(source_ids), dtype=np.float64) for k in (1, 10, 50, 100)
    }
    neighbor_ids = np.empty((len(source_ids), min(top_k, vectors.shape[0] - 1)), dtype=np.int64)
    normalized = None
    if metric == "angle":
        normalized = vectors / torch.linalg.vector_norm(
            vectors, dim=-1, keepdim=True
        ).clamp_min(1e-12)
    elif metric != "hyperbolic":
        raise ValueError(f"Unsupported catalog metric: {metric}")
    candidate_count = vectors.shape[0] - 1
    with torch.no_grad():
        for start in range(0, len(source_ids), batch_size):
            stop = min(len(source_ids), start + batch_size)
            query = sources[start:stop]
            if metric == "angle":
                score = normalized[query] @ normalized.T
                score[torch.arange(stop - start, device=DIAGNOSTIC_DEVICE), query] = -torch.inf
                target_score = (normalized[query] * normalized[targets[start:stop]]).sum(-1)
                better = score > target_score.unsqueeze(-1)
                equal = score == target_score.unsqueeze(-1)
                top = torch.topk(score, k=neighbor_ids.shape[1], dim=-1, largest=True).indices
            else:
                score = _pairwise_poincare_distance_tangents(
                    vectors[query], vectors, 1.0
                )
                score[torch.arange(stop - start, device=DIAGNOSTIC_DEVICE), query] = torch.inf
                target_score = score.gather(
                    1, targets[start:stop, None]
                ).squeeze(1)
                better = score < target_score.unsqueeze(-1)
                equal = score == target_score.unsqueeze(-1)
                top = torch.topk(score, k=neighbor_ids.shape[1], dim=-1, largest=False).indices
            less_count = better.sum(-1)
            equal_count = equal.sum(-1).clamp_min(1)
            ranks[start:stop] = (
                less_count.to(torch.float64)
                + (equal_count.to(torch.float64) + 1.0) / 2.0
            ).cpu().numpy()
            for k in hit_probabilities:
                remaining = (k - less_count).clamp_min(0).to(torch.float64)
                hit_probabilities[k][start:stop] = (
                    torch.minimum(remaining, equal_count) / equal_count
                ).cpu().numpy()
            neighbor_ids[start:stop] = top.cpu().numpy()
    return {
        "source_count": int(len(source_ids)),
        "median_rank": float(np.median(ranks)),
        "mean_rank_percentile": float(np.mean(ranks / candidate_count)),
        "hit_probability_at_k_random_tie_break": {
            str(k): float(np.mean(values)) for k, values in hit_probabilities.items()
        },
    }, neighbor_ids


def d14_d17_diagnostics(
    model: RQVAE,
    encoder_inputs: torch.Tensor,
    tokens: np.ndarray,
    pair_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    free_bytes = total_bytes = 0
    if DIAGNOSTIC_DEVICE.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(DIAGNOSTIC_DEVICE)
        if free_bytes < 8 * 1024**3:
            raise RuntimeError(
                f"{DIAGNOSTIC_DEVICE} has only {free_bytes / 1024**3:.2f} GiB free; refusing to compete"
            )
    analysis_model = copy.deepcopy(model).to(DIAGNOSTIC_DEVICE).eval()
    input_tensor = encoder_inputs.to(DIAGNOSTIC_DEVICE)
    tokens_gpu = torch.as_tensor(tokens, device=DIAGNOSTIC_DEVICE, dtype=torch.long)
    with torch.no_grad():
        encoded = analysis_model.encoder(input_tensor)
        reproduced, _ = analysis_model.rq.get_indices_with_stats(encoded)
    sid_mismatches = {
        f"L{level + 1}": int((reproduced[:, level] != tokens_gpu[:, level]).sum().item())
        for level in range(3)
    }

    representations: dict[str, torch.Tensor] = {"Encoder": encoded}
    residual = encoded.detach().clone()
    cumulative = torch.zeros(
        len(tokens), analysis_model.rq.codebook_dim,
        device=DIAGNOSTIC_DEVICE, dtype=encoded.dtype,
    )
    with torch.no_grad():
        for level, layer in enumerate(analysis_model.rq.vq_layers):
            curvature = layer.get_curvature()
            source = residual
            code_ids = tokens_gpu[:, level]
            if analysis_model.rq.working_radii[level] == 0.0:
                code = layer.embed_code(code_ids)
                cumulative = cumulative + code
                residual = _hyperbolic_residual(source, code, curvature)
            else:
                target_norm = analysis_model.rq._radius_for_level(level, curvature, source)
                pinned = analysis_model.rq._pin_to_radius(source, target_norm)
                code = layer.embed_code(code_ids)
                contribution = analysis_model.rq._restore_norm(code, source, target_norm)
                cumulative = cumulative + contribution
                residual = analysis_model.rq._restore_norm(
                    _hyperbolic_residual(pinned, code, curvature),
                    source, target_norm,
                )
            representations[f"L1-L{level + 1}"] = cumulative.clone()

    pair_frame = pd.DataFrame(pair_rows)
    pair_frame = pair_frame[
        pair_frame["source_item"] != pair_frame["positive_successor"]
    ].reset_index(drop=True)
    pair_arrays = {
        "source": pair_frame["source_item"].to_numpy(dtype=np.int64),
        "positive": pair_frame["positive_successor"].to_numpy(dtype=np.int64),
        "matched_negative": pair_frame["matched_negative"].to_numpy(dtype=np.int64),
        "random_negative": pair_frame["random_negative"].to_numpy(dtype=np.int64),
    }
    classes = ("positive", "matched_negative", "random_negative")
    representation_names = ("Encoder", "L1-L1", "L1-L2", "L1-L3")
    pair_features: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name in representation_names:
        pair_features[name] = {}
        for pair_class in classes:
            pair_features[name][pair_class] = paired_geometry_features(
                representations[name], pair_arrays["source"], pair_arrays[pair_class]
            )

    unshared_mask = pair_frame["positive_lcp"].to_numpy(dtype=np.int8) == 0
    all_rank_sample = (
        pair_frame.sample(frac=1.0, random_state=SEED)
        .drop_duplicates("source_item", keep="first")
        .head(1_000)
        .reset_index(drop=True)
    )
    rank_sources = all_rank_sample["source_item"].to_numpy(dtype=np.int64)
    rank_targets = all_rank_sample["positive_successor"].to_numpy(dtype=np.int64)

    d14: dict[str, Any] = {"pair_count": int(len(pair_frame)), "by_representation": {}}
    angular_ranks: dict[str, dict[str, Any]] = {}
    angular_neighbors: dict[str, np.ndarray] = {}
    for name in representation_names:
        positive = pair_features[name]["positive"]["angle_radians"]
        matched = pair_features[name]["matched_negative"]["angle_radians"]
        random = pair_features[name]["random_negative"]["angle_radians"]
        unshared_positive = positive[unshared_mask]
        unshared_matched = matched[unshared_mask]
        positive_change = positive - pair_features["Encoder"]["positive"]["angle_radians"]
        rank_summary, neighbor_ids = catalog_rank_and_neighbors(
            representations[name], rank_sources, rank_targets, "angle"
        )
        angular_ranks[name] = rank_summary
        angular_neighbors[name] = neighbor_ids
        d14["by_representation"][name] = {
            "positive_angle_radians": percentile_summary(positive),
            "matched_negative_angle_radians": percentile_summary(matched),
            "random_negative_angle_radians": percentile_summary(random),
            "paired_auc_positive_closer_than_matched": paired_lower_auc(positive, matched),
            "paired_auc_positive_closer_than_random": paired_lower_auc(positive, random),
            "l1_unshared_positive_auc_vs_matched": paired_lower_auc(
                unshared_positive, unshared_matched
            ),
            "positive_angle_change_vs_encoder": percentile_summary(positive_change),
            "share_positive_angle_decreased_vs_encoder": float((positive_change < 0).mean()),
            "angular_positive_target_catalog_rank_sample": rank_summary,
        }
    d14["angular_neighborhood_retention_vs_encoder"] = {}
    base_neighbors = angular_neighbors["Encoder"]
    for name in representation_names:
        d14["angular_neighborhood_retention_vs_encoder"][name] = {}
        for k in (10, 50, 100):
            intersection_sizes = np.fromiter(
                (
                    len(np.intersect1d(base_neighbors[row, :k], angular_neighbors[name][row, :k]))
                    for row in range(len(base_neighbors))
                ),
                dtype=np.int64,
                count=len(base_neighbors),
            )
            d14["angular_neighborhood_retention_vs_encoder"][name][str(k)] = {
                "mean_recall": float(np.mean(intersection_sizes / k)),
                "mean_jaccard": float(np.mean(intersection_sizes / (2 * k - intersection_sizes))),
            }

    d15: dict[str, Any] = {
        "curvature": 1.0,
        "exact_cosh_decomposition": (
            "cosh(d_H)=cosh(r_i-r_j)+sinh(r_i)*sinh(r_j)*(1-cos(theta)); "
            "the angular term is coupled to both radii."
        ),
        "by_representation": {},
    }
    for name in representation_names:
        feature = pair_features[name]
        d15["by_representation"][name] = {
            "pair_medians": {
                pair_class: {
                    key: float(np.median(feature[pair_class][key]))
                    for key in (
                        "radial_difference",
                        "angle_radians",
                        "radial_cosh_term",
                        "angular_cosh_term",
                        "angular_cosh_fraction",
                        "hyperbolic_distance",
                        "euclidean_tangent_distance",
                        "angle_only_hyperbolic_distance",
                        "angle_only_euclidean_scale2",
                    )
                }
                for pair_class in classes
            },
            "positive_vs_matched_auc": {
                "hyperbolic": paired_lower_auc(
                    feature["positive"]["hyperbolic_distance"],
                    feature["matched_negative"]["hyperbolic_distance"],
                ),
                "tangent_euclidean": paired_lower_auc(
                    feature["positive"]["euclidean_tangent_distance"],
                    feature["matched_negative"]["euclidean_tangent_distance"],
                ),
                "angle_only_hyperbolic": paired_lower_auc(
                    feature["positive"]["angle_only_hyperbolic_distance"],
                    feature["matched_negative"]["angle_only_hyperbolic_distance"],
                ),
                "angle_only_euclidean_scale2": paired_lower_auc(
                    feature["positive"]["angle_only_euclidean_scale2"],
                    feature["matched_negative"]["angle_only_euclidean_scale2"],
                ),
            },
            "max_law_of_cosines_vs_direct_distance_error": float(
                np.max(feature["positive"]["law_distance_abs_error"])
            ),
        }

    l1_layer = analysis_model.rq.vq_layers[0]
    with torch.no_grad():
        if analysis_model.rq.working_radii[0] == 0.0:
            l1_frame = encoded
        else:
            l1_curvature = l1_layer.get_curvature()
            l1_target_norm = analysis_model.rq._radius_for_level(
                0, l1_curvature, encoded
            )
            l1_frame = analysis_model.rq._pin_to_radius(encoded, l1_target_norm)
        l1_distances = l1_layer._distances(l1_frame)
        l1_probabilities = l1_layer._balanced_assignments(l1_distances)
        l1_assignment = l1_probabilities.argmax(-1)
        raw_nearest_values, raw_nearest_codes = torch.topk(
            l1_distances, k=10, dim=-1, largest=False, sorted=True
        )
        sinkhorn_top_values, _ = torch.topk(
            l1_probabilities, k=2, dim=-1, largest=True, sorted=True
        )
        local_top2_gap = sinkhorn_top_values[:, 0] - sinkhorn_top_values[:, 1]
        raw_gap = raw_nearest_values[:, 1] - raw_nearest_values[:, 0]
        saved_probability = l1_probabilities.gather(
            1, tokens_gpu[:, 0, None]
        ).squeeze(1)
        competing_probability = l1_probabilities.clone()
        competing_probability.scatter_(1, tokens_gpu[:, 0, None], -torch.inf)
        saved_probability_gap = saved_probability - competing_probability.max(-1).values
        assigned_distance = l1_distances.gather(1, tokens_gpu[:, 0, None]).squeeze(1)
        assigned_distance_rank = (
            (l1_distances < assigned_distance.unsqueeze(-1)).sum(-1) + 1
        )
    saved_margin_cpu = saved_probability_gap.cpu().numpy()
    local_margin_cpu = local_top2_gap.cpu().numpy()
    raw_gap_cpu = raw_gap.cpu().numpy()
    rank_cpu = assigned_distance_rank.cpu().numpy()
    src_unshared = pair_arrays["source"][unshared_mask]
    targets_unshared = {
        pair_class: pair_arrays[pair_class][unshared_mask]
        for pair_class in classes
    }
    src_unshared_gpu = torch.as_tensor(src_unshared, device=DIAGNOSTIC_DEVICE, dtype=torch.long)
    targets_unshared_gpu = {
        pair_class: torch.as_tensor(target_ids, device=DIAGNOSTIC_DEVICE, dtype=torch.long)
        for pair_class, target_ids in targets_unshared.items()
    }
    d16: dict[str, Any] = {
        "l1_unshared_positive_pair_count": int(unshared_mask.sum()),
        "local_assignment_mismatches_vs_saved": int(
            (l1_assignment != tokens_gpu[:, 0]).sum().item()
        ),
        "local_assignment_mismatches_vs_cpu_full_reencoding": int(
            (l1_assignment != reproduced[:, 0]).sum().item()
        ),
        "positive_target_vs_control_boundary_proximity": {},
        "positive_vs_control_candidate_proximity": {},
    }
    for pair_class in ("positive", "matched_negative", "random_negative"):
        target_ids = targets_unshared[pair_class]
        saved_gap = saved_margin_cpu[target_ids]
        d16["positive_target_vs_control_boundary_proximity"][pair_class] = {
            "saved_code_probability_gap_signed": percentile_summary(saved_gap),
            "absolute_saved_code_probability_gap": percentile_summary(np.abs(saved_gap)),
            "local_sinkhorn_top1_top2_probability_gap": percentile_summary(
                local_margin_cpu[target_ids]
            ),
            "nearest_codeword_distance_gap": percentile_summary(raw_gap_cpu[target_ids]),
            "saved_assignment_rank_in_raw_distance_order": percentile_summary(rank_cpu[target_ids]),
        }
    for control in ("matched_negative", "random_negative"):
        positive_ids = targets_unshared["positive"]
        negative_ids = targets_unshared[control]
        d16["positive_target_vs_control_boundary_proximity"][f"positive_closer_to_boundary_than_{control}"] = {
            "absolute_saved_code_gap_lower_is_closer_fraction": paired_lower_auc(
                np.abs(saved_margin_cpu[positive_ids]),
                np.abs(saved_margin_cpu[negative_ids]),
            ),
            "local_top2_gap_lower_is_closer_fraction": paired_lower_auc(
                local_margin_cpu[positive_ids], local_margin_cpu[negative_ids]
            ),
            "nearest_distance_gap_lower_is_closer_fraction": paired_lower_auc(
                raw_gap_cpu[positive_ids], raw_gap_cpu[negative_ids]
            ),
        }
    with torch.no_grad():
        candidate_metrics: dict[str, Any] = {}
        for pair_class in classes:
            target_ids = targets_unshared[pair_class]
            candidate_metrics[pair_class] = {}
            for k in (2, 5, 10):
                target_ids_gpu = targets_unshared_gpu[pair_class]
                source_candidates = raw_nearest_codes[src_unshared_gpu, :k]
                target_candidates = raw_nearest_codes[target_ids_gpu, :k]
                assigned_target_code = tokens_gpu[target_ids_gpu, 0]
                assigned_source_code = tokens_gpu[src_unshared_gpu, 0]
                overlap_counts = []
                for start in range(0, len(src_unshared), 20_000):
                    stop = min(len(src_unshared), start + 20_000)
                    matches = (
                        source_candidates[start:stop, :, None]
                        == target_candidates[start:stop, None, :]
                    )
                    overlap_counts.append(matches.any(-1).sum(-1).cpu().numpy())
                overlap = np.concatenate(overlap_counts) if overlap_counts else np.array([], dtype=np.int64)
                candidate_metrics[pair_class][str(k)] = {
                    "target_saved_code_in_source_raw_top_k": float(
                        (source_candidates == assigned_target_code[:, None]).any(-1).float().mean().cpu()
                    ),
                    "source_saved_code_in_target_raw_top_k": float(
                        (target_candidates == assigned_source_code[:, None]).any(-1).float().mean().cpu()
                    ),
                    "mean_candidate_set_intersection_over_k": float(np.mean(overlap / k)) if len(overlap) else 0.0,
                }
        d16["positive_vs_control_candidate_proximity"] = candidate_metrics
    d16["raw_distance_candidate_definition"] = (
        "Top-K codewords by Poincare distance before the corpus-level Sinkhorn balancing; "
        "balanced assignment margins are reported separately."
    )

    before_name, after_name = "L1-L2", "L1-L3"
    d17: dict[str, Any] = {
        "same_behavior_pair_sample_count": int(len(pair_frame)),
        "from": before_name,
        "to": after_name,
        "per_class_distance_changes": {},
    }
    for pair_class in classes:
        before = pair_features[before_name][pair_class]
        after = pair_features[after_name][pair_class]
        angle_delta = after["angle_radians"] - before["angle_radians"]
        distance_delta = after["hyperbolic_distance"] - before["hyperbolic_distance"]
        d17["per_class_distance_changes"][pair_class] = {
            "angle_before_median": float(np.median(before["angle_radians"])),
            "angle_after_median": float(np.median(after["angle_radians"])),
            "angle_delta_after_minus_before": percentile_summary(angle_delta),
            "share_angle_decreased": float((angle_delta < 0).mean()),
            "hyperbolic_distance_before_median": float(np.median(before["hyperbolic_distance"])),
            "hyperbolic_distance_after_median": float(np.median(after["hyperbolic_distance"])),
            "hyperbolic_distance_delta_after_minus_before": percentile_summary(distance_delta),
            "share_hyperbolic_distance_decreased": float((distance_delta < 0).mean()),
        }
    d17["positive_ranking_margin_change"] = {}
    for negative_class in ("matched_negative", "random_negative"):
        before_angle_margin = (
            pair_features[before_name][negative_class]["angle_radians"]
            - pair_features[before_name]["positive"]["angle_radians"]
        )
        after_angle_margin = (
            pair_features[after_name][negative_class]["angle_radians"]
            - pair_features[after_name]["positive"]["angle_radians"]
        )
        before_h_margin = (
            pair_features[before_name][negative_class]["hyperbolic_distance"]
            - pair_features[before_name]["positive"]["hyperbolic_distance"]
        )
        after_h_margin = (
            pair_features[after_name][negative_class]["hyperbolic_distance"]
            - pair_features[after_name]["positive"]["hyperbolic_distance"]
        )
        d17["positive_ranking_margin_change"][negative_class] = {
            "angular_auc_before": paired_lower_auc(
                pair_features[before_name]["positive"]["angle_radians"],
                pair_features[before_name][negative_class]["angle_radians"],
            ),
            "angular_auc_after": paired_lower_auc(
                pair_features[after_name]["positive"]["angle_radians"],
                pair_features[after_name][negative_class]["angle_radians"],
            ),
            "angular_margin_delta_after_minus_before": percentile_summary(
                after_angle_margin - before_angle_margin
            ),
            "share_angular_margin_improved": float(
                ((after_angle_margin - before_angle_margin) > 0).mean()
            ),
            "hyperbolic_auc_before": paired_lower_auc(
                pair_features[before_name]["positive"]["hyperbolic_distance"],
                pair_features[before_name][negative_class]["hyperbolic_distance"],
            ),
            "hyperbolic_auc_after": paired_lower_auc(
                pair_features[after_name]["positive"]["hyperbolic_distance"],
                pair_features[after_name][negative_class]["hyperbolic_distance"],
            ),
            "hyperbolic_margin_delta_after_minus_before": percentile_summary(
                after_h_margin - before_h_margin
            ),
            "share_hyperbolic_margin_improved": float(
                ((after_h_margin - before_h_margin) > 0).mean()
            ),
        }
    hyperbolic_rank_before, _ = catalog_rank_and_neighbors(
        representations[before_name], rank_sources, rank_targets, "hyperbolic"
    )
    hyperbolic_rank_after, _ = catalog_rank_and_neighbors(
        representations[after_name], rank_sources, rank_targets, "hyperbolic"
    )
    d17["catalog_rank_sample"] = {
        "sample_sources": int(len(rank_sources)),
        "angular_before": angular_ranks[before_name],
        "angular_after": angular_ranks[after_name],
        "hyperbolic_before": hyperbolic_rank_before,
        "hyperbolic_after": hyperbolic_rank_after,
        "angular_hit_at_10_delta_after_minus_before": (
            angular_ranks[after_name]["hit_probability_at_k_random_tie_break"]["10"]
            - angular_ranks[before_name]["hit_probability_at_k_random_tie_break"]["10"]
        ),
        "hyperbolic_hit_at_10_delta_after_minus_before": (
            hyperbolic_rank_after["hit_probability_at_k_random_tie_break"]["10"]
            - hyperbolic_rank_before["hit_probability_at_k_random_tie_break"]["10"]
        ),
        "hyperbolic_median_rank_delta_after_minus_before": (
            hyperbolic_rank_after["median_rank"] - hyperbolic_rank_before["median_rank"]
        ),
    }
    del analysis_model, input_tensor, encoded, representations, l1_distances, l1_probabilities
    if DIAGNOSTIC_DEVICE.type == "cuda":
        torch.cuda.synchronize(DIAGNOSTIC_DEVICE)
    return {
        "device": str(DIAGNOSTIC_DEVICE),
        "free_gib_before": (
            free_bytes / 1024**3 if DIAGNOSTIC_DEVICE.type == "cuda" else None
        ),
        "total_gib": (
            total_bytes / 1024**3 if DIAGNOSTIC_DEVICE.type == "cuda" else None
        ),
        "local_reencoding_mismatches_vs_saved": sid_mismatches,
        "D14_angular_loss_and_neighborhood_retention": d14,
        "D15_hyperbolic_radial_angular_coupling": d15,
        "D16_L1_assignment_boundary": d16,
        "D17_L3_geometric_compensation": d17,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rng = np.random.default_rng(SEED)
    del rng

    embedding_path = Path(experiment.EMBEDDING_FILE)
    train_path = Path(experiment.TRAIN_FILE)
    checkpoint_path = Path(experiment.RQVAE_CKPT_PATH)
    raw_sid_path = Path(experiment.RAW_SIDS_NPY)
    hgrec_sid_path = Path(experiment.SIDS_NPY)
    item_sid_path = Path(experiment.ITEM_SIDS_JSON)
    native_metrics_path = SOURCE_DIR / "logs/training_metrics_hyperbolic.jsonl"
    source_files = [
        SOURCE_DIR / "curvature_config.py",
        SOURCE_DIR / "train_rqvae.py",
        SOURCE_DIR / "model/model.py",
        SOURCE_DIR / "model/layers.py",
        Path(__file__).resolve(),
    ]
    input_files = [checkpoint_path, raw_sid_path, hgrec_sid_path, item_sid_path, embedding_path, train_path, native_metrics_path, *source_files]
    for path in input_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    hashes = {relative_to_root(path): sha256(path) for path in input_files}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    native_events = [
        json.loads(line)
        for line in native_metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    snapshots = [event for event in native_events if event.get("event") == "snapshot_saved"]
    matching_snapshots = [
        event for event in snapshots
        if Path(event["checkpoint"]).resolve() == checkpoint_path.resolve()
        and int(event["global_step"]) == int(checkpoint["global_step"])
        and event.get("version") == checkpoint.get("version")
    ]
    matching_snapshot = matching_snapshots[-1] if matching_snapshots else None
    train_start_events = [event for event in native_events if event.get("event") == "train_start"]
    matching_train_start = next(
        (
            event for event in reversed(train_start_events)
            if int(event.get("max_global_steps", -1)) == int(checkpoint["global_step"])
        ),
        None,
    )
    train_end_events = [event for event in native_events if event.get("event") == "train_end"]
    matching_train_end = next(
        (
            event for event in reversed(train_end_events)
            if int(event.get("global_step", -1)) == int(checkpoint["global_step"])
        ),
        None,
    )
    if matching_snapshot is None or matching_train_end is None or matching_train_start is None:
        raise RuntimeError("Native Stage2 metrics do not link this checkpoint and SID snapshot to a completed run")
    for field, expected in (
        ("raw_sids", raw_sid_path),
        ("sids_for_hgrec", hgrec_sid_path),
        ("item_sids", item_sid_path),
    ):
        if Path(matching_snapshot[field]).resolve() != expected.resolve():
            raise RuntimeError(f"Native Stage2 snapshot points to a different {field} file")
    embeddings_np = np.asarray(np.load(embedding_path), dtype=np.float32)
    if embeddings_np.ndim != 2 or not np.isfinite(embeddings_np).all():
        raise ValueError(f"Invalid embedding matrix: {embeddings_np.shape}")
    n_items, embedding_dim = embeddings_np.shape
    train_frame = pd.read_parquet(train_path, columns=["seen_history", "target"])
    targets = train_frame["target"].to_numpy(dtype=np.int64)
    if targets.min() < 0 or targets.max() >= n_items:
        raise ValueError("Training targets do not match embedding item IDs")
    frequencies = np.bincount(targets, minlength=n_items).astype(np.int64)
    sources, successors = trainer._transition_pairs(train_frame)
    if sources.min() < 0 or successors.min() < 0 or sources.max() >= n_items or successors.max() >= n_items:
        raise ValueError("Training transition IDs do not match embedding item IDs")
    item_sid_payload = json.loads(item_sid_path.read_text(encoding="utf-8"))
    sid_ids = sorted(int(item_id) for item_id in item_sid_payload)
    if sid_ids != list(range(n_items)):
        raise ValueError("item_sids.json keys are not exactly the embedding item ID range")
    item_sids = np.asarray([item_sid_payload[str(item_id)] for item_id in range(n_items)], dtype=np.int64)
    raw_sids = np.asarray(np.load(raw_sid_path), dtype=np.int64)
    hgrec_sids = np.asarray(np.load(hgrec_sid_path), dtype=np.int64)
    if item_sids.shape != (n_items, 4) or raw_sids.shape != (n_items, 3) or hgrec_sids.shape != (n_items, 4):
        raise ValueError(f"SID shape mismatch: item={item_sids.shape}, raw={raw_sids.shape}, HGRec={hgrec_sids.shape}")
    if not np.array_equal(item_sids[:, :3], raw_sids) or not np.array_equal(item_sids, hgrec_sids):
        raise ValueError("SID JSON, raw_sids.npy, and HGRec SIDs disagree")
    extension_values = np.unique(item_sids[:, 3])
    if np.any(item_sids[:, :3] < 0) or np.any(item_sids[:, :3] >= 256):
        raise ValueError("Raw SID code is outside the three 256-way codebooks")
    config = trainer._tokenizer_config()
    model = RQVAE(config, in_dim=embedding_dim, context_dim=embedding_dim).to(DEVICE)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict state load mismatch: {incompatible}")
    model.eval()
    embeddings = torch.from_numpy(embeddings_np)
    behaviour_centroid = trainer._behaviour_context(embeddings, sources, successors)
    context_channel = trainer._curvature_context_channel(embeddings, behaviour_centroid)
    encoder_inputs = torch.cat((embeddings, context_channel), dim=1)
    with torch.no_grad():
        encoded = model.encoder(encoder_inputs)
        recomputed_tokens, _ = model.rq.get_indices_with_stats(encoded)
        recomputed_tokens = recomputed_tokens.cpu().numpy().astype(np.int64)
    cpu_mismatch_by_level = {
        f"L{level + 1}": int(np.count_nonzero(recomputed_tokens[:, level] != raw_sids[:, level]))
        for level in range(3)
    }
    cpu_mismatch_cumulative = {
        f"L1-L{depth}": int(np.count_nonzero(np.any(recomputed_tokens[:, :depth] != raw_sids[:, :depth], axis=1)))
        for depth in range(1, 4)
    }
    tokens = raw_sids

    code_rows, code_summary = code_usage(tokens, n_codes=256)
    write_csv(OUTPUT_DIR / "codeword_usage.csv", ["level", "code_id", "item_count", "share"], code_rows)
    histogram_rows = []
    histogram_bins: dict[str, list[float]] = {}
    max_count = max(row["item_count"] for row in code_rows)
    bin_width = max(1, math.ceil((max_count + 1) / 16))
    bin_edges = list(range(0, max_count + bin_width + 1, bin_width))
    for level in ("L1", "L2", "L3"):
        counts = np.asarray([row["item_count"] for row in code_rows if row["level"] == level], dtype=np.int64)
        histogram = np.histogram(counts, bins=bin_edges)[0]
        labels = []
        for index, count in enumerate(histogram):
            lower, upper = bin_edges[index], bin_edges[index + 1] - 1
            label = f"{lower}-{upper}"
            labels.append(label)
            histogram_rows.append({"level": level, "frequency_min": lower, "frequency_max": upper, "codeword_count": int(count)})
        histogram_bins[level] = [float(v) for v in histogram]
    write_csv(OUTPUT_DIR / "codeword_frequency_histogram.csv", ["level", "frequency_min", "frequency_max", "codeword_count"], histogram_rows)

    prefix_rows, branch_rows, prefix_summary = prefix_audit(tokens)
    write_csv(OUTPUT_DIR / "prefix_nodes.csv", ["depth", "code_l1", "code_l2", "code_l3", "cluster_items"], prefix_rows)
    write_csv(OUTPUT_DIR / "branching.csv", ["relation", "parent_l1", "parent_l2", "items_in_parent", "distinct_children"], branch_rows)

    popularity_group, popularity_rule = popularity_groups(frequencies)
    item_ids = np.arange(n_items, dtype=np.int64)
    sorted_ids = np.lexsort((item_ids, -frequencies))
    ranks = np.empty(n_items, dtype=np.int64)
    ranks[sorted_ids] = np.arange(n_items)
    popularity_rows: list[dict[str, Any]] = []
    popularity_summary: dict[str, Any] = {"definition": popularity_rule, "groups": {}}
    for group in ("head", "mid", "tail"):
        mask = popularity_group == group
        group_tokens = tokens[mask]
        l1_counts = np.bincount(group_tokens[:, 0], minlength=256)
        l1l2_unique = len(np.unique(group_tokens[:, :2], axis=0))
        full_unique = len(np.unique(group_tokens[:, :3], axis=0))
        entry = {
            "items": int(mask.sum()),
            "interaction_count": int(frequencies[mask].sum()),
            "interaction_share": float(frequencies[mask].sum() / max(1, frequencies.sum())),
            "target_frequency": percentile_summary(frequencies[mask]),
            "target_frequency_min": int(frequencies[mask].min()),
            "target_frequency_max": int(frequencies[mask].max()),
            "l1_entropy_bits": entropy_bits(l1_counts),
            "used_l1_codes": int((l1_counts > 0).sum()),
            "unique_l1l2_prefixes": int(l1l2_unique),
            "unique_full_prefixes": int(full_unique),
            "full_prefixes_per_item": float(full_unique / max(1, mask.sum())),
        }
        popularity_summary["groups"][group] = entry
        popularity_rows.append({"popularity_group": group, **entry})
    write_csv(OUTPUT_DIR / "popularity_groups.csv", ["popularity_group", "items", "interaction_count", "interaction_share", "target_frequency_min", "target_frequency_max", "l1_entropy_bits", "used_l1_codes", "unique_l1l2_prefixes", "unique_full_prefixes", "full_prefixes_per_item"], popularity_rows)

    pair_rows, behavior_summary_rows, behavior_summary, rank_decile = behavior_pair_audit(
        sources, successors, tokens, ranks, popularity_group
    )
    write_csv(OUTPUT_DIR / "behavior_pair_sample.csv", ["source_item", "positive_successor", "positive_event_count", "positive_successor_group", "positive_rank_decile", "positive_lcp", "matched_negative", "matched_negative_lcp", "random_negative", "random_negative_lcp"], pair_rows)
    write_csv(OUTPUT_DIR / "behavior_prefix_sharing.csv", ["pair_class", "sampled_pairs", "mean_lcp", "share_lcp_ge_1", "share_lcp_ge_2", "share_lcp_ge_3", "lcp_0_fraction", "lcp_1_fraction", "lcp_2_fraction", "lcp_3_fraction"], behavior_summary_rows)

    distance_rows, distance_quantile_rows, distance_summary = distance_audit(encoded, tokens)
    write_csv(OUTPUT_DIR / "geometry_pair_sample.csv", ["item_i", "item_j", "d_hyperbolic_c1", "d_euclidean_tangent", "d_euclidean_scale_matched", "lcp"], distance_rows)
    write_csv(OUTPUT_DIR / "geometry_distance_quantiles.csv", ["metric", "distance_decile", "pair_count", "distance_min", "distance_median", "distance_max", "mean_lcp", "share_lcp_ge_1", "share_lcp_ge_2", "share_lcp_ge_3"], distance_quantile_rows)

    residual_item_rows, residual_summary_rows, residual_summary = residual_audit(
        model, encoded, tokens, popularity_group
    )
    write_csv(OUTPUT_DIR / "item_residual_diagnostics.csv", ["item_id", "popularity_group", "l1_prefix_size", "l1_prefix_size_quartile", "level", "quantization_error_hyperbolic", "residual_tangent_norm_after", "residual_origin_hyperbolic_distance_after"], residual_item_rows)
    write_csv(OUTPUT_DIR / "residual_summary.csv", ["level", "popularity_group", "l1_prefix_size_quartile", "item_count", "mean_quantization_error_hyperbolic", "median_quantization_error_hyperbolic", "q90_quantization_error_hyperbolic", "mean_residual_tangent_norm_after"], residual_summary_rows)

    # Anomaly examples: overloaded/singleton prefixes, behavior transitions split at L1,
    # and random-pair geometry neighbors that disagree at the first SID level.
    anomaly_rows: list[dict[str, Any]] = []
    l1_counts = np.bincount(tokens[:, 0], minlength=256)
    for code in np.argsort(l1_counts)[::-1][:10]:
        members = np.flatnonzero(tokens[:, 0] == code)
        anomaly_rows.append({"case": "dense_L1_prefix", "item_i": int(members[0]), "item_j": None, "score": int(l1_counts[code]), "detail": json.dumps({"l1_code": int(code), "cluster_size": int(l1_counts[code]), "example_item_ids": members[:10].tolist()})})
    for code in np.flatnonzero(l1_counts <= 1)[:100]:
        members = np.flatnonzero(tokens[:, 0] == code)
        anomaly_rows.append({"case": "sparse_L1_prefix", "item_i": int(members[0]) if len(members) else None, "item_j": None, "score": int(l1_counts[code]), "detail": json.dumps({"l1_code": int(code), "cluster_size": int(l1_counts[code])})})
    l1l2, l1l2_counts = np.unique(tokens[:, :2], axis=0, return_counts=True)
    sparse_order = np.lexsort((l1l2[:, 1], l1l2[:, 0], l1l2_counts))
    for index in sparse_order[:100]:
        prefix = l1l2[index]
        item = int(np.flatnonzero(np.all(tokens[:, :2] == prefix, axis=1))[0])
        anomaly_rows.append({"case": "sparse_L1L2_prefix", "item_i": item, "item_j": None, "score": int(l1l2_counts[index]), "detail": json.dumps({"l1_l2": prefix.tolist(), "cluster_size": int(l1l2_counts[index]), "full_sid": tokens[item].tolist()})})
    for row in sorted(pair_rows, key=lambda item: (-item["positive_event_count"], item["source_item"], item["positive_successor"])):
        if row["positive_lcp"] == 0:
            anomaly_rows.append({"case": "frequent_behavior_pair_lcp0", "item_i": row["source_item"], "item_j": row["positive_successor"], "score": row["positive_event_count"], "detail": json.dumps({"source_sid": tokens[row["source_item"]].tolist(), "successor_sid": tokens[row["positive_successor"]].tolist(), "event_count": row["positive_event_count"]})})
        if sum(1 for anomaly in anomaly_rows if anomaly["case"] == "frequent_behavior_pair_lcp0") >= 100:
            break
    geo_df = pd.DataFrame(distance_rows)
    geo_candidates = geo_df[geo_df["lcp"] == 0].nsmallest(100, "d_hyperbolic_c1")
    for row in geo_candidates.itertuples(index=False):
        anomaly_rows.append({"case": "geometry_near_sid_lcp0", "item_i": int(row.item_i), "item_j": int(row.item_j), "score": float(row.d_hyperbolic_c1), "detail": json.dumps({"sid_i": tokens[int(row.item_i)].tolist(), "sid_j": tokens[int(row.item_j)].tolist(), "d_hyperbolic": float(row.d_hyperbolic_c1), "d_euclidean_scale_matched": float(row.d_euclidean_scale_matched)})})
    write_csv(OUTPUT_DIR / "anomaly_examples.csv", ["case", "item_i", "item_j", "score", "detail"], anomaly_rows)

    histogram_labels = [f"{bin_edges[i]}-{bin_edges[i + 1] - 1}" for i in range(len(bin_edges) - 1)]
    write_svg_bar(
        OUTPUT_DIR / "codeword_frequency_histogram.svg",
        "Codeword Usage Frequency Distribution",
        histogram_labels,
        [(level, histogram_bins[level]) for level in ("L1", "L2", "L3")],
        "Codewords per frequency bin",
    )
    cluster_categories = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129+"]
    prefix_hist_series = []
    prefix_depths = [(1, "L1"), (2, "L1-L2"), (3, "L1-L2-L3")]
    for depth, name in prefix_depths:
        _, _, counts = contiguous_unique_rows(tokens[:, :depth])
        bins = [int((counts == 1).sum()), int((counts == 2).sum()), int(((counts >= 3) & (counts <= 4)).sum()), int(((counts >= 5) & (counts <= 8)).sum()), int(((counts >= 9) & (counts <= 16)).sum()), int(((counts >= 17) & (counts <= 32)).sum()), int(((counts >= 33) & (counts <= 64)).sum()), int(((counts >= 65) & (counts <= 128)).sum()), int((counts >= 129).sum())]
        prefix_hist_series.append((name, bins))
    write_svg_bar(OUTPUT_DIR / "prefix_cluster_size_distribution.svg", "Prefix Cluster-Size Distribution", cluster_categories, prefix_hist_series, "Prefix nodes")

    behavior_classes = ["positive", "popularity_matched_negative", "random_negative"]
    write_svg_bar(
        OUTPUT_DIR / "behavior_prefix_sharing.svg",
        "SID Prefix Sharing Across Transition and Control Pairs",
        ["L1", "L1-L2", "L1-L2-L3"],
        [(name, [behavior_summary[name][f"share_lcp_ge_{depth}"] for depth in (1, 2, 3)]) for name in behavior_classes],
        "Share of sampled pairs",
    )
    distance_chart_series = []
    for metric in ("hyperbolic_c1", "euclidean_scale_matched"):
        rows = [row for row in distance_quantile_rows if row["metric"] == metric]
        rows.sort(key=lambda row: row["distance_decile"])
        for depth in range(1, 4):
            distance_chart_series.append((f"{metric}: LCP >= {depth}", [row[f"share_lcp_ge_{depth}"] for row in rows]))
    write_svg_lines(
        OUTPUT_DIR / "distance_quantile_prefix_sharing.svg",
        "Prefix Sharing by Encoder Distance Quantile",
        [str(i) for i in range(1, 11)],
        distance_chart_series,
        "Share of pairs",
        y_min=0.0,
        y_max=1.0,
    )
    residual_chart_series = []
    for group in ("head", "mid", "tail"):
        vals = []
        for level in ("L1", "L2", "L3"):
            row = next(row for row in residual_summary_rows if row["level"] == level and row["popularity_group"] == group and row["l1_prefix_size_quartile"] == "Q4_dense")
            vals.append(row["mean_quantization_error_hyperbolic"])
        residual_chart_series.append((group, vals))
    write_svg_bar(
        OUTPUT_DIR / "residual_error_dense_prefixes.svg",
        "Hyperbolic Quantization Error in Dense L1 Prefixes",
        ["L1", "L2", "L3"],
        residual_chart_series,
        "Mean hyperbolic quantization error",
    )

    metadata = {
        "analysis": "SID-Diag-01 Three-Level Semantic ID Distribution Audit",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "repo_root": str(ROOT),
        "commit": commit,
        "mechanism_name": experiment.MECHANISM_NAME,
        "source_files_sha256": {relative_to_root(path): hashes[relative_to_root(path)] for path in source_files},
        "input_sha256": {relative_to_root(path): hashes[relative_to_root(path)] for path in input_files if path not in source_files},
        "device": "cpu",
        "seed": SEED,
        "checkpoint": {
            "path": relative_to_root(checkpoint_path),
            "global_step": int(checkpoint["global_step"]),
            "local_optimizer_steps": int(checkpoint["local_optimizer_steps"]),
            "geometry": checkpoint["geometry"],
            "version": checkpoint["version"],
            "curvatures": checkpoint["curvatures"],
            "working_radii": checkpoint["working_radii"],
            "effective_epsilons": checkpoint["effective_epsilons"],
            "strict_state_dict_load": "passed",
        },
        "native_run_provenance": {
            "train_start": matching_train_start,
            "snapshot_saved": matching_snapshot,
            "train_end": matching_train_end,
        },
        "data_consistency": {
            "embedding_shape": list(embeddings_np.shape),
            "training_rows": int(len(train_frame)),
            "transition_events": int(len(sources)),
            "item_sids_shape": list(item_sids.shape),
            "raw_sids_shape": list(raw_sids.shape),
            "hgrec_sids_shape": list(hgrec_sids.shape),
            "item_json_equals_raw_and_hgrec": True,
            "strict_checkpoint_load": "passed",
            "cpu_reencoding_mismatch_by_level": cpu_mismatch_by_level,
            "cpu_reencoding_cumulative_mismatch": cpu_mismatch_cumulative,
            "cpu_reencoding_used_as_sid_source": False,
            "sid_source": "native Stage2 full-corpus GPU snapshot; independently linked by the native training metrics log",
            "sid_extension_values": extension_values.tolist(),
            "training_target_items": int(np.count_nonzero(frequencies)),
            "zero_target_frequency_items": int((frequencies == 0).sum()),
        },
        "popularity_rule": popularity_rule,
    }
    summary = {
        "metadata": metadata,
        "codeword_usage": code_summary,
        "prefix_structure": prefix_summary,
        "popularity": popularity_summary,
        "behavior_prefix_sharing": behavior_summary,
        "geometry": distance_summary,
        "residuals": residual_summary,
        "anomaly_example_counts": dict(Counter(row["case"] for row in anomaly_rows)),
    }
    write_json(OUTPUT_DIR / "summary.json", summary)
    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "commit": commit,
        "checkpoint_step": checkpoint["global_step"],
        "exported_artifacts_consistent": True,
        "strict_checkpoint_load": True,
        "cpu_reencoding_cumulative_mismatch": cpu_mismatch_cumulative,
        "unique_prefixes": {key: value["unique_prefixes"] for key, value in prefix_summary.items() if isinstance(value, dict) and "unique_prefixes" in value},
        "behavior_prefix_sharing": behavior_summary,
        "geometry_spearman": distance_summary["spearman_hyperbolic_vs_tangent_euclidean"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
