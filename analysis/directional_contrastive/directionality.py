from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairs_from_frame(frame):
    pairs = []
    for history, target in zip(frame["seen_history"].to_numpy(), frame["target"].to_numpy()):
        if history is not None and len(history):
            pairs.append((int(history[-1]), int(target)))
    return pairs


def build_transition_graph(frame):
    pairs = pairs_from_frame(frame)
    edge_counts = Counter(pairs)
    out_degree, in_degree = Counter(), Counter()
    out_strength, in_strength = Counter(), Counter()
    frequency = Counter(target for _, target in pairs)
    for (source, target), count in edge_counts.items():
        out_degree[source] += 1
        in_degree[target] += 1
        out_strength[source] += count
        in_strength[target] += count
    return {"edge_counts": edge_counts, "negative_edges": set(),
            "reverse_observed": {edge: (edge[1], edge[0]) in edge_counts for edge in edge_counts},
            "out_degree": out_degree, "in_degree": in_degree, "out_strength": out_strength,
            "in_strength": in_strength, "train_frequency": frequency}


def classify_pairs(frame_or_graph, pairs):
    graph = frame_or_graph if isinstance(frame_or_graph, dict) and "edge_counts" in frame_or_graph else build_transition_graph(frame_or_graph)
    edges = graph["edge_counts"]
    return ["SELF" if s == t else "REVERSE-OBSERVED" if (t, s) in edges else "REVERSE-UNOBSERVED" for s, t in pairs]


def stratify_pairs(graph, pairs, quantiles=4):
    categories = classify_pairs(graph, pairs)
    nonsel = [(p, c) for p, c in zip(pairs, categories) if c != "SELF"]
    sources = np.asarray([p[0] for p, _ in nonsel], dtype=np.int64)
    targets = np.asarray([p[1] for p, _ in nonsel], dtype=np.int64)
    def bins(ids, degree):
        vals = np.asarray([degree.get(int(i), 0) for i in ids], dtype=np.float64)
        if len(vals) < 2 or np.all(vals == vals[0]):
            return np.zeros(len(vals), dtype=np.int64)
        return pd.qcut(pd.Series(vals).rank(method="average"), q=min(quantiles, len(np.unique(vals))), labels=False, duplicates="drop").to_numpy(dtype=np.int64)
    sb, tb = bins(sources, graph["out_degree"]), bins(targets, graph["in_degree"])
    groups = {}
    for (pair, category), source_bin, target_bin in zip(nonsel, sb, tb):
        name = f"{category}|source_q{int(source_bin)}|target_q{int(target_bin)}"
        groups.setdefault(name, []).append(pair)
    return groups


class DirectionalHeads(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.f_source = nn.Linear(dim, dim)
        self.f_target = nn.Linear(dim, dim)

    def project_source(self, x):
        return F.normalize(self.f_source(F.normalize(x, dim=-1)), dim=-1)

    def project_target(self, x):
        return F.normalize(self.f_target(F.normalize(x, dim=-1)), dim=-1)


def score_pairs(formula, pairs, heads=None, temperature=0.07):
    """pairs shape (..., 2, D), outputs the directed score for the ordered pair."""
    left, right = pairs[..., 0, :], pairs[..., 1, :]
    if formula == "resample":
        return -torch.linalg.vector_norm(left - right, dim=-1) / temperature
    if heads is None:
        raise ValueError("projection scores require DirectionalHeads")
    forward = (heads.project_source(left) * heads.project_target(right)).sum(-1)
    if formula == "resample_proj_dir":
        return forward / temperature
    if formula == "resample_proj_sym":
        reverse = (heads.project_source(right) * heads.project_target(left)).sum(-1)
        return (forward + reverse) * (0.5 / temperature)
    raise ValueError(f"unknown score formula: {formula}")


def summarize_level_scores(scores, labels=None, batch_size=None):
    """Shape-oriented finite summaries; scores is (levels, pairs)."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("scores must be level-by-pair")
    out = []
    for row in values:
        out.append({"mean": float(np.mean(row)), "median": float(np.median(row)), "n": int(row.size)})
    return out


def prefix_sharing_metrics(levels, positives, negatives):
    levels = np.asarray(levels)
    def share(pairs, depth):
        return float(np.mean([np.all(levels[a, :depth] == levels[b, :depth]) for a, b in pairs])) if pairs else float("nan")
    names = ("L1", "L1+L2", "L1+L2+L3")
    out = {}
    for depth, name in enumerate(names, 1):
        pos, neg = share(positives, depth), share(negatives, depth)
        out[name] = {"positive_share_rate": pos, "negative_share_rate": neg,
                     "gap": pos - neg, "er": pos / neg if neg > 0 else None,
                     "er_undefined_reason": None if neg > 0 else "negative sharing rate is zero"}
    return out


def present_arms(status):
    return {"present": [name for name, exists in status.items() if exists],
            "missing": [name for name, exists in status.items() if not exists]}


def _summary(x):
    a = np.asarray(x, dtype=np.float64)
    return {"mean": float(np.mean(a)) if a.size else None,
            "median": float(np.median(a)) if a.size else None,
            "std": float(np.std(a)) if a.size else None,
            "min": float(np.min(a)) if a.size else None,
            "p10": float(np.percentile(a, 10)) if a.size else None,
            "p25": float(np.percentile(a, 25)) if a.size else None,
            "p75": float(np.percentile(a, 75)) if a.size else None,
            "p90": float(np.percentile(a, 90)) if a.size else None,
            "max": float(np.max(a)) if a.size else None, "n": int(a.size)}


def evaluate_residual_scores(residuals, sources, targets, formula, heads=None, temperature=0.07, batch_size=512,
                             include_self=False):
    """Score each validation minibatch against its in-batch target candidates."""
    device = residuals.device
    n = len(sources)
    result = []
    for level in range(residuals.shape[0]):
        forward_all, reverse_all, gap_all, logitgap_all, recalls = [], [], [], [], {1: [], 5: [], 10: []}
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            src_ids, tgt_ids = sources[start:stop], targets[start:stop]
            valid = np.ones(len(src_ids), dtype=bool) if include_self else src_ids != tgt_ids
            if not np.any(valid):
                continue
            src_ids, tgt_ids = src_ids[valid], tgt_ids[valid]
            x = residuals[level]
            src, candidates = x[torch.as_tensor(src_ids, device=device)], x[torch.as_tensor(tgt_ids, device=device)]
            if formula == "resample":
                logits = -torch.cdist(src, candidates) / temperature
                reverse_logits = logits.T
            else:
                fsrc = heads.project_source(src); ftgt = heads.project_target(candidates)
                logits = fsrc @ ftgt.T
                if formula == "resample_proj_sym":
                    reverse_logits = heads.project_source(candidates) @ heads.project_target(src).T
                    logits = (logits + reverse_logits.T) * 0.5 / temperature
                    reverse_logits = logits.T
                else:
                    reverse_logits = heads.project_source(candidates) @ heads.project_target(src).T
                    logits = logits / temperature
                    reverse_logits = reverse_logits / temperature
            diag = logits.diagonal()
            rev_diag = reverse_logits.diagonal()
            forward_all.extend(diag.detach().cpu().tolist()); reverse_all.extend(rev_diag.detach().cpu().tolist())
            gap_all.extend((diag - rev_diag).detach().cpu().tolist())
            diagonal = torch.eye(len(src_ids), dtype=torch.bool, device=device)
            duplicate_targets = torch.as_tensor(tgt_ids, device=device)[:, None].eq(
                torch.as_tensor(tgt_ids, device=device)[None, :])
            duplicate_targets.fill_diagonal_(False)
            blocked = diagonal | duplicate_targets
            masked_logits = logits.masked_fill(duplicate_targets, float("-inf"))
            if len(src_ids) > 1:
                neg_mean = logits.masked_fill(blocked, float("nan")).nanmean(dim=1)
                logitgap_all.extend((diag - neg_mean).detach().cpu().tolist())
            order = masked_logits.argsort(dim=1, descending=True)
            for k in recalls:
                recalls[k].extend((order[:, :min(k, len(src_ids))] == torch.arange(len(src_ids), device=device)[:, None]).any(dim=1).float().cpu().tolist())
        result.append({"forward_score": _summary(forward_all), "reverse_score": _summary(reverse_all),
                       "direction_gap": _summary(gap_all), "positive_vs_negative_logit_gap": _summary(logitgap_all),
                       "recall": {f"@{k}": float(np.mean(v)) if v else None for k, v in recalls.items()},
                       "residual_norm": float(residuals[level].norm(dim=-1).mean().detach().cpu())})
    return result


def quantized_residuals(model, embeddings):
    encoded = model.encoder(embeddings)
    residual = encoded
    quantized, indices, residual_trace = [], [], []
    for layer in model.rq.vq_layers:
        residual_trace.append(residual)
        codebook = layer.get_code_embs()
        idx = ((residual.square().sum(-1, keepdim=True) - 2 * residual @ codebook.T + codebook.square().sum(-1))
               .argmin(dim=-1))
        q = codebook[idx]
        indices.append(idx); quantized.append(q); residual = residual - q
    return encoded, torch.stack(quantized), torch.stack(indices), torch.stack(residual_trace)
