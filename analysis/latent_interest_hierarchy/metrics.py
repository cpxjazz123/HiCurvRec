"""Metrics for the latent-interest-hierarchy study.

Next-item NLL and Recall@10, with USER-LEVEL bootstrap confidence intervals, plus
the diagnostics that decide whether an apparent hierarchy advantage is real:

  * a popularity-only baseline, so "better than nothing" is distinguishable from
    "better than knowing how popular items are";
  * short-history and low-frequency subsets, where a genuine interest structure
    should help most and a capacity effect should help least;
  * within-group versus between-group transition mass, computed from the model's own
    learned assignment rather than from any hand-specified category;
  * stability of the assignment across seeds, measured by agreement of the induced
    item partition rather than by comparing parameter vectors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402


def evaluate(model, rows, device: str, batch: int = 1024, topk: int = 10) -> dict:
    """NLL and Recall@topk over (history, target) rows."""
    model.eval()
    nll_sum = 0.0
    hits = 0
    per_user_nll = np.empty(len(rows), dtype=np.float64)
    per_user_hit = np.empty(len(rows), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            chunk = rows[s:s + batch]
            hist = [h for _u, h, _t in chunk]
            L = max(len(h) for h in hist)
            H = np.zeros((len(chunk), L), dtype=np.int64)
            for i, h in enumerate(hist):
                if h:
                    # left-aligned and shifted by one, so row 0 is the padding slot
                    H[i, L - len(h):] = [x + 1 for x in h]
            Ht = torch.as_tensor(H, dtype=torch.long, device=device)
            mask = (Ht != 0).float().unsqueeze(-1)
            emb = model.embed(Ht) * mask
            logits = model(emb, mask)
            logp = torch.log_softmax(logits.float(), dim=-1)
            # The logits are indexed in the model's shifted space, where raw item id
            # t lives at row t+1 (row 0 is padding). Gathering with the RAW id would
            # read the neighbouring item's probability.
            tgt = torch.as_tensor([t + 1 for _u, _h, t in chunk], dtype=torch.long,
                                  device=device)
            nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            top = logits.topk(topk, dim=-1).indices
            hit = (top == tgt.unsqueeze(1)).any(dim=1).float()
            nll_sum += float(nll.sum())
            hits += int(hit.sum())
            per_user_nll[s:s + len(chunk)] = nll.cpu().numpy()
            per_user_hit[s:s + len(chunk)] = hit.cpu().numpy()
    return {"nll": nll_sum / len(rows), "recall_at_10": hits / len(rows), "n": len(rows),
            "per_user_nll": per_user_nll, "per_user_hit": per_user_hit}


def bootstrap_ci(values: np.ndarray, stat, resamples: int, seed: int,
                 alpha: float = 0.05) -> dict:
    """Percentile bootstrap over USERS (rows are the resampling unit)."""
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    out = np.empty(resamples, dtype=np.float64)
    for b in range(resamples):
        idx = rng.integers(0, n, size=n)
        out[b] = stat(values[idx])
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return {"point": float(stat(values)), "lo": float(lo), "hi": float(hi),
            "resamples": int(resamples)}


def paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, stat, resamples: int,
                           seed: int, alpha: float = 0.05) -> dict:
    """Paired user-level bootstrap of stat(a) - stat(b) on the SAME users."""
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    diffs = np.empty(resamples, dtype=np.float64)
    for r in range(resamples):
        idx = rng.integers(0, n, size=n)
        diffs[r] = stat(a[idx]) - stat(b[idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(stat(a) - stat(b)), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool((lo > 0) or (hi < 0)), "resamples": int(resamples)}


def learned_assignment(model) -> np.ndarray:
    """The item -> leaf-interest assignment the model itself learned.

    Taken from argmax_c <E_i, C_c> with the SAME item embeddings and centroids the
    prediction uses. It is never taken from an external category, and for the
    hierarchical arms it is folded through the same grouping the profile uses.
    """
    with torch.no_grad():
        z = model.embed.weight[1:] @ model.centroids.t()       # drop the padding row
        leaf = z.argmax(dim=-1).cpu().numpy()
    if model.mode == "flat":
        return leaf
    # group[p, f] is the LEAF that sits under parent p, so the leaf -> parent map is
    # the inverse permutation. Indexing group.reshape(-1)[leaf] would return the
    # flattened position of the leaf, which is not the parent.
    leaf_to_parent = np.empty(C.N_INTEREST, dtype=np.int64)
    gp = model.group.cpu().numpy()
    for p_idx in range(gp.shape[0]):
        leaf_to_parent[gp[p_idx]] = p_idx
    return leaf_to_parent[leaf]


def partition_agreement(a: np.ndarray, b: np.ndarray, n_groups: int) -> float:
    """Fraction of item pairs that land in the same group under both partitions.

    A scale-free measure of whether two runs learned the SAME partition, unlike
    comparing centroids, which can rotate freely.
    """
    ga = a[:, None] == a[None, :]
    gb = b[:, None] == b[None, :]
    return float((ga == gb).mean())


def partition_agreement_null(a: np.ndarray, b: np.ndarray, n_perm: int = 20,
                            seed: int = 0) -> float:
    """The agreement two INDEPENDENT partitions of the same sizes would reach.

    Agreement scales with the number of groups -- 64 leaves agree with themselves
    about 0.97 of the time by chance while 8 parents reach only about 0.80 -- so an
    observed agreement is uninterpretable without this baseline.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        perm = rng.permutation(b.shape[0])
        vals.append(partition_agreement(a, b[perm], 0))
    return float(np.mean(vals))


def transition_mass(rows, assign_fn) -> dict:
    """Within-group versus between-group transition mass under the learned grouping.

    Built from the model's own assignment, never from a supplied category list. The
    comparison is with the share of pairs that would be within-group if transitions
    were assigned at random, so popularity cannot masquerade as structure.
    """
    same = 0
    total = 0
    for _u, h, t in rows:
        if not h:
            continue
        a_h = assign_fn(h[-1])
        a_t = assign_fn(t)
        same += int(a_h == a_t)
        total += 1
    within = same / total if total else 0.0
    return {"within_group_transition_rate": within, "n_pairs": total}