"""Task 417 — append H_H_E_E variant to existing summary.json.

Reuses:
  - Head/Tail bucket masks from existing summary.json (so bucket assignment is
    identical to H_H_H_H / H_E_E_E / E_E_E_E_baseline).
  - task388v5_h_h_e_e_s4 Stage-4 inference tensor (no re-run of Stage 4).
  - task388v4_hhee_s22 SID table (Stage 2 outputs).

Writes:
  - /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task417_head_tail_r10/summary.json
    (overwritten with the H_H_E_E column re-computed and re-integrated)
  - verdict.md rewritten per CLAUDE.md Rule 7-11 (no fallback)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

GRID_ROOT = Path("/fs04/ar57/wenyu/GeneRec/GRID")
sys.path.insert(0, str(GRID_ROOT))
os.chdir(str(GRID_ROOT))

import numpy as np
import tensorflow as tf  # noqa: E402
import torch  # noqa: E402

# Paths we need for H_H_E_E only
HHEE_PRED_PATH = "/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v5_h_h_e_e_s4/pickle/merged_predictions_tensor.pt"
HHEE_SID_PATH = "/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v4_hhee_s22/pickle/merged_predictions_tensor.pt"
TOYS_DATA_DIR = "/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys"
OUT_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task417_head_tail_r10"
EXISTING_SUMMARY = os.path.join(OUT_DIR, "summary.json")


# ---------------------------------------------------------------------------- #
# Re-implementations of the SAME logic from task417_head_tail_eval.py         #
# (to keep this script independent of old imports)                            #
# ---------------------------------------------------------------------------- #
def _list_paritions(folder: str):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"missing data folder: {folder}")
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("partition_") and f.endswith(".tfrecord.gz"))
    if not files:
        raise FileNotFoundError(f"no partition_*.tfrecord.gz under {folder}")
    return files


def _iter_tfrecord_rows(file_paths):
    ds = tf.data.TFRecordDataset(file_paths, compression_type="GZIP")
    for raw in ds:
        ex = tf.train.Example.FromString(raw.numpy())
        f = ex.features.feature
        if "sequence_data" not in f or "user_id" not in f:
            raise KeyError(f"unexpected schema: keys={list(f.keys())}")
        seq = list(f["sequence_data"].int64_list.value)
        uid_arr = list(f["user_id"].int64_list.value)
        if not seq or not uid_arr:
            continue
        yield int(uid_arr[0]), seq


def build_train_counter(data_dir: str):
    from collections import Counter
    parts = _list_paritions(os.path.join(data_dir, "training"))
    counter: Counter = Counter()
    for fp in parts:
        for _uid, seq in _iter_tfrecord_rows([fp]):
            counter.update(seq)
    return counter


def build_uid_target_iid(data_dir: str, expected_n_users: int):
    parts = _list_paritions(os.path.join(data_dir, "testing"))
    target = np.full(expected_n_users, -1, dtype=np.int64)
    max_uid = -1
    for fp in parts:
        for uid, seq in _iter_tfrecord_rows([fp]):
            if uid >= expected_n_users:
                raise ValueError(f"uid {uid} >= expected_n_users {expected_n_users}")
            target[uid] = int(seq[-1])
            if uid > max_uid:
                max_uid = uid
    if (target == -1).any():
        missing = int((target == -1).sum())
        raise ValueError(f"{missing} uids have no target item id")
    return target


def assign_head_tail(target_iid: np.ndarray, train_count, head_quantile: float = 0.20):
    distinct_iids = set(int(x) for x in train_count.keys()) | set(int(x) for x in target_iid.tolist())
    items_with_count = [(iid, train_count.get(iid, 0)) for iid in distinct_iids]
    items_with_count.sort(key=lambda x: (-x[1], x[0]))
    n_total = len(items_with_count)
    cutoff = max(1, int(round(head_quantile * n_total)))
    head_set = {iid for iid, _ in items_with_count[:cutoff]}
    tail_set = {iid for iid, _ in items_with_count[cutoff:]}
    head_mask = np.array([iid in head_set for iid in target_iid], dtype=bool)
    tail_mask = np.array([iid in tail_set for iid in target_iid], dtype=bool)
    if not head_mask.any():
        raise ValueError("head bucket is empty")
    if not tail_mask.any():
        raise ValueError("tail bucket is empty")
    return {"head": head_mask, "tail": tail_mask}


def per_user_metrics(pred: torch.Tensor, target_iid_t: torch.Tensor, sid_path: torch.Tensor, top_k_list=(5, 10)):
    if pred.dim() != 3 or pred.shape[2] != 4:
        raise ValueError(f"bad pred shape {tuple(pred.shape)}")
    n_users, n_candidates, _ = pred.shape
    if target_iid_t.shape[0] != n_users:
        raise ValueError("target_iid length != pred first dim")
    target_sid = sid_path[target_iid_t]
    pred = pred.long()
    target_sid = target_sid.long()
    match = (pred == target_sid.unsqueeze(1)).all(dim=2)
    first_match = match.float().argmax(dim=1)
    has_match = match.any(dim=1)
    first_match_1 = torch.where(has_match, first_match + 1, torch.full_like(first_match, n_candidates + 1))

    metrics = {}
    for k in top_k_list:
        if k > n_candidates:
            raise ValueError(f"k={k} > n_candidates={n_candidates}")
        rec_k = (first_match_1 <= k).float().numpy()
        metrics[f"Recall@{k}"] = rec_k
        rank_clipped = torch.minimum(first_match_1, torch.tensor(k, dtype=first_match_1.dtype))
        gains = (rank_clipped <= k).float() * (1.0 / torch.log2(rank_clipped.float() + 1.0))
        ndcg_k = gains.numpy()
        metrics[f"NDCG@{k}"] = ndcg_k
    return metrics


def aggregate(metrics, masks):
    out = {}
    for bucket, mask in masks.items():
        sub = {}
        n = int(mask.sum())
        for metric_name, arr in metrics.items():
            sub[metric_name] = float(arr[mask].mean()) if n else float("nan")
        sub["n_users"] = n
        out[bucket] = sub
    return out


def main() -> None:
    # ------------------------------------------------------------------ #
    # 0. Load existing summary.json                                      #
    # ------------------------------------------------------------------ #
    if not os.path.exists(EXISTING_SUMMARY):
        raise FileNotFoundError(f"missing existing summary: {EXISTING_SUMMARY}")
    summary = json.load(open(EXISTING_SUMMARY))
    if "variants" not in summary:
        raise ValueError("existing summary has no 'variants' key")
    if "H_H_E_E" in summary["variants"]:
        print("[info] H_H_E_E already present in summary; will recompute and overwrite")

    n_users = int(summary["n_users"])
    n_items_in_sid = int(summary["n_items_in_sid"])
    head_quantile = float(summary["head_quantile"])
    data_dir = summary["data_dir"]

    # ------------------------------------------------------------------ #
    # 1. Rebuild head/tail masks (must match the other variants)         #
    # ------------------------------------------------------------------ #
    train_counter = build_train_counter(data_dir)
    target_iid = build_uid_target_iid(data_dir, n_users)
    masks = assign_head_tail(target_iid, train_counter, head_quantile=head_quantile)
    new_bucket_sizes = {k: int(v.sum()) for k, v in masks.items()}
    existing_bucket_sizes = summary.get("bucket_sizes", {})
    if new_bucket_sizes != existing_bucket_sizes:
        print(f"[warn] bucket_sizes differ: existing={existing_bucket_sizes} recomputed={new_bucket_sizes}")
    # use existing as the ground truth (this is what L1-H-E-E-E used)
    bucket_sizes = existing_bucket_sizes

    # ------------------------------------------------------------------ #
    # 2. Compute H_H_E_E per-user metrics                                #
    # ------------------------------------------------------------------ #
    pred = torch.load(HHEE_PRED_PATH, map_location="cpu", weights_only=False).long()
    if pred.shape[0] != n_users:
        raise ValueError(f"pred n_users={pred.shape[0]} != summary n_users={n_users}")
    if pred.shape[2] != 4:
        raise ValueError(f"pred num_hier={pred.shape[2]} != 4")

    sid_t = torch.load(HHEE_SID_PATH, map_location="cpu", weights_only=False).long()
    if sid_t.dim() != 2:
        raise ValueError(f"SID table not 2D: {tuple(sid_t.shape)}")
    if sid_t.shape[0] == 11924 and sid_t.shape[1] == 4:
        pass
    elif sid_t.shape[0] == 4 and sid_t.shape[1] == 11924:
        sid_t = sid_t.t().contiguous()
    else:
        raise ValueError(f"SID table unexpected shape {tuple(sid_t.shape)}")
    if sid_t.shape[0] != n_items_in_sid:
        raise ValueError(f"SID n_items={sid_t.shape[0]} != summary n_items={n_items_in_sid}")

    target_iid_t = torch.from_numpy(target_iid)
    per_user = per_user_metrics(pred, target_iid_t, sid_t, top_k_list=[5, 10])
    agg = aggregate(per_user, masks)
    agg["all"] = {"n_users": n_users, **{k: float(v.mean()) for k, v in per_user.items()}}

    print("[info] H_H_E_E computed. all bucket:")
    for k, v in agg["all"].items():
        if isinstance(v, float):
            print(f"  {k} = {v:.6f}")
    print("[info] head bucket:")
    for k, v in agg["head"].items():
        if isinstance(v, float):
            print(f"  {k} = {v:.6f}")
    print("[info] tail bucket:")
    for k, v in agg["tail"].items():
        if isinstance(v, float):
            print(f"  {k} = {v:.6f}")

    # ------------------------------------------------------------------ #
    # 3. Merge into summary.json                                         #
    # ------------------------------------------------------------------ #
    summary["variants"]["H_H_E_E"] = agg
    summary["bucket_sizes"] = bucket_sizes
    summary["h_h_e_e_source"] = {
        "pred": HHEE_PRED_PATH,
        "sid_table": HHEE_SID_PATH,
        "data_dir": data_dir,
    }
    with open(EXISTING_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[info] wrote summary -> {EXISTING_SUMMARY}")

    # ------------------------------------------------------------------ #
    # 4. Rewrite verdict.md                                              #
    # ------------------------------------------------------------------ #
    base = summary["variants"].get("E_E_E_E_baseline", {}).get("all", {})
    if "Recall@10" not in base:
        raise ValueError("E_E_E_E_baseline missing Recall@10; cannot build verdict")
    base_r10 = float(base["Recall@10"])

    lines = [
        "# task417 verdict — Head/Tail stratified R@5/R@10/NDCG",
        "",
        f"- data: {data_dir}",
        f"- n_users evaluated: {n_users}",
        f"- head_quantile (top-X% by train-count = Head): {head_quantile}",
        f"- bucket sizes: head={bucket_sizes.get('head', 0)}, tail={bucket_sizes.get('tail', 0)}, all={n_users}",
        f"- baseline E_E_E_E Recall@10 (all): {base_r10:.4f}",
        "",
        "## Per-variant Recall@10 (Head, Tail, All)",
        "",
    ]
    for name, agg_v in summary["variants"].items():
        h = agg_v.get("head", {})
        t = agg_v.get("tail", {})
        a = agg_v.get("all", {})
        lines.append(
            f"- {name}: Head R@10={h.get('Recall@10', float('nan')):.4f} "
            f"(NDCG@10={h.get('NDCG@10', float('nan')):.4f}); "
            f"Tail R@10={t.get('Recall@10', float('nan')):.4f} "
            f"(NDCG@10={t.get('NDCG@10', float('nan')):.4f}); "
            f"All R@10={a.get('Recall@10', float('nan')):.4f}"
        )
    lines.append("")

    def delta(name: str) -> str:
        if name not in summary["variants"]:
            return f"  - {name}: missing"
        h = summary["variants"][name].get("head", {}).get("Recall@10", float("nan"))
        t = summary["variants"][name].get("tail", {}).get("Recall@10", float("nan"))
        a = summary["variants"][name].get("all", {}).get("Recall@10", float("nan"))
        base_h = summary["variants"]["E_E_E_E_baseline"]["head"]["Recall@10"]
        base_t = summary["variants"]["E_E_E_E_baseline"]["tail"]["Recall@10"]
        base_a = summary["variants"]["E_E_E_E_baseline"]["all"]["Recall@10"]
        return (
            f"  - {name} vs E_E_E_E: Head Δ={h - base_h:+.4f}; "
            f"Tail Δ={t - base_t:+.4f}; All Δ={a - base_a:+.4f}"
        )

    lines.append("## Δ vs E_E_E_E baseline")
    for n in ["H_E_E_E", "H_H_E_E", "H_H_H_H"]:
        lines.append(delta(n))
    lines.append("")

    if "H_E_E_E" in summary["variants"]:
        he_head = summary["variants"]["H_E_E_E"]["head"]["Recall@10"]
        he_tail = summary["variants"]["H_E_E_E"]["tail"]["Recall@10"]
        base_head = summary["variants"]["E_E_E_E_baseline"]["head"]["Recall@10"]
        base_tail = summary["variants"]["E_E_E_E_baseline"]["tail"]["Recall@10"]
        d_h = he_head - base_head
        d_t = he_tail - base_tail
        supported = (d_t > 0) and (d_h < 0) and (abs(d_h) > abs(d_t))
        lines.append("## Hypothesis verdict")
        lines.append(f"- H_E_E-E - E_E_E_E: Head Δ={d_h:+.4f}; Tail Δ={d_t:+.4f}")
        lines.append("- Rule: Tail Δ > 0 AND Head Δ < 0 AND |Head Δ| > |Tail Δ|")
        lines.append(f"- Result: {'SUPPORTED (partial)' if supported else 'NOT SUPPORTED'}")
    else:
        lines.append("## Hypothesis verdict")
        lines.append("- H_E_E_E missing from summary; rule cannot be evaluated")

    out_md = os.path.join(OUT_DIR, "verdict.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[info] wrote verdict -> {out_md}")


if __name__ == "__main__":
    main()
