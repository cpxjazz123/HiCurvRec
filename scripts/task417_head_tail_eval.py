"""Task 417: Head/Tail stratified R@5/R@10/NDCG@5/NDCG@10 on existing Stage-4 inference tensors.

Strategy (no retrain; reuse existing s4 outputs):
  1. Iterate over training tfrecord.gz partitions; collect interaction Counter over all
     iids observed in sequence_data (and user_id which is also an int).
  2. Iterate over testing tfrecord.gz partitions; for each row, take the last item id
     in sequence_data as the "next-item target" t_u and record the uid.
  3. Match uid -> target item id (uid is the row index in inference tensor).
  4. Form popularity rank over the union of {target_iid_train, target_iid_test}.
     Head = top-20% by train-count, Tail = bottom-80% (with a stable cut so every
     bucket has positive size).
  5. For each of H_E_E_E, H_H_E_E, E_E_E_E (task396) and H_H_H_H, build a per-user
     vector of (R@5, R@10, NDCG@5, NDCG@10); average within Head/Tail/All buckets.
  6. Write JSON summary + verdict.md.

Usage:
    python task_artifacts/scripts/task417_head_tail_eval.py \
        --out_dir /home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task417_head_tail_r10
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

GRID_ROOT = Path("/fs04/ar57/wenyu/GeneRec/GRID")
sys.path.insert(0, str(GRID_ROOT))
os.chdir(str(GRID_ROOT))

import numpy as np
import tensorflow as tf  # type: ignore  # noqa: E402
import torch  # noqa: E402


TOYS_DATA_DIR = "/fs04/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys"

VARIANT_PATHS: Dict[str, str] = {
    "H_H_H_H": "/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v5_h_h_h_h_s4/pickle/merged_predictions_tensor.pt",
    "H_H_E_E": "/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v5_h_h_e_e_s4/pickle/merged_predictions_tensor.pt",
    "H_E_E_E": "/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v5_h_e_e_e_s4/pickle/merged_predictions_tensor.pt",
    "E_E_E_E_baseline": "/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task396_baseline_s4_rerun/pickle/merged_predictions_tensor.pt",
}

# Stage 2 semantic-id table for each variant: (n_items=11924, num_hier=4)
VARIANT_SID_TABLE_PATHS: Dict[str, str] = {
    "H_H_H_H": "/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v4_hhhh_s22/pickle/merged_predictions_tensor.pt",
    "H_H_E_E": "/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v4_hhee_s22/pickle/merged_predictions_tensor.pt",
    "H_E_E_E": "/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task388v4_hee_s22/pickle/merged_predictions_tensor.pt",
    "E_E_E_E_baseline": "/fs04/ar57/wenyu/GeneRec/GRID/logs/inference/runs/task13_group_a_s22/pickle/merged_predictions_tensor.pt",
}


# -----------------------------------------------------------------------------#
# Step 1: gather popularity                                                    #
# -----------------------------------------------------------------------------#
def _list_paritions(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"missing data folder: {folder}")
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("partition_") and f.endswith(".tfrecord.gz"))
    if not files:
        raise FileNotFoundError(f"no partition_*.tfrecord.gz under {folder}")
    return files


def _iter_tfrecord_rows(file_paths: List[str]):
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


def build_train_item_counter(data_dir: str) -> Counter:
    parts = _list_paritions(os.path.join(data_dir, "training"))
    counter: Counter = Counter()
    t0 = time.time()
    for fp in parts:
        for _uid, seq in _iter_tfrecord_rows([fp]):
            counter.update(seq)
    print(f"[info] train partitions scanned={len(parts)}, distinct_iids={len(counter)}, total_interactions={sum(counter.values())}, t={time.time() - t0:.1f}s")
    return counter


# -----------------------------------------------------------------------------#
# Step 2: gather uid->target_iid from testing partitions                      #
# -----------------------------------------------------------------------------#
def build_uid_target_iid(data_dir: str, expected_n_users: int) -> Tuple[np.ndarray, int]:
    """Return (target_iid[uid], n_unique_uids_seen).

    For each row, target is the last non-padding item id in sequence_data. (We
    treat 0 and 1 as semantic-id placeholders introduced by SID preprocessing; we
    keep them only if they appear in raw tfrecord. To stay faithful to the raw
    count, we use the literal final item id in the int64_list.)
    """
    parts = _list_paritions(os.path.join(data_dir, "testing"))
    target = np.full(expected_n_users, -1, dtype=np.int64)
    n_seen = 0
    max_uid = -1
    t0 = time.time()
    for fp in parts:
        for uid, seq in _iter_tfrecord_rows([fp]):
            n_seen += 1
            if uid >= expected_n_users:
                raise ValueError(f"uid {uid} >= expected_n_users {expected_n_users}")
            target[uid] = int(seq[-1])
            if uid > max_uid:
                max_uid = uid
    print(f"[info] testing partitions scanned={len(parts)}, rows={n_seen}, max_uid={max_uid}, t={time.time() - t0:.1f}s")
    if (target == -1).any():
        missing = int((target == -1).sum())
        raise ValueError(f"{missing} uids have no target item id; cannot score those rows")
    return target, n_seen


# -----------------------------------------------------------------------------#
# Step 3: bucket assignment                                                   #
# -----------------------------------------------------------------------------#
def assign_head_tail(target_iid: np.ndarray, train_count: Counter, head_quantile: float = 0.20) -> Dict[str, np.ndarray]:
    """Top head_quantile of items by train count = head; rest = tail.

    Returns boolean masks aligned to uids:
      - head_mask[u] = (target_iid[u] in head bucket)
      - tail_mask[u] = (target_iid[u] in tail bucket)
    Anything not seen in training is treated as tail (count=0).
    """
    distinct_iids = set(int(x) for x in train_count.keys()) | set(int(x) for x in target_iid.tolist())
    if not distinct_iids:
        raise ValueError("no items observed in train+test; cannot bucket")
    # rank items by (train_count desc, iid asc) to break ties deterministically
    items_with_count = [(iid, train_count.get(iid, 0)) for iid in distinct_iids]
    items_with_count.sort(key=lambda x: (-x[1], x[0]))
    n_total = len(items_with_count)
    cutoff = max(1, int(round(head_quantile * n_total)))
    head_set = {iid for iid, _ in items_with_count[:cutoff]}
    tail_set = {iid for iid, _ in items_with_count[cutoff:]}
    head_mask = np.array([iid in head_set for iid in target_iid], dtype=bool)
    tail_mask = np.array([iid in tail_set for iid in target_iid], dtype=bool)
    print(f"[info] total unique items across train+test={n_total}; head_bucket_size={len(head_set)}; tail_bucket_size={len(tail_set)}")
    if not head_mask.any():
        raise ValueError("head bucket is empty; raise head_quantile or add items")
    if not tail_mask.any():
        raise ValueError("tail bucket is empty; lower head_quantile")
    return {"head": head_mask, "tail": tail_mask}


# -----------------------------------------------------------------------------#
# Step 4: per-user metric computation from generated_ids tensor                #
# -----------------------------------------------------------------------------#
TOP_K_LIST = [5, 10]


def _dcg_discount(top_k: int) -> torch.Tensor:
    # discount = log2(rank+1), rank starts at 1
    return 1.0 / torch.log2(torch.arange(2, top_k + 2, dtype=torch.float64))


def per_user_metrics(pred: torch.Tensor, target_iid_t: torch.Tensor, sid_path: torch.Tensor, top_k_list: List[int] = TOP_K_LIST) -> Dict[str, np.ndarray]:
    """Return per-user arrays of {Recall@K, NDCG@K}.

    pred: (n_users, top_candidates=10, num_hier=4) int64
    target_iid_t: (n_users,) int64 target item id
    sid_path: (n_items, num_hier) int64 semantic id table (item catalog), same
              num_hier as pred.
    """
    if pred.dim() != 3 or pred.shape[2] != 4:
        raise ValueError(f"bad pred shape {tuple(pred.shape)}")
    n_users, n_candidates, _num_hier = pred.shape
    if target_iid_t.shape[0] != n_users:
        raise ValueError("target_iid length != pred first dim")
    target_sid = sid_path[target_iid_t]  # (n_users, num_hier)
    pred = pred.long()
    target_sid = target_sid.long()

    # match: (n_users, n_candidates) bool
    match = (pred == target_sid.unsqueeze(1)).all(dim=2)
    # rank of the FIRST match (-1 if none)
    first_match = match.float().argmax(dim=1)
    has_match = match.any(dim=1)
    # ranks in 0-indexed; convert to 1-indexed; -1 -> inf when no match
    first_match_1 = torch.where(has_match, first_match + 1, torch.full_like(first_match, n_candidates + 1))

    metrics_per_user: Dict[str, np.ndarray] = {}
    for k in top_k_list:
        if k > n_candidates:
            raise ValueError(f"k={k} > n_candidates={n_candidates}")
        # Recall@K = 1 iff target first rank <= k
        rec_k = (first_match_1 <= k).float().numpy()
        metrics_per_user[f"Recall@{k}"] = rec_k
        # NDCG@K with binary gain = 1 if hit, discount 1/log2(rank+1)
        # If no hit within K -> NDCG = 0 (no relevance at all within window)
        rank_clipped = torch.minimum(first_match_1, torch.tensor(k, dtype=first_match_1.dtype))
        # ideal DCG for single relevant in top-K = 1/log2(2) = 1
        gains = (rank_clipped <= k).float() * (1.0 / torch.log2(rank_clipped.float() + 1.0))
        ndcg_k = gains.numpy()
        metrics_per_user[f"NDCG@{k}"] = ndcg_k
    return metrics_per_user


def aggregate(metrics: Dict[str, np.ndarray], masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for bucket, mask in masks.items():
        sub: Dict[str, float] = {}
        idx = np.where(mask)[0]
        n = int(mask.sum())
        for metric_name, arr in metrics.items():
            sub[metric_name] = float(arr[mask].mean()) if n else float("nan")
        sub["n_users"] = n
        out[bucket] = sub
    return out


# -----------------------------------------------------------------------------#
# Main                                                                        #
# -----------------------------------------------------------------------------#
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=TOYS_DATA_DIR)
    p.add_argument("--out_dir", default="/home/wlia0047/ar57/wenyu/GeneRec/GRID/result/task417_head_tail_r10")
    p.add_argument("--head_quantile", type=float, default=0.20)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load all prediction tensors first so we agree on n_users.
    preds: Dict[str, torch.Tensor] = {}
    for name, path in VARIANT_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing pred tensor for {name}: {path}")
        t = torch.load(path, map_location="cpu", weights_only=False).long()
        if t.dim() != 3:
            raise ValueError(f"{name} pred has wrong dim {t.shape}")
        preds[name] = t
    n_users = next(iter(preds.values())).shape[0]
    if not all(t.shape[0] == n_users for t in preds.values()):
        raise ValueError("variant pred tensors disagree on n_users")
    print(f"[info] {len(preds)} variant tensors loaded, n_users={n_users}, top_k={next(iter(preds.values())).shape[1]}")

    # popularity from training partitions
    train_counter = build_train_item_counter(args.data_dir)

    # uid -> next target iid (from testing partitions)
    target_iid, n_rows = build_uid_target_iid(args.data_dir, n_users)
    print(f"[info] gathered {n_rows} test rows; first 5 target iids: {target_iid[:5].tolist()}")

    # bucket masks
    masks = assign_head_tail(target_iid, train_counter, head_quantile=args.head_quantile)

    # need semantic id table per variant: catalog shape is either (n_items, num_hier)
    # OR (num_hier, n_items) -- normalize to (n_items, num_hier).
    sid_tables: Dict[str, torch.Tensor] = {}
    n_items_in_sid = None
    for name, sid_p in VARIANT_SID_TABLE_PATHS.items():
        if not os.path.exists(sid_p):
            raise FileNotFoundError(f"missing SID table for {name}: {sid_p}")
        sid_t = torch.load(sid_p, map_location="cpu", weights_only=False).long()
        if sid_t.dim() != 2:
            raise ValueError(f"SID table for {name} must be 2D, got shape {tuple(sid_t.shape)}")
        if sid_t.shape[0] == 11924 and sid_t.shape[1] == 4:
            pass  # already (n_items, num_hier)
        elif sid_t.shape[0] == 4 and sid_t.shape[1] == 11924:
            sid_t = sid_t.t().contiguous()  # transpose to (n_items, num_hier)
        else:
            raise ValueError(
                f"SID table for {name} has unexpected shape {tuple(sid_t.shape)}; "
                "expected (11924, 4) or (4, 11924)"
            )
        sid_tables[name] = sid_t
        if n_items_in_sid is None:
            n_items_in_sid = sid_t.shape[0]
        elif sid_t.shape[0] != n_items_in_sid:
            raise ValueError(f"SID table row count mismatch: {name}={sid_t.shape[0]} vs {n_items_in_sid}")
    if n_items_in_sid is None:
        raise RuntimeError("no SID tables loaded")
    if (target_iid >= n_items_in_sid).any() or (target_iid < 0).any():
        bad = target_iid[(target_iid >= n_items_in_sid) | (target_iid < 0)]
        raise ValueError(f"{int(len(bad))} target iids out of SID catalog range (max={n_items_in_sid - 1})")
    print(f"[info] all 4 SID tables loaded; n_items={n_items_in_sid}")

    # Compute per-user metrics for every variant
    target_iid_t = torch.from_numpy(target_iid)
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_user_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for name, tensor in preds.items():
        sid_t = sid_tables[name]
        m = per_user_metrics(tensor, target_iid_t, sid_t, top_k_list=TOP_K_LIST)
        per_user_arrays[name] = m
        summary[name] = aggregate(m, masks)
        all_mask = np.ones(n_users, dtype=bool)
        all_agg: Dict[str, float] = {"n_users": n_users}
        for metric_name, arr in m.items():
            all_agg[metric_name] = float(arr.mean())
        summary[name]["all"] = all_agg

    summary_dict = {
        "task": "task417_head_tail_r10",
        "data_dir": args.data_dir,
        "head_quantile": args.head_quantile,
        "n_users": n_users,
        "n_items_in_sid": n_items_in_sid,
        "bucket_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "variants": summary,
    }
    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\n[info] wrote summary -> {out_json}")
    print(json.dumps(summary_dict, indent=2))

    # verdict.md
    base = summary.get("E_E_E_E_baseline", {}).get("all", {})
    base_r10 = base.get("Recall@10")
    if base_r10 is None:
        raise ValueError("baseline E_E_E_E missing in summary; cannot build verdict")
    lines = [
        "# task417 verdict — Head/Tail stratified R@5/R@10/NDCG",
        "",
        f"- data: {args.data_dir}",
        f"- n_users evaluated: {n_users}",
        f"- head_quantile (top-X% by train-count = Head): {args.head_quantile}",
        f"- bucket sizes: head={int(masks['head'].sum())}, tail={int(masks['tail'].sum())}, all={n_users}",
        f"- baseline E_E_E_E Recall@10 (all): {base_r10:.4f}",
        "",
        "## Per-variant Recall@10 (Head, Tail, All)",
        "",
    ]
    for name, agg in summary.items():
        h = agg.get("head", {})
        t = agg.get("tail", {})
        a = agg.get("all", {})
        lines.append(
            f"- {name}: Head R@10={h.get('Recall@10', float('nan')):.4f} "
            f"(NDCG@10={h.get('NDCG@10', float('nan')):.4f}); "
            f"Tail R@10={t.get('Recall@10', float('nan')):.4f} "
            f"(NDCG@10={t.get('NDCG@10', float('nan')):.4f}); "
            f"All R@10={a.get('Recall@10', float('nan')):.4f}"
        )
    lines.append("")

    # Head Δ / Tail Δ for H_E_E_E vs baseline
    def delta(name: str) -> str:
        if name not in summary:
            return f"  - {name}: missing"
        h = summary[name].get("head", {}).get("Recall@10", float("nan"))
        t = summary[name].get("tail", {}).get("Recall@10", float("nan"))
        a = summary[name].get("all", {}).get("Recall@10", float("nan"))
        base_h = summary["E_E_E_E_baseline"]["head"]["Recall@10"]
        base_t = summary["E_E_E_E_baseline"]["tail"]["Recall@10"]
        base_a = summary["E_E_E_E_baseline"]["all"]["Recall@10"]
        d_h = h - base_h
        d_t = t - base_t
        d_a = a - base_a
        return (
            f"  - {name} vs E_E_E_E: Head Δ={d_h:+.4f}; Tail Δ={d_t:+.4f}; All Δ={d_a:+.4f}"
        )

    lines.append("## Δ vs E_E_E_E baseline")
    for n in ["H_E_E_E", "H_H_E_E", "H_H_H_H"]:
        lines.append(delta(n))
    lines.append("")

    # hypothesis check (only meaningful if H_E_E_E and E_E_E_E are both present)
    if "H_E_E_E" in summary and "E_E_E_E_baseline" in summary:
        he_head = summary["H_E_E_E"]["head"]["Recall@10"]
        he_tail = summary["H_E_E_E"]["tail"]["Recall@10"]
        base_head = summary["E_E_E_E_baseline"]["head"]["Recall@10"]
        base_tail = summary["E_E_E_E_baseline"]["tail"]["Recall@10"]
        d_h = he_head - base_head
        d_t = he_tail - base_tail
        supported = (d_t > 0) and (d_h < 0) and (abs(d_h) > abs(d_t))
        lines.append("## Hypothesis verdict")
        lines.append(f"- H_E_E-E - E_E_E_E: Head Δ={d_h:+.4f}; Tail Δ={d_t:+.4f}")
        lines.append(f"- Rule: Tail Δ > 0 AND Head Δ < 0 AND |Head Δ| > |Tail Δ|")
        lines.append(f"- Result: {'SUPPORTED (partial)' if supported else 'NOT SUPPORTED'}")
    else:
        lines.append("## Hypothesis verdict")
        lines.append("- insufficient data; H_E_E_E or E_E_E_E_baseline missing")

    out_md = os.path.join(args.out_dir, "verdict.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[info] wrote verdict -> {out_md}")


if __name__ == "__main__":
    main()
