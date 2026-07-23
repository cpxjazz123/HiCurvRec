"""Task 444 — Stage 4 item-level R@5/R@10/NDCG@5/NDCG@10 evaluation.

Usage:
    python3 task444_s4_item_eval.py \\
        --variant U_E_E  # only for logging \\
        --pred_pt /path/to/s4/merged_predictions_tensor.pt \\
        --sid /path/to/s22/merged_predictions_tensor.pt \\
        --ckpt /path/to/s3.ckpt  # only for logging/trace \\
        --out_json /path/to/result.json \\
        --data_dir /home/.../data/amazon_data/toys

pred_pt: shape (n_users, n_candidates=10, num_hier=4) int64, from Stage 4 inference
sid: shape (11924, 4) 或 (4, 11924) int64, from Stage 2.2 (unified ckpt inference)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf  # type: ignore
import torch

GRID_ROOT = Path("/fs04/ar57/wenyu/GeneRec/GRID")
sys.path.insert(0, str(GRID_ROOT))
os.chdir(str(GRID_ROOT))

TOP_K_LIST = [5, 10]


def _list_paritions(folder: str) -> List[str]:
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                   if f.startswith("partition_") and f.endswith(".tfrecord.gz"))
    if not files:
        raise FileNotFoundError(f"no partition_*.tfrecord.gz under {folder}")
    return files


def _iter_test_rows(file_paths):
    ds = tf.data.TFRecordDataset(file_paths, compression_type="GZIP")
    for raw in ds:
        ex = tf.train.Example.FromString(raw.numpy())
        f = ex.features.feature
        if "sequence_data" not in f or "user_id" not in f:
            continue
        seq = list(f["sequence_data"].int64_list.value)
        uid_arr = list(f["user_id"].int64_list.value)
        if not seq or not uid_arr:
            continue
        yield int(uid_arr[0]), int(seq[-1])


def build_uid_target_iid(data_dir: str, n_users: int) -> np.ndarray:
    parts = _list_paritions(os.path.join(data_dir, "testing"))
    target = np.full(n_users, -1, dtype=np.int64)
    t0 = time.time()
    for fp in parts:
        for uid, last in _iter_test_rows([fp]):
            if uid >= n_users:
                continue
            target[uid] = last
    n_missing = int((target == -1).sum())
    print(f"[s4 eval] test partition scanned t={time.time()-t0:.1f}s, n_missing={n_missing}")
    if n_missing:
        raise ValueError(f"{n_missing} uids have no target item id")
    return target


def per_user_metrics(pred: torch.Tensor, target_sid: torch.Tensor,
                     top_k_list: List[int] = TOP_K_LIST) -> Dict[str, np.ndarray]:
    """R@K + NDCG@K per user.
       pred: (n_users, n_candidates, 4) int64
       target_sid: (n_users, 4) int64
    """
    if pred.dim() != 3 or pred.shape[2] != 4:
        raise ValueError(f"bad pred shape {tuple(pred.shape)}")
    pred = pred.long()
    target_sid = target_sid.long()

    match = (pred == target_sid.unsqueeze(1)).all(dim=2)
    first_match = match.float().argmax(dim=1)
    has_match = match.any(dim=1)
    n_candidates = pred.shape[1]
    first_match_1 = torch.where(has_match, first_match + 1,
                                 torch.full_like(first_match, n_candidates + 1))

    out = {}
    for k in top_k_list:
        if k > n_candidates:
            raise ValueError(f"k={k} > n_candidates={n_candidates}")
        rec_k = (first_match_1 <= k).float().numpy()
        out[f"Recall@{k}"] = rec_k
        rank_clipped = torch.minimum(first_match_1, torch.tensor(k, dtype=first_match_1.dtype))
        gains = (rank_clipped <= k).float() * (1.0 / torch.log2(rank_clipped.float() + 1.0))
        out[f"NDCG@{k}"] = gains.numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True, help='Variant name (only for logging)')
    p.add_argument('--pred_pt', required=True, help='Stage 4 merged_predictions_tensor.pt')
    p.add_argument('--sid', required=True, help='Stage 2.2 SID table')
    p.add_argument('--ckpt', default='', help='Optional ckpt path for trace')
    p.add_argument('--out_json', required=True)
    p.add_argument('--data_dir', default='/home/wlia0047/ar57/wenyu/GeneRec/GRID/data/amazon_data/toys')
    args = p.parse_args()

    print(f'[s4 eval {args.variant}] pred={args.pred_pt} sid={args.sid}', flush=True)

    pred = torch.load(args.pred_pt, map_location='cpu', weights_only=False).long()
    print(f'  pred shape: {tuple(pred.shape)}', flush=True)
    n_users, n_candidates, _ = pred.shape

    sid = torch.load(args.sid, map_location='cpu', weights_only=False).long()
    print(f'  sid shape: {tuple(sid.shape)}', flush=True)
    if sid.dim() != 2:
        raise ValueError(f'SID table must be 2D')
    if sid.shape[0] == 11924 and sid.shape[1] == 4:
        pass
    elif sid.shape[0] == 4 and sid.shape[1] == 11924:
        sid = sid.t().contiguous()
    else:
        raise ValueError(f'SID table wrong shape {tuple(sid.shape)}; expected (11924, 4) or (4, 11924)')
    print(f'  sid normalized: {tuple(sid.shape)}', flush=True)

    target_iid = build_uid_target_iid(args.data_dir, n_users)
    if (target_iid >= sid.shape[0]).any() or (target_iid < 0).any():
        raise ValueError('target_iid out of SID range')

    target_sid = sid[target_iid]  # (n_users, 4)
    metrics_per_user = per_user_metrics(pred, target_sid)
    mean_metrics = {k: float(v.mean()) for k, v in metrics_per_user.items()}

    result = {
        'variant': args.variant,
        'constrained_pt': args.pred_pt,
        'sid': args.sid,
        'ckpt': args.ckpt,
        'n_users_evaluated': n_users,
        'n_users_tensor': n_users,
        'n_users_missing': 0,
        **mean_metrics,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(mean_metrics, indent=2), flush=True)
    print(f'  saved → {args.out_json}', flush=True)


if __name__ == '__main__':
    main()
