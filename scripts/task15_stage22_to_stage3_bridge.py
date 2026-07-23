#!/usr/bin/env python
"""
task5_stage22_to_stage3_bridge.py — Task #65 Stage 2.2 → Stage 3 桥接脚本

功能:
    - 读取 Stage 2.2 推断产物 merged_predictions_tensor.pt (shape: (11924, 3) 或 (12288, 3))
    - 截取前 11924 行（catalog items）
    - 对 L3 code（col index 2）构造 dedup column:
        * unique items → 0
        * 重复项 → 1, 2, ... (one-hot within L3 group)
    - 拼接 → (11924, 4) → 转置为 (4, 11924)
    - 保存为 *_dedup.pt 给 Stage 3 使用 (num_hierarchies=4)

参考: scripts/task6_l4_sid_tensor.py
"""

import argparse
import sys
import torch
from pathlib import Path


def build_dedup_col(l3_codes: torch.Tensor) -> torch.Tensor:
    """对每个 unique L3 code，组内分配 0, 1, 2, ..."""
    n = l3_codes.shape[0]
    dedup_col = torch.zeros(n, dtype=torch.long)
    counters = {}
    for i in range(n):
        code = l3_codes[i].item()
        cnt = counters.get(code, 0)
        dedup_col[i] = cnt
        counters[code] = cnt + 1
    return dedup_col


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Stage 2.2 merged_predictions_tensor.pt 路径",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Stage 3 输入文件 (4, 11924) 保存路径",
    )
    parser.add_argument(
        "--n-catalog",
        type=int,
        default=11924,
        help="catalog item 数（Toys = 11924）",
    )
    args = parser.parse_args()

    print(f"[task65-bridge] input:  {args.input}")
    print(f"[task65-bridge] output: {args.output}")
    print(f"[task65-bridge] n_catalog: {args.n_catalog}")

    sid = torch.load(args.input, map_location="cpu", weights_only=False).long()
    print(f"[task65-bridge] loaded shape: {sid.shape}, dtype: {sid.dtype}")

    if sid.shape[0] < args.n_catalog:
        raise ValueError(
            f"input 第一维 {sid.shape[0]} < n_catalog {args.n_catalog}，"
            f"上游 SID tensor 数量不足，预期外失败 → 终止"
        )

    sid_catalog = sid[: args.n_catalog]  # (n_catalog, 3)
    print(f"[task65-bridge] sid_catalog shape: {sid_catalog.shape}")

    # 用 col 2 (L3 code) 做 dedup
    dedup_col = build_dedup_col(sid_catalog[:, 2])
    n_dedup = int((dedup_col > 0).sum().item())
    print(
        f"[task65-bridge] dedup_col: unique values = "
        f"{torch.unique(dedup_col).tolist()}, n_items_with_dedup>0 = {n_dedup}"
    )

    sid4 = torch.cat([sid_catalog, dedup_col.unsqueeze(1)], dim=1)  # (n_catalog, 4)
    sid4 = sid4.t().contiguous()  # (4, n_catalog)
    print(f"[task65-bridge] sid4 final shape: {sid4.shape}")

    for i in range(4):
        u = torch.unique(sid4[i])
        print(
            f"[task65-bridge] hierarchy {i}: "
            f"unique={len(u)}, range={u.min().item()}..{u.max().item()}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sid4, out_path)
    print(f"[task65-bridge] saved → {out_path}")


if __name__ == "__main__":
    main()