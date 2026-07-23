#!/usr/bin/env python3
"""
Task #78 — sid_tiger_text.pt (24588, 3) → Instruments_tiger.index.json (4 tokens per item)

PAPER 标准 (GRID Section 3.3 + R5 硬约束):
- Stage 2 num_hierarchies=3 → SID tensor (N, 3)
- 推断后追加 1 列去重 digit → (N, 4), 解决 collision
- Stage 3/4 用 num_hierarchies=4

去重规则 (方案 B, R11.3 自主决策):
- 对 (N, 3) tensor 按第 1-3 列分组 (用 tuple 做 key)
- 每组内追加 0, 1, 2, ... 的索引作为第 4 层 (d)
- 不碰撞 item 直接置 d=0

输出格式: LETTER Instruments.index.json
{"<item_id_str>": ["<a_X>", "<b_Y>", "<c_Z>", "<d_W>"], ...}

Why (无 fallback): 严格 paper-aligned, 不允许用其他近似方式隐藏 collision
"""

import torch
import json
import sys
from collections import defaultdict
from pathlib import Path

INPUT_PT = "/home/wlia0047/ar57/wenyu/GeneRec/products/task73/sid_tiger_text.pt"
OUTPUT_JSON = "/home/wlia0047/ar57/wenyu/GeneRec/external/LETTER/LETTER-TIGER/Instruments_tiger.index.json"


def build_dedup_digit(sid_3layer: torch.Tensor) -> torch.Tensor:
    """对 (N, 3) tensor 逐组碰撞加 dedup digit (d), 返回 (N, 4) tensor."""
    assert sid_3layer.dim() == 2 and sid_3layer.shape[1] == 3, \
        f"expected (N, 3), got {tuple(sid_3layer.shape)}"

    n = sid_3layer.shape[0]
    sid_4layer = torch.zeros((n, 4), dtype=sid_3layer.dtype)
    sid_4layer[:, :3] = sid_3layer[:, :3]

    # 用 groupby 按 (a, b, c) tuple 分组, 给每组加 0, 1, 2, ...
    groups = defaultdict(list)
    for i, row in enumerate(sid_3layer.tolist()):
        groups[tuple(row)].append(i)

    for key, idxs in groups.items():
        if len(idxs) == 1:
            continue  # d=0 (已默认)
        for d, idx in enumerate(idxs):
            sid_4layer[idx, 3] = d

    return sid_4layer


def main():
    if not Path(INPUT_PT).exists():
        raise FileNotFoundError(
            f"sid_tiger_text.pt 不存在: {INPUT_PT}\n"
            "→ 必须先跑 Task #73 Stage 2.2 RQ-VAE 推断生成 (24588, 3) SID tensor"
        )

    sid_3 = torch.load(INPUT_PT, map_location="cpu", weights_only=False)

    print(f"[INFO] Input sid_tiger_text.pt shape: {tuple(sid_3.shape)}, dtype: {sid_3.dtype}")
    print(f"[INFO] SID 范围: min={sid_3.min().item()}, max={sid_3.max().item()}")

    sid_4 = build_dedup_digit(sid_3)
    print(f"[INFO] Output sid_4 shape: {tuple(sid_4.shape)}")

    # 验证 collision rate 0% — 每个 4-tuple 唯一
    unique_4 = len(set(tuple(row.tolist()) for row in sid_4))
    n = sid_4.shape[0]
    if unique_4 != n:
        raise AssertionError(
            f"去重后 4-tuple 仍有碰撞: {n - unique_4} 个 (expected 0 collisions)"
        )
    print(f"[CHECK] 4-tuple 唯一性: PASS (unique={unique_4}/{n})")

    # 验证 collision rate 改善 — 计算原 3-tuple 碰撞数
    unique_3 = len(set(tuple(row.tolist()) for row in sid_3))
    n_collisions_3 = n - unique_3
    print(f"[INFO] 3-tuple collisions: {n_collisions_3} ({100*n_collisions_3/n:.2f}%)")
    print(f"[INFO] 4-tuple collisions: 0 (dedup digit 解决)")

    # 转 LETTER JSON 格式
    out = {}
    for i, row in enumerate(sid_4.tolist()):
        a, b, c, d = row
        out[str(i)] = [f"<a_{a}>", f"<b_{b}>", f"<c_{c}>", f"<d_{d}>"]

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f)

    print(f"[OUTPUT] {OUTPUT_JSON}")
    print(f"[OUTPUT] {n} items × 4 tokens written")

    # 抽样打印前 5 个
    print("\n[SAMPLE] 前 5 项:")
    for k in list(out.keys())[:5]:
        print(f"  {k!r}: {out[k]}")


if __name__ == "__main__":
    main()
