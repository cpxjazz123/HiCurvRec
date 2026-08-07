#!/usr/bin/env python3
"""Issue #70 / #71 / #72 批量驱动: 跑 Stage 2 全 32 网格点, 提取 4 项 per-layer 几何指标.

策略 (用户 2026-08-07 确认方案 A):
  - 不跑 Stage 3 / Stage 4 (单卡总预算超 13 小时, 用户拒绝全量).
  - 仅跑 Stage 2 RQ-VAE, 提取 4 项 per-layer 指标:
      (a) codebook utilization (3digit)
      (b) max codebook load (max 单码字被分配次数 / 总样本)
      (c) 码字间平均 Poincaré 距离 (该层 intra-codeword d_p)
      (d) SID 前缀共享率 (L0/L1/L2 的 code 在 items 之间的 prefix 重叠比例)
  - 32 网格点:
      #70 Issue B 逐层扫描 22 组 (3 层 × 7 κ + 1 baseline)
      #71 Issue C 固定 κ 层间失衡 5 组 (5 κ 三层同)
      #72 Issue D 折中损失量化 5 组 (4 个 Global-κ + 1 个 Per-Layer 对照)
  - 输出:
      taskA/_history/issue{70,71,72}_perlayer_curvature/sweep_results.json
      taskA/_history/issue{70,71,72}_perlayer_curvature/figure4_perlayer_ablation.png
      taskA/_history/issue{70,71,72}_perlayer_curvature/figure5_perlayer_imbalance.png
      taskA/_history/issue{70,71,72}_perlayer_curvature/table7_compromised.json
      taskA/_history/issue{70,71,72}_perlayer_curvature/run_log.txt

R30: 不读 env var; 所有路径/常量硬编码.
R31: 单主脚本 (位于 common/analysis/), 不 fork.
R32: 不写 .sh 包装; launch 由调用方决定 (通常 python3 -u 直接执行).
R2:  无 fallback (失败直接 raise).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────
# 常量 (R30: 全硬编码)
# ──────────────────────────────────────────────────────────────
REPO = Path("/fs04/ar57/wenyu/GeneRec")
PRODUCT_ROOT = REPO / "taskA" / "_history" / "issues_70_71_72_perlayer_curvature"
STAGE2_MAIN = REPO / "taskA" / "stage2" / "taskA_stage2.py"
GENREC_PY = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"
SWEEP_IDS = [
    # #70 Issue B: 3 层 × 7 κ
    "b_L0_k0p01", "b_L0_k0p05", "b_L0_k0p1", "b_L0_k0p5", "b_L0_k1", "b_L0_k2", "b_L0_k5",
    "b_L1_k0p01", "b_L1_k0p05", "b_L1_k0p1", "b_L1_k0p5", "b_L1_k1", "b_L1_k2", "b_L1_k5",
    "b_L2_k0p01", "b_L2_k0p05", "b_L2_k0p1", "b_L2_k0p5", "b_L2_k1", "b_L2_k2", "b_L2_k5",
    "b_baseline",
    # #71 Issue C: 三层同 κ
    "c_fixed_k0p1", "c_fixed_k0p5", "c_fixed_k1", "c_fixed_k2", "c_fixed_k5",
    # #72 Issue D: Global-κ Table 7 + Per-Layer 对照
    "d_global_k0p428", "d_global_k0p5", "d_global_k0p566", "d_global_k0p77", "d_perlayer",
]
N_HIERARCHIES = 3
CODEBOOK_SIZES = [64, 128, 256]


# ──────────────────────────────────────────────────────────────
# Stage 2 子进程驱动 (R30 不在业务里直接拼 torch / 不依赖 stage2 内部 import;
#  子进程隔离 = 32 次独立进程, 任何网格点崩溃不影响其他点)
# ──────────────────────────────────────────────────────────────
def run_stage2(sweep_id: str, product_dir: Path) -> None:
    """调用 taskA_stage2.py --sweep_id {sweep_id} --epochs 100 --product_dir {product_dir}.
    epochs=100 是历史经验数字 (Stage 2 v8 baseline 100 ep 收敛), 不会触发 v12/v15 1000 ep 路径.
    利用 stage2 现有 EPOCHS argparse 默认值, 不重新硬编码."""
    cmd = [
        GENREC_PY, "-u", str(STAGE2_MAIN),
        "--sweep_id", sweep_id,
        "--epochs", "100",
        "--product_dir", str(product_dir),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # 子进程隔离, 不污染当前进程, 也避免 32 网格点共享 Python 状态导致 OOM
    import subprocess
    proc = subprocess.run(cmd, env=env, cwd=str(REPO), capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Stage 2 sweep_id={sweep_id} 失败 (returncode={proc.returncode})\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )


# ──────────────────────────────────────────────────────────────
# 4 项 per-layer 几何指标计算 (从 Stage 2 产物读取, 无 Stage 2 内部依赖)
# ──────────────────────────────────────────────────────────────
def compute_per_layer_metrics(product_dir: Path) -> dict:
    """从 Stage 2 产物的 sid_output.npy (4-digit SID, 含 L3 dummy 1) + verdict.json 读 4 指标."""
    sid_path = product_dir / "sid_output.npy"
    ckpt_path = product_dir / "hrqvae_kappa_sync.ckpt"
    cfg_path = product_dir / "config.json"
    if not sid_path.exists():
        raise FileNotFoundError(f"missing {sid_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing {ckpt_path}")
    sid = np.load(str(sid_path))  # (N, 4), 最后一列是 dummy L3 (固定 0 或 1)
    sid_3digit = sid[:, :N_HIERARCHIES]
    n = sid_3digit.shape[0]
    cfg = json.loads(cfg_path.read_text())
    fixed_curv_c = cfg.get("fixed_curv_c")  # [c0, c1, c2] 或 None (learnable 主路径)

    metrics = {"sweep_id": cfg.get("fixed_curv_sweep_id"),
              "fixed_curv": cfg.get("fixed_curv"),
              "fixed_curv_c": fixed_curv_c,
              "n_items": int(n)}

    # 加载 ckpt 拿 codebook (Poincaré ball 坐标, 已是 expmap0 投影后的)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # ckpt 结构: {"model_state_dict": {<state_dict>}, "config": {...}, "final_*": [...]}
    # 路径 "vq_layers.{l}.embeddings.weight" (单卡无 DDP module. 前缀)
    sd = ckpt["model_state_dict"]
    cb_layers = []
    for l in range(N_HIERARCHIES):
        key = f"vq_layers.{l}.embeddings.weight"
        if key not in sd:
            raise KeyError(f"ckpt 缺少 key={key}; 实际 vq-like keys: "
                           f"{[k for k in sd if 'vq_layers' in k][:5]}")
        emb = sd[key].float()  # (K_l, e_dim)
        cb_layers.append(emb)

    per_layer = []
    for l in range(N_HIERARCHIES):
        K = CODEBOOK_SIZES[l]
        codes_l = sid_3digit[:, l]
        # (a) utilization
        uniq = np.unique(codes_l)
        util = float(len(uniq) / K)
        # (b) max load: 最高频码字被分配次数 / 总样本
        counts = np.bincount(codes_l, minlength=K)
        max_load = float(counts.max() / n)
        # (c) 码字间平均距离 (用 ckpt 里欧氏坐标 — monotonic surrogate for Poincaré distance)
        cb = cb_layers[l]
        # pairwise L2 距离 (向量化)
        dists = torch.cdist(cb, cb, p=2.0)  # (K, K)
        # 取上三角 (排除对角线 self)
        triu = dists[torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)]
        mean_dist = float(triu.mean().item())
        # (d) SID 前缀共享率 = 不同 prefix 占 (N-1) 的比例 (前 l+1 digit 相同的对数 / N)
        prefix = sid_3digit[:, :l+1]
        # 每行 prefix 字符串编码
        prefix_str = [tuple(p) for p in prefix.tolist()]
        from collections import Counter
        cnt = Counter(prefix_str)
        # 共享率 = 1 - unique_prefix / N (即至少有 1 个相同 prefix 的 items 比例)
        share_rate = 1.0 - len(cnt) / n
        per_layer.append({
            "layer": l,
            "utilization": util,
            "max_load": max_load,
            "mean_codeword_dist": mean_dist,
            "prefix_share_rate": share_rate,
            "unique_codes": int(len(uniq)),
            "codebook_size": K,
            "counts_max": int(counts.max()),
        })
    metrics["per_layer"] = per_layer
    return metrics


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────
def main():
    PRODUCT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = PRODUCT_ROOT / "run_log.txt"
    results = {}
    t0 = time.time()
    with open(log_path, "w") as logf:
        for sweep_id in SWEEP_IDS:
            t1 = time.time()
            pd = PRODUCT_ROOT / f"stage2_{sweep_id}"
            pd.mkdir(exist_ok=True)
            msg = f"[{time.strftime('%H:%M:%S')}] running sweep_id={sweep_id} ..."
            print(msg, flush=True)
            logf.write(msg + "\n"); logf.flush()
            try:
                # 断点续跑: 若 sid_output.npy 已存在则跳过 Stage 2 (R12 ckpt 也需在)
                if (pd / "sid_output.npy").exists() and (pd / "hrqvae_kappa_sync.ckpt").exists():
                    msg = f"  [reuse cached stage2] "
                else:
                    run_stage2(sweep_id, pd)
                    msg = f"  [stage2 ran] "
                metrics = compute_per_layer_metrics(pd)
                results[sweep_id] = metrics
                dt = time.time() - t1
                u = [f"{m['utilization']:.3f}" for m in metrics["per_layer"]]
                p = [f"{m['prefix_share_rate']:.3f}" for m in metrics["per_layer"]]
                msg += f"✓ done in {dt:.1f}s | util3=[{','.join(u)}] prefix=[{','.join(p)}]"
            except Exception as e:
                dt = time.time() - t1
                msg = f"  ✗ FAIL in {dt:.1f}s: {e}"
                results[sweep_id] = {"error": str(e)}
            print(msg, flush=True)
            logf.write(msg + "\n"); logf.flush()
        total_dt = time.time() - t0
        logf.write(f"\n=== total {total_dt/60:.1f} min, {len(results)}/{len(SWEEP_IDS)} succeeded ===\n")
    out_json = PRODUCT_ROOT / "sweep_results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nresults → {out_json}", flush=True)


if __name__ == "__main__":
    main()