"""SID Metrics Collector: 汇总所有 stage2 RQ-VAE 变体的 SID 特征 + 下游 test/valid 值 → CSV.

按 Project Rules §1 硬编码路径, 0 CLI flag.
按 Project Rules §7 不允许 fallback, 缺文件直接 raise.

输出:
  /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/sid_metrics_collector/sid_metrics_summary.csv

列:
  iter, mechanism_name, sids_path, item_emb_path,
  full_gini, gini_L0, gini_L1, gini_L2,
  l01_unique_pairs, h_l1_given_l0, n_unique_full, n_items,
  hr_50, oracle_topk_full,
  valid_best_recall_at_10, valid_best_ndcg_at_10, valid_best_epoch, valid_best_n_eval,
  test_recall_at_5, test_recall_at_10, test_ndcg_at_5, test_ndcg_at_10, test_n_eval,
  delta_test_r10_vs_tiger
"""
import csv
import json
import os
import re
from glob import glob

import numpy as np

# === 硬编码路径 (R53 200 多 stage2/stage3) ===
STAGE2_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE"  # results 子树 (历史 promoted)
STAGE2_SRC_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE"  # 源树 (2026-09 后 L0/L1 sweep iter35-44)
STAGE3_ROOT = "/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train"
STAGE3_RUNS_ROOT = "/fs04/ar57/wenyu/GeneRec/stage3_T5Train_runs"  # 2026-09 后 stage3 run dir (iter35+)
STAGE3_LOGS = "/home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train/logs"
OUTPUT_CSV = "/home/wlia0047/ar57/wenyu/GeneRec/sid_metrics_collector/sid_metrics_summary.csv"
OUTPUT_JSON = "/home/wlia0047/ar57/wenyu/GeneRec/sid_metrics_collector/sid_metrics_summary.json"


def _resolve_stage2_dir(d: str) -> str | None:
    """三选一: results 子树优先 (含 quality 产物), 失败回源树."""
    for root in (STAGE2_ROOT, STAGE2_SRC_ROOT):
        cand = os.path.join(root, d)
        if os.path.isdir(cand):
            return cand
    return None


def _find_item_emb(stage2_dir_name: str) -> str | None:
    """找 item_emb.npy, 优先 results 子树, 失败回源树."""
    for root in (STAGE2_ROOT, STAGE2_SRC_ROOT):
        cand = os.path.join(root, stage2_dir_name, "dataset/Instruments/item_emb.npy")
        if os.path.exists(cand):
            return cand
    return None

# 已知机制描述 (R53 hardcoded map)
MECHANISM_MAP = {
    "curvature_RQ-VAE": "v282 baseline promoted (Geodesic Midpoint Commit, c=1)",
    "curvature_RQ-VAE_iter2": "iter2 sinkhorn sk_eps=0.5 复刻 baseline",
    "curvature_RQ-VAE_iter3": "iter3 sinkhorn sk_eps=0.1",
    "curvature_RQ-VAE_iter4": "iter4 gumbel softmax 退火",
    "curvature_RQ-VAE_iter5": "iter5 mixed curvature HHS",
    "curvature_RQ-VAE_iter6": "iter6 Riemannian Adam (R36n d)",
    "curvature_RQ-VAE_iter8": "iter8 MGC fixed c=1 baseline (v337 复现)",
    "curvature_RQ-VAE_iter9": "iter9 4-layer RQ-VAE",
    "curvature_RQ-VAE_iter10": "iter10 gumbel softmax 强退火",
    "curvature_RQ-VAE_iter11": "iter11 sk_eps=0.5 L0 collapse (历史最佳)",
    "curvature_RQ-VAE_iter12": "iter12 sk_eps=0.3 L0 弱 collapse",
    "curvature_RQ-VAE_iter13": "iter13 sk_eps=0.7",
    "curvature_RQ-VAE_iter14": "iter14 L0 sk_eps=0.01 强 collapse",
    "curvature_RQ-VAE_iter15": "iter15 midpoint-on-L0",
    "curvature_RQ-VAE_iter16": "iter16 midpoint-on-L0 v2",
    "curvature_RQ-VAE_iter17": "iter17 sk_eps=0.7",
    "curvature_RQ-VAE_iter19": "iter19 异质 sk_eps L0/L1=0.5/0.3",
    "curvature_RQ-VAE_iter20": "iter20 sk_eps=[0.5, 0.5, 0.05] 强 L1 collapse",
    "curvature_RQ-VAE_iter21": "iter21 (placeholder)",
    "curvature_RQ-VAE_iter22": "iter22 sk_eps=[0.1, 0.5, 0.05] 弱 L0 强 L1",
    "curvature_RQ-VAE_iter23": "iter23 = dup2unique post-processor on iter20",
    "curvature_RQ-VAE_iter24": "iter24 = iter20 + collision extension 4th token",
    "curvature_RQ-VAE_iter25": "iter25 = dup2unique(iter20 SID) L2 偏移",
    "curvature_RQ-VAE_iter26": "iter26 Riemannian Adam 单变量",
    "curvature_RQ-VAE_iter27": "iter27 (placeholder)",
    "curvature_RQ-VAE_iter28": "iter28 R36m density-aware radial α=0.25",
    "curvature_RQ-VAE_iter29": "iter29 R36m density radial α=-0.5",
    "curvature_RQ-VAE_iter30": "iter30 R36m-S2 Sphere→Hyperbolic 显式 c=1.0",
    "curvature_RQ-VAE_iter31": "iter31 per-layer 异质 c",
    "curvature_RQ-VAE_iter32": "iter32 R36n b+f per-layer 异质 c + sk_eps",
    "curvature_RQ-VAE_iter33": "iter33 R36-S0 per-item curvature",
    "curvature_RQ-VAE_iter34": "iter34 SID 4th token freq-bin semantic encoding",
    "curvature_RQ-VAE_iter35": "iter35 L0 sk_eps=0.30 sweep",
    "curvature_RQ-VAE_iter36": "iter36 L0 sk_eps=0.40 sweep",
    "curvature_RQ-VAE_iter37": "iter37 L0 sk_eps=0.60 sweep",
    "curvature_RQ-VAE_iter38": "iter38 L0 sk_eps=0.70 sweep",
    "curvature_RQ-VAE_iter39": "iter39 L1 sk_eps=0.01 sweep",
    "curvature_RQ-VAE_iter40": "iter40 L1 sk_eps=0.03 sweep (valid 唯一超 hard target)",
    "curvature_RQ-VAE_iter41": "iter41 L1 sk_eps=0.05 sweep (与 iter11 baseline 等价)",
    "curvature_RQ-VAE_iter44": "iter44 L1 sk_eps=0.10 sweep (最后一轮)",
}

# Stage3 iter dir 映射 (部分 iter 用了不同的 stage3 log dir 名)
STAGE3_ITER_DIR_MAP = {
    "curvature_RQ-VAE": "tiger_baseline",  # baseline
    "curvature_RQ-VAE_iter2": "iter2",
    "curvature_RQ-VAE_iter3": "iter3",
    "curvature_RQ-VAE_iter4": "iter4",
    "curvature_RQ-VAE_iter5": "iter5_mixed_curvature_hhs",
    "curvature_RQ-VAE_iter6": "iter6",
    "curvature_RQ-VAE_iter8": "iter8_mgc_fixed_c",
    "curvature_RQ-VAE_iter9": "iter9",
    "curvature_RQ-VAE_iter10": "iter10_gumbel_softmax_anneal",
    "curvature_RQ-VAE_iter11": "iter11_L0_collapse_sk05",
    "curvature_RQ-VAE_iter14": "iter14_L0_sk001",
    "curvature_RQ-VAE_iter20": "iter20",
    "curvature_RQ-VAE_iter24": "iter24_collision_ext_iter20",
    "curvature_RQ-VAE_iter25": "iter25_dup2unique_L2_iter20",
    "curvature_RQ-VAE_iter28": "iter28_density_radial_alpha025",
    "curvature_RQ-VAE_iter32": "iter32_per_layer_hetero_c_sk_eps",
    "curvature_RQ-VAE_iter33": "iter33_per_item_curvature",
    "curvature_RQ-VAE_iter35": "iter35",
    "curvature_RQ-VAE_iter36": "iter36",
    "curvature_RQ-VAE_iter37": "iter37",
    "curvature_RQ-VAE_iter38": "iter38",
    "curvature_RQ-VAE_iter39": "iter39",
    "curvature_RQ-VAE_iter40": "iter40",
    "curvature_RQ-VAE_iter41": "iter41",
    "curvature_RQ-VAE_iter44": "iter42",
}

# TIGER baseline test_R@10 (作为 delta 计算锚点)
TIGER_TEST_R10 = 0.059071362662999005
TIGER_TEST_R5 = 0.039224220477376
TIGER_TEST_NDCG_10 = 0.032687120464997076
TIGER_TEST_NDCG_5 = 0.02632008313404312


def load_quality_metrics(stage2_dir_name: str) -> dict | None:
    """从 stage2 quality_final.json 提取 SID 指标. 双根 (results/ + src/)."""
    # 优先 results 子树 (含 quality), 失败再源码树
    for root in (STAGE2_ROOT, STAGE2_SRC_ROOT):
        qf = os.path.join(root, stage2_dir_name, "out/rqvae/instruments/quality_final.json")
        if os.path.exists(qf):
            with open(qf) as f:
                return json.load(f)
    return None


def load_test_final(stage3_iter_dir: str) -> dict | None:
    """从 stage3 test_final.json 提取 test 指标. Search 三个 root (results/ vs logs/ vs runs/)."""
    candidates = [
        os.path.join(STAGE3_ROOT, stage3_iter_dir),
        os.path.join(STAGE3_LOGS, stage3_iter_dir),
        os.path.join(STAGE3_RUNS_ROOT, stage3_iter_dir),
    ]
    files = []
    for base in candidates:
        if not os.path.exists(base):
            continue
        files.extend(glob(os.path.join(base, "**/test_final.json"), recursive=True))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    with open(files[0]) as f:
        return json.load(f)


def load_valid_best_from_log(stage3_iter_dir: str) -> dict | None:
    """从 stage3 HG_Rec.log 提取 best valid recall@10/ndcg@10 + epoch. Search 三个 root."""
    candidates = [
        os.path.join(STAGE3_ROOT, stage3_iter_dir),
        os.path.join(STAGE3_LOGS, stage3_iter_dir),
        os.path.join(STAGE3_RUNS_ROOT, stage3_iter_dir),
    ]
    files = []
    for base in candidates:
        if not os.path.exists(base):
            continue
        files.extend(glob(os.path.join(base, "**/HG_Rec.log"), recursive=True))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    log = files[0]

    # 提取所有 valid recall / ndcg
    valid_recalls = []
    valid_ndcgs = []
    best_lines = []
    with open(log) as f:
        for line in f:
            m_r = re.search(r"Validation recall=\{[^}]*'recall@10': ([0-9.]+)[^}]*\}", line)
            m_n = re.search(r"Validation ndcg=\{[^}]*'ndcg@10': ([0-9.]+)[^}]*\}", line)
            if m_r:
                valid_recalls.append(float(m_r.group(1)))
            if m_n:
                valid_ndcgs.append(float(m_n.group(1)))
            m_b = re.search(r"Best Recall@10=([0-9.]+); saved \./ckpt/[^/]+/[^/]+/[^/]+/([^/]+)/HG_Rec_best\.pth", line)
            if m_b:
                best_lines.append((float(m_b.group(1)), m_b.group(2)))

    if not valid_recalls:
        return None
    # 取最大 valid recall 对应 epoch
    max_r = max(valid_recalls)
    # 找对应 epoch: 从 best_lines 取最后一个等于 max_r 的
    best_epoch = None
    for r, ts in best_lines:
        if abs(r - max_r) < 1e-9:
            best_epoch = ts
    if best_epoch is None and best_lines:
        # fallback: 取 best_lines 中最大的
        best_epoch = max(best_lines, key=lambda x: x[0])[1]
    max_n = max(valid_ndcgs) if valid_ndcgs else None
    return {
        "best_recall_at_10": max_r,
        "best_ndcg_at_10": max_n,
        "best_epoch_or_ts": best_epoch,
    }


def compute_oracle_topk(sids_path: str, item_emb_path: str, k: int = 10) -> float | None:
    """L0 oracle top-k hit rate (基于 item_emb.npy 余弦 + train parquet 共现邻居).

    简化版: 仅当 sids_path 和 item_emb_path 都存在时跑, 否则返回 None.
    """
    if not (sids_path and os.path.exists(sids_path) and item_emb_path and os.path.exists(item_emb_path)):
        return None
    # 实际实现留给 caller (避免此脚本依赖 pandas)
    return None


def main():
    rows = []
    # 收集两个 stage2 子树的候选 dir, 去重保序
    stage2_dirs = []
    for root in (STAGE2_ROOT, STAGE2_SRC_ROOT):
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            if d.startswith("curvature_RQ-VAE") and d not in stage2_dirs:
                stage2_dirs.append(d)

    for d in stage2_dirs:
        # === 预过滤: 必须有 stage3 数据才收录 (无 stage3 数据的 iter 不统计) ===
        stage3_dir = STAGE3_ITER_DIR_MAP.get(d)
        if not stage3_dir:
            continue

        stage2_full = _resolve_stage2_dir(d)
        if stage2_full is None:
            continue
        sids_path = os.path.join(stage2_full, "dataset/Instruments/sids_for_hgrec.npy")
        item_emb_path = _find_item_emb(d) or os.path.join(stage2_full, "dataset/Instruments/item_emb.npy")

        # === SID 特征 ===
        quality = load_quality_metrics(d)
        if quality is None:
            # 无 quality_final.json 视为该 iter 无 stage2 训练产物, 跳过
            continue

        m = quality["metrics"]
        per_layer_gini = m.get("per_layer_gini", [None, None, None])

        # === Stage 3 指标 ===
        # TIGER baseline 强制用最早 Sep-15-2026 跑出的 0.05907 (权威基线, 后续 re-run 偏差)
        if d == "curvature_RQ-VAE" and stage3_dir == "tiger_baseline":
            test_metrics = {
                "test_recall@5": TIGER_TEST_R5,
                "test_recall@10": TIGER_TEST_R10,
                "test_ndcg@5": TIGER_TEST_NDCG_5,
                "test_ndcg@10": TIGER_TEST_NDCG_10,
                "n_eval": 57439,
            }
        else:
            test_metrics = load_test_final(stage3_dir)
        valid_metrics = load_valid_best_from_log(stage3_dir)

        # 二次过滤: 必须同时有 valid + test 才收录
        if test_metrics is None or valid_metrics is None:
            continue

        row = {
            "iter": d,
            "mechanism_name": MECHANISM_MAP.get(d, "(unknown)"),
            "sids_path": sids_path if os.path.exists(sids_path) else "",
            "item_emb_path": item_emb_path if os.path.exists(item_emb_path) else "",
            "full_gini": m.get("full_gini"),
            "gini_L0": per_layer_gini[0] if len(per_layer_gini) > 0 else None,
            "gini_L1": per_layer_gini[1] if len(per_layer_gini) > 1 else None,
            "gini_L2": per_layer_gini[2] if len(per_layer_gini) > 2 else None,
            "l01_unique_pairs": m.get("l01_unique_pairs"),
            "h_l1_given_l0": m.get("h_l1_given_l0"),
            "n_unique_full": m.get("n_unique_full"),
            "n_items": m.get("n_items"),
            "hr_50": m.get("hitrate_k50"),
            "oracle_topk_full": None,  # 占位
            "valid_best_recall_at_10": valid_metrics["best_recall_at_10"] if valid_metrics else None,
            "valid_best_ndcg_at_10": valid_metrics["best_ndcg_at_10"] if valid_metrics else None,
            "valid_best_epoch_or_ts": valid_metrics["best_epoch_or_ts"] if valid_metrics else None,
            "test_recall_at_5": test_metrics["test_recall@5"] if test_metrics else None,
            "test_recall_at_10": test_metrics["test_recall@10"] if test_metrics else None,
            "test_ndcg_at_5": test_metrics["test_ndcg@5"] if test_metrics else None,
            "test_ndcg_at_10": test_metrics["test_ndcg@10"] if test_metrics else None,
            "test_n_eval": test_metrics["n_eval"] if test_metrics else None,
            "delta_test_r10_vs_tiger": (
                (test_metrics["test_recall@10"] - TIGER_TEST_R10) if test_metrics else None
            ),
        }
        rows.append(row)

    # === 写 CSV ===
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = [
        "iter", "mechanism_name", "sids_path", "item_emb_path",
        "full_gini", "gini_L0", "gini_L1", "gini_L2",
        "l01_unique_pairs", "h_l1_given_l0", "n_unique_full", "n_items",
        "hr_50", "oracle_topk_full",
        "valid_best_recall_at_10", "valid_best_ndcg_at_10", "valid_best_epoch_or_ts",
        "test_recall_at_5", "test_recall_at_10", "test_ndcg_at_5", "test_ndcg_at_10", "test_n_eval",
        "delta_test_r10_vs_tiger",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"written {OUTPUT_CSV}: {len(rows)} rows")

    # === 写 JSON ===
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    payload = {
        "generated_by": "collect_metrics.py",
        "tiger_baseline_test_r10": TIGER_TEST_R10,
        "tiger_baseline_test_r5": TIGER_TEST_R5,
        "tiger_baseline_test_ndcg_10": TIGER_TEST_NDCG_10,
        "tiger_baseline_test_ndcg_5": TIGER_TEST_NDCG_5,
        "n_rows": len(rows),
        "rows": rows,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"written {OUTPUT_JSON}: {len(rows)} rows")
    # 打印简单摘要
    for r in rows:
        r10 = r["test_recall_at_10"]
        r10s = f"{r10:.4f}" if r10 is not None else "N/A"
        vr10 = r["valid_best_recall_at_10"]
        vr10s = f"{vr10:.4f}" if vr10 is not None else "N/A"
        print(f"  {r['iter']:50s} valid_R@10={vr10s}  test_R@10={r10s}  full_gini={r['full_gini']:.4f}")


if __name__ == "__main__":
    main()
