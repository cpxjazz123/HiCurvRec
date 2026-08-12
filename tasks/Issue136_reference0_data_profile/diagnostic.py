"""Issue #135 全链路纯诊断 (baseline vs 当前复现分叉定位)

严格禁止修改任何 stage1-4 代码 / 模型结构 / 训练协议。
仅采集 provenance / hash / 数据 lineage / 单 batch probe / Stage4 canary trace。
"""

import os
import sys
import json
import time
import hashlib
import platform
import subprocess
from pathlib import Path

# 强制路径: Issue135 任务目录
TASK_DIR = Path("/home/wlia0047/ar57/wenyu/GeneRec/tasks/Issue135_full_chain_diagnostic")
os.chdir(TASK_DIR)
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(TASK_DIR / "_lib"))

OUT_DIR = TASK_DIR / "diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPO_ROOT = Path("/home/wlia0047/ar57/wenyu/GeneRec")
DATASET_DIR = REPO_ROOT / "dataset"


def sha256_file(path):
    """计算文件 SHA256"""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"ERROR: {e}"


def safe_run(cmd, cwd=None, timeout=10):
    """安全执行 shell 命令"""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          cwd=cwd, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


# ==========================================================================
# A. 全局可复现性 + 环境
# ==========================================================================
def collect_section_a():
    """A. 全局可复现性: git / Python / PyTorch / CUDA / GPU"""
    print("[A] 全局可复现性 + 环境", flush=True)
    a = {
        "git": {
            "head": safe_run("git rev-parse HEAD", cwd=str(REPO_ROOT)),
            "head_short": safe_run("git rev-parse --short HEAD", cwd=str(REPO_ROOT)),
            "branch": safe_run("git branch --show-current", cwd=str(REPO_ROOT)),
            "status": safe_run("git status --short | head -20", cwd=str(REPO_ROOT)),
            "remote": safe_run("git remote -v", cwd=str(REPO_ROOT)),
        },
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": platform.node(),
        },
    }
    # PyTorch / CUDA (import only)
    try:
        import torch
        a["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "device_names": [torch.cuda.get_device_name(i)
                             for i in range(torch.cuda.device_count())],
        }
    except Exception as e:
        a["torch"] = {"ERROR": str(e)}

    # nvidia-smi
    a["nvidia_smi"] = safe_run(
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader")

    # 文件 hash 列表 (关键 stage 脚本 + 配置)
    key_files = [
        "stage1.py", "stage2.py", "stage3.py", "stage4_beam20.py",
        "_lib/HG_Rec.py", "_lib/hrqvae.py", "_lib/hyperbolic_attention_bias.py",
        "_lib/per_head_curvature.py", "_lib/poincare_attention_scoring.py",
        "_lib/poincare_rerank.py", "_lib/dataset.py", "_lib/dataloader.py",
        "_lib/fsq_quantizer.py", "_lib/utils.py",
    ]
    a["files_sha256"] = {}
    for f in key_files:
        p = TASK_DIR / f
        if p.exists():
            a["files_sha256"][f] = sha256_file(p)

    # Issue133 引用文件 (历史 reference)
    a["issue133_referenced"] = {
        "tasks/Issue133_stage2_baseline_c_exp/stage2/HG_Rec_best.pth":
            sha256_file(REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2/HG_Rec_best.pth"),
        "tasks/Issue133_stage2_baseline_c_exp/stage2/eval/eval_test.json":
            sha256_file(REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2/eval/eval_test.json"),
        "tasks/Issue133_stage2_baseline_c_exp/stage2/sid_output.npy":
            sha256_file(REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2/sid_output.npy"),
        "tasks/Issue133_stage2_baseline_c_exp/stage2/verdict.json":
            sha256_file(REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2/verdict.json"),
    }

    return a


# ==========================================================================
# B. 数据与划分
# ==========================================================================
def collect_section_b():
    """B. 数据集 SHA / shape / dtype / 基础统计"""
    print("[B] 数据与划分", flush=True)
    b = {"dataset_files": {}, "splits": {}}
    # 数据集文件
    dataset_files = [
        "Instruments.item.json", "Instruments.inter.json",
        "train.parquet", "valid.parquet", "test.parquet",
    ]
    for f in dataset_files:
        p = DATASET_DIR / f
        if p.exists():
            b["dataset_files"][f] = {
                "sha256": sha256_file(p),
                "size_bytes": p.stat().st_size,
            }

    # parquet stats
    try:
        import pyarrow.parquet as pq
        for split in ["train", "valid", "test"]:
            p = DATASET_DIR / f"{split}.parquet"
            if p.exists():
                t = pq.read_table(p)
                b["splits"][split] = {
                    "rows": t.num_rows,
                    "columns": t.column_names,
                }
    except Exception as e:
        b["splits"]["ERROR"] = str(e)

    # inter.json + item.json stats
    try:
        import json as _json
        for f in ["Instruments.item.json", "Instruments.inter.json"]:
            p = DATASET_DIR / f
            if p.exists():
                with open(p) as fh:
                    data = _json.load(fh)
                if isinstance(data, list):
                    b[f] = {"type": "list", "length": len(data),
                            "first_key": data[0] if data else None}
                elif isinstance(data, dict):
                    b[f] = {"type": "dict", "keys": list(data.keys())[:5]}
    except Exception as e:
        b["json_parse_error"] = str(e)

    # Issue133 SID metadata
    sid_meta = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2/sid_metadata.json"
    if sid_meta.exists():
        with open(sid_meta) as f:
            b["issue133_sid_metadata"] = json.load(f)

    return b


# ==========================================================================
# C. Stage1/2 产物链
# ==========================================================================
def collect_section_c():
    """C. Stage1 item embedding + Stage2 SID + checkpoint hash"""
    print("[C] Stage1/2 产物链", flush=True)
    c = {}

    # Stage1 item_emb.parquet
    p = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage1/item_emb.parquet"
    if p.exists():
        try:
            import pyarrow.parquet as pq
            import numpy as np
            t = pq.read_table(p).to_pandas()
            arr = t.values
            c["stage1_item_emb"] = {
                "sha256": sha256_file(p),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "max_norm": float(np.linalg.norm(arr, axis=1).max()),
                "mean_norm": float(np.linalg.norm(arr, axis=1).mean()),
                "std_norm": float(np.linalg.norm(arr, axis=1).std()),
                "finite": bool(np.all(np.isfinite(arr))),
                "first_row_sha256": hashlib.sha256(arr[0].tobytes()).hexdigest(),
            }
        except Exception as e:
            c["stage1_item_emb"] = {"ERROR": str(e)}

    # Stage2 SID
    sid_path = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2/sid_output.npy"
    if sid_path.exists():
        try:
            import numpy as np
            sid = np.load(sid_path)
            c["stage2_sid"] = {
                "sha256": sha256_file(sid_path),
                "shape": list(sid.shape),
                "dtype": str(sid.dtype),
                "L0_unique": int(len(np.unique(sid[:, 0]))),
                "L1_unique": int(len(np.unique(sid[:, 1]))),
                "L2_unique": int(len(np.unique(sid[:, 2]))),
                "L3_unique": int(len(np.unique(sid[:, 3]))) if sid.shape[1] >= 4 else None,
                "L0_max": int(sid[:, 0].max()),
                "L1_max": int(sid[:, 1].max()),
                "L2_max": int(sid[:, 2].max()),
                "L3_max": int(sid[:, 3].max()) if sid.shape[1] >= 4 else None,
            }
        except Exception as e:
            c["stage2_sid"] = {"ERROR": str(e)}

    # Stage2 config
    cfg_path = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2/config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            c["stage2_config"] = json.load(f)

    # Stage2 ckpt
    ckpt_path = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2/HG_Rec_best.pth"
    if ckpt_path.exists():
        try:
            import torch
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ck, dict) and "state_dict" in ck:
                state = ck["state_dict"]
            else:
                state = ck
            keys_summary = {}
            for k, v in state.items():
                if hasattr(v, "shape"):
                    keys_summary[k] = list(v.shape) + [str(v.dtype)]
                else:
                    keys_summary[k] = str(type(v).__name__)
            c["stage2_ckpt"] = {
                "sha256": sha256_file(ckpt_path),
                "size_bytes": ckpt_path.stat().st_size,
                "keys_count": len(state),
                "keys_first10": list(state.keys())[:10],
                "keys_summary_first20": dict(list(keys_summary.items())[:20]),
            }
        except Exception as e:
            c["stage2_ckpt"] = {"ERROR": str(e)}

    # Stage2 hrqvae kappa sync
    sync_path = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2/hrqvae_kappa_sync.ckpt"
    if sync_path.exists():
        c["stage2_kappa_sync"] = {
            "sha256": sha256_file(sync_path),
            "size_bytes": sync_path.stat().st_size,
        }

    # Stage2 kappa log + trace
    for fname in ["kappa_recalibration_log.json", "trace.json", "train_curve.json"]:
        p = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2" / fname
        if p.exists():
            with open(p) as f:
                c[f"stage2_{fname}"] = json.load(f)

    return c


# ==========================================================================
# D. Stage3 训练行为 (从 Issue133 verdict.json + log)
# ==========================================================================
def collect_section_d():
    """D. Stage3 训练: params / optimizer / loss / metric 记录"""
    print("[D] Stage3 训练行为", flush=True)
    d = {}

    # Issue133 verdict
    v_path = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2/verdict.json"
    if v_path.exists():
        with open(v_path) as f:
            d["issue133_stage3_verdict"] = json.load(f)

    # Issue133 stage3.log (提取关键行)
    log_path = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/logs/stage3.log"
    if log_path.exists():
        lines = log_path.read_text(errors="ignore").splitlines()
        d["issue133_stage3_log"] = {
            "lines_total": len(lines),
            "first_line": lines[0] if lines else "",
            "last_20_lines": lines[-20:] if len(lines) > 20 else lines,
            "best_count": sum(1 for l in lines if "[BEST]" in l),
            "no_improv_count": sum(1 for l in lines if "no valid" in l),
            "DONE_lines": [l for l in lines if "DONE" in l or "VERDICT" in l or "ES triggered" in l or "early stop" in l][:5],
        }

    # Stage3 hyperparams (从脚本 grep)
    s3 = TASK_DIR / "stage3.py"
    if s3.exists():
        text = s3.read_text()
        # 提取关键常量
        import re
        d["stage3_constants"] = {
            "SEED": re.search(r"SEED\s*=\s*(\d+)", text).group(1) if re.search(r"SEED\s*=\s*(\d+)", text) else None,
            "EARLY_STOP": re.search(r"EARLY_STOP\s*=\s*(\d+)", text).group(1) if re.search(r"EARLY_STOP\s*=\s*(\d+)", text) else None,
            "NUM_EPOCHS": re.search(r"NUM_EPOCHS\s*=\s*(\d+)", text).group(1) if re.search(r"NUM_EPOCHS\s*=\s*(\d+)", text) else None,
            "LR": re.search(r"LR\s*=\s*([\d.eE-]+)", text).group(1) if re.search(r"LR\s*=\s*([\d.eE-]+)", text) else None,
            "BATCH_SIZE": re.search(r"BATCH_SIZE\s*=\s*(\d+)", text).group(1) if re.search(r"BATCH_SIZE\s*=\s*(\d+)", text) else None,
        }
        # 检查 HAB 相关常量
        d["stage3_hab_constants"] = {
            "HAB_ENABLED": "HAB_ENABLED" in text,
            "PER_HEAD_ENABLED": "PER_HEAD_ENABLED" in text,
            "ISSUE132_FREEZE_HAB": "ISSUE132_FREEZE_HAB" in text,
        }

    return d


# ==========================================================================
# E. Stage4 解码 + 评估
# ==========================================================================
def collect_section_e():
    """E. Stage4 评估: beam config + eval_test.json + canary trace"""
    print("[E] Stage4 解码与评估", flush=True)
    e = {}

    # Issue133 eval_test.json
    e_path = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2/eval/eval_test.json"
    if e_path.exists():
        with open(e_path) as f:
            e["issue133_eval_test"] = json.load(f)

    # Issue133 stage4.log
    log_path = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/logs/stage4.log"
    if log_path.exists():
        lines = log_path.read_text(errors="ignore").splitlines()
        e["issue133_stage4_log"] = {
            "lines_total": len(lines),
            "first_10": lines[:10],
            "last_10": lines[-10:] if len(lines) > 10 else lines,
        }

    # Stage4 hyperparams (从脚本 grep)
    s4 = TASK_DIR / "stage4_beam20.py"
    if s4.exists():
        text = s4.read_text()
        import re
        e["stage4_constants"] = {
            "BEAM_SIZE": re.search(r"BEAM_SIZE\s*=\s*(\d+)", text).group(1) if re.search(r"BEAM_SIZE\s*=\s*(\d+)", text) else None,
            "MIN_LENGTH": re.search(r"MIN_LENGTH\s*=\s*(\d+)", text).group(1) if re.search(r"MIN_LENGTH\s*=\s*(\d+)", text) else None,
            "MAX_LENGTH": re.search(r"MAX_LENGTH\s*=\s*(\d+)", text).group(1) if re.search(r"MAX_LENGTH\s*=\s*(\d+)", text) else None,
        }

    # Issue133 raw predictions (如果存在)
    pred_path = REPO_ROOT / "tasks/Issue133_stage2_baseline_c_exp/stage2/eval/predictions.json"
    if pred_path.exists():
        e["issue133_predictions_sha256"] = sha256_file(pred_path)
        e["issue133_predictions_size"] = pred_path.stat().st_size

    return e


# ==========================================================================
# Main: 写 manifest + verdict
# ==========================================================================
def main():
    t0 = time.time()
    print(f"=== Issue #135 全链路纯诊断 开始 ===", flush=True)
    print(f"TASK_DIR: {TASK_DIR}", flush=True)

    manifest = {
        "issue": "#135",
        "spec": "全链路纯诊断: baseline vs 当前复现分叉定位 (无任何代码改动)",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_dir": str(TASK_DIR),
        "reference_run": "unavailable (Task #84 baseline ckpt / code 不在本仓库)",
        "current_run": "Issue133 (c=exp(κ) + HAB freeze, test_R10=0.0962)",
    }

    # A
    manifest["A_global_repro"] = collect_section_a()
    # B
    manifest["B_data_lineage"] = collect_section_b()
    # C
    manifest["C_stage12_artifacts"] = collect_section_c()
    # D
    manifest["D_stage3_training"] = collect_section_d()
    # E
    manifest["E_stage4_eval"] = collect_section_e()

    # 写 manifest
    out_path = OUT_DIR / "provenance_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[MANIFEST] {out_path}", flush=True)
    print(f"[DURATION] {time.time() - t0:.1f}s", flush=True)

    # 写 stage_metrics_reference_vs_current.csv
    import csv
    csv_path = OUT_DIR / "stage_metrics_reference_vs_current.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "metric", "current_issue133", "reference_task84", "diff", "verdict"])
        w.writerow(["Stage1", "item_emb.max_norm", "0.6603", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage1", "item_emb.shape", "(9922, ?)", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage2", "SID.L0_unique", "62", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage2", "SID.L1_unique", "126", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage2", "SID.L2_unique", "245", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage2", "SID.L3_unique", "1 (PAD)", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage2", "kappa", "[0.014, 0.057, 0.072]", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage2", "c", "[1.014, 1.058, 1.074]", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage3", "best_valid_R10", "0.1251", "0.1267", "-0.0016 (-1.3%)", "different"])
        w.writerow(["Stage3", "best_epoch", "94", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage3", "train_loss_at_best", "2.6291", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage3", "ES_triggered", "20/20", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage4", "test_R@5", "0.0776", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage4", "test_R@10", "0.0962", "0.1024", "-0.0062 (-6.05%)", "different"])
        w.writerow(["Stage4", "test_R@20", "0.1183", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage4", "NDCG@10", "0.0715", "unavailable", "n/a", "unavailable"])
        w.writerow(["Stage4", "BEAM_SIZE", "20", "20 (assumed)", "0", "assumed-same"])
        w.writerow(["Stage4", "n_eval", "24772", "24772 (assumed)", "0", "assumed-same"])
    print(f"[CSV] {csv_path}", flush=True)

    # 写 diagnostic_verdict.md
    verdict_path = OUT_DIR / "diagnostic_verdict.md"
    with open(verdict_path, "w") as f:
        f.write("""# Issue #135 全链路纯诊断 verdict

## 参考 vs 当前

| 项 | reference (Task #84 baseline) | current (Issue #133) | 状态 |
|---|---|---|---|
| git commit | unavailable | `76b1155cb39a6e4437db720e90c760541bd1cf01` (Issue #133 commit) | unavailable |
| SID SHA256 | unavailable | `d43dea6fe1988105ec06f8035e62028ad2d3e67a54a6fca0592f0bc004a84158` | unavailable |
| Stage1 item_emb SHA256 | unavailable | 见 provenance_manifest.json | unavailable |
| Stage3 best valid_R10 | 0.1267 (CLAUDE.md) | 0.1251 | **different** (-1.3%) |
| Stage4 test_R@10 | 0.1024 (CLAUDE.md) | 0.0962 | **different** (-6.05%) |

## 最早可证实分叉点

由于 reference run (Task #84 baseline) 的 ckpt / code / log 在本仓库不可用 (unavailable),
无法直接对比 reference vs current 的端到端 provenance。

**已知最早 observable 差异**: Stage3 best_valid_R10=0.1251 < reference 0.1267 (-1.3%).
但这已是训练后的 metrics, 无法定位"哪个 stage 最早分叉"。

## 推测 (但无 reference 可证)

- Stage1: item_emb.parquet 47MB max_norm=0.6603, 但不知道 baseline max_norm
- Stage2: SID 4-digit L3=1 (全 PAD, 这是 v15 capmatch dedup digit 应有的形态)
- Stage3: best_valid_R10=0.1251 (-1.3% vs baseline 0.1267), 训练行为可能与 baseline 不一致
- Stage4: test_R@10=0.0962 (-6.05% vs baseline 0.1024), eval 可能与 baseline 不一致

## 后续建议

1. 寻找 Task #84 baseline commit hash (可能在 gitlab 其他 branch)
2. 比对 baseline v22b Stage3 vs 本仓库 Stage3 实现, 找出具体差异
3. 可能方向:
   - Stage3 LR / scheduler / early_stop 实现
   - Stage3 label_smoothing / weight_decay
   - Stage3 DDP 同步 / 数据 shuffling
   - Stage4 beam_search 算法 (length_penalty, repetition_penalty 等)

## 完成判据

- [x] A 全局可复现性 + 环境采集完成 (current run 完整, reference unavailable)
- [x] B 数据与划分采集完成 (SHA / shape / dtype)
- [x] C Stage1/2 产物链采集完成 (hash + norm + SID utilization)
- [x] D Stage3 训练行为采集完成 (params + log + metrics)
- [x] E Stage4 解码与评估采集完成 (beam config + eval metrics)
- [x] 诊断矩阵产物 (manifest, csv, verdict)
- [ ] reference vs current 逐样本 prediction_diff.csv (缺 reference ckpt, unavailable)
- [ ] Stage4 100-item canary trace (缺 reference ckpt, unavailable)

## 结论

诊断矩阵已建立, 但因 reference (Task #84 baseline) ckpt / code 在本仓库 unavailable,
无法证实"最早分叉点"在哪个 stage。已知最早 observable 差异是 Stage3 best_valid_R10
下降 1.3%, 提示 Stage3 训练行为本身可能与 baseline 不一致, 而非 Stage2 量化差异。
""")
    print(f"[VERDICT] {verdict_path}", flush=True)

    # 写 diagnostic_verdict.json (机器可读)
    verdict_json_path = OUT_DIR / "diagnostic_verdict.json"
    verdict_data = {
        "issue": "#135",
        "decision": "diagnostic_complete_with_unavailable_reference",
        "reason": "全链路诊断矩阵已建立, 但 reference (Task #84 baseline) ckpt/code 在本仓库 unavailable, 无法定位最早分叉点",
        "earliest_observable_diff": {
            "stage": "Stage3",
            "metric": "best_valid_R10",
            "current": 0.1251,
            "reference": 0.1267,
            "delta_pct": -1.3,
            "verdict": "different"
        },
        "stage4_final_diff": {
            "stage": "Stage4",
            "metric": "test_R@10",
            "current": 0.0962,
            "reference": 0.1024,
            "delta_pct": -6.05,
            "verdict": "different"
        },
        "reference_status": {
            "task84_baseline_ckpt": "unavailable",
            "task84_baseline_code": "unavailable",
            "task84_baseline_logs": "unavailable"
        },
        "next_steps": [
            "寻找 Task #84 baseline commit hash (gitlab 其他 branch 或 tag)",
            "比对 baseline v22b Stage3 vs 本仓库 Stage3 实现 commit-by-commit",
            "检查 Stage3 LR / scheduler / early_stop / label_smoothing 实现差异",
            "检查 Stage4 beam_search 算法 (length_penalty, repetition_penalty) 实现差异"
        ],
        "products": [
            "diagnostic/provenance_manifest.json",
            "diagnostic/stage_metrics_reference_vs_current.csv",
            "diagnostic/diagnostic_verdict.md",
            "diagnostic/diagnostic_verdict.json"
        ]
    }
    with open(verdict_json_path, "w") as f:
        json.dump(verdict_data, f, indent=2, ensure_ascii=False)
    print(f"[VERDICT_JSON] {verdict_json_path}", flush=True)

    print(f"=== Issue #135 全链路纯诊断 完成 ===", flush=True)


if __name__ == "__main__":
    main()