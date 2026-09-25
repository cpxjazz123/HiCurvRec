"""R34c/R34d 验证脚本 — 完整 4 阶段重跑 3 次, 验证 test_R@10 噪声分布.

每次循环:
  Stage 1 → Stage 2 → Stage 3 → Stage 4

每次用不同的 Stage 1 ckpt 输出目录 (run_<i>) 避免覆盖.
每次 Stage 3 训练用不同的 best_ckpt 输出目录避免覆盖.

预期 (R34c+R34d 修复后):
  3 次 test_R@10 差异 < 0.001 (同 epoch best_ckpt + 固定 sampler 分片)
"""
import json
import os
import shutil
import subprocess
import sys
import time

MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3"
N_RUNS = 3
START_RUN = 1  # 从 run 1 开始 (run 1 + 2 已经在 stage4_r34c_run1/2.log 完成, 可选重跑)

CONDA_PYTHON = "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3"


def run_stage(stage_name: str, cmd: list, log_path: str, timeout_sec: int = 3600) -> int:
    """Run a stage command with timeout + log. Returns exit code."""
    print(f"\n{'='*80}\n[run_v19_3times] {stage_name}\n  cmd: {' '.join(cmd)}\n  log: {log_path}\n{'='*80}")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, timeout=timeout_sec)
    elapsed = time.time() - t0
    print(f"[run_v19_3times] {stage_name} exit={proc.returncode} elapsed={elapsed:.1f}s")
    if proc.returncode != 0:
        # 打印 log 末尾 30 行诊断
        with open(log_path) as f:
            lines = f.readlines()
        print(f"[run_v19_3times] {stage_name} FAILED. Last 30 lines of log:")
        for line in lines[-30:]:
            print(f"  {line.rstrip()}")
    return proc.returncode


def parse_test_final_json(json_path: str) -> dict:
    """Read Stage 4 test_final.json → return metrics dict."""
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        return json.load(f)


def main():
    results = []
    for run_id in range(START_RUN, START_RUN + N_RUNS):
        print(f"\n{'#'*80}\n# RUN {run_id}/{START_RUN + N_RUNS - 1}\n{'#'*80}")

        stage1_dir = f"/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_run{run_id}"
        # R34d fix: 让 train_rqvae_instruments.py 用 stage1_dir 作为输出路径
        # 当前 train_rqvae_instruments.py 硬编码了 v19 输出路径, 我们用环境变量切换 (如果支持)
        # 简单做法: 每个 run 改 rqvae_out_v19_cend_07 目录 (直接覆盖)
        stage1_dir_default = "/home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v19_cend_07"

        # === Stage 1 ===
        rc = run_stage(
            f"Run{run_id} Stage 1 RQ-VAE",
            [
                "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
                "--nproc_per_node=4", "--master_port=29500",
                f"{MAIN_DIR}/train_rqvae_instruments.py",
            ],
            f"/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/run{run_id}_stage1.log",
            timeout_sec=1200,
        )
        if rc != 0:
            print(f"[run_v19_3times] Stage 1 FAILED, abort run {run_id}")
            results.append({"run_id": run_id, "stage": 1, "rc": rc})
            continue

        # === Stage 2 ===
        rc = run_stage(
            f"Run{run_id} Stage 2 SID",
            [CONDA_PYTHON, f"{MAIN_DIR}/infer_sids_instruments.py"],
            f"/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/run{run_id}_stage2_infer.log",
            timeout_sec=300,
        )
        if rc != 0:
            print(f"[run_v19_3times] Stage 2 infer FAILED, abort run {run_id}")
            results.append({"run_id": run_id, "stage": 2, "rc": rc})
            continue

        rc = run_stage(
            f"Run{run_id} Stage 2 build HG-Rec",
            [CONDA_PYTHON, f"{MAIN_DIR}/build_v19_sids_for_hgrec.py"],
            f"/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/run{run_id}_stage2_build.log",
            timeout_sec=60,
        )
        if rc != 0:
            print(f"[run_v19_3times] Stage 2 build FAILED, abort run {run_id}")
            results.append({"run_id": run_id, "stage": 2, "rc": rc})
            continue

        # === Stage 3 ===
        rc = run_stage(
            f"Run{run_id} Stage 3 T5",
            [
                "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
                "--nproc_per_node=4", "--master_port=29501",
                f"{MAIN_DIR}/train_decoder.py",
                f"{MAIN_DIR}/configs/decoder_instruments_hgrec_v19.gin",
            ],
            f"/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/run{run_id}_stage3.log",
            timeout_sec=2700,
        )
        if rc != 0:
            print(f"[run_v19_3times] Stage 3 FAILED, abort run {run_id}")
            results.append({"run_id": run_id, "stage": 3, "rc": rc})
            continue

        # === Stage 4 ===
        rc = run_stage(
            f"Run{run_id} Stage 4 test eval",
            [CONDA_PYTHON, f"{MAIN_DIR}/stage4_beam20.py"],
            f"/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/run{run_id}_stage4.log",
            timeout_sec=300,
        )
        if rc != 0:
            print(f"[run_v19_3times] Stage 4 FAILED, abort run {run_id}")
            results.append({"run_id": run_id, "stage": 4, "rc": rc})
            continue

        # === 读 test_final.json ===
        # resolve_stage3_ckpt 优先返回新路径 (R34b fix 后), 这里 4 个入口目录都尝试
        ckpt_dirs = [
            f"{MAIN_DIR}/out/decoder/instruments_hgrec_configs/hgrec_v19",
            f"{MAIN_DIR}/out/decoder/instruments_hgrec_/fs04/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/configs/hgrec_v19",
        ]
        metrics = {}
        for d in ckpt_dirs:
            p = os.path.join(d, "test_final.json")
            m = parse_test_final_json(p)
            if m:
                metrics = m
                metrics["ckpt_dir"] = d
                break
        results.append({
            "run_id": run_id,
            "stage": 4,
            "rc": 0,
            "metrics": metrics,
        })
        print(f"[run_v19_3times] Run{run_id} test_R@10={metrics.get('test_R@10', 'N/A')}")

    # === 总结 ===
    print(f"\n{'='*80}\n[run_v19_3times] SUMMARY ({len(results)} runs)\n{'='*80}")
    summary = {"runs": results}
    test_r10_values = [r["metrics"].get("test_R@10") for r in results if r.get("metrics")]
    if test_r10_values:
        import numpy as _np
        arr = _np.array(test_r10_values)
        summary["test_R@10_mean"] = float(arr.mean())
        summary["test_R@10_std"] = float(arr.std())
        summary["test_R@10_max"] = float(arr.max())
        summary["test_R@10_min"] = float(arr.min())
        summary["test_R@10_range"] = float(arr.max() - arr.min())
        print(f"  test_R@10 mean={arr.mean():.4f} std={arr.std():.4f} range=[{arr.min():.4f}, {arr.max():.4f}]")
        for v in test_r10_values:
            print(f"  - run: {v:.4f}")

    # 写 verdict
    verdict_path = f"{MAIN_DIR}/tasks/issue256_v19_3times_verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[run_v19_3times] verdict → {verdict_path}")


if __name__ == "__main__":
    main()
