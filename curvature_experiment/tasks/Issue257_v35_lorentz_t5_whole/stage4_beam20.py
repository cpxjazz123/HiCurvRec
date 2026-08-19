"""v35 Stage 4 — HG-Rec T5 + Lorentz RMSNorm test eval (DDP 4 卡, beam=20).

用 Stage 3 v35 best_ckpt.pt 跑 test 集评估.
输出 test_R@10, test_R@20, test_NDCG@10/20.

启动方式:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29510 \
    /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/tasks/Issue257_v35_lorentz_t5_whole/stage4_beam20.py

R35: 单 ckpt + beam=20, 禁 Borda Rank Fusion.
R35b: DDP 4 卡各自分片不重复评估 test 集, all_reduce SUM.
R42: 必须 torchrun --nproc_per_node=4 (DDP 4 卡).
R30/R43: 超参硬编码, 无 CLI 数值超参.

⚠️ 重要: 必须设 FORCE_HGREC=1 环境变量 (绕开 test_eval_only.py gin binding bug).

输入:
  - Stage 3 产物: best_ckpt.pt
  - gin config:    /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/configs/
                  decoder_instruments_hgrec_v35.gin

产物:
  - test_final.json (与 best_ckpt.pt 同目录)

R37 决策点:
  test_R@10 ≥ 0.1113 (v19 baseline) → R37 PASS (v35 突破 ceiling)
  test_R@10 < 0.1113 → R37 FAIL (回退 v19, R50 revert)
"""
import os
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3"

STAGE4_TORCHRUN_BASE = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29511",
]

STAGE3_CKPT = os.path.join(MAIN_DIR, "out/decoder/instruments_hgrec_configs/hgrec_v35/best_ckpt.pt")


def build_stage4_cmd(ckpt_path):
    return STAGE4_TORCHRUN_BASE + [
        os.path.join(MAIN_DIR, "test_eval_only.py"),
        os.path.join(MAIN_DIR, "configs/decoder_instruments_hgrec_v35.gin"),
        ckpt_path,
    ]


def check_stage3_artifact():
    if not os.path.exists(STAGE3_CKPT):
        raise FileNotFoundError(
            f"v35 Stage 3 best_ckpt 不存在: {STAGE3_CKPT}\n"
            f"请先跑 stage3.py 训练 v35 HG-Rec T5 (Lorentz RMSNorm)."
        )
    print(f"[v35 stage4] Stage 3 best_ckpt OK: {STAGE3_CKPT}")


def main():
    check_stage3_artifact()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    env["FORCE_HGREC"] = "1"

    cmd = build_stage4_cmd(STAGE3_CKPT)
    print(f"[v35 stage4] launching DDP 4-card HG-Rec T5 + Lorentz RMSNorm test eval (beam=20)")
    print(f"[v35 stage4] command: FORCE_HGREC=1 CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
          f"{' '.join(cmd)}")
    print(f"[v35 stage4] FORCE_HGREC={env['FORCE_HGREC']} (REQUIRED — 绕开 gin binding bug)")
    print(f"[v35 stage4] best_ckpt: {STAGE3_CKPT}")
    print(f"[v35 stage4] expected test_R@10 ≥ 0.1113 (v19 baseline)")

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        print(f"[v35 stage4] FAIL exit={result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    out_dir = os.path.dirname(STAGE3_CKPT)
    out_json = os.path.join(out_dir, "test_final.json")
    # test_eval_only.py 内置 hardcode 输出路径: out/decoder/instruments/test_final.json
    default_json = os.path.join(MAIN_DIR, "out/decoder/instruments/test_final.json")
    if not os.path.exists(out_json) and os.path.exists(default_json):
        # 把默认输出复制到 v35 目录,便于 R37 决策路径
        import shutil
        shutil.copy(default_json, out_json)
        print(f"[v35 stage4] copied {default_json} -> {out_json}")
    if os.path.exists(out_json):
        import json
        with open(out_json) as f:
            metrics = json.load(f)
        r10 = metrics.get("test_R@10", 0)
        print(f"\n[v35 stage4] test_R@10 = {r10:.4f}")
        if r10 >= 0.1113:
            print(f"[v35 stage4] R37 PASS (test_R@10 ≥ 0.1113 v19 baseline, v35 Lorentz RMSNorm 突破 ceiling)")
        else:
            print(f"[v35 stage4] R37 FAIL (test_R@10 < 0.1113 v19 baseline, 触发 R50 revert)")
    print(f"[v35 stage4] done → {out_json}")


if __name__ == "__main__":
    main()