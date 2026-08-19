"""v35 Stage 3 — HG-Rec T5 + Lorentz RMSNorm 训练 (DDP 4 卡).

主目录默认入口: 用 Stage 2 v19 SID (Instruments_v19_sids_for_hgrec.npy)
跑 HG-Rec T5 训练 (T5 24 个 RMSNorm 全部切 Lorentz hyperboloid model,
Chen 2022 Fully Hyperbolic NN 风格).

启动方式:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29509 \
    /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/tasks/Issue257_v35_lorentz_t5_whole/stage3.py

实际执行 → 顶层 train_decoder.py + configs/decoder_instruments_hgrec_v35.gin
(R40 自包含, 训练期 valid 评估 per-epoch, EARLY_STOP=20).
R42: 必须 torchrun --nproc_per_node=4 (DDP 4 卡).
R30/R43: 超参硬编码, 无 CLI 数值超参.
R41: EARLY_STOP=20, R41b: EVAL_INTERVAL=1.
R35: 单 ckpt + beam=20 (valid 阶段用 beam=20 选 best).
R36: 曲率机制: Lorentz RMSNorm (Chen 2022 Fully Hyperbolic NN),
     不调 LR/dropout/wd/batch_size.

输入:
  - Stage 2 产物: /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/dataset/
                  Instruments/Instruments_v19_sids_for_hgrec.npy

产物:
  - /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/out/decoder/
    instruments_hgrec_v35/best_ckpt.pt (~22MB)
  - /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/out/decoder/
    instruments_hgrec_v35/train_log.json

R37 决策点:
  test_R@10 ≥ 0.1113 (v19 baseline) → R37 PASS (v35 突破 0.1113 ceiling)
  test_R@10 < 0.1113 → R37 FAIL (回退 v19, R50 revert)
"""
import os
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3"

# ⚠️ R34b fix: 必须 cd 到 MAIN_DIR 后用相对路径传 config
STAGE3_TORCHRUN = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29509",
    os.path.join(MAIN_DIR, "train_decoder.py"),
    "configs/decoder_instruments_hgrec_v35.gin",
]

# R40 自包含: Stage 2 产物必须存在
STAGE2_OUT = "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/dataset/Instruments/Instruments_v19_sids_for_hgrec.npy"


def check_stage2_artifact():
    if not os.path.exists(STAGE2_OUT):
        raise FileNotFoundError(
            f"Stage 2 SID 不存在: {STAGE2_OUT}\n"
            f"请先跑 stage2.py 跑 RQ-VAE 推理 + 格式转换."
        )
    print(f"[v35 stage3] Stage 2 SID OK: {STAGE2_OUT}")


def main():
    check_stage2_artifact()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    print(f"[v35 stage3] launching DDP 4-card HG-Rec T5 training (Lorentz RMSNorm, Chen 2022)")
    print(f"[v35 stage3] command: cd {MAIN_DIR} && {' '.join(STAGE3_TORCHRUN)}")
    print(f"[v35 stage3] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print(f"[v35 stage3] expected best_ckpt: out/decoder/instruments_hgrec_v35/best_ckpt.pt")
    print(f"[v35 stage3] expected LorentzRMSNorm count: 24 (T5 6 encoder + 4 decoder blocks)")
    print(f"[v35 stage3] EARLY_STOP=20, EVAL_INTERVAL=1 (R41/R41b)")

    # ⚠️ R34b fix: 必须 cwd=MAIN_DIR 让 train_decoder.py save_dir_root fallback 解析为正确 tag
    result = subprocess.run(STAGE3_TORCHRUN, cwd=MAIN_DIR, env=env, check=False)
    if result.returncode != 0:
        print(f"[v35 stage3] FAIL exit={result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[v35 stage3] done → out/decoder/instruments_hgrec_v35/best_ckpt.pt")


if __name__ == "__main__":
    main()