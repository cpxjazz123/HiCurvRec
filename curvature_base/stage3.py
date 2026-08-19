"""Stage 3 — HG-Rec T5 training (DDP 4 卡).

主目录默认入口: 用 Stage 2 v19 SID (Instruments_v19_sids_for_hgrec.npy)
跑 HG-Rec T5 训练 (HALC v2 + v16 differential schedule).

启动方式 (任选其一):
  1) python3 stage3.py
  2) CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 \
       /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/stage3.py

实际执行 → 顶层 train_decoder.py + configs/decoder_instruments_hgrec_v19.gin
(R40 自包含, 训练期 valid 评估 per-epoch, EARLY_STOP=20).
R42: 必须 torchrun --nproc_per_node=4 (DDP 4 卡).
R30/R43: 超参硬编码, 无 CLI 数值超参.
R41: EARLY_STOP=20, R41b: EVAL_INTERVAL=1.
R35: 单 ckpt + beam=20 (valid 阶段用 beam=20 选 best).
R36: 曲率机制: HALC v2 sigmoid c 调度 (v16 differential schedule),
     不调 LR/dropout/wd/batch_size.

输入:
  - Stage 2 产物: /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/dataset/
                  Instruments/Instruments_v19_sids_for_hgrec.npy

产物:
  - /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/out/decoder/
    instruments_hgrec_configs/hgrec_v19/best_ckpt.pt (~22MB)
  - /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/out/decoder/
    instruments_hgrec_configs/hgrec_v19/train_log.json

预期 best valid ndcg@10 = 0.097 (vs Issue239 0.097, -0.001 noise 内).
"""
import os
import subprocess
import sys

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# ⚠️ R34b fix: 必须 cd 到 MAIN_DIR 后用相对路径传 config,
# 否则 train_decoder.py save_dir_root fallback 把 config 绝对路径当 tag,
# 输出到 out/decoder/instruments_hgrec_<full_path>/hgrec_v19/ 嵌套错目录
STAGE3_TORCHRUN = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29501",
    os.path.join(MAIN_DIR, "train_decoder.py"),
    "configs/decoder_instruments_hgrec_v19.gin",
]

# R40 自包含: Stage 2 产物必须存在
STAGE2_OUT = "/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/dataset/Instruments/Instruments_v19_sids_for_hgrec.npy"


def check_stage2_artifact():
    if not os.path.exists(STAGE2_OUT):
        raise FileNotFoundError(
            f"Stage 2 SID 不存在: {STAGE2_OUT}\n"
            f"请先跑 stage2.py 跑 RQ-VAE 推理 + 格式转换."
        )
    print(f"[stage3] Stage 2 SID OK: {STAGE2_OUT}")


def main():
    check_stage2_artifact()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    print(f"[stage3] launching DDP 4-card HG-Rec T5 training (HALC v2 + v16)")
    print(f"[stage3] command: cd {MAIN_DIR} && {' '.join(STAGE3_TORCHRUN)}")
    print(f"[stage3] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print(f"[stage3] expected best_ckpt: out/decoder/instruments_hgrec_configs/hgrec_v19/best_ckpt.pt")
    print(f"[stage3] expected best valid ndcg@10 = 0.0970")
    print(f"[stage3] EARLY_STOP=20, EVAL_INTERVAL=1 (R41/R41b)")

    # ⚠️ R34b fix: 必须 cwd=MAIN_DIR 让 train_decoder.py save_dir_root fallback 解析为正确 tag
    result = subprocess.run(STAGE3_TORCHRUN, cwd=MAIN_DIR, env=env, check=False)
    if result.returncode != 0:
        print(f"[stage3] FAIL exit={result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[stage3] done → out/decoder/instruments_hgrec_configs/hgrec_v19/best_ckpt.pt")


if __name__ == "__main__":
    main()