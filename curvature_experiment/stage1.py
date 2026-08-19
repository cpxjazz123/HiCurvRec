"""Stage 1 — RQ-VAE training (DDP 4 卡).

主目录默认入口: v19 baseline 配置 (c_end=0.7, USE_M3_TRANSPORT=True,
HALC v2 + v16 differential schedule).
输出 ckpt: /home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/
           rqvae_out_v19_cend_07/rqvae_final.pt

启动方式 (任选其一):
  1) python3 stage1.py
  2) CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \
       stage1.py
  3) bash -c "$(cat <<'EOS'
     CUDA_VISIBLE_DEVICES=0,1,2,3 /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun \\
       --nproc_per_node=4 --master_port=29500 stage1.py
     EOS
     )"

实际执行 → scripts/train_rqvae_instruments.py (R40 自包含, 所有超参硬编码).
R42: 必须 torchrun --nproc_per_node=4 (DDP 4 卡).
R30/R43: 超参硬编码, 无 CLI 数值超参.
R36: 仅曲率机制 (c_end=0.7 curriculum), 不调 LR/dropout/wd.
R44b: HF 缓存到 hj82_scratch2, /home 不写.

产物:
  - rqvae_final.pt (~13.8MB)
  - rqvae_step{10000..100000}.pt (中间 ckpt, 每 10k step)
  - codes_per_layer 预期 [47, 234, 234]
"""
import os
import subprocess
import sys

# R42: 默认 4 卡 DDP 启动 (master_port 29500 不与 Stage 3/4 冲突)
STAGE1_TORCHRUN = [
    "/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/torchrun",
    "--nproc_per_node=4",
    "--master_port=29500",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "scripts/train_rqvae_instruments.py"),
]


def main():
    # GPU 选择: 默认 0,1,2,3 (R7 需确认空闲)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    # 提示用户: 必须先确认 GPU 空闲
    print("[stage1] launching DDP 4-card RQ-VAE training (c_end=0.7)")
    print(f"[stage1] command: {' '.join(STAGE1_TORCHRUN)}")
    print(f"[stage1] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print(f"[stage1] expected output: rqvae_out_v19_cend_07/rqvae_final.pt")
    print(f"[stage1] expected codes_per_layer: [47, 234, 234]")

    # 实际执行
    result = subprocess.run(STAGE1_TORCHRUN, env=env, check=False)
    if result.returncode != 0:
        print(f"[stage1] FAIL exit={result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[stage1] done → /home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v19_cend_07/rqvae_final.pt")


if __name__ == "__main__":
    main()