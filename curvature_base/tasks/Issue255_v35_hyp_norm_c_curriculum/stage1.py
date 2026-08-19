"""v35 Issue255 Stage 1: RQ-VAE training (DDP 4 卡).

v35 备胎 (Issue255): HyperbolicRMSNorm + HALC v2 c curriculum.

Stage 1 复用 v19 RQ-VAE ckpt (rqvae_out_v19_cend_07/rqvae_final.pt).
Stage 1 改动 = 0 (v35 创新仅在 Stage 3 RMSNorm c 调度).

R36 严格化: v35 改动仅 Stage 3 (R36 新曲率正则项, RMSNorm 曲率运行时调度),
Stage 1/2 RQ-VAE 训练产物 (SID npy, codebook ckpt) 复用 v19 baseline.
R40 自包含: 任务目录可独立运行 (Stage1 → Stage2 → Stage3 → Stage4_beam20).
本脚本仅作为 stage 1 wrapper, 实际执行 → 顶层 train_rqvae_instruments.py.

启动命令:
  cd /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 \\
      /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/train_rqvae_instruments.py

R42: 必须 torchrun --nproc_per_node=4 (DDP 4 卡).
R30/R43: 超参硬编码, 无 CLI 数值超参.

预期产物: /home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v19_cend_07/rqvae_final.pt
"""