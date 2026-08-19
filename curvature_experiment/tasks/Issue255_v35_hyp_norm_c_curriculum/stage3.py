"""v35 Issue255 Stage 3: HG-Rec T5 训练 (DDP 4 卡).

v35 备胎 (Issue255): HyperbolicRMSNorm + HALC v2 c curriculum.
- 在 v34 HyperbolicRMSNorm (c=1.0 静态) 基础上, 加 HALC v2 sigmoid c curriculum:
  c(t) = c_min + (c_max - c_min) * sigmoid((t - t_start) / t_scale)
  c_min=0.5 (C10 扩展下界, 避免 c→0 数值不稳定)
  c_max=1.0 (强 Poincaré 约束, 与 v34 终态一致)
  t_start=15 (前 15 epoch 让 model 学习基础语义, c 仍≈0.5)
  t_scale=5 (epoch 15-20 平滑过渡到 c≈1.0)

实际执行 → 顶层 train_decoder.py + configs/decoder_instruments_hgrec_v35.gin
(顶层 train_decoder.py 通过 sys.argv path-detect 检测 'v35' 字符串, 触发 c curriculum)

启动命令:
  cd /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 \\
      /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/train_decoder.py \\
      /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/configs/decoder_instruments_hgrec_v35.gin \\
      > /home/wlia0047/hj82_scratch2/wenyu/claude_tmp/v35_stage3.log 2>&1

R42: 必须 torchrun --nproc_per_node=4 (DDP 4 卡).
R30/R43: 超参硬编码, 无 CLI 数值超参.
R41: EARLY_STOP=20, R41b per-epoch valid eval.
R35: 单 ckpt + beam=20.
R36: 改动仅 RMSNorm c curriculum (新曲率正则项), c_min/c_max/t_start/t_scale 全部硬编码.

预期产物: /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/out/decoder/instruments_hgrec_configs/hgrec_v35/best_ckpt.pt
"""