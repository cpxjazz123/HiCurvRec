"""v35 Issue255 Stage 2: SID inference (单卡).

v35 备胎 (Issue255): HyperbolicRMSNorm + HALC v2 c curriculum.

Stage 2 复用 v19 SID npy (Instruments_v19_sids_for_hgrec.npy, shape 9922x4).
Stage 2 改动 = 0 (v35 创新仅在 Stage 3).

R40 自包含: 任务目录可独立运行. 本脚本仅作为 stage 2 wrapper,
实际推理 → 顶层 /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/build_sids_for_hgrec.py

预期产物: /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/dataset/Instruments/Instruments_v19_sids_for_hgrec.npy

启动命令:
  cd /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3
  python3 /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/build_sids_for_hgrec.py \\
      --ckpt /home/wlia0047/hj82_scratch2/wenyu/rqvae_dataset/instruments/rqvae_out_v19_cend_07/rqvae_final.pt \\
      --out /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/dataset/Instruments/Instruments_v19_sids_for_hgrec.npy

单卡运行即可 (推理不是瓶颈).
"""