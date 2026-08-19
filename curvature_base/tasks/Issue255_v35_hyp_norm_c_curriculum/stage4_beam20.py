"""v35 Issue255 Stage 4: HG-Rec T5 test eval (DDP 4 卡).

v35 备胎 (Issue255): HyperbolicRMSNorm + HALC v2 c curriculum.

Stage 4 用 stage3 训练得到的 best_ckpt.pt, 跑 test 集评估.
R35: 单 ckpt + beam=20, 禁 Borda Rank Fusion.
R35b: DDP 4 卡各自分片不重复评估 test 集, all_reduce SUM.

实际执行 → 顶层 test_eval_only.py + best_ckpt.pt

启动命令:
  cd /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29502 \\
      /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/test_eval_only.py \\
      /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/configs/decoder_instruments_hgrec_v35.gin \\
      /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/out/decoder/instruments_hgrec_configs/hgrec_v35/best_ckpt.pt \\
      > /home/wlia0047/hj82_scratch2/wenyu/claude_tmp/v35_stage4.log 2>&1

预期产物: /home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/out/decoder/instruments_hgrec_configs/hgrec_v35/test_final.json

R37 决策点:
  test_R@10 ≥ 0.1077 (v34 baseline) → R37 PASS, 保留 v35 改动
  test_R@10 < 0.1077 (v34 baseline) → R37 FAIL, R50 revert v35 改动
"""