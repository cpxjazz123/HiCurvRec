# Issue #240 — infra

| 项 | 命令 | 输出摘要 | sha256 | 判定 |
|---|---|---|---|---|
| 数据 train | `sha256sum results/stage0_build_parquet/train.parquet` | 存在 | `80597e38d081…` | PASS |
| 数据 items | `sha256sum results/stage0_build_parquet/items.parquet` | 存在 | `6408e3c0f98a…` | PASS |
| Stage1 嵌入 | `sha256sum stage1_GeneEmbedding/output/sentence_t5.npy` | 24587×768 | `6490c7175395…` | PASS |
| 冻结基线源码 | `sha256sum stage2_RQ-VAE/baseline/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/train_rqvae.py` | 只读 | `f10d57c8946c…` | PASS |
| 基线 model | `model/model.py` / `model/layers.py` | 只读 | `22f8752db264…` / `ac52f7a598c5…` | PASS |
| 历史 resample SID | `results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample/item_sids.json` | 24,587 项 | `5e4562c05c7d…` | PASS |
| 历史 checkpoint | `.../resample/rqvae_best.pth` | 4.6 MB | `66df180564755…` | PASS |
| 历史 Stage3 recall | `.../TIGER_BEHAVIOR_SAMPLE_RQ-VAE__resample/logs/.../test_final.json` | `test_recall@10=0.0633019`, n_eval 57,439 | — | PASS（issue 引用 0.0637372，差 4.4e-4，已记录） |
| 解释器 | `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9`（§7） | torch 2.x + CUDA | — | PASS |
| 硬件 | `nvidia-smi` | 4 × L40S 46 GB，端口 50202 空闲 | — | PASS |
| 诊断树 | `stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE_diag` | 与基线 diff：46 行新增 / 1 行改动 | `train_rqvae.py 0e08e7bcb968…`、`diag_trace.py ddecfa9bc9d7…` | PASS |
| 本轮 checkpoint | `results/stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE_diag/resample/rqvae_best.pth` | 3000 epoch 终点 | `0b44272f0f6d…` | PASS |
