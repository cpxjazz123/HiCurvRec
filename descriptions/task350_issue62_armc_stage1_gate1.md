# Task #350 — Issue #62 Gate 1 Arm C Stage 1 (#30+#43 联合 + Sinkhorn ON)

## 来源

承接 Issue #62 §Gate 1 Arm C spec (Issue #62 owner 显式 opened, 2026-07-30 16:07, §实验设计 Arm C):
- Stage 1 = #30 per-layer Codebook Transforms + #43 HypPreEncoder (NO K0=256, NO Sinkhorn OFF)
- K=[64,128,256] baseline (Issue #30 baseline K, 不是 K0=256)
- Sinkhorn ON (sk_eps=[0.003,0.003,0.003])
- 1000 epoch batch=1024 + lr=1e-3 AdamW (跟 Arm D 同步)
- --hyp_c 0.74 (Issue #43 HypPreEncoder)
- --radius_list 0.1 1.0 10.0 --scale_list 2.0 2.0 2.0 (Issue #30)

承接 task349 Arm D NO-GO 收口 (本 verdict task 上一阶段).

## 实施 spec

```bash
nohup bash scripts/task350_issue62_gate1_armc_stage1_train.sh
# num_emb_list 64 128 256 (Issue #30 baseline K)
# num_epochs 1000 --batch_size 1024 --lr 1e-3 --sk_eps 0.003
# hyp_c 0.74, radius_list 0.1 1.0 10.0, scale_list 2.0 2.0 2.0
# beta 0.25 --save_limit 1 --seed 42
# GPU 0
```

## 决策矩阵

- 若 Arm C PASS Gate 1 (L0/L1/L2 usage ≥ 20% @ ep30) → 进入 Stage 2 Sinkhorn + Stage 3 T5-mini + Stage 4 R@10 eval
- 若 Arm C 也 USAGE-KILL @ ep30 → Issue #62 整体 NO-GO 收口 (跟 #30+#43 K0=256 + Sinkhorn OFF 同模式)
- 若 Arm C PASS Gate 1 但 Stage 4 R@10 ≤ 0.1042 → Issue #62 联合 ablation 不优于 #43 单点, NO-GO 收口

## Arm C vs Arm D 关键差异

| 维度 | Arm C (本任务) | Arm D (NO-GO) |
|------|---------------|---------------|
| K0 | 64 (baseline) | 256 |
| Sinkhorn | ON (sk_eps=0.003) | OFF (sk_eps=0) |
| 期望 | 跟 Issue #30 baseline 同模式 + #43 HypPre | 大 K 探索 |

## 物理产物

| 类型 | 路径 |
|------|------|
| Trainer wrapper | `scripts/task350_issue62_gate1_armc_stage1_train.py` |
| Launcher | `scripts/task350_issue62_gate1_armc_stage1_train.sh` |
| 训练日志 | `logs/task350/stage1_gate1_armc_20260731_114535.log` |
| ckpt 路径 | `products/task350/hrqvae_issue62_gate1_armc/Jul-31-2026_*_beta_0.250_codebook_[64,128,256]_sk_0.003/` |