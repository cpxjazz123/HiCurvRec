# Task #403 / Issue #109 [Stage 3 短训] task401 ckpt + Stage 3 30 epoch 短训 → Stage 4 R@K

**日期**: 2026-07-31
**任务**: task401 30-epoch Stage 1 ckpt → Stage 2 Sinkhorn (复用 task402 SID) → Stage 3 T5-mini **30 epoch 短训** → Stage 4 R@K eval
**目的**: 验证 task402 verdict §4 关键发现 #4: "训练时长 ≤ 30 epoch 杠杆 Stage 3 是否成立"
**基线对照**:
- HG-Rec Task #84: R@10 = 0.1020 (200 ep Stage 1 + 200 ep Stage 3)
- task402: R@10 = 0.0991 (30 ep Stage 1 + 200 ep Stage 3, -2.8% vs baseline, +14.7% vs task396b)
- 决策阈值: R@10 > 0.1020 GO, ≤ 0.1020 NO-GO

## 假设

**H1**: Stage 3 30 epoch 短训 + Stage 1 30 epoch 短训 (task401 recipe) → R@10 > 0.1020 GO
**H2 (refuted)**: Stage 3 短训会过拟合 (训练数据不足) → R@10 ≪ 0.1020 NO-GO
**H3 (refuted)**: Stage 3 短训对 Stage 1 长训 baseline (task84) 有提升 (recipe 不适用 task84 baseline, 此 task 是 task401 recipe 联合)

## Gate 0 (实施前预检)

- [x] task401 Stage 1 ckpt 已落盘 (`products/task401_issue108_best_epoch_verify/ckpt/issue108_ckpt.pt`)
- [x] task402 Stage 2 SID 已落盘 (`HG-Rec/dataset/Instruments/Instruments_t5_rqvae_task402.npy`, 9922 × 4)
- [x] task402 Stage 3 训练 config 已验证可跑通 (`scripts/task84_hgrec_stage3_train.py` + Issue #97 patch)
- [x] GPU 0 全空闲 (util 0%, mem 0 MiB) — task403 分配 GPU 0
- [x] R12 ckpt 强制落盘逻辑已就绪 (task84_hgrec_stage3_train.py 内置)

## Gate 1 (Stage 1 ckpt 复用)

- 直接 load task401 ckpt, 不重新训练 Stage 1 (避免重复 task401 工作)

## Gate 2 (Stage 2 SID 复用)

- 直接复用 task402 SID (`Instruments_t5_rqvae_task402.npy`), 不重新跑 Sinkhorn

## Gate 3 (Stage 3 T5-mini 30 epoch 短训)

- config 完全跟 task402 Stage 3 一致 (T5-mini 5.5M, 200 ep config → 改 num_epochs=30)
- early_stop = 5 (vs task402 = 20), 因训练时长短, 早停阈值相应缩短
- LR scheduler 跟 task402 一致
- ckpt 输出路径: `products/task403_issue109_stage3_short_train/ckpt/Instruments/<TS>/HG_Rec_best.pth`
- R12 强制: 每 epoch 末 save ckpt + delete old (跟 task402 一致)
- 预计耗时: ~15 min (30 epoch vs task402 200 epoch)

## Gate 4 (Stage 4 R@K eval)

- 复用 `scripts/task402_stage4_rk_eval.sh` 模板 (改 paths → task403)
- 决策阈值: R@10 > 0.1020 GO, ≤ 0.1020 NO-GO
- 跟 task402 (R@10=0.0991) + task396b (R@10=0.0864) + HG-Rec baseline (R@10=0.1020) 对比

## 期望

| 任务 | Stage 1 ep | Stage 3 ep | R@10 | Δ vs baseline |
|------|-----------|-----------|------|---------------|
| task84 (HG-Rec) | 200 | 200 | 0.1020 | baseline |
| task396b | 200 | 200 | 0.0864 | -15.3% |
| task402 | 30 | 200 | 0.0991 | -2.8% |
| **task403 (待)** | **30** | **30** | **?** | **?** |

**期望 1 (H1)**: task403 R@10 > 0.1020 → 训练时长杠杆双 Stage 都成立, 推翻 HG-Rec baseline recipe
**期望 2 (H2)**: task403 R@10 ≤ 0.1020 → Stage 3 短训稀释 Stage 1 杠杆, 但仍比 task402 (-2.8%) 略好
**期望 3 (H3)**: task403 R@10 ≪ 0.0991 → Stage 3 训练时长真的影响 T5 学习, 短训过拟合严重

## R11.5 自主决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| GPU 分配 | GPU 0 | 4 卡全空闲, GPU 0/1/2/3 任选, 默认 GPU 0 (跟 task396b 错开) |
| Stage 3 num_epochs | 30 (跟 task401 Stage 1 同步) | 验证"训练时长 ≤ 30 epoch"杠杆 Stage 3 是否成立 |
| early_stop | 5 (vs task402=20) | 30 epoch 短训, 早停阈值相应缩短 |
| Stage 3 ckpt dir | `products/task403_issue109_stage3_short_train/` | 跟 task401/task402 命名一致 |
| Stage 4 script | 复用 task402 模式, 独立 `task403_stage4_rk_eval.sh` | 避免 task398 hardcoded paths bug 重复 |

## 预期产物

- verdict: `verdicts/task403_issue109_stage3_short_train_v2.md`
- metrics: `verdicts/task403_stage4_rk_eval_metrics.json`
- Stage 3 ckpt: `products/task403_issue109_stage3_short_train/ckpt/Instruments/<TS>/HG_Rec_best.pth`
- Stage 4 log: `logs/task403_stage3_train_<TS>.log` + `logs/task403_stage4_eval_<TS>.log`
