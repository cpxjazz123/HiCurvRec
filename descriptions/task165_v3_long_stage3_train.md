# Task #165 — v3 Stage 3 长训 200 epoch

> **任务目的**: 复用 Task #164 已训 Stage 1 ckpt (Phase A 100ep + Phase B 100ep κ-decouple) + κ-Stereographic SID, 长训 T5-small 60M 200 epochs + early_stop=30, 关闭 val/test gap (~18%), 期望 Stage 4 test R@10 ≥ 0.1058 baseline.
>
> **完成日期**: (in progress)
> **状态**: 🟢 在跑 (PID 4022824, GPU 0)

---

## 1. 背景

承接 Task #164 verdict (κ-Stereographic + Phase A/B 部分成功):
- ✅ Stage 1 codebook collapse 修复: util L0/L1/L2 = 90%/100%/97%
- ⛔ Stage 4 test R@10 = 0.0964 (-8.9% vs baseline 0.1058)
- ✅ Stage 3 val R@10 = 0.1177 (+11.2% over baseline, 12 epoch 时已超)
- 关键诊断: val/test gap ≈ 18%, **Stage 3 训练 epoch 不足** (12/200 early-killed)

假设 R1: 长训 Stage 3 (200 epoch) 可以关闭 val/test gap
假设 R2: test R@10 应能突破 0.1058 baseline (val R@10=0.1177 表明模型学到了 pattern)
假设 R3: 如果 test R@10 仍 < 0.1058, 需进一步诊断 (T5 容量 / κ_max 探测 / Sinkhorn 配置)

---

## 2. 实验设计

**变量**: Stage 3 训练 epoch 数 (12 → 200) + early_stop patience (20 → 30)

**保持不变**:
- Stage 1 ckpt: `products/task164/phase_b_kappa_decouple/jul-25-2026_01-28-22/best_loss_model.pth` (4.5 MB)
- Stage 2 SID: `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_phase_b_kappa_decouple.npy` (9922 items, util 90/100/97%)
- Stage 3 模型: T5-small 60M (6 enc + 6 dec, d_model=512, d_ff=2048, 8 heads d_kv=64, ~44.6M params)
- lr=1e-4, batch_size=256, seed=42, beam_size=20

**启动命令**:
```bash
bash scripts/task165_v3_stage3_longtrain.sh
# CUDA_VISIBLE_DEVICES=0, TRITON_CACHE_DIR=/home/wlia0047/.triton/cache_task165_v3_s3
# num_epochs=200, early_stop=30, lr_theta 不适用 (Phase A/B 已完成)
```

**Stage 4 (自动触发)**:
```bash
bash scripts/task165_v3_stage4_eval.sh
# R88 daemon: training PID 退出后自动调用
# beam_size=20, 写 verdicts/task165_v3_metrics.json
```

---

## 3. 决策触发 (vs baseline)

| 条件 | test R@10 | 决策 |
|------|-----------|------|
| val R@10 ≥ 0.12, test R@10 ≥ 0.1058 | ✅ Stop hook 达成 | 写 verdict + paper §6.2 修订 + 标记完成 |
| val R@10 ≥ 0.12, test R@10 < 0.1058 | ⛔ val/test gap 仍存 | 进一步诊断 (T5 容量 / SID 质量) |
| val R@10 < 0.12 | ⛔ Stage 3 训练失败 | 检查过拟合 / lr 调度 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| Stage 3 训练 | ~3-4 hour (200 epoch × 1.27 min/epoch 实测) |
| Stage 4 推断 | ~5 min |
| R@10 评估 | ~1 min |
| **总计** | ~3-4 hour |

---

## 5. 风险与缓解

**风险 1**: GPU 0 与其他任务冲突 → 当前 nvidia-smi 全空闲, 监控 R7
**风险 2**: 长训过拟合 → early_stop=30 (vs v1 的 20) 留更多 buffer
**风险 3**: TRITON cache 累积 → 独立 TRITON_CACHE_DIR 路径
**风险 4**: Stage 3 训练崩溃 → R12 save_strategy 强制每 epoch 保存, 不会全部丢失

---

## 6. 完成度跟踪

- [x] paper.md §5.7.1 第 6 条修订 (Phase A/B 修复 + Stage 4 gap)
- [x] paper.md §6.2 §6.2.1 新增 ("R3 falsified" 子节)
- [x] task165_v3_stage3_longtrain.sh 创建 (chmod +x)
- [x] task165_v3_stage4_eval.sh 创建 (chmod +x)
- [x] R88 daemon (_STAGE4_TRIGGER.sh) 配置
- [x] products/task165/_TRAINING_PID 写入 (PID 4022824)
- [x] GPU 0 启动 v3 Stage 3 (200 epoch, early_stop=30)
- [ ] Stage 3 训练完成 (200 epoch 或 early_stop)
- [ ] Stage 4 自动 eval 完成
- [ ] verdicts/task165_v3_long_stage3_result.md 写完
- [ ] loop.md §16 R8 清理 (任务完成时立即)