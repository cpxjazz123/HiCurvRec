# Task #31 — Baseline 多 seed 扩展 (seed=7, 用户中断)

> **任务目的**: 同一 baseline SID 跑多 seed 隔离 seed variance；本次 seed=7 由用户指令中断
> **完成日期**: 2026-07-19
> **状态**: ⏸️ 用户中断（非完成）
> **执行人**: Claude

---

## 1. 一句话结果

用户明确指令 "先把我结束Task #31" 后，PID 2853080 (launcher) + PID 2853361 (train.py parent) + 8 个 pt_data_worker 子进程已被 SIGTERM 优雅停止，GPU 0 完全释放（0 MiB），**best ckpt 在 step 1300 (val/recall@5=0.03364) 保留**。

---

## 2. 执行时间线

| 步骤 | 时间 | 备注 |
|------|------|------|
| 启动 Stage 3 (seed=7) | 2026-07-19 16:15 | PID 2853080 launcher, grid_toys env, cuda:0 |
| 训练 step 1000 | 2026-07-19 ~17:33 | val/recall@5=0.02777（首次 top-1） |
| 训练 step 1100 | 2026-07-19 ~17:38 | val/recall@5=0.02797 |
| 训练 step 1200 | 2026-07-19 ~17:43 | val/recall@5=0.03086 |
| **训练 step 1300** | 2026-07-19 ~17:53 | **val/recall@5=0.03364 (best)** |
| 训练 step 1400-1500 | 2026-07-19 17:53-18:07 | val 未进 top-1, training 继续 |
| 用户指令中断 | 2026-07-19 18:07 | SIGTERM → 10s 后 train.py parent 退出 |
| GPU 0 验证释放 | 2026-07-19 18:08 | nvidia-smi: 27081 MiB → 0 MiB |

**总训练时长**: 1h52m (实际训练 step 0 → 1500)

---

## 3. 关键指标

### 3.1 val/recall@5 曲线

| Step | val/recall@5 | Status |
|-----:|-------------:|--------|
| 1000 | 0.02777 | best (ckpt saved) |
| 1100 | 0.02797 | best |
| 1200 | 0.03086 | best |
| **1300** | **0.03364** | **best (final ckpt)** |
| 1400 | (not in top 1) | - |
| 1500 | (not in top 1) | - |

### 3.2 与 Task #87 (seed=42) 对比

| | seed=42 (Task #87) | seed=7 (Task #31, 本次) |
|---|---|---|
| 100k-step best val/recall@5 | 0.02050 (step 66000) | 0.03364 (step 1300, 1.3%) |
| 训练进度 | 86% (86000/100000) | 1.5% (1500/100000) |
| 总训练时长 | ~6-8h | 1h52m (中断) |

**关键观察**: seed=7 在仅 step 1300 就已经达到 0.03364，远高于 seed=42 在 step 66000 的 0.02050。这可能是：
- (a) seed=7 训练初期 val recall 虚高（Lightning 早期 eval 间隔 1000 steps + 数据集顺序随机）→ 后续会回归
- (b) seed=7 的 SID → TIGER 隐空间更易学 → 全程 val 都高
- **无法判断** — 需要继续训练至少 10-20k steps 看是否稳定

### 3.3 训练参数（与 Task #87 完全一致）

- Stage 1 SID: `products/task87_tiger_baseline/stage2_rqvae_infer/sid_dedup.pt`
- 模型: snap-default TIGER (4 enc/dec, 6 heads, d_model=128)
- Optimizer: Adafactor lr=0.01 + inverse_sqrt schedule
- Batch=256, sequence_length=120, num_hierarchies=4
- num_user_bins=2000 (启用 user tokens)
- seed=7 (vs Task #87 seed=42, Task #107 seed=123)

---

## 4. 产物清单

| 路径 | 大小 | 说明 |
|------|------|------|
| `logs/task31_seed7_s3/runs/2026-07-19/16-15-47/checkpoints/checkpoint_epoch=000_step=001300.ckpt` | 159 MB | best ckpt, val/recall@5=0.03364 @ step 1300 |
| `logs/task31_seed7_s3/run.log` | 7.9 MB | 完整 training log |
| `logs/task31_seed7_s3/runs/2026-07-19/16-15-47/` | - | TensorBoard events |

**未产出**:
- 无 Stage 4 inference
- 无 final test R@5 数字
- 无 merged_predictions_tensor.pt

---

## 5. 用户中断原因

用户提供指令: "先把我结束Task #31"。本次执行结果仅作为"早期 val 趋势"参考，**不能视为完整 seed=7 baseline**。若需继续：

1. **继续训练至 100k steps** (~5-6h 剩余) — 可能 val 稳定在 ~0.020-0.030 区间
2. **改用更激进 early-stop** — 每 1000 steps eval，连续 5k steps 无改进就停
3. **改用 ckpt resume** — `python -m src.train trainer.resume_from_checkpoint=path ckpt_path=...`

---

## 6. 结论

⏸️ **Task #31 seed=7 由用户指令中断**，best val/recall@5=0.03364 @ step 1300 保留，但仅 1.3% 进度，**无 test 评估**。
后续若需 seed=7 baseline，需 resume 训练或重启。

---

当前任务已完成，请做下一个任务的指示。

result: Task #31 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
