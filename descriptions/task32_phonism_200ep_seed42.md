# Task #32 — phonism/genrec TIGER 200 epochs × seed=42 (run #1/4)

> **任务目的**: 跑满 phonism TIGER 训练至 200 epochs（vs Task #126 仅 10 epochs），验证 R@5=0.034 上限假设
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

> 同系列任务: Task #33 (seed=123), Task #34 (seed=7), Task #35 (seed=2024)
> GPU 分配: cuda:0 (Task #32) / cuda:1 (Task #33) / cuda:2 (Task #34) / cuda:3 (Task #35)
> 全部并行跑，约 5-6h 完成

---

## 1. 背景

### 1.1 Task #126 现状
- Task #126 仅跑 10 epochs of 200，已达 R@5=0.0200（仍在上升）
- phonism 实跑报告 200 epochs + patience=10 early stop → R@5=0.0340（与 paper Table 1 完全一致）
- 用户指令: "新建一个任务，然后四个种子。四个任务，帮我跑满。看看结果。"

### 1.2 本次任务假设
- 假设 R1: 跑满 200 epochs 后 R@5 接近 0.030-0.034（与 phonism 一致）
- 假设 R2: 4 个 seed 的 R@5 方差 CV ≤ 20%（与 Task #87 / Task #107 seed variance 一致）
- 假设 R3: 4 个 seed 的 R@5 均值 ≫ Task #87 baseline (0.01937)

---

## 2. 实验设计

**变量**: 仅 seed (42 / 123 / 7 / 2024)
**保持不变** (与 Task #126 完全一致):
- RQ-VAE ckpt: `/home/wlia0047/ar57/wenyu/genrec/out/tiger/amazon/toys/rqvae/checkpoint_19999.pt` (30000 iter)
- TIGER config: phonism `config/tiger/amazon/tiger.gin`
- Optimizer: Adam lr=1e-4, no scheduler, fp16
- batch_size=256, beam_size=30, dropout=0.1
- epochs=200, early_stop_patience=10, eval_test_every_epoch=2
- num_layers=4, num_decoder_layers=4, d_model=128, d_ff=1024
- num_heads=6, d_kv=64, codebook_size=256, sem_id_dim=3, max_seq_len=50
- 评估: full ranking（phonism 默认）

**trainer 改动**:
- `genrec/trainers/tiger_trainer.py:130-141` 新增 `seed=42` 参数
- `torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)` 在 train() 起始

**启动命令** (Task #32 = seed=42):
```bash
cd /home/wlia0047/ar57/wenyu/genrec
PYTHONPATH="/home/wlia0047/ar57/wenyu/genrec:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0 \
python -m genrec.trainers.tiger_trainer \
    --config config/tiger/amazon/tiger.gin \
    --split toys \
    --gin "train.epochs=200" \
    --gin "train.early_stop_patience=10" \
    --gin "train.wandb_logging=False" \
    --gin "train.eval_test_every_epoch=2" \
    --gin "train.save_dir_root='out/tiger/amazon/toys/seed42/'" \
    --gin "train.pretrained_rqvae_path='out/tiger/amazon/toys/rqvae/checkpoint_19999.pt'" \
    --gin "train.seed=42"
```

---

## 3. 决策触发

| 指标条件 | 结果 | 决策 |
|----------|------|------|
| 4 seed 均值 R@5 ≥ 0.030 | ✅ 假设 R1 确认 | phonism R@5=0.034 上限可信，Task #87 真天花板需重新审视 |
| 4 seed 均值 R@5 ∈ [0.020, 0.030] | ⚠️ 部分确认 | phonism 实跑 ≠ paper 数字，需查 phonism 内部机制 |
| 4 seed 均值 R@5 < 0.020 | ❌ 假设 R1 否证 | Task #87 0.01937 反而是上限，phonism 数字虚高 |
| 4 seed CV ≤ 20% | ✅ 假设 R2 确认 | seed variance 在正常区间 |
| 4 seed CV > 30% | ⚠️ seed variance 主导 | 结论需 seed-aggregate 后取均值 |

---

## 4. 预算

| 阶段 | 估算时间 (单 seed) |
|------|-------------------|
| TIGER 训练至 200 epochs / early-stop | ~5-6h (full ranking eval 每个 epoch) |
| 4 seed 并行总时长 | ~5-6h（受限于最慢 seed） |
| 评估报告 + verdict | ~30 min |

---

## 5. 风险与缓解

**风险 1**: GPU OOM（phonism TIGER 跑 batch_size=256 + beam_size=30，可能 OOM）→ fallback 改 batch_size=128
**风险 2**: HF model 下载慢（首次 sentence-t5-base）→ 已 Task #126 缓存
**风险 3**: early_stop 触发过早 → patience=10 给足够缓冲
**风险 4**: 全 4 seed 同时 OOM → 串行 fallback（但用户指令是"四个任务，帮我跑满"，串行违反"并行"精神）

---

## 6. 完成度跟踪

- [ ] Trainer seed 参数 patch + py_compile verify
- [ ] 4 个 description 写完 (#32-#35)
- [ ] 4 个 launcher script 写完
- [ ] 4 PIDs 启动 (cuda:0/1/2/3)
- [ ] §16 更新为 4 活跃任务
- [ ] 4 PIDs 完成训练 + test eval
- [ ] 4 verdicts 写完 (#32-#35)
- [ ] 综合对比表 (4 seed vs Task #87 vs Task #126)
