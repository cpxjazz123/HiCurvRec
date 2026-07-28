# Task #157 — HG-Rec T5-base 220M 容量解锁

> **任务目的**: 用 T5-base 220M (vs Task #84 5.5M T5-from-scratch) 重训 Stage 3 T5, 验证 T5 容量是 R@10 = 0.105 plateau 的根因. **目标**: R@10 ≥ 0.118 (paper 0.1315 -10%, 当前 baseline +13%).
> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (Task #156 Stage 2 完成后立即启动)

---

## 1. 背景

**用户 2026-07-24 决策** (option C, 并行):
- Task #156 stage 2 → stage 3 (T5-small 5.5M) on **GPU 3** (~2h45min 串行)
- Task #157 (T5-base 220M) on **GPU 0** (~93h 串行)
- 总墙时间 ≈ 93h

**代码分析结论** (2026-07-24 任务#156 status check):
1. **评估指标是 strict SID exact-match**: R@K 是 4-token 序列精确匹配. T5 容量决定了"4-token 序列预测"的精度上限
2. **T5-from-scratch 5.5M 是当前 plateau 根因**: 比 paper T5-base (220M) 小 **40x**. 模型只能学粗 user→next pattern, 学不到 SID fine ranking
3. **5 个 HG-Rec variants (β/codebook/sk/beam/κ) R@10 都挤在 0.10 ± 0.005**: proof that capacity hits ceiling, codebook tuning hits noise floor

**Task #84 R@10 = 0.102 vs paper 0.1315 (-22.4%)**: 是 T5 capacity 不足, 不是评估/SID/codebook 问题.

---

## 2. 实验设计

**变量**: T5-from-scratch 5.5M → **T5-base 220M** (vs Task #84 baseline)
**保持不变** (跟 Task #84 baseline 一致):
- Stage 1 RQ-VAE: 复用 Task #156 code-default (β=0.25, [32,64,256], sk=0.5) 或 Task #84 baseline (β=1.0, [64,128,256], sk=0)
- Stage 2 SID: 复用 Task #156 待生成 (`Instruments_t5_rqvae_code_default.npy`)
- Stage 3 T5: `num_epochs=200`, `early_stop=20`, `seed=42`, `batch_size=256`, `lr=1e-4`
- Stage 4 eval: `beam_size=20`, `topk_list=[5,10,20]` (跟 Task #84 baseline 一致)

**T5-base config (跟 HG-Rec 官方 T5-base 标准)**:
- `num_layers=12`, `num_decoder_layers=12`
- `d_model=768`, `d_ff=3072`, `num_heads=12`, `d_kv=64`
- `vocab_size=1025`, `pad_token_id=0`, `eos_token_id=0`
- `feed_forward_proj='relu'`
- `dropout_rate=0.1`
- params ≈ **220M** (vs Task #84 5.5M, ratio 40x)

**R11.3 决策: 用 Task #156 的 code-default SID 复用, 不重训 Stage 1**
- 单变量原则: 只换 T5, 其它不变
- Stage 1 + Stage 2 已经跑完, 不重复劳动
- 对照组: Task #84 (T5-small 5.5M, β=1.0)

---

## 3. 决策触发 (vs Task #84 baseline 0.1020)

| R@10 区间 | Δ vs baseline | Δ vs paper | 解读 |
|-----------|---------------|-----------|------|
| ≥ 0.118 | +15% | -10% | T5 容量是主因 ✅, 启动 paper Section 5.4 撰写 |
| [0.105, 0.118) | +3% ~ +15% | -20% ~ -10% | 部分缓解, 仍有数据/协议层因素 |
| [0.0973, 0.105) | -5% ~ +3% | -25% ~ -20% | 没效果, T5-base 也撞墙, 转向其他调查 |
| < 0.0973 | < -5% | < -25% | ❌ 异常, T5-base 训练失败或 eval bug |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| T5-base Stage 3 train (200 ep, 40x params) | **~93h** (4 days, GPU 0) |
| Stage 4 eval (T5-base 220M, beam=20) | ~2-3 min |
| **总计 (GPU 0 占用)** | **~93h** |
| **并行 Task #156 (GPU 3)** | ~2h45min, 早完成 |

**为什么 93h 估算**:
- T5-small 5.5M 训练 140 min (含 200 epoch + early stop)
- T5-base 220M = 40x params = 40x FLOPs ≈ 40x 训练时间
- 140 min × 40 = 5600 min = ~93h
- 可能更短 (early stop 触发在 ep 50-100) 或更长 (lr 调度没 warmup)

---

## 5. 风险与缓解

**风险 1**: T5-base 220M 训练 OOM on L40S 46GB
- 缓解: 模型 ~880 MB, Adam 优化器 ~3.5 GB, activations + bs=256 ≈ 5 GB → 总 ~10 GB, 不会 OOM
- 风险: `max_length=20` × `d_model=768` × batch=256 → activation 可能 30-40 GB → 接近 OOM
- 缓解: 用 `batch_size=128` 或 `gradient_accumulation_steps=2` 降低 activation, 等价 batch=256

**风险 2**: T5-base from-scratch 难收敛
- 缓解: paper 说 HG-Rec 就这么干的, R@10 = 0.13 能跑出来 → 走 paper 同样的 recipe
- 缓解: `early_stop=20` 保证不会 over-train, lr=1e-4 已经是合理值

**风险 3**: 93h 训练期间 GPU 0 长期占用
- 缓解: 用户已确认并行方案 (option C), 知悉 GPU 0 占用 ~4 天
- 缓解: R13v (R12) ckpt save 每 1000 step + 每 epoch 末, 防止中途崩溃
- 缓解: 如果中途发现 NO-GO 信号, 可 kill + 立即释放 GPU 0

**风险 4**: T5-base from-scratch 220M 在 10K item 训练数据下不收敛
- 这是为什么 user 选 C 而不是单跑 T5-base: Task #156 (T5-small) 是平行的诊断对照组, 如果 Task #156 R@10 = 0.105 (plateau) 且 Task #157 R@10 = 0.102 (无提升) → 说明数据/协议层, 不是容量
- 反向: Task #157 R@10 ≥ 0.118 → 容量论确认, paper-aligned recipe 阻塞点定位

---

## 6. 完成度跟踪

- [x] Task #156 Stage 1 best_loss ckpt 落盘 ✅ (2026-07-24 20:51)
- [ ] Task #156 Stage 2 SID codebook (in_progress, GPU 3)
- [ ] Task #156 Stage 3 T5-small train (in queue)
- [ ] Task #156 Stage 4 eval (in queue)
- [ ] **Task #157 Stage 3 T5-base 220M train (待 Task #156 Stage 2 完成, GPU 0)**
- [ ] Task #157 Stage 4 eval
- [ ] 写 verdict `verdicts/task157_t5_base_capacity_result.md`
- [ ] 写 composite verdict (Task #84+#156+#157 对比) `verdicts/task157_capacity_unlock_synthesis.md`
- [ ] 更新 loop.md §16 (R8 清理)

---

## 7. 关键决策点 (R11.3)

| 决策 | 选择 | 理由 |
|------|------|------|
| Stage 1 SID 来源 | 复用 Task #156 (code-default) | 单变量原则, 只换 T5 |
| T5-base 初始化 | From-scratch (不预训练) | HG-Rec paper 同样做法, paper-aligned |
| Stage 4 eval beam_size | 20 | 跟 Task #84 baseline 一致 |
| batch_size Stage 3 | 256 | 跟 baseline 一致; OOM 风险用 gradient_accumulation 缓解 |
| R12 ckpt save | 每 epoch 末 + 删旧 (save_limit=1) | R12 硬规则 |
| 并行 Task #156 | ✅ GPU 3 同时跑 T5-small 5.5M | user 选 C |
| early_stop patience | 20 (跟 Task #84 一致) | paper default |
| 数据并行 degree | 1 (无 DDP) | 单 GPU, 简化 |

---

## 8. 后续 (ROI 排序)

1. **等 Task #156 Stage 4 落盘** (~2h45min 内)
2. **Task #157 T5-base 220M 训练** (93h, GPU 0 单独占用)
3. **Composite verdict**: Task #84 baseline + Task #156 + Task #157 三方对比, 验证 capacity lift 是否有效
4. **Paper Section 5.4 候选输入**: HG-Rec code-faithful recipe + T5-base capacity unlock 双因素分析
5. **若 Task #157 R@10 ≥ 0.118**: paper-aligned recipe 完成, 进入下一 phase (升级数据集到更大的? 改进 SID quality?)

---

## 9. 结论 (待写)

result: (待 Stage 4 R@10 落盘后 fill)
