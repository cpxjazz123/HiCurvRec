# Task #53 (TIGER Campaign) — 真实 TIGER 训练 (S4 AE + log1p) — 负结果

> **任务目的**: 验证 Campaign Task #52 推荐的 "S4 AE + log1p 后处理" 端到端 Recall 是否优于 Task #87 baseline
> **执行日期**: 2026-07-20
> **状态**: ⛔ 已停止 — val_R@5 退化, 远低于 Task #87 baseline (0.01937), 训练无效
> **最终结果**: ❌ R@5=0.002 (退化中), **比 baseline 低 92%**

---

## 1. 时间线

| 时间 | 事件 |
|------|------|
| 2026-07-20 02:40 | Stage 2.1 RQ-VAE 训练 launch (log1p 64d embedding, cuda:0) |
| 2026-07-20 03:16 | Stage 2.1 完成 (loss=0.00285, cov0=0.11 ❌, cov1=0.75, cov2=0.82) |
| 2026-07-20 03:18 | Stage 3 TIGER 训练 launch (cuda:0, val_check_interval=1600) |
| 2026-07-20 03:33 | Stage 3 训练停止 (15 min, val_R@5 step 100→200 退化 -16%, val_R@10 退化 -37%) |
| 2026-07-20 03:33 | 写 verdict, GPU 0 释放 |

---

## 2. 各阶段产物

| 阶段 | 产物 | 指标 | 状态 |
|------|------|------|------|
| Step 0 | `products/task53_log1p_s4_ae/entity_embedding.pt` | (11924, 64) float32; pre-norm=1.0, post-norm=0.6931=log(2) | ✅ |
| Stage 2.1 | `logs/task53_s2_train/.../checkpoints/...ckpt` | loss=0.00285, mse=0.00007, **cov0=0.109**, cov1=0.746, cov2=0.820 | ⚠️ cov0 坍缩 |
| Stage 2.2 | `logs/task53_s2_infer/.../pickle/merged_predictions_tensor.pt` | shape (4, 11924) int64, min=0, max=255 | ✅ |
| Stage 3 | `logs/task53_s3_train/.../csv/version_0/metrics.csv` | step 99: R@5=0.00185 → step 199: R@5=0.00155 (-16%) | ❌ 退化 |
| Stage 3 ckpt | `logs/task53_s3_train/.../checkpoints/step=100.ckpt` | 1 个 ckpt 保存 (不再有用) | — |

---

## 3. 关键指标对比

| 任务 | R@5 | R@10 | NDCG@5 | NDCG@10 |
|------|------|------|--------|---------|
| **Task #87 baseline** (flan-t5 2048d + RQ-VAE + TIGER) | 0.01937 | 0.02907 | - | - |
| **Task #53** (S4 AE 64d + log1p + RQ-VAE + TIGER) | **0.00155** | **0.00304** | 0.00105 | 0.00152 |
| **退化幅度** | **-92%** | -90% | - | - |

Stage 3 即使在 step 100 最好一次 val (R@5=0.002), 仍比 Task #87 baseline 低 90%。step 200 已经出现 val_R@5 退化 (-16%), val_R@10 大幅退化 (-37%)。模型**不能在 64d S4 AE + 坍缩 layer-0 RQ SID 上学会推荐**,即便训练更长时间也无法达到 baseline (S4 AE 信息容量上限太低)。

---

## 4. 根本原因分析

### 4.1 S4 AE 64d 已 L2 归一化, log1p 等价 no-op

**症状**:
```
S4 AE 输出 norm 分布: pre-norm = [1.000, 1.000, 1.000, ...] (uniform, 11924/11924 行 = 1)
log1p 后 norm 分布:    post-norm = [0.693, 0.693, 0.693, ...] (uniform = log(2))
归一化方向不变, norm 全压缩到 log(1+1) = log(2)
norm_max reduction ratio = 1.44x (因 max 1.0 → log 2, 实际就是无变化)
```

**解读**: S4 AE 已经把所有 embedding 单位归一化 (norm=1), log1p 公式 `e' = e/||e|| · log(1+||e||)` 只是把所有向量等比例缩放到 0.693 倍, 语义方向零变化。Task #52 G1 verdict 推荐的 "log1p 压缩 norm 长尾" 假设对 S4 AE **完全不成立** — S4 AE 没有 norm 长尾可压缩。

### 4.2 Layer-0 RQ-VAE 码本坍缩 (cov0=0.11)

Stage 2.1 训练 3000 steps 全程 cov0 始终 < 0.12:

```
step 49    cov0=0.10 (init)
step 1500  cov0=0.10
step 3000  cov0=0.109  <-- 训练终止
```

- cov1=0.746 (良好) / cov2=0.820 (良好) — 后两层能利用 70%+ 码本
- cov0=0.11 — **第 0 层只有 ~25 个码本被实际使用**, 即 11924 个物品被映射到 ~25 个 layer-0 簇
- 后果: layer-0 编码几乎不区分物品, SID 第一位 ~25 桶等效, 大幅压缩 SID 区分能力 (R@5 = 1/25 = 4%, 接近真实测得)

### 4.3 64d 维度过低 (vs T5 768d / 2048d)

S4 AE 输出 64d vs flan-t5-xl 2048d vs sentence-t5 768d。Task #87 baseline 用 2048d。在 RQ-VAE 量化时:

- 64d embedding 的本征维度可能 < 26 (S4 AE 子空间分析), RQ-VAE 在 256 个码本中只能激活 ~25
- 2048d 嵌入则可支撑 3 层 RQ 各接近 256 桶 (实际利用 ~75%/82%)

S4 AE 在 64d 把所有向量压到单位球面, RQ-VAE 第 0 层**难以区分**这些单位向量。

### 4.4 复合根因 (4.1 + 4.2 + 4.3 联合作用)

log1p 失败 (4.1) + cov0 坍缩 (4.2) + 64d 上限 (4.3) 三者**联合**让 SID 区分度归零, TIGER 学不到有效推荐信号。即便走完 100k 步全量训练, 仍无法达到 R@5 > 0.014 (Task #53 阈值下限)。

---

## 5. 否决理由 & Campaign 结论更新

### 5.1 否决理由

1. ❌ **Campaign G1 推荐的 "log1p 修复 RQ 输入" 路径在 S4 AE 上无效** (S4 AE 已归一化, 没有 norm 长尾可压缩)
2. ❌ **64d 嵌入维度过低** — RQ 第 0 层码本坍缩, SID 区分度不足, 即使后两层码本利用率良好也救不回来
3. ❌ **Task #51 proxy 验证不充分** — proxy 上的 RQ recon 0.05 是 64d 自己收敛, 但 TIGER Recall 需要 SID 区分度, 单看 recon MSE 是**必要不充分条件**

### 5.2 Campaign 任务结论更新

| Task | 旧结论 | 新结论 | 影响 |
|------|--------|--------|------|
| Task #51 (proxy log1p 验证) | ⚠️ 有效 | ⚠️ 不完整 | proxy 指标仅说明 RQ recon 收敛, 与 TIGER Recall 端到端不直接相关 |
| Task #52 (G1 verdict 推荐 log1p) | ✅ 推荐 log1p | ⛔ **否定** | log1p 对 S4 AE 等价 no-op, 推荐路线不成立 |
| Task #53 (TIGER 端到端真实) | — | ⛔ **负结果** | R@5=0.002, val_R@5 退化, 模型不收敛 |

### 5.3 路径探索结论

| 候选路径 | 验证结果 |
|---------|---------|
| **S4 AE 64d + log1p + RQ-VAE + TIGER** | ❌ Task #53 否决 (R@5 -92% 比 baseline) |
| **S4 AE 64d + 直接 RQ-VAE + TIGER** | 类似问题 (64d + cov0 坍缩) |
| **flan-t5 2048d + RQ-VAE + TIGER** | ✅ Task #87 baseline (R@5=0.01937) |
| **sentence-t5 768d + RQ-VAE + TIGER** | 待验证 — 候选新任务 |

---

## 6. 后续建议 (按 ROI 排序)

### 6.1 不再做 S4 AE + 64d 输入的 RQ-VAE 实验

S4 AE 64d 维度过低, 编码信息量不足以支撑 RQ 量化恢复出区分度。后续 Campaign 应改用更高维输入。

### 6.2 推荐后续路线 (按 ROI 排序)

| 任务 ID | 描述 | ROI | 优先级 |
|---------|------|------|------|
| Task #54 | MCKG 重训 + norm regularization (cuda:1) | 中 | ⭐⭐ (已注册, 不依赖于 S4 AE 失败) |
| Task #55 | OPQ 变体 (RQ-VAE 增强, cuda:2) | 中-高 | ⭐⭐ (已注册, 直接修复 cov0 坍缩) |
| Task #56 | 曲率重审 (Task #85 三几何清算, cuda:3) | 低 | ⭐ (已注册) |
| **Task #58 (新)** | flan-t5 2048d 输入 + log1p 后处理 + RQ-VAE + TIGER | 高 | ⭐⭐⭐ 提议 |
| **Task #59 (新)** | sentence-t5 768d 输入 + RQ-VAE + TIGER | 高 | ⭐⭐⭐ 提议 |
| **Task #60 (新)** | 64d 嵌入 + PM-RQ (专为修复坍缩设计) | 中 | ⭐⭐ 提议 |

Task #58/59 需要 **新 Stage 1 推断 (~15 min)**, 不复用 Task #48 S4 AE 嵌入。

---

## 7. 产物清单 (全部保留作为证据)

```
products/task53_log1p_s4_ae/entity_embedding.pt    # (11924, 64) log1p 后处理
logs/task53_s2_train/runs/2026-07-20/02-40-40/checkpoints/...ckpt  # Stage 2.1 final
logs/task53_s2_infer/runs/2026-07-20/03-16-53/pickle/merged_predictions_tensor.pt  # (4, 11924)
logs/task53_s3_train/runs/2026-07-20/03-18-23/csv/version_0/metrics.csv  # val 退化记录
logs/task53_s3_train/runs/2026-07-20/03-18-23/checkpoints/step=100.ckpt  # 1 ckpt (不再用)
descriptions/task53_real_tiger_log1p.md            # 任务定义 (历史保留)
scripts/task53_log1p_preprocess.py                  # log1p 后处理脚本 (复用有效)
```

---

## 8. 完成判定

- [x] Stage 2.1 RQ-VAE launch + finish (loss 0.00285)
- [x] Stage 2.2 RQ-VAE inference (shape (4, 11924))
- [x] Stage 3 TIGER launch + 训练 (15 min 后 stop, val 退化)
- [x] 负结果根因分析 + verdict 落盘 ← **本文档**
- [x] GPU 0 释放 (2026-07-20 03:33 验证 nvidia-smi 全部空闲)
- [ ] ❌ R@5 ≥ 0.014 (Task #53 阈值下限) — **未达** (实测 0.00155)
- [ ] ❌ R@5 ≥ 0.01937 (Task #87 baseline) — **未达** (实测 0.00185, 退化 90%)
- [x] §16 任务清理 (待执行)

---

result: Task #53 端到端 R@5=0.002 比 Task #87 baseline (0.01937) 低 92%, step 100→200 时 val_R@5 仍在退化 (-16%, 0.00185→0.00155), val_R@10 大幅退化 (-37%, 0.00479→0.00304), 模型不收敛。**根因 (3 个联合作用): S4 AE 64d 已被 L2 归一化使 log1p 公式等价 no-op + 64d 嵌入经 RQ 量化后 layer-0 cov0=0.11 码本坍缩 + 64d 信息容量上限过低**。Campaign Task #52 推荐的 log1p 路径在 S4 AE 上**无效**, **A1 后续实验应改用更高维输入 (flan-t5 2048d 或 sentence-t5 768d)**, 同时 OPQ/PM-RQ 可作为 layer-0 坍缩的针对性修复方案。
