# Task #71 方案 E Stage 2.1 执行结果 — 单层三流形 RQ-VAE 训练可行性

> **任务名**: Task #71 Exp E Stage 2.1 — 单层 decoder 框架测试
> **完成日期**: 2026-07-17
> **状态**: ✅ **Stage 2.1 完成（单层验证）**，🟡 **Stage 2.2 多层验证待启动**
> **Decision**: **R8_CONFIRMED** (训练框架可行) + **R10_OBSERVATION** (单层无差异，几何需多层验证)

---

## 1. 任务目标

在 Exp A/B/D/F/G 全部 CONFIRMED/MARGINAL 的前提下，启动 Task #71 的最终验证：

**关键问题**：用三种残差几何（r_E / r_H / r_S）训练完整 RQ-VAE，Stage 4 推断的 R@10 是否不同？

Stage 2.1 是其中**第一阶段**（单层 decoder 框架测试），目的是验证：
1. 训练脚本能跑通三种几何而不崩溃
2. K-Means 初始化的 codebook 健康（131/256 使用率）
3. EMA 更新无 Task #62 那样的 codeword 0 坍缩

---

## 2. 实验设置

| 参数 | 值 |
|------|---|
| 数据 | `products/task62/item_embeddings.pt` (11924, 768) |
| 模型 | 单层 RQ-VAE：`decoder = Linear(768, 768)` |
| Codebook | K-Means 初始化 (MiniBatchKMeans, n_init=3) |
| 残差 | E: x-q ; H: poinc_log_map ; S: sph_log_map（**只在 forward 中计算，不参与 loss**） |
| 损失 | recon_loss + commitment_loss + codebook_loss |
| n_clusters | 256 |
| n_steps | 2000 |
| batch_size | 512 |
| lr | 1e-3 |
| Device | E→cuda:0, H→cuda:2, S→cuda:3（并行）|
| log_interval | 200 |

---

## 3. 关键结果

### 3.1 三种几何最终指标（全部相同）

| 残差 | train_loss | train_recon | val_recon | codebook_usage |
|------|-----------|-------------|-----------|----------------|
| **E** | 0.0005 | **0.000180** | **0.000188** | 131/256 (51.2%) |
| **H** | 0.0005 | **0.000180** | **0.000188** | 131/256 (51.2%) |
| **S** | 0.0005 | **0.000180** | **0.000188** | 131/256 (51.2%) |

### 3.2 Codebook 数值一致性

| Codebook pair | avg min pairwise distance |
|---------------|---------------------------|
| E ↔ H | **0.0001** |
| E ↔ S | **0.0001** |
| H ↔ S | **0.0001** |

三组 codebook 的 codeword 完全一致（差异 < 浮点精度）。

### 3.3 Codebook 健康度

- 训练后 codebook usage = **131/256 = 51.2%**（远好于 Task #62 EMA 的 1/256 坍缩）
- 验证了 **R3（K-Means 初始化）** 的修复效果
- 51.2% 表明：仍有近一半 codeword 是死码，但已足够为 SID 生成提供足够多样性

---

## 4. 物理解释

### 4.1 为什么三组结果完全相同？

**R10_OBSERVATION**: 单层 RQ-VAE 训练只用了 `decoder(q)` 重建 `x`，**几何路径的残差没有被反向传播**。

```python
# forward()
d = cdist(x, codebook)        # 只用欧氏距离选最近码字
idx = d.argmin(dim=1)         # 三种几何此处完全相同
q = codebook[idx]             # 选出来的 q 完全相同
r = compute_residual(x, q)    # 计算残差（E/H/S 不同），但 r 不参与 loss
x_hat = decoder(q)            # 只用 q 重建
recon_loss = MSE(x_hat, x)    # loss 只取决于 q
```

K-Means 初始化 → 三个 codebook 起点一致 → EMA 更新只依赖 `||x - q||²` 梯度 → 三组 codebook 始终保持数值一致。

**结论**：单层框架下，几何选择完全不影响结果。这是**正确的数学事实**，不是 bug。

### 4.2 几何差异什么时候才显现？

**Stage 2.2（多层 RQ-VAE）** 是几何差异真正起作用的阶段：

```python
# L=3 层 RQ-VAE
q1 = nearest(x, C1, metric='E')  # 三几何此层相同
r1 = compute_residual(x, q1, geom)  # E: x-q1, H: Mobius, S: log_map

# 关键差异：r1 输入到下一层
x2 = x + r1   # 或者直接 x2 = r1（取决于实现）
q2 = nearest(x2, C2, metric=?)  # 第二层用什么 metric？
r2 = compute_residual(x2, q2, geom)
# ...
```

如果不同几何产生的 r1 分布不同（如 norm、sparsity），就会导致 C2 的输入分布不同，最终 C2 学到的 codebook 不同 → 重建 MSE 不同 → R@10 不同。

### 4.3 与 Task #70 的关系

Task #70 已经在 L1 残差层面证明：**几何选择导致 argmin 决策不一致**（24.61% 不匹配 r_E vs r_H 选不同码字）。

但这仅是 L1 残差分析。Task #71 Exp E 是把这种不一致**级联到 L=3 层**后的累积效应。

---

## 5. 决策

**R8_CONFIRMED**（单层框架可行）

| 触发条件 | 实测 | 决策 |
|---------|------|------|
| 三种几何崩溃 | ❌（都跑完 2000 步） | — |
| EMA 坍缩（usage < 10%）| ❌（51.2%） | — |
| NaN/Inf | ❌（0） | — |
| 全稳定 | ✅ | **R8_CONFIRMED → Stage 2.1 完成** |

**R10_OBSERVATION**（单层 = 几何不变量）

| 指标 | 实测 | 含义 |
|------|------|------|
| 三个 codebook min dist | 0.0001 | 三组 codeword **数值完全一致** |
| train_loss 轨迹 | 完全重合 | 损失函数不依赖残差几何 |
| val_recon | 完全重合 | 重建质量相同 |

**解读**：单层 decoder + K-Means init 是 **几何不变量**，无法区分三种方案。要测差异 → **必须升到 L=3 层**。

---

## 6. P5 paper 影响

### 6.1 不能用单层结果下结论

之前担心的问题：
> "Poincaré 几何在 RQ-VAE 中是否有用？"

**单层实验无法回答这个问题**。P5 paper 必须明确写：

> "We evaluate the full L=3 RQ-VAE training pipeline to measure the cumulative effect of geometric residuals. Single-layer tests are invariant to the choice of geometry by construction."

### 6.2 Stage 2.2 多层 RQ-VAE 是必经之路

**预算估算**（基于 Exp G 数据）：
- 单流形单层训练（2000 steps）：~5min（实测）
- 三流形 L=3 训练：~50min × 3 = **~2.5h GPU**（并行）
- Stage 2.2 SID 生成：~10min × 3
- Stage 3 TIGER 训练：~1.5h × 3 = **~4.5h GPU**（并行）
- Stage 4 推断 + R@10：~5min × 3
- **总预算**：~7h GPU（3 卡并行）

### 6.3 决策阈值

| 结果 | 决策 |
|------|------|
| 三流形 R@10 差距 < 1% | ❌ **R10_FAILED**：几何在 RQ-VAE 中无实际价值 → P5 放弃三流形主题 |
| 三流形 R@10 差距 1-5% | ⚠️ **R10_MARGINAL**：写入 paper 作为 ablation，无强 claim |
| 某个几何 R@10 > baseline +5% | ✅ **R10_CONFIRMED**：作为 P5 main result |

---

## 7. 累计 Task #71 结论

| 方案 | 结论 | 状态 |
|------|------|------|
| **A** (形式) | R1_CONFIRMED (cos < 0.95) | ✅ |
| **B** (probe) | R2_MARGINAL (diff 3.77%) | ⚠️ |
| **D** (L2 量化) | R4_CONFIRMED (match 19%) | ✅ |
| **F** (数值稳定) | R6_CONFIRMED (三几何梯度相同) | ✅ |
| **G** (计算成本) | R7_CONFIRMED (H/E=1.60×) | ✅ |
| ~~C~~ (聚类) | 跳过 | — |
| **E** Stage 2.1 (单层) | R8_CONFIRMED + R10_OBSERVATION | ✅ |
| **E** Stage 2.2+ (多层) | **待启动**，~7h GPU（3 卡并行）| 🟡 |

---

## 8. 产物清单

| 产物 | 路径 |
|------|------|
| 任务脚本 | `scripts/task71_exp_e_s21_train.py` |
| Codebook (E) | `products/task71/exp_e_s21/codebook_E.pt` (3.0 MB) |
| Codebook (H) | `products/task71/exp_e_s21/codebook_H.pt` (3.0 MB) |
| Codebook (S) | `products/task71/exp_e_s21/codebook_S.pt` (3.0 MB) |
| History (E/H/S) | `products/task71/exp_e_s21/history_{E,H,S}.json` |
| 训练日志 | `logs/task71_exp_e_s21/run_{E,H,S}.log` |
| Verdict | `verdicts/task71_exp_e_s21.md` |

---

## 9. 下一步

**启动 Stage 2.2：L=3 层三流形 RQ-VAE**

具体步骤：
1. 写 `scripts/task71_exp_e_s22_train_multilayer.py`
2. 实现 cascade: q_1 → r_1 → q_2 → r_2 → q_3 → r_3 → final_recon
3. 残差几何只影响 r_1 → r_3 的计算
4. 三个几何并行训练（GPU 0/2/3）

启动条件：
- GPU 1 仍被 Task #68 Stage 3 占 → 不影响
- 三个空闲 GPU（0/2/3）可用
- 预计 ~50min 完成 Stage 2.2 训练

---

## 10. 完成度

- [x] 写 `scripts/task71_exp_e_s21_train.py` (含 `--residual_type` 参数)
- [x] 修复 `'float' object has no attribute 'item'` bug
- [x] 三几何各 2000 步单层训练（GPU 0/2/3 并行）
- [x] 决策：R8_CONFIRMED + R10_OBSERVATION
- [x] 写 verdict

**Stage 2.1 完成 — 框架可行，几何差异待 Stage 2.2 多层验证。**
