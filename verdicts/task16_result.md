# Task #60 执行结果与分析

> **任务名**: HHHH Stage 3 long-train (5120 step) 验证 R1 假设
> **完成日期**: 2026-07-16
> **状态**: ✅ 完成（R1 部分确认，训练预算有帮助但仍不足）
> **执行人**: Claude（loop tick 自动化恢复 + 执行）

---

## 1. 任务目标

验证 `task59` verdict 中的 R1 根因假设——"Stage 3 max_steps=2560 不够"。
- 如果 Stage 3 val/recall@10 ≥ 0.04 且 Stage 4 R@10 ≥ 0.05 → R1 确认，HHHH 可救活
- 如果 Stage 4 R@10 ∈ [0.02, 0.05) → R1 部分确认
- 如果 Stage 4 R@10 < 0.02 → R1 否证

---

## 2. 执行过程

| Stage | 状态 | 关键参数 | 时间 |
|-------|------|---------|------|
| Stage 3 训练 | ✅ **之前已完成** | max_steps=5120, patience=20, best val/recall@10=0.0419 @ step 3679 | 已在产物中 |
| Stage 4 推断 | ✅ 本场执行 | `tiger_inference_flat`, 19412 rows, A40 GPU | 24s |
| R@10 评估 | ✅ 本场执行 | `task388v4_s4_item_eval.py`, 19412 users | ~3min |

**恢复操作**：
- 从 `snap-research/GRID` 上游重新获取已删除的 `configs/` 目录
- 创建 `GRID → .` 符号链接以满足 eval 脚本路径依赖
- 创建 `.project-root` 标记文件以满足 `rootutils.setup_root()`

---

## 3. 关键结果指标

| 指标 | Task59 (2560 step) | Task60 (5120 step) | 提升 | Paper Baseline | 目标 |
|------|:----:|:----:|:----:|:----:|:----:|
| Stage 3 val/recall@10 | 0.0125 | **0.0419** ✅ | +235% | - | ≥ 0.04 |
| Stage 4 R@10 | 0.0117 | **0.0284** | +143% | 0.0973 | ≥ 0.05 |
| Stage 4 R@5 | 0.0086 | 0.0183 | +113% | - | - |
| Stage 4 NDCG@10 | 0.0073 | 0.0146 | +99% | - | - |

---

## 4. 分析与解读

### 4.1 主要发现

1. **R1 部分确认 — 训练预算有帮助但不充分**：
   - 从 2560→5120 step，Stage 4 R@10 提升 +143%（0.0117→0.0284），效果显著
   - 但 R@10=0.0284 仍远低于 baseline 0.0973（仅达到 29%）
   - 按任务决策表，处于 `[0.02, 0.05)` → R1 **部分确认**

2. **Stage 3 训练已达标**：
   - `val/recall@10 = 0.0419 ≥ 0.04` ✅ 说明 TIGER 模型在足够步数下可学会 HHHH SID
   - 但 Stage 3 val → Stage 4 实际 R@10 存在**显著差距**（0.0419 → 0.0284）

3. **与 Task59 对比的本质提升**：
   - R@10 从 0.0117→0.0284，跨越了"几乎随机"到"有信号"的临界点
   - 所有指标一致提升 2-2.4×，说明训练预算确实是主要瓶颈之一
   - 但差距表明还有**非训练预算问题**在限制

### 4.2 为什么 Stage 4 R@10 没到 0.05？

可能存在以下几种原因（对应 R2-R4 备选假设）：

| 假设 | 可能性 | 证据 |
|------|--------|------|
| **R2: SID recipe 不匹配** | 高 | HHHH 的 Poincaré ball SID 与 TIGER 的欧氏语义空间存在根本冲突 |
| **R3: monitor metric 稀疏** | 中 | `val/recall@10` 在训练早期接近 0，early stopping 可能在噪声中触发 |
| **R4: HHHH 算法端到端无效** | 高 | 即使两倍预算，R@10=0.0284 离 baseline 和论文目标都很远 |

### 4.3 与 Task62 的关联

Task #62 发现了一个关键洞见：**L2 归一化后欧氏/双曲/球面距离等价**。这暗示：
- HHHH 的 Poincaré ball SID（使用双曲距离训练 RQ-VAE）与标准 TIGER（使用欧氏 embedding）可能没有本质上的几何区分
- HHHH 的 R@10 低下可能不是"训练不够"的问题，而是 SID 本身的信息容量不足

---

## 5. 产物清单

| 产物 | 路径 |
|------|------|
| Stage 3 best ckpt | `products/task60/ckpts/task60_hhhh_best.ckpt` |
| Stage 3 训练日志 | `products/task60/train/task60_hhhh_s3_long/` |
| Stage 4 inference tensor | `products/task60/inference/pickle/merged_predictions_tensor.pt` |
| Stage 4 推断日志 | `products/task60/inference.log` |
| R@10 评估结果 JSON | `products/task60/eval/task60_hhhh_s4_eval.json` |

---

## 6. 后续建议

1. **R1 已确认有帮助，但不充分** → 无需在"更长的训练"方向继续投入
2. **建议优先调查 R2**（SID recipe 不匹配）：
   - HHHH 的 v3 RQ-VAE recipe（Poincaré + z-score + KMeans-on-z）生成的 SID 语义是否与 TIGER 的文本解码兼容？
   - 可通过分析 SID 的 token 分布和碰撞率初步判断
3. **放弃 HHHH 方向也是合理选择**：
   - 两倍预算只达到 baseline 的 29%
   - Task62 发现距离函数等价性，动摇了 HHHH 的几何分离理论基础
   - 建议与用户讨论是否继续 HHHH 路线或转向其他方向

result: Task #16 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
