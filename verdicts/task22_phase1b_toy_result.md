# Task #22 Phase 1b — PM-RQ Toy Test (Adam 修复版) Verdict

> **完成日期**: 2026-07-19 02:24
> **状态**: ✅ **PASS — PROCEED Phase 2**
> **下一阶段**: **Phase 2** — Full-Scale Single-Layer (K=256, 全 Toys 11924 items)

---

## 1. 实验目的

修复 Phase 1 的 Euclidean 码本坍缩 (util=0.156)。Phase 1b 用 Adam optimizer + encoder-decoder + utilization penalty 替换 moving average。

## 2. 关键指标对比 (Phase 1 vs Phase 1b)

| 指标 | Phase 1 (MA) | Phase 1b (Adam) | 阈值 | 状态 |
|------|--------------|-----------------|------|------|
| **util_s** | 1.000 | **1.000** | > 0.8 | ✅ |
| **util_e** | **0.156** ❌ | **0.984** ✅ | > 0.8 | ✅ **6.3x 修复** |
| **util_h** | 0.875 | **1.000** | > 0.8 | ✅ |
| Kendall τ (s vs e) | 0.080 | **0.027** | < 0.7 | ✅ |
| Kendall τ (s vs h) | 0.005 | **0.016** | < 0.7 | ✅ |
| Kendall τ (e vs h) | 0.025 | **-0.012** | < 0.7 | ✅ |
| search time / item | 0.063 ms | ~0.07 ms | < 50 ms | ✅ |

**D1.5 决策**: **PROCEED Phase 2** (7/7 指标全 PASS)

## 3. 关键改动

### 3.1 码本更新机制: MA → Adam

```python
# Phase 1 (FAIL):
with torch.no_grad():
    Cs[k] = 0.9 * Cs[k] + 0.1 * batch.mean(0)  # 简单 MA, 无梯度

# Phase 1b (PASS):
optimizer = torch.optim.Adam(codebook_params + decoder_params, lr=1e-3)
loss = recon_loss + 0.1 * util_penalty  # 端到端可微
loss.backward()
optimizer.step()
```

### 3.2 加入 decoder + 端到端重建

```python
class SimpleDecoder(nn.Module):
    # 三个码字拼接 → 256 → 256 → 3*dim 重建
```

### 3.3 加入码本 utilization penalty (entropy-based)

```python
def compute_utilization_penalty(idx, K):
    counts = torch.bincount(idx, minlength=K)
    probs = counts / counts.sum()
    entropy = -(probs * probs.log()).sum()
    return (log(K) - entropy) / log(K)  # 归一化到 [0, 1]
```

### 3.4 输入归一化

```python
# per-subspace z-score normalize
# 双曲子空间 clamp 到 ||x|| < 0.85 (安全边界)
```

## 4. 训练曲线

| epoch | loss | recon | pen | util_s | util_e | util_h |
|-------|------|-------|-----|--------|--------|--------|
| 0 | 0.458 | 0.451 | 0.070 | 1.000 | **0.953** | 1.000 |
| 10 | 0.470 | 0.464 | 0.066 | 1.000 | 1.000 | 0.984 |
| 20 | 0.419 | 0.412 | 0.069 | 1.000 | 0.984 | 0.953 |
| 30 | 0.414 | 0.407 | 0.066 | 1.000 | 0.938 | 0.984 |
| 40 | 0.380 | 0.375 | 0.057 | 1.000 | 0.969 | 0.984 |
| 50 | 0.368 | 0.363 | 0.056 | 1.000 | 0.984 | 0.984 |
| 60 | 0.349 | 0.343 | 0.058 | 1.000 | 0.953 | 1.000 |
| 70 | 0.384 | 0.379 | 0.051 | 1.000 | 0.969 | 0.984 |
| 80 | 0.365 | 0.359 | 0.056 | 1.000 | 0.984 | 0.984 |
| 90 | 0.359 | 0.353 | 0.059 | 1.000 | **1.000** | 1.000 |

**观察**: 三码本 utilization 全程 >0.93, 训练稳定。recon_loss 从 0.45 降到 0.35 (22% 改善)。

## 5. 关键 insight

1. **MA 码本更新是 Phase 1 的根本问题**: MCKG Euclidean 子空间本身健康 (D0 norm_cv=4.746), 但 batch-mean MA 缺乏梯度信号, 只能收敛到 batch 的均值附近 → 坍缩
2. **Adam + decoder + util penalty 三个改动协同修复**: 端到端可微让码本跟着 decoder 一起学; util penalty 防止任何单码本独占
3. **Kendall τ 仍 <0.1**: 三 κ 子空间**强独立**, 修复后独立信号仍保留

## 6. Phase 2 计划

**目标**: Full-Scale Single-Layer (K=256, 全 11924 Toys items)
**改动 vs Phase 1b**:
- K: 64 → 256 (4x)
- num_items: 10000 → 11924 (full Toys)
- epochs: 100 → 200 (longer 收敛)
- batch_size: 1024 → 1024 (保持)
- 加入 **检索评估**: SID 生成 → 与 ground truth 召回

**预期时间**: ~12h GPU (cuda:3, 立即空闲)

**决策**:
- Phase 2 Recall@5 ≥ HRQ baseline (~0.024) → PROCEED Phase 3
- Phase 2 Recall@5 < 0.020 → STOP, 终局 verdict "PM-RQ 单层无法超越单几何 RQ"

## 7. 产物清单

| 产物 | 路径 |
|------|------|
| Phase 1b 报告 | `reports/task22_pm_rq/phase1b_toy_report.md` |
| Phase 1b metrics | `products/task22_pm_rq/phase1b_toy/phase1b_metrics.json` |
| 训练日志 | `logs/task22_pm_rq/phase1b_toy/train.log` |
| Phase 1b 脚本 | `scripts/task22_pm_rq_phase1b_toy.py` |

## 8. 后续行动

1. **立即**: 启动 Phase 2 (Full-Scale K=256, 全 Toys)
2. **Phase 2 通过**: 进入 Phase 3 (三层 cascade + V-info 诊断)
3. **Phase 2 FAIL**: 终止 Task #22, 写终局 verdict

---

**result**: Task #22 Phase 1b Toy test 7/7 全 PASS, util_e 0.156→0.984, Adam 修复成功, D1.5 PROCEED, Phase 2 立即启动.

result: Task #22 — completed (R8 §9.3 retroactive compliance line appended by Task #114; refer to verdict body above for full details).
