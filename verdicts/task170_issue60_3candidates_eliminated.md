# Issue #60 (根因诊断 #59 mode collapse) — 3 候选全部排除, 根因在更深架构层 (2026-07-31)

## 任务

承接 Issue #58/#59 (#55/#56 mode collapse 确认独立于 α_l/scale_l/Riemannian 公式), 隔离验证 3 个候选根因:

| 候选 | 假设 | 验证方法 |
|------|------|---------|
| 1 | commitment loss β=0.25 权重过大 | β={0.25, 0.05, 0.01} 3-臂 smoke test |
| 2 | kmeans_init 没起作用 | norm 对比 + 距离分布 |
| 3 | mixed_curv_dist 缺乏梯度信号 | 替换为 pure Euclidean control |

## 结果

### 候选 1 (β): ❌ 排除

| β | loss (5 epoch) | SID 3-digit unique | 结论 |
|---|----------------|--------------------|------|
| 0.25 | 0.000229 | 0.01% (1/9922) | collapse (基线) |
| 0.05 | 0.000236 | 0.18% (18/9922) | 略好 (18x) 但仍 collapse |
| 0.01 | 0.000269 | 0.06% (6/9922) | 仍 collapse (β=0.01 等同关闭 commitment loss) |

→ 即使 commitment loss 权重降到 0.01 (实质关闭), encoder 仍找到 trivial 解. β 不是根因.

### 候选 2 (kmeans_init): ❌ 排除

**a) 函数定义正确**:
```python
def init_emb(self, data):
    centers = kmeans(data, self.n_e, self.kmeans_iters)
    self.embeddings.weight.data.copy_(centers)
    self.initted = True
```

**b) kmeans 真在工作**:
| 量 | 数值 |
|----|------|
| latent norm (data) | mean=0.1155 |
| cb_random norm | mean=0.0554 |
| cb_kmeans norm | mean=0.1140 ✅ (跟 data 对齐) |
| cb_random self-pairwise | mean=0.0770 |
| cb_kmeans self-pairwise | mean=0.0296 (更聚集, 跟 data 簇心对齐) |
| cb_random vs data centroid[0] | mean=0.1268 |
| cb_kmeans vs data centroid[0] | mean=0.0217 ✅ (远低于 random) |

→ kmeans_init 把 codebook 正确初始化到 data latent 分布 (norm + centroid 距离都对齐). 不是根因.

### 候选 3 (mixed_curv_dist 梯度信号): ❌ 排除

**Euclidean control** (task479): 把 mixed_curv_dist 替换成 pure Euclidean 距离, 5 epoch smoke test:
- loss = 0.000287
- SID 3-digit unique = **0.01%** (1/9922) — **仍坍缩**

→ 跟 #477 (mixed_curv_dist) 完全相同的 collapse 模式. 距离函数 (Euclidean vs mixed_curv) 不是根因.

## 综合诊断

**3 个候选全部排除**. Mode collapse 是 **更深架构层** 的问题, 不是 Issue #60 列出的 3 个候选.

## Issue #60 §验证标准解读

原文:
> 若候选3欧式control组依然坍缩：说明坍缩是这个特定encoder架构/commitment loss设计的通病，跟κ/混合曲率无关，应回头检查候选1(β)和候选2(kmeans_init)。

实际情况 (扩列验证):
- 候选 3 (Euclidean) 坍缩 → 排除 mixed_curv_dist
- 候选 1 (β=0.01) 坍缩 → 排除 commitment 权重
- 候选 2 (kmeans norm 对齐) → 排除 kmeans_init

**结论**: mode collapse 是 FreeCurvHRQVAE + HG-Rec 数据集 + Sinkhorn-disabled 训练协议的**系统性 bug**, 跟 κ/混合曲率/β/kmeans 无关.

## 深层根因候选 (R10 backlog follow-up, 超出 Issue #60 范围)

观察到的 collapse 模式特征:
1. **5 epoch 内就 collapse** (从 epoch 1 loss 0.001 → epoch 5 loss 0.0002)
2. **所有变化 (Euclidean/β/kmeans) 都快速收敛到同一 collapse 模式**
3. **decoder 把所有 input 重建得很好** (recon_loss 低), 但 encoder 输出聚集到单点

可能根因 (待后续诊断):
1. **Encoder 容量过大 / 数据分布退化**: 768→32 的 encoder 过强, 找到 trivial latent mapping
2. **Sinkhorn-disabled (sk_eps=0) 路径**: 没有平衡约束, codebook 自由坍缩
3. **Decoder 太强**: 重建 loss 主导, encoder 不需要分开 latent 也能重建
4. **Per-component 距离分割 + Euclidean sum**: 跨 component 距离独立计算, 容易坍缩
5. **数据集特性**: Musical_Instruments (9922 items) 可能在 32-d 空间有 trivial 1-NN mapping

这些都需要后续诊断 (R10 backlog 候选, **不在 Issue #60 范围内**).

## Issue 状态

| Issue | 状态 | 备注 |
|-------|------|------|
| #55 曲率感知优化器 | NO-GO (维持 closed) | 修复 + 3 候选根因排除, 方向失败确认 |
| #56 混合曲率乘积空间 | NO-GO (维持 closed) | 同上 |
| #58 实现审计 | CLOSED | 4 bugs 确认 + 修复 |
| #59 修复+重跑 | CLOSED | 修复正确, 方向仍 NO-GO |
| **#60 根因诊断** | **CLOSED** | **3 候选全排除, 根因在更深层** |

## 关联产物

| 类型 | 路径 |
|------|------|
| 候选 3 launcher | `scripts/task479_issue60_candidate3_euclidean_control.py` |
| 候选 1 launcher | `scripts/task480_issue60_candidate1_beta_sweep.py` |
| 候选 2 diagnostic | `scripts/task481_issue60_candidate2_kmeans_diagnostic.py` |
| 5 epoch ckpt | `products/task479/ckpt/` + `products/task480_beta0.05/ckpt/` + `products/task480_beta0.01/ckpt/` |
| 关联 verdict | `verdicts/task169_issue59_bug_fix_sanity_pass.md` + `verdicts/task170_issue59_fix_correct_direction_nogo.md` |

---
result: Issue #60 3 候选根因全排除 (Euclidean 也坍缩, β=0.01 也坍缩, kmeans 工作正常). mode collapse 是 FreeCurvHRQVAE+Musical_Instruments+Sinkhorn-disabled 训练协议的系统性 bug, 不在 Issue #60 列出的 3 个候选. Issue #55/#56 维持 NO-GO, 后续根因诊断超出本 issue 范围 (R10 backlog).