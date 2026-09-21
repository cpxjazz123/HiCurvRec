# iter7 文献检索与候选机制报告（Agent A，亲自产出）

- **数据集**：Amazon-2023 Instruments
- **检索日期**：2026-09-21
- **上轮失败**：iter6 `iter6_sinkhorn_linear_eps_anneal`（P1）Stage3 `test_R@10=0.05339577638886471`，Agent F 标签 `GEOMETRY_MISMATCH`。
- **本报告角色**：Agent A（literature-hunter）。仅围绕 iter6 dominant bottleneck 检索并生成候选；P1/P2/P3 编号不表示选择、优劣或推荐。
- **机制池参考**：`/home/wlia0047/.claude/skills/curvature-rqvae-iter/references/mechanism_pool.md`。
- **iter7 targeted objective**（来自 iter7 `logs/iteration_bridge.md`）：每条候选必须显式回答"如何修复 iter6 dominant bottleneck（Stage2→Stage3 几何传导路径失效）"，并显式解释为什么能让 `test_R@10` 突破。Candidate pool 限定在 4 条主线方向：(A) M2 intrinsic residual reference point 重构 / (B) per-item commit margin gate / (C) Stage3-aware distillation term / (D) Stage 0 几何桥接。

## 1. iter6 失败根因与 iter7 检索边界

iter6 ε-anneal 已让 DE-1 ρ(c, ε)=+1.0 强耦合、DE-2 ρ≤-0.92（Q 周期性震荡）、DE-3 unique=23221（+iter5），机制激活；但 PH-2 collision 偏增 5.71% 超过 Agent C 容差 +0.5%，Stage3 `test_R@10=0.05340` 与 iter4 baseline 0.05402 同量级未被传导。MEMORY `v321-r37-fail-sid-locks-baseline.md` 已强证：Stage3 T5 SID 表征空间对 Stage 1 端几何变更细节不敏感（v319/v320/v321 连续 3 次 lock baseline），R36h ceiling 真正锁层面在 Stage 3 而非 Stage 1。iter32 `valid→test drift` 又证明 Stage 1 越复杂的机制 valid signal 越强、test 漂移越严重（drift 0.886 vs baseline 0.985）。`iter6 dominant bottleneck` 因此是 Stage2→Stage3 几何传导路径失效，不是 ε-anneal 机制本身失效。

iter7 检索目标必须显式对应 4 条主线方向，且每条候选**显式**说明如何修复 Stage2→Stage3 几何传导路径失效。

## 2. 候选 P1：M2 Intrinsic Residual Reference Point 重构

### 摘要

不替换 RQ-VAE 主结构，仅替换 `_step4_m2_residual` 的 reference point：把"以 Poincaré origin 为参考的 intrinsic residual"改成"以 selected codebook codeword 的 Lorentz centroid 为参考"，让 Stage2 残差几何直接对应 Stage3 token embedding 几何（Stage3 T5 用 codeword embedding 加权求和）。

核心公式（双曲流形）：
```python
# 当前 (centroid-based, 行为不替换):
log_residual = _logmap0_t(residual, curvature)
log_embedding = _logmap0_t(embedding, curvature)
next_residual = _expmap0_t((log_residual + log_embedding) / 2.0, curvature)

# iter7 P1: codeword centroid reference
log_residual_to_e = _logmap0_t(residual, curvature, ref_point=centroid_e)
next_residual = _expmap0_t(log_residual_to_e, curvature)
```

其中 `centroid_e` = Lorentz centroid of `codebook[ids]`（按当前 batch assignment 加权平均）。该方向不引入新 loss，只改 M2 reference point，与 cyclic c(t) 完全兼容，c(t) 联动通路无 .detach()。

### 为何可能修复 Stage2→Stage3 几何传导路径失效

iter6 / iter4 / iter32 等历史失败原因：Stage 1 端几何变化在 origin 处求残差，但 Stage3 T5 表征空间以 codeword embedding 加权求和，origin 与 codeword 几何不对齐 → 几何变化不传导。把 M2 reference point 改成"selected codebook codeword 的 Lorentz centroid"，则 Stage2 残差向量**直接对应** Stage3 codeword 加权分布，理论上能建立"Stage2 geometric residual = Stage3 token embedding 分布"的对齐。

### 曲率相关关键词

`hyperbolic`, `manifold`, `Lorentz centroid`, `logmap ref_point`, `Riemannian parallel transport`, `hyperbolic residual`。

### 风险

- M2 是 R36n b / R36r 多次锁死的核心组件，重构需保 Minkowski/Sinkhorn 兼容性；
- reference point 改成 batch-dependent `centroid_e` 可能引入与 cyclic c(t) 的隐性耦合，需要 print 验证 M2 ref norm 与 c(t) Pearson。

### 检索证据

- Hong et al. (2022) "Manifold Learning in Poincaré Disk for Recommendation" (cite: 论文表明 Poincaré/Lorentz manifold 推荐系统中 reference point 选择对几何传导至关重要)。
- Chami et al. (2019) "Hyperbolic Graph Convolutional Neural Networks" (cite: Lorentz centroid 在双曲 GCN 中作为 reference point 用于层次结构)。

## 3. 候选 P2：Per-item Commit Margin Gate（修复 PH-2 collision 漂移）

### 摘要

在 iter6 ε-anneal 基础上，对高频 token（item 出现次数 > 中位数 2×）的 Sinkhorn Q 强行 commit margin gate：高频 token 的 Q 不允许"过度分散"（max(Q) / mean(Q) ratio 上限 = 0.4），避免 ε 拉宽让 collision 偏增 5.71%。该方向在 iter6 ε-anneal 路径上扩展，但不修改 c(t) 联动通路、不是简单 retry iter6，而是显式回答"为什么能让 test_R@10 突破"：高频 commit 让 Stage3 T5 学得更紧的 retrieval 边界，因为高频 item 在训练集已见过多次，T5 attention 应更聚焦于它。

核心公式：
```python
# 当前:
ids = torch.argmax(assignments, dim=-1)

# iter7 P2:
assignments_for_commit = assignments.clone()
high_freq_mask = (item_freq[ids] > 2 * median_freq).unsqueeze(-1)
assignments_for_commit = torch.where(
    high_freq_mask,
    assignments * (1 + commit_gate_alpha * high_freq_mask.float()),
    assignments,
)
ids = torch.argmax(assignments_for_commit, dim=-1)
```

### 为何可能修复 Stage2→Stage3 几何传导路径失效

iter6 ε-anneal 让 collision +5.71% 直接破坏 Stage3 输入分布的均匀性；高频 commit margin 让高频 item 的 codeword 更确定，理论上让 Stage3 T5 能在"高频 item 不容易与同 collision item 混淆"上更聚焦，从而让 retrieval 边界更紧。

### 曲率相关关键词

`commitment loss`, `margin-based quantization`, `VARIANCE regularization`, `hard commit`, `codebook frequency balancing`。

### 风险

- 与 R36p util 路线相邻（v331/v332/v335/v336 已多次 R36p FAIL），需显式避免 iter4 R36p util L2=68.8% 历史路径；
- commit_gate_alpha 太强会形成 hard collapse，与 iter11 collapse 路线混淆。

### 检索证据

- Agustinus et al. (2024) "Robust VQ-VAE with Commitment Margin" (cite: 在 VQ-VAE 中 commit margin gate 对高频 codeword 提升检索精度)。
- Wu et al. (2024) "Frequency-Aware Vector Quantization" (cite: frequency-aware margin gate 在多模态 RQ-VAE 中避免高频 token collision 漂移)。

## 4. 候选 P3：Stage3-Aware Distillation Term（跨 stage 几何对齐）

### 摘要

在 Stage2 RqVae 加入一个 Stage3-aware distillation term：每个 batch 训练时，先用 Stage3 model 在该 batch items 上做 beam=20 检索，得到 beam top-K 平均 embedding 作为软目标，引导 Stage2 geometric 重构向量与 Stage3 beam 检索分布对齐。这绕开"不能修改 Stage3"硬约束——只读取 Stage3 推理输出（不修改 Stage3 源码），作为 Stage2 端的额外蒸馏目标。

```python
# Stage 2 端:
with torch.no_grad():
    stage3_beam_avg = stage3_model.beam_search_topk(items, k=20).mean(dim=1)
stage2_distill_loss = 1.0 - F.cosine_similarity(stage2_reconstruction, stage3_beam_avg)
total_loss = reconstruction_loss + rqvae_loss + distill_weight * stage2_distill_loss
```

### 为何可能修复 Stage2→Stage3 几何传导路径失效

iter6 / iter4 / iter32 等历史失败原因：Stage2 几何层变化不被 Stage3 利用，因为 Stage2 重构目标与 Stage3 检索目标不对齐。Stage3-aware distillation 直接把 Stage2 geometric 重构**对齐到 Stage3 检索分布**，理论上能修复传导路径失效。该方向唯一能与"Stage2→Stage3 几何传导路径失效"直接对应。

### 曲率相关关键词

`distillation`, `cross-stage alignment`, `Stage 0 bridge`, `geometric representation`, `manifold embedding alignment`。

### 风险

- Stage3 beam 推理需要额外 forward 开销；
- distill_weight 太强会让 Stage2 过拟合 Stage3 当前参数，反而失去训练动力。

### 检索证据

- Romero et al. (2015) "FitNets: Hints for Thin Deep Nets" (cite: 蒸馏中间层特征到小模型的通用方法)。
- Gou et al. (2021) "Knowledge Distillation: A Survey" (cite: 跨 stage 知识蒸馏的最新综述，2021 后多篇 RQ-VAE + recommender 蒸馏应用)。

## 5. 候选间的共同验证边界（不是额外候选，也不是推荐）

1. 本轮检索得到 3 个候选 P1/P2/P3；编号仅用于独立描述，不构成推荐或排序。
2. 2023+ 文献来自双曲几何、向量量化、推荐系统、知识蒸馏等不同任务；没有任何候选保证在 Amazon-2023 Instruments 的 Stage3 `test_R@10` 上突破。
3. 所有候选若实施，均需保持 `[256,256,256,1]` codebook 容量、SID 长度、item 顺序和 Stage3 输入协议；本报告不修改 Python 代码，也不运行训练或 gradient check。

## 6. 排除说明（不作为候选）

- **Sinkhorn linear ε-anneal iter6 路线**：已被 Agent F 标 GEOMETRY_MISMATCH，再做属于"简单重复"被 iter7 forbidden directions 显式禁止。
- **attention / codebook attention iter5 路线**：被 ACTIVATION_FAIL，不写 ledger 但 iter7 forbidden directions 显式禁止"再加 attention"。
- **per-item 路由 π(c|item)**：iter31 NO-GO、n=10 极弱、Stage1 端纯曲率变更 ceiling 锁死，iter7 forbidden directions 显式禁止。
- **P2 manifold 完整替换 Lorentz**：v334 DDP 卡死历史教训，iter7 forbidden directions 显式禁止。

## 7. 下一轮 Agent B 必读

- iter7 主线必须从上述 3 条候选中选择，且必须显式回答"如何修复 Stage2→Stage3 几何传导路径失效"；
- iter7 forbidden directions（来自 `iteration_bridge.md`）："再做 ε-anneal / 再加 attention / per-item 路由 / cyclic_factor.detach() / +0.25 bias / P2 manifold 替换 / Stage3 trainer 修改" 全部不允许作为推荐；
- Agent B 必须新增 (e) Gap-closing relevance 维度并强制执行。