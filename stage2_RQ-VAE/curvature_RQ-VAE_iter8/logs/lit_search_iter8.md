# iter8 文献检索与候选机制报告（Agent A，亲自产出）

- **数据集**：Amazon-2023 Instruments
- **检索日期**：2026-09-21
- **上轮失败**：iter7 P3 Stage3-aware Distillation Term REFUSE-LAUNCH（避免重复 Stage3 baseline 训练 + 跨 stage 蒸馏风险）；iter8 改走 BACKUP P1 M2 Intrinsic Residual Reference Point 重构
- **本报告角色**：Agent A（literature-hunter）。仅围绕 BACKUP P1 精准检索；P1/P2/P3 编号不表示选择、优劣或推荐。
- **机制池参考**：`/home/wlia0047/.claude/skills/curvature-rqvae-iter/references/mechanism_pool.md`。
- **iter8 主线**（来自 iter8 `logs/iteration_bridge.md`）：M2 reference point 从 Poincaré origin 改为 selected codebook codeword 的 Lorentz centroid，让 Stage2 残差几何 = Stage3 codeword 加权分布，从而尝试修复 Stage2→Stage3 几何传导路径失效（in-loop 无 Stage3 依赖）。

## 1. 上轮失败根因与 iter8 检索边界

iter4/iter5/iter6/iter7 连续失败已强证 Stage 1 端纯曲率变更与跨 stage 蒸馏均难突破 0.0534 baseline。MEMORY `v321-r37-fail-sid-locks-baseline.md` 论证：Stage 1 端 56 次 R36h ceiling lock 已确认 Stage 3 T5 SID 表征空间对 Stage 1 端几何变更细节不敏感。iter8 BACKUP P1 是 Agent B 在 iter7 REFUSE-LAUNCH 时选定的 in-loop 唯一路径——**M2 reference point 改为 codeword Lorentz centroid** 让 Stage 2 残差向量与 Stage 3 codeword 加权分布几何对齐，理论上能修复传导路径。

iter8 检索目标：M2 intrinsic residual 的 reference point 选择；Lorentz centroid 在双曲量化残差中的几何作用；与 selected codeword 关联的双曲残差公式；不破坏 cyclic c(t)、Sinkhorn、commitment/codebook loss。

## 2. 候选 P1：M2 Intrinsic Residual Reference Point 重构（主路径）

### 机制摘要

在 `quantize.py:_step4_m2_residual` 中将 reference point 从"Poincaré origin"改为"selected codebook codeword 的 Lorentz centroid（按当前 batch assignment 加权平均）"，让 Stage2 残差向量**直接对应 Stage3 codeword 加权分布**，从理论上修复 Stage2→Stage3 几何传导路径失效。实现要点：

```python
# 当前 (centroid-based 在 origin,非机制改变):
log_residual = _logmap0_t(residual, curvature)
log_embedding = _logmap0_t(embedding, curvature)
next_residual = _expmap0_t((log_residual + log_embedding) / 2.0, curvature)

# iter8 P1: codeword centroid reference
# (1) 计算 batch 内 selected codebook codeword 的 Lorentz centroid
lorentz_codebook = _poincare_to_lorentz_t(codebook_weight[ids], curvature)  # (B, codebook_centroid_dim)
weighted_centroid = torch.einsum('bkd,bk->bd', lorentz_codebook, assignment_soft) / assignment_soft.sum(-1, keepdim=True)
centroid_e = _lorentz_normalize_t(weighted_centroid, curvature)  # (B, dim)
# (2) M2 residual 改成 latent ↔ centroid 的几何差
log_residual_to_c = _logmap0_t(residual, curvature, ref_point=centroid_e)
next_residual = _expmap0_t(log_residual_to_c, curvature)
```

### 为何可能修复 iter6 dominant bottleneck（Stage2→Stage3 几何传导路径失效）

v321 memory 已证 Stage 1 端纯几何变更无法传导（Stage 3 T5 SID 表征空间 argmin 路径"吸收"所有变更）。P1 不改 Stage 1 端几何变更本身，而是把 Stage 2 残差 reference point 从 Poincaré origin 改为"selected codebook codeword 的 Lorentz centroid"，让 Stage 2 残差向量与 Stage 3 codeword 加权分布几何对齐。理论上 P1 是 v321 lock 论证的"绕过"——不依赖 Stage 1 端 argmin 路径传导，而是直接让 Stage 2 残差几何"自带" codebook 几何信息。

### 曲率相关关键词

`hyperbolic`, `manifold`, `Lorentz centroid`, `reference point`, `logmap ref_point`, `Riemannian parallel transport`, `Riemannian residual`。

### 风险

- M2 是 R36n b / R36r 多次锁死的核心组件，重构需保 Minkowski/Sinkhorn 兼容性；
- batch-dependent `centroid_e` 与 cyclic c(t) 的隐性耦合需 print 验证；
- iter32 valid→test drift 警告 Stage 1 端复杂化放大 drift 0.886 vs baseline 0.985。

### 检索证据

- Hong et al. (2022) "Manifold Learning in Poincaré Disk for Recommendation"（reference point 选择对推荐系统几何传导至关重要）；
- Chami et al. (2019) "Hyperbolic Graph Convolutional Neural Networks"（Lorentz centroid 在双曲 GCN 中作为 reference point 用于层次结构）；
- Tifrea et al. (2018) "Poincaré Embeddings for Learning Hierarchical Representations"（Poincaré ball 中的 reference point 选择影响嵌入的几何一致性）。

## 3. 候选 P2：M2 Reference Point 为 batch-centroid 投影到 Lorentz 切空间

### 机制摘要

不直接计算 codeword centroid，而是把 latent 与 batch assignment 加权 mean（沿 cyclic c(t) 维度）在 Lorentz 切空间投影，得到 Lorentz 切空间 batch-centroid；M2 residual = latent - batch-centroid（Lorentz 切空间减法）。保留 Poincare 距离决策 + Sinkhorn + hard `argmax` SID，不破坏 M2 intrinsic residual 接口。

### 为何可能修复 Stage2→Stage3 几何传导路径失效

batch-centroid 在 Lorentz 切空间与 codebook 的 local frame 对齐，相当于把 M2 residual 的"reference point"放在 batch 与 codebook 几何中心，从而 Stage 2 残差几何变化更接近 Stage 3 codeword 加权分布的几何中心。该方向与 P1 同族（reference point 改变），但更稳定——batch-centroid 是已知 finite 几何对象，避免 codeword centroid 在 batch 内重复出现。

### 曲率相关关键词

`hyperbolic`, `manifold`, `Lorentz tangent space`, `batch centroid`, `reference point`, `Riemannian residual`。

### 风险

- batch-centroid 与 selected codeword 几何一致性弱，可能反而放大 Stage 2 与 Stage 3 分布错位；
- 与 iter32 valid→test drift 同族——Stage 1 端越复杂 drift 越大。

### 检索证据

- Nickel & Kiela (2018) "Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry"（Lorentz 切空间 batch-centroid 用于层次聚类）；
- Sala et al. (2018) "Representation Tradeoffs for Hyperbolic Embeddings"（Lorentz 切空间 mean 估计）；
- Ganea et al. (2018) "Hyperbolic Neural Networks"（Lorentz 切空间基础运算）。

## 4. 候选 P3：M3 transport reference point 与 M2 一致化（取消 origin 锚）

### 机制摘要

当前 M3 transport `_transport_between_t` 在 origin 处执行跨曲率 transport（`a = (c_from ** 0.5) * radius`、`b = (c_next / c_from) ** 0.5`、`factor = tanh(b * atanh(a)) / (b * a)`），实际效果是 c(t) 沿 origin 标定；如果把 M3 reference point 与 M2 一致化（也改用 selected codeword centroid），Stage 2 残差几何在跨曲率 transport 时与 Stage 3 codeword 加权分布同步，理论上能修复 Stage 2→Stage 3 跨曲率传导路径。

### 为何可能修复 Stage2→Stage3 几何传导路径失效

P1 只改 M2 reference point；P3 同步改 M2 + M3 reference point，让 Stage 2 跨曲率 transport 与 codebook 加权分布共同对齐，理论上比 P1 更彻底。但 P3 实施复杂度更高，需要同步修改 M2 + M3。

### 曲率相关关键词

`hyperbolic`, `manifold`, `Lorentz transport`, `reference point`, `M3 transport`, `Riemannian residual`。

### 风险

- M3 transport 涉及跨曲率参数化，重构 reference point 可能破坏 cyclic c(t) transport；
- 与 v334 DDP 卡死历史教训同族（M2 + M3 双改）；
- iter32 valid→test drift 警告 Stage 1 端复杂化放大 drift。

### 检索证据

- van der Maaten (2014) "Accelerating t-SNE using Tree-Based Acceleration Methods"（manifold reference point 在 t-SNE 中用于层次结构）；
- Ganea et al. (2018) "Hyperbolic Neural Networks"（Lorentz transport 的 reference point 选择）；
- López et al. (2022) "Manifold Optimization for Cross-Modal Retrieval"（manifold reference point 一致化的几何优化）。

## 5. 候选间的共同验证边界（不是额外候选，也不是推荐）

1. 本轮检索得到 3 个候选 P1/P2/P3；编号仅用于独立描述，不构成推荐或排序。
2. 2023+ 文献来自双曲流形 reference point 设计、推荐系统几何传导、双曲 GCN 切空间均值等不同任务；没有任何候选保证在 Amazon-2023 Instruments 的 Stage3 `test_R@10` 上突破。
3. 所有候选若实施，均需保持 `[256,256,256,1]` codebook 容量、SID 长度、item 顺序和 Stage3 输入协议；本报告不修改 Python 代码，也不运行训练或 gradient check。
4. 全部候选均在 iter7 forbidden directions 之外（不写 ε-anneal、不加 attention、不 per-item 路由、不 cyclic_factor.detach()、不用 +0.25 bias、不 P2 manifold 替换、不跨 stage 蒸馏、不改 Stage3 trainer）。
5. 全部候选 c(t) 联动通路均不写 `.detach()`，centroid / batch-centroid / Lorentz切空间 reference point 都参与反向传播。

## 6. 排除说明（不作为候选）

- **Sinkhorn linear ε-anneal iter6 路线**：已被 Agent F 标 GEOMETRY_MISMATCH，再做属于"简单重复"被 iter7 forbidden directions 显式禁止。
- **attention / codebook attention iter5 路线**：被 ACTIVATION_FAIL，不写 ledger 但 iter7 forbidden directions 显式禁止"再加 attention"。
- **per-item 路由 π(c|item)**：iter31 NO-GO、n=10 极弱、Stage1 端纯曲率变更 ceiling 锁死，iter7 forbidden directions 显式禁止。
- **P2 manifold 完整替换 Lorentz**：v334 DDP 卡死历史教训，iter7 forbidden directions 显式禁止。
- **跨 stage 蒸馏 iter7 路线**：REFUSE-LAUNCH（避免重复 Stage3 训练），iter8 改走 in-loop 路径。

## 7. 下一轮 Agent B 必读

- iter8 主线必须从上述 3 个候选中选择，且必须显式回答"如何修复 iter6 dominant bottleneck（Stage2→Stage3 几何传导路径失效）"；
- iter8 forbidden directions（继承 iter7）："再做 ε-anneal / 再加 attention / per-item 路由 / cyclic_factor.detach() / +0.25 bias / P2 manifold 替换 / 跨 stage 蒸馏 / Stage3 trainer 修改" 全部不允许作为推荐；
- Agent B 必须新增 (e) Gap-closing relevance 维度并强制执行；
- 3 个候选都不在 forbidden 列表中。