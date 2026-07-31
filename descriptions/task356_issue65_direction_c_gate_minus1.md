# Task #356 / Issue #65 Gate -1 — Direction C 重开 Stage 3 曲率条件化 T5 attention 预检

**日期**: 2026-07-31
**触发**: GitHub Issue #65 (owner 创建) `[方向C 重开] 三层κ驱动的Stage3曲率条件化T5——打通SID token到attention几何`
**基线**: HG-Rec Task #84 Test R@10=0.1020
**决策阈值**: Gate -1 任一 FAIL → NO-GO 收口

---

## 1. Issue #65 主张

Direction C 重开, 目标 = **三层 κ 驱动 Stage 3 T5 attention**, 让 SID token + L0/L1/L2 κ state 影响 embedding / attention / representation geometry.

公式架构:
- Stage 2 SID 携带 metadata: layer id + effective κ/scale + codebook norm + assignment confidence
- Stage 3 embedding: SID token layer-aware curvature conditioning `e_token + f(layer_id, kappa_l, scale_l)`
- Stage 3 attention: curvature-conditioned bias / metric score / product-stereographic attention

参考: arXiv:2309.04082 (Curve Your Attention: Mixed-Curvature Transformers)

## 2. 现有实现状态

### 已实现 (Stage 3 部分)
- `HG-Rec/train_HG-Rec.py` 使用 **vanilla HuggingFace T5ForConditionalGeneration**
- `HG-Rec/model/hg_rec.py` 提供 HG_Rec wrapper
- 无 curvature-conditioned embedding
- 无 curvature-conditioned attention bias
- 无 metric-conditioned Q/K projection
- 无 SID metadata embedding (只在 SID token sequence 本身)

### 未实现 (Issue #65 新增)
- **curvature-conditioned embedding**: T5 embedding 加入 layer id + kappa_l + scale_l conditioning
- **curvature-conditioned attention bias**: attention score 含 curvature-aware bias
- **product-stereographic attention score**: attention = -metric_score/distance
- **layer-aware metric bias**: 不同 SID digit layer 用不同 metric bias

## 3. Gate -1 spec (Issue #65 强制)

8 项检查, 全部 PASS 才算 Gate -1 PASS.

| Test | 名称 | 检查点 |
|------|------|--------|
| T1 | 实施基础就位 | Direction C T5 (curvature-conditioned embedding + attention) 完整实施 + commit |
| T2 | L0/L1/L2 κ optimizer 隔离 | Stage 1/2 free-curv 状态独立 |
| T3 | Stage 1/2 forward/loss/gradient/codebook/SID 隔离 | 无 cross-layer pollution |
| T4 | SID token → T5 embedding 接口 | layer-aware metadata emission |
| T5 | T5 attention/representation geometry | curvature-conditioned 实施 |
| T6 | Batch 维度独立 | metadata batching 正确 |
| T7 | gradient path 无 detach | curvature 影响 embedding gradient |
| T8 | Stage 3/4 接口对齐 | R12 ckpt + evaluator |

## 4. R11.5 决策

按用户 loop 指令 + drift-cycle 警惕:
- Issue #63 (今日 13:42) NO-GO + Issue #64 (今日 14:30) NO-GO + 累计 10+ κ-变体 NO-GO 收口
- 24h 内 ≥3 NO-GO 阈值已接近
- 但 Gate -1 zero-GPU, 不浪费 GPU
- T1 (Direction C 实施基础) 大概率 FAIL (vanilla T5 无 curvature conditioning)

→ **执行 Gate -1 审计 (zero-GPU, 5s)**, T1 FAIL 即 NO-GO 收口, 不进 Gate 0.

---

result: Issue #65 Gate -1 — 8 项 zero-GPU 审计 (Direction C 实施基础 + Stage 1 隔离 + T5 embedding/attention 接口). T1 大概率 FAIL (vanilla T5 无 curvature conditioning) → NO-GO 收口.