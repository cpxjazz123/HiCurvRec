# Task #391 / Issue #98 [方向B Gate1] product component distance 尺度审计

**日期**: 2026-07-31
**触发**: Issue #98 [方向B Gate1] product component distance 尺度审计
**前置**:
- Issue #95 task388 product HypPreEncoder FAIL (step1 agree3=100%, util 4.69%/0.78%/0.39%)
- Issue #92 task385 product kmeans_init+β=0 FAIL (step1 agree3=100%)
- Issue #89 task378 product 分离诊断 PASS 6/6

**任务**: 审计 product-space 每 component 的 distance 公式和归一化尺度, 找出三 component 同步坍缩根因.

**验收 (per Issue #98 spec)**:
- 列出每层每 component 的实际 distance 公式、scale/weight 应用位置、tensor shape、broadcast 维度、detach 情况
- synthetic component-separation audit: 构造让三 component 理应选不同码字的输入, 验证 argmin ≠ agree3=100%
- 真实小批 no-training audit: report component argmin agreement, top-k margin, distance variance, mixing weight, κ/scale, codebook norm
- PASS: 定位并修复 ≥1 个 component distance/scale/broadcast/detach 问题, synthetic 三 component 可产生不同 argmin, 真实 step0/step1 agree3<95%, 无 NaN/Inf
- FAIL: agree3=100% / 无法证明 component distance 尺度可比 / 修复靠删除 product 结构

**实施**: `scripts/task391_issue98_product_component_dist_audit.py` (R4 py_compile OK)

**预期**: `_per_component_dist_sq` (hrqvae_free_curv.py line 346-389) 算 sqrt(Σ_m d²_m) 没有 per-component std 归一化 + init_emb 用全空间 kmeans (未按 component 切片) → 三 component 距离尺度不一致 + codebook entries 耦合 → argmin 始终指向同一码字.

**结果**: ⏳ pending — 跑 audit 后落 verdict