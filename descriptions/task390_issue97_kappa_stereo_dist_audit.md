# Task #390 / Issue #97 [方向A Gate1] κ-Stereographic distance 公式审计与最小替换验证

**日期**: 2026-07-31
**触发**: Issue #97 [方向A Gate1] κ-Stereographic distance 公式审计与最小替换验证
**前置**:
- Issue #94 task387 HypPreEncoder FAIL (step1 max_load=99.39%, util 1.56%/0.78%/0.39%)
- Issue #91 task384 kmeans_init + β=0 FAIL (step1 max_load=82.56%)
- Issue #43 task334 HypPreEncoder 机制 PASS (c=0.74 Ollivier mean)

**任务**: 找出当前 assignment distance 的公式 bug, 解释 #94 step1 99.39% 坍缩根因.

**验收 (per Issue #97 spec)**:
- 写出当前公式 vs Issue #47 unified vs arXiv:2405.13979 差异表
- synthetic tensor 数值审计 (κ→0, κ>0, κ<0, boundary)
- 真实 Stage1 输入 no-training assignment audit (L0/L1/L2 argmin spread)
- PASS: 找到并修复 ≥1 个公式/尺度/广播/detach 问题, step0/step1 argmin 不再单码字占比 >50%
- FAIL: 公式与 #47 完全一致 / argmin 单码字占比 >50% / 无法解释 #94 max_load=99.39%

**实施**: `scripts/task390_issue97_kappa_stereo_dist_audit.py` (R4 py_compile OK)

**预期**: 当前 HG-Rec utils.py `HVectorQuantization` 硬编码 c=1.0 + init_emb 在 train 模式 + uniform_(-0.01,0.01) codebook init + expmap0/proj_to_ball 强制 norm < 1.0, 多个尺度问题叠加 → step1 单码字坍缩根因.

**结果**: ⏳ pending — 跑 audit 后落 verdict