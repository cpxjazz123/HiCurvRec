# Task #392 / Issue #99 [方向C Gate2] 基于公式修复的 SID/metadata 多样性准入

**日期**: 2026-07-31
**触发**: Issue #99 [方向C Gate2] 基于公式修复的 SID/metadata 多样性准入
**前置**:
- Issue #96 task389 HypPreEncoder SID FAIL (unique=1/9922, collision=99.99%, util 1.56%/0.78%/0.39%, metadata kappa/scale var=0)
- Issue #93 task386 metadata uniform FAIL (on_off_diff=1.179 真信号, shuffle_diff=0 跟 #90 一致)
- 依赖: Issue #97 (κ-Stereographic 公式修复) + Issue #98 (product dist 修复) PASS

**任务**: 建立 Stage2 SID/metadata 多样性准入的 audit 框架, 验证 baseline collapse, 定义 #97/#98 PASS 后的 post-fix framework.

**验收 (per Issue #99 spec)**:
- PASS 条件: SID unique ≥ 9500/9922 + collision ≤ 0.20 + 三层 utilization ≥ 90% + metadata kappa/scale/conf variance 非零 + shuffle metadata distance 非零 + item↔SID 1-1 对齐
- FAIL 条件: SID unique < 9500 / metadata variance ≈ 0 / SID 不是由三层几何框架产生
- Gate 1: 可复用 #86 PASS 但必须记录依赖风险 (#87/#96 证明原 SID 坍缩)
- Gate 2: 本 issue 核心验收 (公式修复后跑 Stage2 SID + sid_metadata, 报告 8 项指标)
- Gate 3/4: ⏸ STOP per spec, 直到 Gate 2 PASS

**实施**: `scripts/task392_issue99_formula_based_sid_diversity.py` (R4 py_compile OK)

**当前状态**: ❌ FAIL — 依赖 #97/#98 修复, 当前 baseline 确认坍缩 (跟 #96 一致, baseline_collapse_confirmed=True). 

**next_action**: 等 #97/#98 修复 PASS 后, 跑 post_fix framework → 写 Gate 2 verdict → commit+push+comment+close.

**结果**: ⏳ pending — baseline audit 跑完, 等 #97/#98 修复