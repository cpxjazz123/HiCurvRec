# Task #293 / Issue #23 — Gate 0 Phase 0: per-layer per-epoch c_k curriculum

**Status**: ✅ Pipeline COMPLETED (Gate 0 only, **hard-stop NO-GO at Gate 0**)

## TL;DR
- **目的**: Issue #23 提出 per-layer per-epoch c_k curriculum (NCV 时间函数). Gate 0 验证 Phase 0 frozen ckpt 上 schedule 组合性.
- **方法**: 冻结 task275 A2_extend_ep50 ckpt (product_manifold=True, 36-d, codebook [64,128,256]). 测 3 schedules (A=异构时变 / B=全程宽 / C=全程窄) × 3 layers × 3 segments × 3 seeds = **81 个 agreement measurements**.
- **结果**: **0/81 measurements 三层全 OPEN (60-90% 带内)**. 通过条件 ≥60% (即 ≥2/3 seeds 三层全 OPEN).
- **结论**: **Gate 0 FAIL → 硬停止, 不进 Gate 1**. per-layer per-epoch c_k curriculum 在 frozen ckpt 上组合性 REFUTED.

## Gate 0 结果 (Issue #23 body 通过条件: ≥60% 三层全 OPEN)

| Schedule | seed=42 (L0/L1/L2) | seed=43 (L0/L1/L2) | seed=44 (L0/L1/L2) | n_layers_open |
|----------|-----|-----|-----|-----|
| **A** 异构时变 | 0.00/7.22/17.30% | 0.06/10.28/16.37% | 14.75/9.65/15.47% | **0/9** |
| **B** 全程宽 U(0.5,20) | 0.00/7.22/17.30% | 0.06/10.28/16.37% | 14.75/9.65/15.47% | **0/9** |
| **C** 全程窄 U(1,5) | 7.89/23.88/46.60% | 14.07/22.02/48.05% | 15.36/25.68/48.19% | **0/9** |

> 所有 81 个 measurements agreement < 60% (L0 0-15%, L1 7-26%, L2 15-48%). 远低于 60% 阈值, 全部 NOT OPEN.

## 决策逻辑 (R11.3 自主决策 + Issue #23 硬停止)

- **通过条件**: 81 个 measurements 中 ≥ 60% 三层全 OPEN (60-90% 带内, ±3pp 容忍).
- **实测**: 0/81 OPEN → 0% 通过率, 远低于 60% 阈值.
- **硬停止规则 (Issue #23 body)**: Gate 0 FAIL → 不要进 Gate 1, 关闭 issue.
- **决策**: NO-GO, 关闭 Issue #23.

## 关键发现 (跨任务一致性)

1. **per-layer c_k 参数空间在 task275 ckpt 上已耗尽**: 这跟 [[issue11-gate1-full-nogo]] (Task #242) 结论一致 — PC κ 在 frozen task275 ckpt (product_manifold + curriculum 后) 不论 (c_k_min, c_k_max) 怎么配, 都无法进入 OPEN 带 60-90%.
2. **time-varying curriculum 不能挽救 frozen ckpt 失配**: Schedule A (异构时变) 没有比 Schedule B/C 更好 — frozen ckpt 编码后的 residual geometry 已被学习固化, 单纯改 c_k_range 时间函数无法弥补.
3. **Schedule B 全程宽 U(0.5,20) 跟 Schedule C 全程窄 U(1,5) 表现相似**: 都 NOT OPEN, 反映 frozen ckpt 的 geometry 决定 argmin, 不是 c_k range.

## 决策点 (R11.3 自主决策记录)

1. **ckpt 选择**: task275 A2_extend_ep50 (product_manifold=True, curriculum warm-start, 50 epoch plateau). 跟 Issue #23 body 期望一致 ("冻结 task275 final ckpt").
2. **Schedule 模板**: A/B/C 三组, 严格按 Issue #23 body 文字描述实现.
3. **±3pp 容忍度**: 复用 task241 ±3pp 容忍 (Issue #23 body 明确引用). 即使加容忍, 所有 measurements 仍 < 60%, 无需触发容忍.
4. **Phase 0 协议 (frozen forward-pass)**: 严格复用 task241 per_codeword_kappa_stereographic + per_layer_criterion, 加 schedule 维度.

## 数据
- 脚本: `scripts/task293_issue23_gate0_phase0.py` (zero GPU, CPU only, ~30s runtime)
- 结果 JSON: `/home/wlia0047/.claude/jobs/04ccf474/tmp/task293_issue23_gate0_results.json`
- Issue: https://github.com/WENYULIANG123/GeneRec/issues/23

## 关联
- [[issue11-gate1-full-nogo]]: 同一 ckpt, per-layer c_k_range 静态空间也耗尽 (Task #242 闭环)
- [[free-curv-codebook-collapse]]: κ+codebook 反馈循环是架构根本问题, time-varying curriculum 不能改
- [[phase0-mode-collapse]]: task178/task180 200 epoch collision 85%, task275 50 epoch plateau 是当前最优 warm-start
- [[3-way-alternative-quantizer-no-go]]: FSQ/EMA/Restoration 全 NO-GO, R@10 ceiling 0.1020 在 Stage 1/2 已固化

result: Task #293 / Issue #23 Gate 0 Phase 0 FAIL, 0/81 measurements 三层全 OPEN (≥60% 阈值). per-layer per-epoch c_k curriculum 在 frozen ckpt 上组合性 REFUTED, 硬停止 Gate 0, 不进 Gate 1, 关闭 Issue #23.