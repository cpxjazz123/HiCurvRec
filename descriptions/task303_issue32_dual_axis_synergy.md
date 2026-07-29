# Task #303 / Issue #32 — per-layer 异构 Codebook Transforms r_l + s_l + per-layer c_k range 双轴协同

**日期**: 2026-07-30
**状态**: 🟡 准备启动 Gate 0
**Issue**: [#32 per-layer 异构 Codebook Transforms + c_k range 协同](https://github.com/WENYULIANG123/GeneRec/issues/32)

---

## 1. 来源

承接 Issue #31 closure comment (2026-07-29 owner-verdict「码字几何是 Phase 0 mode collapse 关键, 后续候选必须在架构层」) + Issue #30 (per-layer 异构 Codebook Transforms r_l + s_l 唯一 Gate 1+2 PASS, Gate 3 Stage 4 GO +0.2%) + Issue #29 NO-GO + Issue #28 NO-GO + 12 方向 × 16 verdict NO-GO 收口.

**AI 自主决策依据 (R11.5)**: Issue #30 Gate 4 GO = 首个击败 baseline R@10 端点 (R@10=0.1022, +0.2%). Issue #32 是 owner-verdict 关键发现「码字几何是 Phase 0 mode collapse 关键」+ task242 per-layer c_k range 思路的下一协同机制. 双轴 (码字几何 + metric) 协同验证 R@10 增益.

---

## 2. 假设

**H1 (双轴协同贡献 R@10)**: task242 c_k_range 单独 Stage 1 FAIL (L0=23.44%); #30 r_l + s_l 单独 Gate 1+2 PASS. 双轴协同预期能让 L0 utilization ≥ 95% + R@10 > 0.1020.

**H2 (per-layer c_k range 不引入 commit loss 主导)**: #30 r_l + s_l 已让码字在不同几何尺度, c_k range 在异构几何上不再被 commit loss 主导.

**H3 (双轴异构不撞 baseline Stage 1 几何天花板)**: 双轴独立机制叠加, 预期突破 baseline Stage 1 几何天花板.

---

## 3. 实验设计

**自变量** (per-layer 异构 4 维):
1. per-layer r_l = [0.5, 1.0, 2.0] (沿用 Issue #30 思路但数值不同, 任务 #32 自有参数)
2. per-layer s_l = [1.0, 1.0, 1.0]
3. per-layer R_l = [I, I, I] (identity)
4. per-layer c_k_range_list = [(1.0, 5.0), (0.5, 20.0), (0.5, 20.0)] (task242 Arm A)

**受控因素**: baseline K=[64,128,256], e_dim=32, seed=42, Musical_Instruments 5-core, T5-mini 200 epoch.

**配置区别** (Issue #32 vs Issue #30):
- Issue #30: r_l=[0.1, 1.0, 10.0] + s_l=[2.0, 2.0, 2.0] + c_k_range=[(1,5),(0.5,20),(0.5,20)]
- Issue #32: r_l=[0.5, 1.0, 2.0] + s_l=[1.0, 1.0, 1.0] + c_k_range=[(1,5),(0.5,20),(0.5,20)]

Issue #32 故意把 r_l + s_l 调到中间值 (跟 #30 区分), 验证「r_l + s_l + c_k_range 双轴」是否独立于「r_l 极端值」贡献 R@10.

---

## 4. 阶段闸门

**Gate 0 (实现)**: scripts/task303_issue32_gate0_codebook_transforms_ckrange.py 提交 + 回归测试 (Issue #30 wrapper 在 r_l=[0.5,1.0,2.0]+s_l=[1,1,1]+c_k_range 输入下运行).

**Gate 1 (Stage 1 训练, GPU)**: 100 epoch, 通过条件 L0/L1/L2 util ≥ 90% + collision ≤ 0.20.

**Gate 2 (Sinkhorn 推断)**: 5 iter, 4-digit unique ≥ 9500.

**Gate 3 (Stage 3 T5 + Stage 4 eval)**: R@10 > 0.1020 → GO.

每一 Gate 失败 → hard-stop, 不跨 Gate.

---

## 5. GPU 资源

- Gate 1: GPU 0 (Stage 1 100 epoch, ~30 min)
- Gate 2: CPU (Sinkhorn 推断 cheap)
- Gate 3: GPU 2 (Stage 3 T5 + Stage 4 eval, ~120 min)

---

## 6. 关联

- [[task301-issue30-stage4-result]]: Issue #30 Gate 4 GO 🎉 (R@10=0.1022 +0.2%) — 唯一 PASS 端点
- [[task242-arm-a]]: per-layer c_k range Stage 1 FAIL (L0=23.44%) — Issue #32 协同对象
- [[cross-task-c-k-range-no-go-exhausted]]: 8 方向 c_k range NO-GO 收口, Issue #32 是协同升级
- [[issue32-dual-axis-synergy]]: Issue #32 body 来源

---

## 7. 物理产物 (待写)

- `scripts/task303_issue32_gate0_codebook_transforms_ckrange.py`
- `scripts/task303_issue32_gate1_stage1_train.sh`
- `scripts/task303_issue32_gate2_stage2_codebook.py`
- `scripts/task303_issue32_gate3_stage3_train.sh`
- `scripts/task303_issue32_gate4_stage4_eval.sh`
- `verdicts/task303_issue32_*.md`
- `products/task303/`