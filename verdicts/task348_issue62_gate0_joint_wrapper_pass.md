# Task #348 / Issue #62 Gate 0 — #30+#43 联合 wrapper 设计 + 5/5 sanity ✅ PASS

**日期**: 2026-07-31
**触发**: Issue #62 [放弃方向 C + #30+#43 联合 ablation] 27+ 方向 NO-GO 收口后唯一 ROI > 0 路径
**类型**: Issue #62 Gate 0 (零 GPU, low-cost sanity)
**状态**: ✅ **PASS** — 5/5 联合 wrapper sanity 通过, 联合 Issue #30 + Issue #43 不引入新 bug

---

## 1. 设计目的

**组合 Issue #30 + Issue #43 两个 GO 端点**:
- **Issue #30** (per-layer Codebook Transforms r_l=[0.1,1,10]+s_l=[2,2,2]): Stage 1/2 per-layer 异构 codebook 几何, R@10=0.1022 marginal GO
- **Issue #43** (HypPreEncoder expmap0 c=0.74): Stage 1/2 预量化双曲感知映射, R@10=0.1042 best GO ⭐

两者从未联合测试过. 假设两者在 Stage 1/2 范畴内互补增益.

---

## 2. 联合 wrapper 实施

**脚本**: `scripts/task348_issue62_gate0_joint_wrapper.py` (~280 行)

**架构**:
```
x (768d) → [Issue #43] HypPreEncoder(expmap0 c=0.74) → encoder
       → [Issue #30] per-layer Codebook Transforms (radius+rotation+scale)
       → hrq (RQ-VAE) → decoder → out (768d)
```

**实施方式 (R11.4 critical decision)**:
- 不修改 HG-Rec upstream (`HG-Rec/model/hrqvae.py`)
- 复用 Issue #43 wrapper (`HRQVAEWithHypPre`) 跟 Issue #30 wrapper (`PerLayerCodebookTransformHRQVAE`) 的核心组件
- 新建组合 wrapper `HRQVAEWithHypPreAndPerLayerTransforms`, 通过 monkey-patch 在 forward 调用前应用 per-layer transform matrix 到 vq_layers.embeddings.weight

**回归保证**:
- 所有变换 opt-in via flag (hyp_enabled, radius_list, scale_list)
- identity 配置 (hyp_enabled=False, radius=[1,1,1], scale=[1,1,1]) → 输出与 baseline 完全一致

---

## 3. Gate 0 Sanity 结果 (5/5)

| Test | 内容 | 实测 | 决策 |
|------|------|------|------|
| T1 | baseline HRQVAE forward pass | out.shape=(4,768), rq_loss=0.0297 (finite) | ✅ PASS |
| T2 | HypPre ON + per-layer identity | max\|out_wrapped - out_direct\|=0.0, ‖x_hyp‖_max=0.7243 (boundary=1.1625) | ✅ PASS |
| T3 | per-layer identity only (HypPre OFF) | max\|out_wrapped - out_base\|=0.0 (完全一致) | ✅ PASS |
| T4 | Both wrappers active (Issue #30 design + Issue #43) | mean\|out_wrapped - out_base\|=6.84e-3, indices [3,79] valid | ✅ PASS |
| T5 | monkey-patch 干净恢复 | max\|out_1 - out_2\|=0.0, baseline no leakage | ✅ PASS |

**关键观察**:
- T2/T3 的零差异验证: identity 变换与 baseline 数学等价 (无副作用)
- T4 的差异 (6.84e-3) 验证: 设计配置 (r=[0.1,1,10]+s=[2,2,2]+HypPre ON) 显著生效, 不是 identity
- T5 的零差异验证: monkey-patch 干净恢复, 多次 forward 一致, 无 baseline leakage

---

## 4. Gate 1 启动条件

Gate 0 PASS → 进入 Gate 1 (Stage 1/2 重新训练):

| Arm | 配置 | 预期 R@10 | GPU |
|-----|------|-----------|-----|
| Arm A | #30 端点 (baseline) | 0.1022 | 已知 ckpt |
| Arm B | #43 端点 (baseline) | 0.1042 | 已知 ckpt |
| **Arm C** | **#30 + #43 联合 (新)** | **待测 > 0.1042 目标** | 1 GPU (~3h Stage 1 + ~3h Stage 2) |
| **Arm D** | **#30 + #43 + K0=256 三联合 (新)** | **待测 > 0.1042 目标** | 1 GPU (~3h Stage 1 + ~3h Stage 2) |

**总 GPU 时间**: ~6h Stage 1/2 + ~3h Stage 3+4 = ~9h

**决策矩阵**:
- 任一 Arm R@10 > 0.1042 (+0.5pp) → 联合 ablation 突破天花板, GO
- R@10 > 0.1053 (+1pp) → 接近 task194_k0256 仓库最高, GO
- Arm C ≈ Arm B, Δ < +0.5pp → 维持 R10 backlog 候选 #5 (Stage 2 SID 几何健康化)
- Arm C < Arm B → #30+#43 联合 NO-GO, 27+ 方向全 NO-GO 收口

---

## 5. R11.5 透明决策

**选了**: 自主启动 Issue #62 Gate 0 (low-cost zero GPU 5/5 sanity)
**为什么**:
- Issue #62 已 owner-side accepted (issue body 写明 "本 issue 等 owner-side Stage 4 结果出来立即启动 Gate 0")
- Issue #158/#159 Stage 4 已完成 (commit 217710a 16:00:17Z), 条件已满足
- 联合 wrapper 是 Issue #30 + Issue #43 两个 PASS wrapper 的组合, 不引入新公式 bug
- 5/5 sanity 闭环, 验证组合不破坏 regression

**备选**: 等 owner 显式指示 → R11.4 不允许等 (Issue #62 自身已 issue 化)

**不可逆性**: 本 Gate 0 不修改 upstream, 不删除数据, 无需 dry-run. 实施文件是 wrapper script, 可随时 revert.

---

## 6. 物理产物

| 类型 | 路径 |
|------|------|
| Gate 0 wrapper 脚本 | `scripts/task348_issue62_gate0_joint_wrapper.py` (~280 行) |
| Gate 0 verdict JSON | `verdicts/task348_issue62_gate0_joint_wrapper.json` |
| Gate 0 verdict markdown | `verdicts/task348_issue62_gate0_joint_wrapper_pass.md` (本文件) |
| Description | `descriptions/task348_issue62_joint_ablation_gate0.md` |

---

## 7. R14 闭环

- Issue #62 Gate 0 5/5 sanity PASS ✅
- 联合 wrapper `HRQVAEWithHypPreAndPerLayerTransforms` 实施完成
- identity 配置与 baseline 数学等价, 设计配置与 baseline 显著不同
- 不引入新公式 bug (跟 #47/#55/#57 同模式检查)
- 下一步 Gate 1 Stage 1/2 重新训练 (~6h) 待 owner 拍板 (per Issue #62 §R11.4 critical decision on Stage 1 GPU)

---

result: Task #348 / Issue #62 Gate 0 — #30+#43 联合 wrapper 5/5 sanity PASS. 联合 wrapper 实施完成, 不引入新公式 bug, identity 配置与 baseline 数学等价, 设计配置与 baseline 显著不同 (mean diff 6.84e-3). 下一步 Gate 1 Stage 1/2 重新训练 (~6h, 2 arms) 待 owner 拍板.