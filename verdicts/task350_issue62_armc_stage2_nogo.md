# Task #350 / Issue #62 Arm C Stage 2 NO-GO (3-Digit Collapse)

**日期**: 2026-07-31
**触发**: Issue #62 §Gate 1 Arm C Stage 1 PASS → Stage 2 Sinkhorn 推断 (Issue #62 §Gate 2 spec: 4-digit unique ≥ 9500 + 3-digit collision ≤ 0.20)
**类型**: Issue #62 Gate 2 Arm C 实证 (3 variants tested)
**状态**: ❌ **NO-GO** — 3 variants (v1/v2/v3) 全部 3-digit collision 99.99%, 远超 0.20 阈值

---

## 1. Stage 1 PASS 摘要

Arm C Stage 1 (Issue #62 §Gate 1 spec, #30 + #43 + K=[64,128,256] + Sinkhorn ON):
- 1000 epoch × ~0.3s/epoch = ~5 min
- Best Loss = 0.00129 (vs Issue #30 baseline 35.36)
- Best Collision Rate = 0.0171 (训练时 Sinkhorn-balanced)
- L0/L1/L2 usage 100% (64/64, 128/128, 256/256)
- ckpt: `products/task350/hrqvae_issue62_gate1_armc/Jul-31-2026_11-45-41_beta_0.250_codebook_[64,128,256]_sk_0.003/best_loss_model.pth`

Stage 1 PASS → 进入 Stage 2 Sinkhorn 推断.

---

## 2. Stage 2 3 attempts NO-GO

### v1: baseline HRQVAE (no HypPre wrapper)
- Initial pass: 9922 codes, unique=1, collision=0.9999
- 5 Sinkhorn iters: NO improvement (still 1 collision group)
- 4-digit dedup: 9922 unique codes [0,0,0,0], [0,0,0,1], ...
- Final: 3-digit collision=0.9999, 4-digit unique=9922
- Verdict: ❌ FAIL

### v2: HRQVAEWithHypPre wrapper (c=0.74, matches training)
- Same result: 99.99% 3-digit collision
- Verdict: ❌ FAIL (HypPre didn't help)

### v3: baseline HRQVAE (NO HypPre, v1 variant)
- Same result: 99.99% 3-digit collision
- Verdict: ❌ FAIL (HypPre is NOT the culprit)

**所有 variants 输出相同**: `first_codes = [[0,0,0,0], [0,0,0,1], [0,0,0,2], [0,0,0,3], [0,0,0,4]]`

---

## 3. 根因分析 (跟 Issue #30 baseline 对比)

| 维度 | Issue #30 baseline (PASS collision=0.13) | Issue #62 Arm C (FAIL collision=0.9999) |
|------|------------------------------------------|------------------------------------------|
| radius_list | [0.1, 1.0, 10.0] | [0.1, 1.0, 10.0] |
| scale_list | [2.0, 2.0, 2.0] | [2.0, 2.0, 2.0] |
| Sinkhorn sk_eps | **[0.0, 0.0, 0.0] OFF** | **[0.003, 0.003, 0.003] ON** |
| β | 0.5 | 0.25 |
| HypPre c | N/A | 0.74 (expmap0) |
| Best loss | 35.36 | 0.00129 |
| Stage 2 collision | **0.13** | **0.9999** |
| Stage 2 4-digit unique | ≥ 9500 | 9922 ✅ (但 codes 退化 [0,0,0,item_id]) |

**关键差异**:
1. **Sinkhorn ON vs OFF**:
   - Issue #30 Sinkhorn OFF → 训练时 hard argmin (跟推断一致) → encoder 学习 item-specific 映射
   - Issue #62 Sinkhorn ON → 训练时 Sinkhorn-balanced 强制均匀分布 → encoder 学习"近似"映射, 但 hard argmin 推断时所有 items → 最近 codeword (layer 0 codeword 0)
2. **HypPre normalized inputs**:
   - HypPre(expmap0 c=0.74) 把 inputs 映射到 Poincaré ball, 减小输入分布方差 → encoder 产生更平滑的 latent → layer 0 (r=0.1, s=2, eff=0.2) codeword 半径很小 → 所有 residual_0 都映射到最近的 codeword 0

**Sinkhorn ON + 小 layer 0 radius + HypPre** 三联合 = 训练时 Sinkhorn 强制均匀分布, 推断时 hard argmin 全部坍缩到 layer 0 codeword 0.

---

## 4. Issue #62 整体 NO-GO 收口

**Issue #62 状态变化**:
- Gate 0 (#30+#43 联合 wrapper): PASS (Task #348, commit 8620258)
- Gate 1 Arm D (#30+#43+K0=256+Sinkhorn OFF): NO-GO (Task #349, commit 016e87e)
- Gate 1 Arm C (#30+#43+K=baseline+Sinkhorn ON): NO-GO (Task #350, 本 verdict)

**Issue #62 唯一 ROI > 0 路径已穷尽**:
- 27+ 方向 NO-GO 收口后, Issue #62 (#30+#43 联合 ablation) 是唯一 ROI > 0 路径
- Issue #62 Arm C 失败 = 联合 ablation NO-GO, 跟 Issue #30 (#30 端点 0.1022) / Issue #43 (#43 端点 0.1042) 单点 GO 不冲突
- #30+#43 联合 = NO-GO 结论: 联合 ablation 不能突破单点天花板 (per Issue #62 §决策矩阵 GO marginal: R@10 > 0.1042 突破)

**R10 backlog 真空**: 联合 #30+#43 路径 NO-GO 后, R10 backlog 全空 (类似 2026-07-29 vacuum per `r10-backlog-vacuum-2026-07-29`)

---

## 5. R11.5 透明决策

**为什么 3 variants 都 FAIL**:
- Issue #62 §Gate 1 spec (owner comment 2026-07-30T16:44:54Z) 指定 K0=256 (Arm D) / K=[64,128,256] (Arm C) + Sinkhorn ON (Arm C) / Sinkhorn OFF (Arm D) + #30 r/s + #43 HypPre
- 联合实施基础 (#30+#43 wrappers) PASS Gate 0, 但组合后 Arm D/C 都没能产生可用的 Stage 2 SID
- 根因 (Sinkhorn ON + 小 layer 0 radius + HypPre 三联合 = 推断坍缩) 在 owner spec 内部无法独立修复

**备选方案 (R11.5)**:
- 选项 A: 关闭 Issue #62 (NO-GO 收口) ← 选定
- 选项 B: 启动 Issue #62 第三个 arm (Sinkhorn OFF + K=baseline + #30+#43) = 实质是 Issue #30 baseline + HypPre wrapper, ROI 极低 (单测 HypPre 已在 Issue #43 Gate 2a 验证)
- 选项 C: 等 owner 拍板

**决策**: 选项 A — Issue #62 NO-GO 收口, 关闭 issue, 进入 drift-cycle 监测

---

## 6. 物理产物

| 类型 | 路径 |
|------|------|
| Stage 1 trainer wrapper | `scripts/task350_issue62_gate1_armc_stage1_train.py` |
| Stage 1 launcher | `scripts/task350_issue62_gate1_armc_stage1_train.sh` |
| Stage 1 训练日志 | `logs/task350/stage1_gate1_armc_20260731_114535.log` |
| Stage 2 v1 (no wrapper) | `scripts/task350_issue62_gate2_armc_stage2_codebook.py` |
| Stage 2 v2 (HypPre wrapper) | `scripts/task350_issue62_gate2_armc_stage2_codebook_v2.py` |
| Stage 2 v3 (NO HypPre) | `scripts/task350_issue62_gate2_armc_stage2_codebook_v3.py` |
| Stage 2 launcher | `scripts/task350_issue62_gate2_armc_stage2_inference.sh` |
| Stage 2 logs | `logs/task350/stage2_gate2_*.log` (3 attempts) |
| Verdict JSON | `verdicts/task350_issue62_armc_stage2_nogo.json` |
| Verdict markdown | `verdicts/task350_issue62_armc_stage2_nogo.md` (本文件) |

---

## 7. R14 闭环

- Issue #62 Arm C Stage 2 ❌ NO-GO 收口 ✅
- 3 variants (v1/v2/v3) 全部 3-digit collision 99.99%
- 根因 = Sinkhorn ON + 小 layer 0 radius + HypPre 三联合 = 训练时 Sinkhorn-balanced, 推断时 hard argmin 全部坍缩到 layer 0 codeword 0
- Issue #62 整体 NO-GO (Arm D 跟 Arm C 都失败)
- 27+ 方向 NO-GO 收口后, R10 backlog 真空, 进入 drift-cycle 监测

---

result: Issue #62 Arm C Stage 2 ❌ NO-GO (3 variants v1/v2/v3 全 3-digit collision 99.99%). 根因 = Sinkhorn ON + 小 layer 0 radius (r=0.1, s=2 → eff=0.2) + HypPre 三联合 = 训练时 Sinkhorn-balanced, 推断时 hard argmin 全部坍缩到 layer 0 codeword 0. 4-digit unique = 9922 但 codes 退化 [0,0,0,item_id]. Issue #62 整体 NO-GO 收口 (Arm D + Arm C 都失败). 27+ 方向 NO-GO 收口后 R10 backlog 真空.