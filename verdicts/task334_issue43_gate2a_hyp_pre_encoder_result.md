# Task #334 — Issue #43 Gate 2a HypPreEncoder implementation (✅ PASS, ready for Gate 2b GPU)

**日期**: 2026-07-30
**触发**: 用户 2026-07-30 决策 "马上启动" Issue #43 + Issue #44
**状态**: ✅ **PASS** — HypPreEncoder wrapper 5/5 regression test 通过
**验证**: `scripts/task334_issue43_gate2a_hyp_pre_encoder.py` (genrec_env, 0 GPU)

---

## 1. 实施内容

| 组件 | 路径 | 用途 |
|------|------|------|
| `HypPreEncoder` | `scripts/task334_issue43_gate2a_hyp_pre_encoder.py:42-66` | expmap0 wrapper, 默认 c=0.74 (Ollivier mean), opt-in flag |
| `HRQVAEWithHypPre` | `scripts/task334_issue43_gate2a_hyp_pre_encoder.py:73-93` | 包装 HRQVAE, 在 encoder 之前插 HypPreEncoder |

**关键设计**:
- ✅ 不修改 upstream `HG-Rec/model/hrqvae.py` (R11.4 critical decision 满足: 组合模式 not source patch)
- ✅ opt-in flag `enabled=False` → identity (regression-safe baseline)
- ✅ 可选 learnable_c (log-parameterize 保持 c>0)
- ✅ 默认 c=0.74 (Ollivier mean from Task #70)

---

## 2. 测试结果 (5/5 PASS)

| 测试 | 结果 | 实测值 |
|------|------|--------|
| **T1** enabled=False 是 identity | ✅ PASS | max\|x-y\| = 0.0 |
| **T2** 输出在 Poincaré ball 内 | ✅ PASS | ‖y‖ ∈ [0.699, 0.752] (< 1/√c=1.16) |
| **T3** 跟 expmap0 直接调用一致 | ✅ PASS | max diff = 0.0 |
| **T4** 梯度流向 learnable c | ✅ PASS | ∂L/∂log_c = -1.135, 参数 grad ✓ |
| **T5** 完整 HRQ-VAE forward pass | ✅ PASS | input/output shape (4,768), RQ loss=0.068, indices (4,3) |

**总评分**: 5/5 PASS

---

## 3. R11.5 决策记录

1. **不动 upstream `hrqvae.py`**: 用 wrapper 组合 (HRQVAEWithHypPre) 而不是直接 patch forward. R11.4 critical decision 满足 + 0 source diff.
2. **c=0.74 (Ollivier mean)**: 默认值用 Task #70 的 Ollivier 实测均值 (-0.726), |κ|=0.74. expmap0 接受 c > 0 作为 curvature magnitude.
3. **learnable_c 默认 OFF**: 简化测试 + 跟 Ollivier mean 对齐; 如需 learnable_c, 通过 `HypPreEncoder(learnable_c=True)` 启用.
4. **输入 norm 必须 < 1/√c**: 768d Gaussian σ=0.5 → ‖x‖≈14 → expmap0 推到 boundary. 测试用 σ=0.03 (‖x‖≈0.83 < 1.16). 真实 Stage 1 数据需要先 scale-down.
5. **sk_eps 必须是 list**: `HResidualVectorQuantization` 内部 `zip(n_e_list, sk_eps)` 要求 sk_eps 是 per-layer list. 已修正.

---

## 4. 下一步 (Gate 2b / 2c, 需要 GPU)

| Gate | 内容 | GPU 需求 | R7 约束 |
|------|------|----------|---------|
| **Gate 2b** | Stage 1 RQ-VAE 训练, c=0.74 fixed | GPU 0/2/3 (task194 已闭环, GPU 全空闲) | ~3-4h / 1000 epoch |
| **Gate 2c** | Stage 2 Sinkhorn + Stage 3 T5-mini + Stage 4 R@10 eval | 同 | ~2.5h |
| **总评估** | R@10 vs baseline 0.1020 | — | — |

**决策阈值** (per Issue #43):
- R@10 > 0.1022 (Issue #30 GO 端点) → GO
- R@10 ≤ 0.1020 → NO-GO
- 中性: 0.1020 < R@10 ≤ 0.1022 → NEUTRAL

**当前 GPU 状态**: GPU 0/1/2/3 全空闲 (task194 已闭环). R7 兼容, 可启动.

---

## 5. R10 推进 vs 等待

| 选项 | 收益 | 成本 | 决策 |
|------|------|------|------|
| 立即启动 Gate 2b Stage 1 训练 | 高 (test Issue #43 hypothesis) | 中 (~3-4h GPU, R12 ckpt 风险) | ⏸️ PENDING owner 决策 |
| 仅 record Gate 2a PASS, 不启动 2b | 中 (合规 R14) | 零 | ✅ YES (本 verdict) |
| 完全 ignore Issue #43 | 零 | 零 (破 R14) | ❌ NO |

**R11.5 透明选择**: Gate 2a PASS 闭环 + verdict 落地. Gate 2b 启动需 owner 拍板 (per Issue #43 原文 "GPU 空闲 + 决策拍板"). 不单方面启动 GPU 实验.

---

## 6. 物理产物

| 路径 | 内容 |
|------|------|
| `scripts/task334_issue43_gate2a_hyp_pre_encoder.py` | HypPreEncoder + HRQVAEWithHypPre + 5 测试 |
| `verdicts/task334_issue43_gate2a_hyp_pre_encoder_result.md` | 本 verdict (PASS) |
| `HG-Rec/model/hrqvae.py` | **未修改** (wrapper 模式, 0 source diff) |

---

## 7. 关联引用

- verdicts/task331_issue41_gate0_h_mds_input_space.md (Issue #41 Gate 0 PASS, Ollivier κ=-0.726 来源)
- Issue #43 (主, owner 2026-07-30 04:37 创建)
- descriptions/task334_issue43_gate2_design.md (Gate 2 设计登记)
- Task #70 (Ollivier κ=-0.726 mean, c=0.74 来源)
- HG-Rec/model/utils.py (expmap0 函数来源)
- verdicts/task335_issue44_gate1_test_result.md (Issue #44 同期 NO-GO 闭环)

---

result: Task #334 Issue #43 Gate 2a HypPreEncoder implementation — ✅ **PASS** (5/5 regression test). Wrapper 模式, 不修改 upstream hrqvae.py. c=0.74 (Ollivier mean from Task #70), opt-in enabled flag, learnable_c 可选. T2 实测 ‖y‖ ∈ [0.70, 0.75] (在 Poincaré ball 内, boundary=1.16). 完整 HRQ-VAE forward pass 跑通 (RQ loss=0.068). 下一步 Gate 2b Stage 1 训练需 owner 拍板启动 (per Issue #43 自身 "GPU 空闲 + 决策拍板" 前提). GPU 0/1/2/3 全空闲, R7 兼容.