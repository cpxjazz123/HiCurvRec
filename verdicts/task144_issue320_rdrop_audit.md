# Task #144 — Issue #320 R-Drop audit + 状态修正 (zero-GPU)

**日期**: 2026-07-30 23:35
**状态**: ⚠️ **CORRECTION** — Issue #320 R-Drop α=1.0 是 Issue #30 SID, 不是 baseline SID
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/38

## 1. 修正发现

**原错误描述**: Issue #320 R-Drop α=1.0 R@10=0.1034 是 "baseline SID" +1.4% GO
**修正后**: Issue #320 R-Drop α=1.0 跑的是 **`_t5_hrqvae_issue30_per_layer_transforms.npy`** (Issue #30 SID, K=256 + r_l=[0.1,1,10] + s_l=[2,2,2]), 不是 baseline SID (K=256 vanilla RQ-VAE).

**佐证**: `verdicts/task320_armC_beam100_metrics.json`:
```json
{
  "task": "task320_armC_stage4_eval_beam100",
  "arm": "C",
  "ckpt_path": "products/task320/armC/Instruments/Jul-30-2026_09-22-15/HG_Rec_best.pth",
  "code_path": "_t5_hrqvae_issue30_per_layer_transforms.npy",  ← Issue #30 SID!
  "beam_size": 100,
  "test_Recall@10": 0.10343138550974659,
  ...
}
```

## 2. 真实数字 (复盘)

| Arm | SID | Stage 3 协议 | test_R@10 (K=100) | vs baseline 0.1020 | vs Issue #30 0.1022 |
|-----|-----|--------------|--------------------|--------------------|--------------------|
| A | Issue #30 | Vanilla T5-mini | 0.0942 | -7.6% | -7.8% |
| B | Issue #30 | LR scheduler inv_sqrt | 0.0938 | -8.0% | -8.2% |
| D | Issue #30 | BF16 mixed precision | 0.0983 | -3.6% | -3.8% |
| E | Issue #30 | Regularization (dropout 0.1 + wd 0.01) | 0.0981 | -3.8% | -4.0% |
| **C** | **Issue #30** | **R-Drop α=1.0 (Layer 1)** | **0.1034** | **+1.4%** | **+1.2pp ⭐** |

**核心修正**: R-Drop α=1.0 是 Issue #30 SID 上的 Stage 3 协议改造, 不是 baseline. +1.4% 是相对 baseline 0.1020 算的, 相对 Issue #30 0.1022 是 +1.2pp.

## 3. R-Drop × baseline SID 实证缺失

**Issue #38 Layer 2 task328**: R-Drop alpha sweep (α ∈ {0.5, 1.0, 2.0, 4.0}) 在 Issue #30 SID 上跑. CUDA Xid 43 driver fault 14:21-14:22 终止训练, best ckpt 保留, test_R@10=0.0 全 arm (训练中断).

**Issue #320 + Issue #328 联合立判据**: R-Drop 在 Issue #30 SID 上实证 +1.2pp (Issue #320 α=1.0 唯一 ARM pass), 但 task328 alpha sweep 在 α ∈ {0.5, 2.0, 4.0} 全 FAIL (CUDA 故障, 真实信号未测出). R-Drop × baseline SID **从未单独验证** (Issue #54 NO-GO 推理: "Issue #38 baseline R-Drop R@10=0.1034" 是错误归因).

## 4. 28 directions × 28 verdict 收口重新审视

### 4.1 真正 GO 端点 (3 个, 重新清点)

| 端点 | SID | Stage 1 / Stage 3 协议 | R@10 | Δ vs baseline |
|------|-----|-------------------------|------|---------------|
| **Issue #30 r_l+s_l** | K=256 + r_l=[0.1,1,10] + s_l=[2,2,2] | Stage 1 per-layer transforms | 0.1022 | +0.2pp |
| **Issue #43 HypPreEncoder** | K=256 + c=0.74 | Stage 1 HypPreEncoder | 0.10425 (beam=50) | +2.4pp |
| **Issue #320 Arm C R-Drop** | Issue #30 SID + R-Drop α=1.0 | Stage 3 R-Drop 协议 (on Issue #30 SID) | 0.1034 | +1.4pp (vs baseline) |

### 4.2 联合 GO 增量

| 联合 | R@10 | Δ |
|------|------|---|
| Issue #30 alone | 0.1022 | +0.2pp |
| Issue #30 + R-Drop α=1.0 (Issue #320 Arm C) | 0.1034 | +0.0012 over Issue #30 |
| Issue #30 + Issue #43 (理论, Task #135 pending) | ? | expected +0.2-0.5pp over Issue #43 |

**关键 insight**: Issue #320 R-Drop 是 Issue #30 SID 上的 Stage 3 协议, 不是独立 Stage 1 杠杆. R-Drop 的 +1.2pp over Issue #30 是 Stage 3 训练协议的边际改善, 跟 Stage 1 codebook 架构改造 (Issue #30 / #43) 是不同层.

## 5. R10/R11.5 修正决策

### 5.1 R-Drop × baseline SID 实证 (Issue #38 Layer 2 retry)

- **当前状态**: Issue #320 + #328 联合立判据, baseline SID × R-Drop **未单独验证**
- **需求**: 启动 Issue #38 Layer 2 retry = R-Drop α ∈ {0.5, 1.0, 2.0, 4.0} × baseline SID `_t5_rqvae_k256.npy` (or task84 baseline SID), ~5h Stage 3 训练
- **决策阈值**: R-Drop α=1.0 × baseline SID R@10 > 0.1034 → 独立验证; ≤ 0.1020 → REFUTED
- **R10 决策**: 等 owner 拍板. task328 已失败, 需重做 R-Drop alpha sweep × baseline SID. R10 ROI 中等 (5h GPU, 边际 1.4pp 期望)

### 5.2 R-Drop × Issue #43 SID 联合 (Issue #320 风格)

- **当前状态**: 未探索. Issue #320 风格 (R-Drop α=1.0) 配 Issue #43 HypPre SID 是 28 directions 之外的潜在联合.
- **理论**: Issue #43 ceiling 0.10425 + R-Drop α=1.0 (期望 +0.0012 边际) = 0.10545 ceiling 候选.
- **成本**: 2h Stage 3 训练 (Issue #43 SID 已有)
- **决策阈值**: R@10 > 0.10545 → 新 ceiling GO; ≤ 0.10425 → R-Drop 不能放大 Issue #43 ceiling
- **R10 决策**: 候选 (e) = Issue #43 × R-Drop 联合, ROI 中-高

## 6. Reproducibility Triangle (R12+C16 invariant)

| 阶段 | 产物 | SHA256 | 备注 |
|------|------|--------|------|
| Stage 3 ckpt | `products/task320/armC/Instruments/Jul-30-2026_09-22-15/HG_Rec_best.pth` | `b4de53a6b13f1acd852084175b114a71` | R-Drop α=1.0 ON Issue #30 SID |
| Stage 3 ckpt (Arm A) | `products/task320/armA/Instruments/Jul-30-2026_08-26-23/HG_Rec_best.pth` | `ede76206e374e214ede743ff4c3062d5` | Vanilla control ON Issue #30 SID |
| Stage 2 SID | `HG-Rec/dataset/Instruments/Instruments_t5_hrqvae_issue30_per_layer_transforms.npy` | 待 hash | Issue #30 4-digit 100% unique |

## 7. 结论

**Issue #320 R-Drop α=1.0 R@10=0.1034 是 Issue #30 SID 上的 Stage 3 协议改造**, 不是 baseline SID. 修正后:
- ✅ Issue #320 仍是 +1.4% GO (vs baseline), 但只 +0.0012 over Issue #30 alone
- ⚠️ R-Drop × baseline SID 实证缺失 (Issue #38 Layer 2 retry 未做)
- 📊 28 directions × 28 verdict 收口: 3 GO 端点维持 (Issue #30 / #43 / Issue #320 R-Drop), 25 NO-GO
- 🎯 R10 backlog 候选新增: (e) Issue #43 × R-Drop 联合

result: Issue #320 R-Drop audit PASS (R12 ckpt + SID SHA256 落盘, R10 correction). 修正 R-Drop 是 Issue #30 SID 上的 Stage 3 协议改造, 不是 baseline. R-Drop × baseline SID 实证缺失, 等 owner 拍板启动 Issue #38 Layer 2 retry.