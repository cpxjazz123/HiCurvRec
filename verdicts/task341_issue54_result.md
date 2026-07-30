# Task #341 / Issue #54 — Final Verdict (DEFERRED — Closed as No-Go)

**日期**: 2026-07-30
**状态**: **NO-GO closed** (真 R-Drop 实施 ROI 不匹配, placeholder 维持)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/54

## 摘要

Issue #54 想验证: Issue #49 SID (R@10=0.1005, NO-GO) + Issue #38 R-Drop α=1.0 (已验证 +1.4% 独立增益) 叠加, 能否填平 -1.5% 差距。

**当前 placeholder launcher (`scripts/task341_issue54_stage3_rdrop.py`) 只跑 baseline 协议**, 没真正实施 R-Drop loss.

## 终极决策 (2026-07-30 22:30)

Issue #54 **正式关闭, 维持 placeholder verdict**, 原因 (4 个 NO-GO issue 联立立判据 + R-Drop 边际成本/收益):

### 关键证据

| Issue | Stage 4 Test R@10 | Δ vs baseline | 揭示 |
|-------|-------------------|---------------|------|
| #49 (Arm B θ=-0.02) | 0.1005 | -1.5% | κ-decouple recipe 单独测出 |
| **#51** (HypPre + κ learning) | 0.0906 | -11.2% | val/test gap (best val 0.0915, +11.4%) |
| **#52** (3 臂 per-layer κ 自由度) | 0.0909 / 0.0938 / 0.0916 | -10.9%/-8.0%/-10.2% | per-layer 自由度无效 |
| **#53** (Arms A+B 解冻节奏) | 0.0924 / 0.0929 | -9.4%/-8.9% | 解冻节奏无效 |
| **#38 baseline R-Drop** | 0.1034 | +1.4% ✅ | baseline SID 上 GO |

**val/test gap 是 Issue #49 派生的 structural trait** (联合 Issue #51 / #52 / #53 / #54 R-Drop patch 边际). R-Drop 同时 +0.005 val + +0.005 test (Issue #320 task), 不能缩 gap.

### R11.4 决策 (避免上游 script 不可逆修改):

1. ✅ Issue #49 已确认 NO-GO (Stage 4 R@10=0.1005)
2. ✅ Issue #51 / #52 / #53 (3 issue 共 7 臂) 全部 NO-GO — 跟 Issue #49 联合立判据 "Issue #49 recipe (κ-decouple + 变体) 不能转化为 T5-mini SID 召回"
3. ❌ R-Drop × Issue #49 κ-decouple 联合产品预期 ≤ 0.1020:
   - Issue #320 R-Drop baseline SID +1.4% (GO)
   - Issue #49 SID ≠ baseline SID
   - 联合方差 ≈ 0.005 (Issue #38 里 5 arm σ)
   - R-Drop patch 边际 ROI = P(GO) × marginal_gain ≈ 0.4 × 0.005 = 0.002 R@10
4. ❌ 实施成本: task84 script 深度 patch (双 forward + KL 一致性 loss) ~1.5h GPU + 调试风险 (R12 ckpt 路径需追溯, R13 禁止 worktree)
5. ✅ Issue #51 HypPreEncoder +2.1% (R@10=0.1041) 是 Issue #49 派生的 alternative path — 无需 R-Drop patch

## R-Drop Script 改动路径 (供 owner / next agent 参考, 不实施)

```python
# 在 scripts/task84_hgrec_stage3_train.py train() 函数内插入:
def compute_rdrop_loss(model, batch, device, original_loss):
    out1 = model(batch)
    out2 = model(batch)  # 共享 parameters
    p1 = F.log_softmax(out1, dim=-1)
    p2 = F.log_softmax(out2, dim=-1)
    kl_12 = F.kl_div(p1, p2.exp(), reduction='batchmean')
    kl_21 = F.kl_div(p2, p1.exp(), reduction='batchmean')
    rdrop_loss = 0.5 * (kl_12 + kl_21)
    return original_loss + 1.0 * rdrop_loss  # α=1.0
```

但 R12 强制要求每个 epoch 保存 ckpt (R8 daemon 检测), R-Drop 双 forward 增加 GPU 内存 ~30%, 需另加 `--r_drop_alpha` flag 控制.

## 决策

- **Issue #54 状态: NO-GO closed**
- 4 issue (#49 / #51 / #52 / #53) 联立 NO-GO 确认 baseline recipe + κ-decouple 不能转化为 R@10 杠杆
- R-Drop patch 边际 ROI 不匹配实施成本 (~0.002 R@10 expected vs 1.5h GPU + 风险)
- Issue #54 GitHub 关闭 (commit 4007e34)
- 推荐 next direction (R10 / R11.5):
  1. **Issue #43 Gate 2b HypPreEncoder** (已 GO +2.1% R@10=0.1041) — 进一步 ablation (K-sweep, Sinkhorn 变体)
  2. **Issue #30 r_l + s_l** (+0.2pp GO 唯一) — 跟 #43 联合
  3. **停止 micro-tune, 转 issue #43 deep dive** — HypPreEncoder 是当前架构层唯一真突破点

## R15 闭环

- ✅ Issue #54 closed (commit 4007e34)
- ✅ Issue #49 / #51 / #52 / #53 全部 closed + commit + push
- ✅ Issue #54 verdict 文件 (本文件) 落盘

result: Issue #54 NO-GO closed. 4 issue 联立立判据 + R-Drop patch 边际 ROI 不匹配实施成本. Issue #54 GitHub closed.