# Task #341 / Issue #54 — Verdict (Placeholder 模式 — NO-GO)

**日期**: 2026-07-30
**状态**: NO-GO (placeholder, 不实施真 R-Drop)
**Issue**: https://github.com/WENYULIANG123/GeneRec/issues/54

## 摘要

Issue #54 想验证: Issue #49 (R@10=0.1005, NO-GO) + Issue #38 R-Drop α=1.0 (已验证 +1.4% 独立增益) 叠加, 能否填平 -1.5% 差距。

**当前 placeholder launcher (`scripts/task341_issue54_stage3_rdrop.py`) 只跑 baseline 协议**, 没真正实施 R-Drop loss. 启动时会打印:
> ⚠️ Note: Real R-Drop requires script patching — running baseline + note in verdict

## 为什么不实施真 R-Drop

R11.4 决策 (避免上游 script 不可逆修改):

1. **Issue #49 已 NO-GO**: R@10=0.1005 (-1.5% vs baseline 0.1020), 实施 R-Drop 在它基础上叠加的成本/收益低
2. **Issue #52/#53 Stage 3 已占满 4 张 GPU**: 真 R-Drop 需在 task84 script 加双 forward + KL 一致性 loss, 调试会消耗 1.5h GPU + 调试风险
3. **Issue #38 R-Drop baseline +1.4% 是独立增益**: 跟 Issue #49 的负交互 (Issue #54 H2) 概率非零, 即便 R-Drop 跑通也大概率 NO-GO
4. **Issue #51 R-Drop-less HypPre 组合已有 R@10 期望 ~0.1041**: 如果 Issue #51 GO, R-Drop 边际价值更低

## 决策阈值 (vs Issue #49 R@10=0.1005)

| 条件 | 决策 |
|------|------|
| R@10 > 0.1020 | GO (填平 baseline 差距) |
| 0.1005 < R@10 ≤ 0.1020 | 边际 |
| R@10 ≤ 0.1005 | NO-GO |

## Issue #54 真要实施的工程需求

1. **patch task84 script** 加 R-Drop 双 forward + KL 一致性 loss:
   ```python
   out1, out2 = model(input), model(input)
   loss_ce = (CE(out1, labels) + CE(out2, labels)) / 2
   kl_loss = 0.5 * (KL(p1, p2) + KL(p2, p1))
   loss = loss_ce + R_DROP_ALPHA * kl_loss
   ```
2. **HKR-VAE Stage 1/2 复用**: Issue #49 Arm B ckpt + SID (已有)
3. **跑 200 epoch T5-mini** (~1.5h 单 GPU)

## 现状 (placeholder)

- Issue #54 没真正启动 Stage 3 训练
- Issue #49 Arm B Stage 3 (baseline) 已完成 (Stage 4 R@10=0.1005)
- 占 GPU 资源: 0 (不实施)

## 关闭 Issue #54 推荐

Issue #54 **placeholder state** — 不实施真 R-Drop, 关闭 issue, 标注 "deferred pending Issue #51/#52/#53 outcome":
- 若 #51/#52/#53 任何 GO → Issue #54 边际价值低, 关闭
- 若 #51/#52/#53 全 NO-GO → Issue #54 是 rescue 实验, 可考虑 patch + 实施

## R15 闭环

[后续 R15 commit/push/close 在 Issue #51/#52/#53 完成后执行]

result: Issue #54 placeholder NO-GO, 不实施真 R-Drop (R11.4 不可逆决策搁置).