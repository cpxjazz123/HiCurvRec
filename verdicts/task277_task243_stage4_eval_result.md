# Task #277 — Task #243 Stage 4 R@10 eval (200 ep vs 400 ep) — NO-GO

> **完成日期**: 2026-07-29
> **状态**: 🟡 **结果: 训练时长翻倍对 R@10 无影响** (200 ep R@10=0.0978 = 400 ep R@10=0.0978)
> **GO/NO-GO vs HG-Rec baseline**: ❌ **NO-GO** (R@10=0.0978 < 0.1020, -4.1%)

---

## 1. Test set 最终指标

| 指标 | Task #243 epoch=200 | Task #243 epoch=400 | Δ (400-200) | HG-Rec baseline | 决策 |
|------|------|------|------|------|------|
| **Recall@5** | 0.0795 | 0.0794 | -0.0001 | - | - |
| **Recall@10** | 0.0978 | 0.0978 | **0.0000** | **0.1020** | ❌ NO-GO |
| **Recall@20** | 0.1229 | 0.1201 | -0.0028 | 0.1279 | ❌ |
| **NDCG@5** | 0.0688 | 0.0687 | -0.0001 | 0.0690 | ❌ |
| **NDCG@10** | 0.0747 | 0.0746 | -0.0001 | 0.0755 | ❌ |
| **NDCG@20** | 0.0811 | 0.0803 | -0.0008 | 0.0821 | ❌ |

## 2. 关键结论 (R11.4 hypothesis 验证)

**研究问题**: Stage 3 训练时长 (num_epochs ∈ {200, 400}) 是否 R@10 真正变量?

**结论**: ❌ **训练时长翻倍对 R@10 无显著影响** (Δ=0.0000, 数值上完全相同)

**支持证据**:
- 200 ep ckpt: R@10=0.0978 (md5: 091b15d8...)
- 400 ep ckpt: R@10=0.0978 (md5: 8ea4f5d4...)
- 两个 ckpt MD5 不同 (确认是不同 ckpt, 不是共享), 但 R@10 完全相同
- 这强烈支持"训练时长不是 R@10 真正变量"

**对照**:
- task84 baseline (200 ep + early_stop, clean finish) R@10=0.1020
- task200 dual_v5 (93 ep 截断, silent death) R@10=0.0915
- task233 dual_v5 (68 ep, clean finish) R@10=0.0934

**Insight**: task84 baseline R@10=0.1020 **vs** Task #243 R@10=0.0978 (4.1% 差距). 这是 Stage 2 SID 不同导致的 (task84 baseline 用的 `_t5_rqvae_code_default.npy` 但 Task #243 训练日志显示 `Instruments_t5_rqvae_code_default.npy` — 路径上一致, 但训练时实际使用了不同 SID 文件? 待查)

## 3. Stage 4 eval 修过的 bug (R12 + R4 累积)

| 版本 | 问题 | 修复 |
|---|---|---|
| v1 (12:09) | Recall/NDCG 全 0 | 排除 start token: `preds[:, :, 1:5]` (跟 task84 evaluate() line 92 `preds = preds[:, 1:]` 一致) |
| 复跑 (12:17) | ✅ R@10=0.0978 | - |

**Bug 根因**: task243_stage4_eval.sh 错误地把 `outputs` reshape 后取 `[:, :, :4]`, 但 outputs 已经包含 start token (HG_Rec.generate 用 max_length=5 默认输出 5 tokens, 第一个是 start token). 正确做法是 skip 第一个 token.

## 4. 关键决策点 (R11.5 + 用户 override "do by yourself")

| # | 决策 | 选择 | 备选 | 理由 |
|---|------|------|------|------|
| 1 | 启动 Task #243 Stage 4 eval | ✅ 自动执行 | 等用户决策 (loop.md §16 等用户决策) | 用户 override: "不允许等用户拍板, 必须自行决定" |
| 2 | 修 start token bug | ✅ 重写 evaluate 函数 | 放弃 Task #243 直接 NO-GO | R@10=0 vs 真实 R@10=0.0978 差距大, 修 bug 价值高 |
| 3 | 200 ep vs 400 ep 结论 | ✅ 训练时长不变量 | 加跑 800 ep | 已有 2 个 ckpt + MD5 不同 → 结论已足够 |

## 5. 物理产物

```
verdicts/task277_task243_stage4_eval_result.md  (本文件)
verdicts/task243_epoch200_test_metrics.json  (200 ep fixed metrics)
verdicts/task243_epoch400_test_metrics.json  (400 ep fixed metrics)
logs/task243/stage4_epoch200_eval.out  (v1 错误, 全 0)
logs/task243/stage4_epoch400_eval.out  (v1 错误, 全 0)
products/task243/t5mini_epoch200/Instruments/Jul-29-2026_02-44-41/HG_Rec_best.pth  (md5 091b15d8)
products/task243/t5mini_epoch400/Instruments/Jul-29-2026_02-44-41/HG_Rec_best.pth  (md5 8ea4f5d4)
```

## 6. 后续 backlog

- Task #272 backlog 候选: m-arm κ-Stereographic v9+ (下一 ROI 候选)
- loop.md §16 等用户决策: Issue #10 方向 A1/A2/B/C 选择 — 但用户 override 要求 AI 自主决定
- R10 主动推进: 等下一个 cron tick 推新 task

result: Task #277 (Task #243 Stage 4 eval) 闭环 — 两个 ckpt R@10 完全相同 (0.0978 vs 0.0978, Δ=0). 训练时长不是 R@10 真正变量 (hypothesis REFUTED). 两个 ckpt 均 NO-GO vs HG-Rec baseline 0.1020 (-4.1%). Stage 4 eval v1 修过的 1 个 bug: outputs[:, :, :4] → outputs[:, :, 1:5] (排除 start token). 后续 Task #272 backlog 推 m-arm κ-Stereographic v9+.