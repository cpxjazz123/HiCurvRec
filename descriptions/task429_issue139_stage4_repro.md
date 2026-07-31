# Task #429 / Issue #139 [方向C Gate4] Stage4 真实 history-SID 协议来源与双重复跑证据 (R18 repro)

## 目标

Issue #136 task428 关闭时缺 verdict (TypeError exit) + Issue #139 R18 强制 repro audit 要求 5 件套 (原始日志 + 配置 + ckpt SHA256 + verdict + commit) 落地。重做 task428 + 修复 WrappedHGRec.forward 路由 bug + 跑 2×control + 2×adapter 双重复跑证据。

## 实施

- 脚本: `scripts/task429_issue139_stage4_repro.py`
- AdapterHookedHGRec 类: 用 `self.hgrec.generate(input_ids=...)` 走 HG_Rec wrapper 路径, 内部 hook `self.hgrec.model.shared(input_ids)` → adapter → `self.hgrec.model.generate(inputs_embeds=adapted, ...)` (修复前次 task428 的 self.model.model 路由错误)
- 双重复跑: control 2 次 (seed=43, seed=44) + adapter 2 次 (seed=143, seed=144), 报告 6 metrics 平均 + 复跑 delta
- R20+R21 强制 5 件套: config.json + SHA256 三件套 + 原始 log + verdict.json + commit (R15 push)
- R139 reproducibility triangle: T5 ckpt + SID npy + adapter ckpt 三件套 SHA256 落盘

## 预期产物

- `products/task429_issue139_stage4_repro/config.json` (R20+R21 配置 5 件套)
- `products/task429_issue139_stage4_repro/verdict.json` (R21 verdict 含 commit hash)
- `logs/task429_issue139_stage4_repro.log` (R20 原始日志)
- `verdicts/task429_issue139_gate4_*.md`
- SHA256 三件套 + 6 metrics + control vs adapter R@10 + baseline R@10 vs 0.1020

## Gate 4 决策

- PASS: control R@10 > 0.05 (合理复现, baseline 0.1020) + adapter R@10 > 0.1020 (超过 baseline)
- PARTIAL: control R@10 > 0.05 但 adapter R@10 ≤ 0.1020 (协议重建成功, dual-gate 无效)
- FAIL: control R@10 ≤ 0.05 (协议仍未正确重建)

## 关联

- Issue #139: Stage4 真实 history-SID 协议来源与双重复跑证据 (R18 强制 repro audit)
- 联立 #133 (proxy R@K=0) + #136 (WrappedHGRec 路由 bug) → #139 (R18 修复 + 5 件套审计)
- R7: GPU 2 空闲 (前次 task428 已退出, 4×L40S 全部空闲)
- R12: 训练 ckpt 强制保存到 products/task429/
- R17 + R20: commit + verdict 必须详细 4 Gate 回答
- R139 reproducibility triangle: SHA256 ckpt + SID + eval script 三件套必须落盘
- R18 强制: 重做实证 (precheck/GPU/eval), 不能"沿用判决" NO-GO 收口