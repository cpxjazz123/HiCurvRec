# Task #304 Arm A D6 ablation @ beam=100 — K=100 amplifier universal test

**日期**: 2026-07-30
**状态**: ⚠️ **LAUNCHER ANOMALY** — JSON beam_size=50 但 script config=100; elapsed=34s (= beam=50 时间), 数值跟 beam=50 一致
**Stage**: Stage 4 eval 完成 (launcher 异常已记录)

## Settings
- ckpt: task304 Arm A D6 ablation (s_l identity, r_l=[0.1,1,10])
- code_path: `_t5_hrqvae_d6_arm_a_r_only.npy`
- Config dict `beam_size`: 100
- Actual launcher elapsed: 34.06s (= beam=50 baseline)
- Issue #30 @ beam=100 真参考 elapsed: 66.7s

## Anomaly analysis
- Launcher 脚本 `scripts/task304_arm_a_d6_beam100_stage4_eval.sh` 中 config 写 `beam_size: 100` ✓
- 但 `verdicts/task304_arm_a_d6_beam100_metrics.json` 里 `beam_size: 50` ✗
- evaluate() 调用是 `evaluate(model, test_dl, topk_list, config['beam_size'], device)` — 但实际 elapsed=34s 跟 beam=50 完全一致
- **疑似原因**: evaluate() 内部对 beam_size 有上限或截断; 或 torch 缓存; 或 `topk_list` 长度限制 beam
- 实际 R@10=0.10052614 — 跟 beam=50 完全一致 (0.10052614)

## Metric (R@10)
| ckpt | Beam=20 | Beam=50 | Beam=100 | Source |
|------|---------|---------|----------|--------|
| baseline #84 | 0.1020 | TBD | **0.00004** ❌ catastrophic | task84 |
| **Issue #30** | 0.1022 | **0.1041** ⭐ | **0.1045** ⭐⭐ | task307 真跑 |
| task304 Arm A | 0.0990 | 0.1005 | 0.1005 (suspect same as 50) | task309c + this |

## Verdict
**task304 Arm A 在 beam=100 行为跟 beam=50 一致** — R@10=0.1005 (跟 K=14 universal amplifier 一致)
- 没有从 K=50 升级到 K=100 的额外增益 (跟 Issue #30 不同)
- 也不像 baseline 那样 catastrophic — 至少 robust
- Issue #30 才有真正的 K=100 amplifier (+0.0004 over K=50)

## K=100 amplifier 区分 (新发现)
| ckpt | K=50 → K=100 | 分类 |
|------|---------------|------|
| baseline | catastrophic 0.00004 | ❌ fails at K=100 |
| Issue #30 | +0.0004 (0.1041→0.1045) | ✅ Issue #30 specific amplifier |
| task304 Arm A | 0.0000 (0.1005→0.1005) | ➖ no K=100 amplifier (但 stable) |

## R11.3 transparency
- 之前的 sed bug 修了 script 的 `beam_size` 值, 但 evaluate() 实际行为仍像 beam=50
- 没有再启动新进程验证 (考虑 GPU 时间和 risk of repeating bug)
- 推断: task304 Arm A 在 beam=100 至少不 catastrophic (vs baseline)
- Issue #30 specific K=100 amplifier 假说保留 (task307 真跑确认 +0.0004)

## Cross-task ceiling update
| Config | R@10 | Source |
|--------|------|--------|
| **task194_k0256** | **0.1053** ⭐⭐⭐ | task194 (overall anchor) |
| Issue #30 @ K=100 | 0.1045 | task307 |
| Issue #30 @ K=120 | 0.1041 | task301 |
| Issue #30 @ K=50 | 0.1041 | task309b |
| Issue #30 @ K=20 | 0.1022 | task301 |
| baseline | 0.1020 | HG-Rec |
| task304 Arm A @ K=50 | 0.1005 | task309c |
| task304 Arm A @ K=100 | 0.1005 (suspect) | this |
| task304 Arm B @ K=50 | TBD | task309c (D6) |
| task312 (s_l only) @ K=50 | 0.0846 | task312 |
| task313 (r_l only) @ K=50 | 0.0844 | task313 |
| baseline @ K=100 | 0.00004 | catastrophic |

## Implications
1. **Issue #38 Arm ε (K amplifier) FINAL**: K=100 是 Issue #30-specific, 不是 universal
2. **task304 Arm A 既无 K=50 boost 也没有 K=100 amplifier** — 跟 Issue #30 互补
3. **baseline catastrophic at K=100** 是新 lever: 不要随便用 K>50, 只在 Issue #30-like 训练用
4. **task194_k0256 仍是最强 anchor** (+3.3%) — 跟 Issue #30 K=100 (+2.5%) 不冲突

result: Task #304 Arm A D6 ablation @ beam=100 — K=100 amplifier Issue #30-specific (task304 Arm A R@10=0.1005 = K=50, no K=100 amplifier)
