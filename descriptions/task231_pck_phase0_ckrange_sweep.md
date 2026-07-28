# Task #231 — Issue #6 调参版 (Per-Codeword κ c_k range sweep)

## 来源
- Task #218 verdict next-steps: "缩窄 c_k 范围 [0.5, 20] → [0.5, 5] 或 [1, 5] 调参版"
- GitHub Issue #6 (2026-07-28): [Escape Route 1] Per-Codeword kappa - tune c_k range to open L0
- Issue #6 提议 Re-run Task #218's Phase 0 script with c_k ~ Uniform(0.5,5) and Uniform(1,5), 3 seeds each

## 背景
Task #218 已 done (Stage 1 训练 verdict PARTIAL SUCCESS, 2026-07-26):
- L0 (K=64) 28.90% ± 2.15% TOO_STRONG
- L1 (K=128) 67.15% ± 0.22% OPEN
- L2 (K=256) 75.69% ± 0.79% OPEN

Issue #6 假设: L0 一致率太低是因为 c_k spread [0.5, 20] 太宽 (少量大 c_k codewords 几何主导). 缩窄到 [0.5, 5] 应该让 L0 进 60-90% open band, 同时 L1/L2 保持.

## 核心想法
攻命题前提 (b)+(d), 但在**离线检查**阶段调 c_k range (不训练, 仅 forward-pass-only):
- score_k = (1/sqrt(c_k)) · arccosh( 1 + 2*c_k*||z-e_k||^2 / [(1-c_k*||z||^2)(1-c_k*||e_k||^2)] )
- sweep c_k ~ Uniform(0.5, 5) 和 Uniform(1, 5), 3 seeds each

## 配置
| 项 | 值 |
|----|-----|
| baseline ckpt | Task #84 HG-Rec (R@10=0.1020) |
| ckpt path | products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth |
| num_emb_list | 64 128 256 (跟 baseline 一致) |
| 测量方法 | forward-pass-only (不训练), reuse task218 脚本逻辑 |
| c_k sweep | Uniform(0.5, 20) [baseline 复现] / Uniform(0.5, 5) / Uniform(1, 5) |
| n_seeds | 3 per range |
| GPU | 0 (CPU only) |

## 一致率判据 (跟 task218 一致)
- > 95% → FAIL (per-codeword κ 影响被淹没)
- 60-90% → ✅ OPEN (per-codeword κ 有区分力)
- < 50% → TOO_STRONG (几何主导过头)

## 决策阈值
| 指标 | GO | NO-GO |
|------|-----|-------|
| L0 一致率 (新 range) | 进 60-90% open band | 仍 < 50% 或 > 95% |
| L1/L2 一致率 (新 range) | 保持在 60-90% (不退) | 退化到 > 95% 或 < 50% |

## 产物
- scripts/task231_pck_phase0_ckrange_sweep.py (复用 task218 script 改 sweep)
- /home/wlia0047/.claude/jobs/04ccf474/tmp/task231_pck_phase0_results.json
- verdicts/task231_pck_phase0_ckrange_sweep_result.md

## 决策逻辑
- 全部 sweep 后写 verdict, 跟 task218 baseline 对比, 报告 c_k range 对 L0/L1/L2 一致率的影响
- Issue #6 关闭附 verdict 引用 (如果通过推荐 Issue #5 follow-up, 如果失败推荐 Gromov hybrid)

## Status
Not yet implemented. Issue #6 提议的 Phase 0 forward-pass-only sweep, Phase 1 完整 Stage 1 训练由 Task #222 (task220 verdict PARTIAL SUCCESS 推荐) 接手.
