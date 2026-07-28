# Task #232 — Issue #7 调参版 (Gromov ρ_e weight/entropy/spread-norm sweep)

## 来源
- Task #219 verdict next-steps §9: "调参版 (Task #221, 30 min)" — 三个候选 fix (A) weight decay (B) entropy reg (C) spread norm
- GitHub Issue #7 (2026-07-28): [Escape Route 2] Gromov product argmax - dampen rho_e weight to open L1/L2
- Issue #7 提议: Implement (A)/(B)/(C) as variants of Task #219's script, sweep + evaluate consistency per layer

## 背景
Task #219 已 done (Stage 1 训练 verdict NO-GO, 2026-07-26):
- L0 (K=64) 79.65% ✅ OPEN
- L1 (K=128) 50.21% TOO_STRONG
- L2 (K=256) 40.48% TOO_STRONG

Issue #7 假设: L1/L2 fail 是因为 ρ_e 贡献没按层 spread 缩放. 三个 fix 候选:
- (A) weight decay: score = ρ_e · weight_l − d(z,e), sweep weight_l ∈ {0.3, 0.5, 0.7} per layer
- (B) entropy reg: 乘以 exp(-H)/K 抑制深层码字过度选用
- (C) spread norm: 用 |ρ_e − ρ_mean| 归一化

## 核心想法
攻命题前提 (a) — Gromov product argmax, 离线检查阶段调 ρ_e 影响:
- 公式: score_k = ρ_e · weight_l − d(z, e_k) (argmax)
- default weight_l = 1.0 (task219 baseline)
- sweep: weight_l ∈ {0.3, 0.5, 0.7}, 同时测 (B) entropy reg 和 (C) spread norm

## 配置
| 项 | 值 |
|----|-----|
| baseline ckpt | Task #84 HG-Rec (R@10=0.1020) |
| ckpt path | products/task84/ckpt/Instruments/Jul-23-2026_20-08-06_beta_1.000_codebook_[64,128,256]_sk_0.000/best_loss_model.pth |
| num_emb_list | 64 128 256 (跟 baseline 一致) |
| 测量方法 | forward-pass-only (不训练), reuse task219 脚本逻辑 |
| Variants | (A) weight decay / (B) entropy reg / (C) spread norm |
| Sweep | weight_l ∈ {0.3, 0.5, 0.7} × (A); (B) + (C) 单独 |
| GPU | 0 (CPU only) |

## 一致率判据 (跟 task219 一致)
- > 95% → FAIL (Gromov 影响被淹没)
- 60-90% → ✅ OPEN (Gromov 有区分力)
- < 50% → TOO_STRONG (几何主导过头)

## 决策阈值
| 指标 | GO | NO-GO |
|------|-----|-------|
| L1/L2 一致率 (new variant) | 进 60-90% open band | 仍 < 50% 或 > 95% |
| L0 一致率 (new variant) | 保持在 79.65% (不退) | 退化到 < 50% 或 > 95% |

## 产物
- scripts/task232_gromov_phase0_weight_sweep.py (复用 task219 script 加 ABC variants)
- /home/wlia0047/.claude/jobs/04ccf474/tmp/task232_gromov_phase0_results.json
- verdicts/task232_gromov_phase0_weight_sweep_result.md

## 决策逻辑
- 全部 sweep 后写 verdict, 报告 (A)/(B)/(C) 三个 fix 对 L0/L1/L2 一致率的影响
- 互补性: 跟 Task #231 (Per-Codeword κ c_k range sweep) 互补 — #231 L0 难开 L1/L2 易开, #232 L0 易开 L1/L2 难开
- Hybrid 可能性: L0(Gromov) + L1/L2(per-codeword κ) — 推荐为 Issue #5 follow-up

## Status
Not yet implemented. Issue #7 提议的 Phase 0 forward-pass-only sweep, Phase 1 完整 Stage 1 训练由 hybrid 路线接手 (如果 #231 和 #232 各自打开不同层).
