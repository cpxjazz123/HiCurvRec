# Task #242 — Issue #11 Gate 1/1b: Per-layer c_k range 训练 (Issue #11 Per-layer kappa range)

## 来源

- GitHub Issue #11 (2026-07-28): [Per-layer kappa range] 逐层独立 c_k 区间 (L0 U(1,5) / L1,L2 U(0.5,20))
- Issue #9 已 FULL NO-GO, Issue #11 是其后续候选路径
- 直接上游: Task #235 (Issue #9 Gate 1) --assignment_mode_list 改动已落地, 通路打通

## 背景

Issue #11 取 task231 sweep 表中每层各自最优 c_k 区间拼成配置:
- L0: U(1,5) (Phase 0 一致率 82.68%)
- L1: U(0.5,20) (Phase 0 一致率 67.15%)
- L2: U(0.5,20) (Phase 0 一致率 75.69%)

H2 假设: 逐层独立区间能解开 task231 §关键发现 2 的 joint constraint 陷阱, 让 L0 利用率突破 90%.

## 步骤

1. ✅ Gate 0 Phase 0 组合验证 (Task #241 verdict): PASS, 三层一致率 81.65% / 64.86% / 75.69% (偏差 ≤ 3pp)
2. ✅ Gate 1 Arm A Stage 1 训练 (40 epoch): L0 utilization = 23.44%, collision = 0.9385 → **Gate 1 NO-GO**
3. ✅ Gate 1b Arm A+ dead_revive Stage 1 训练: L0 utilization = 3.12%, collision = 0.9945 → **Gate 1b FULL NO-GO**
4. ✅ Issue #11 close with FULL NO-GO (Gate 1/1b 双双失败)
5. ✅ 上游 `--c_k_range_list` CLI flag 改动保留 (5/5 sanity test 通过, default 行为不变)

## 产物

- verdicts/task241_issue11_gate0_perlayer_ck_combo_pass.md (Gate 0 PASS)
- verdicts/task242_issue11_gate1_perlayer_stage1_result.md (Gate 1 NO-GO)
- verdicts/task242_issue11_gate1b_perlayer_deadrevive_result.md (Gate 1b FULL NO-GO)
- scripts/task242_issue11_gate1_perlayer_stage1.sh
- scripts/task242_issue11_gate1b_perlayer_deadrevive_stage1.sh
- products/task242/hrqvae_perlayer_ck/ (Arm A ckpts)
- products/task242/hrqvae_perlayer_ck_deadrevive/ (Arm A+ ckpts)
- HG-Rec upstream patch: --c_k_range_list CLI flag (4 files)

## Status

✅ **CLOSED — FULL NO-GO** (2026-07-29)

Issue #11 关闭. PC κ 参数空间正式判为耗尽 (历次扫描 task218/220/222/231/242 Arm A/A+ 全部不达 L0 util ≥ 90%). L0 利用率天花板似乎在 23% 左右, 跟 c_k range 选择关系不大, 这是 PC κ 的内生限制不是参数空间问题.

后续 R10 推进候选:
- Issue #12 Gate 0 (分布层诊断) - 已启动 (Task #244)
- 跨架构 LETTER/S3Rec paper-aligned fix 后的 R@10 重新基线
- kmeans_init 重做 (Issue #9 关闭评论末项)
