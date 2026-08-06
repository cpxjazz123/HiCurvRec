---
type: index
title: Verdicts MOC — HG-Rec 复现任务索引
created: 2026-08-05
tags: [moc, vault-entry]
---

# Verdicts — Map of Content (Obsidian Vault)

> 本目录是 HG-Rec 复现 + κ-Stereographic 变体实验的 **Obsidian vault**。
> 根目录 `verdicts/` 包含 228 个产物 (.md / .json / .log)，已按主题分组 + 加 YAML frontmatter。
> 用 Obsidian 直接打开 `verdicts/` 作为 vault 根。

> **基线**: Task #84 valid R@10=**0.1267** / test R@10=**0.1024** (Musical_Instruments 9922 items)
> **历史最佳**: hyp v2 端到端 test R@10=**0.1048** (+0.0024, [Task #210 收口](#b-hyp-端到端突破-task-210))
> **决策阈值**: test R@10 > **0.1024** → GO；失败 → NO-GO

---

## 目录

- [A. Baseline 复现](#a-baseline-复现-task-84)
- [B. Hyp 端到端突破 (taskA hyp v2/v5/v6/v7)](#b-hyp-端到端突破-task-210-task-215-task-259)
- [C. Issue N 系列裁定 (issue12-30, 40, 55-57, 76)](#c-issue-n-系列裁定-issue12-30-40-55-57-76)
- [D. T5 容量 / 训练 (Task #157-#161)](#d-t5-容量--训练-task-157-161)
- [E. 复现审计 (Task #82/88/89/90/91/117/118/207)](#e-复现审计)
- [F. 几何 NO-GO 系列 (7 方向几何变体)](#f-几何-no-go-系列)
- [G. 方向 C 适配器 (Task #448-#475, issue157-187)](#g-方向-c-适配器-task-448-475)
- [H. Issue #30 α-sweep / beam20 reeval (大文件集)](#h-issue-30-α-sweep--beam20-reeval)
- [I. 诊断 / 分析报告](#i-诊断--分析报告)
- [J. Idle / Loop tick log](#j-idle--loop-tick-log)
- [K. Paper / 文档收尾](#k-paper--文档收尾)
- [L. 关键数字速查](#l-关键数字速查)

---

## A. Baseline 复现 (Task #84)

| 任务 | 主题 | 状态 | Verdict |
|---|---|---|---|
| Task #84 | HG-Rec baseline 端到端复现 | ✅ DONE | test R@10=0.1024 (paper 0.1315, Δ -22.4%) |

> baseline 由 Task #84 起算；后续所有变体与 baseline 对比均使用此值 (test R@10=0.1024)。
> 关联: [[taskA_hyp_v5_capmatch_relstruct_bf16_bs1024_1000ep_verdict]] 是首个稳定超基线变体。

---

## B. Hyp 端到端突破 (Task #210, #215, #259)

> **核心结论**: taskA hyp v2 (Task #210) 是迄今**唯一**稳定超基线的端到端变体。后续 v6/v7 (#41/#43 anchored κ & distance-bucket neg) **未超基线**。

| 文件 | 主题 | 状态 | 关键数字 |
|---|---|---|---|
| [[taskA_hyp_v5_capmatch_relstruct_bf16_bs1024_1000ep_verdict]] | hyp v5 capmatch REL_STRUCT (历史最佳) | ✅ PASS | test R@10=**0.1048**, NDCG@20=0.0860 |
| [[taskA_hyp_v6_issue41_anchored_kappa_verdict]] | v6 锚定 κ (issue #41) | ❌ FAIL | test R@10=0.0993 (-0.003) |
| [[taskA_hyp_v7_issue43_distance_bucket_neg_verdict]] | v7 距离区间负样本 (issue #43) | ❌ FAIL | test R@10=**0.0968** (-0.0056) |
| [[taskA_hyp_v6_issue36_margin_limited_info_nce_verdict]] | v6 issue #36 margin-limited InfoNCE | (系列) | |
| [[taskA_hyp_v6_issue37_layered_semantic_neg_verdict]] | v6 issue #37 KMeans cluster neg | PASS | test R@10=0.1044 |
| [[taskA_hyp_v6_issue38_tangent_codebook_audit_verdict]] | v6 issue #38 tangent codebook audit | (系列) | |
| [[taskA_hyp_v6_issue39_kappa_ema_tr_verdict]] | v6 issue #39 κ EMA + trust region | FAIL | test R@10=0.1002 |

**5% 超基线不可达**: taskA hyp 路线已 **9 次确认** (v22/v26/v15/v20/v25/#39/#41/#43 + 历史多次) 物理不可达 5% (R@10=0.1075) 目标。

---

## C. Issue N 系列裁定 (issue12-30, 40, 55-57, 76)

### C.1 Issue #12-15 (Stage2 κ synchronization 早期裁定)

| 文件 | 主题 | 状态 |
|---|---|---|
| [[issue12_precheck_verdict]] | Issue #12 precheck | verdict |
| [[issue12_p2_token_sid_mapping]] | issue #12 P2 token-SID mapping 证据 | JSON |
| [[issue12_step2_step3_canary_result]] | issue #12 step2/3 canary | JSON |
| [[issue12_step4_beam_search_result]] | issue #12 step4 beam search | JSON |
| [[issue13_precheck_verdict]] | Issue #13 precheck | verdict |
| [[issue13_step1_protocol_audit]] | issue #13 protocol audit | JSON |
| [[issue13_step2_step3_canary_result]] | issue #13 step2/3 canary | JSON |
| [[issue13_step5_mixing_diagnostic]] | issue #13 mixing diagnostic | JSON |
| [[issue13_step6_beam_search_result]] | issue #13 step6 beam search | JSON |
| [[issue13_step1_step2_step3_step4_step5_step6_verdict]] | issue #13 全 6 步裁定 | verdict |
| [[issue14_precheck_verdict]] | Issue #14 precheck | verdict |
| [[issue14_full_eval_result]] | issue #14 full eval (DDP) | JSON |
| [[issue14_root_cause]] | issue #14 根因分析 | JSON |
| [[issue14_step1_step2_step3_verdict]] | issue #13 全 3 步裁定 | verdict |
| [[issue14_kappa_codebook_sync]] | issue #14 κ-codebook sync | JSON |
| [[issue15_precheck_verdict]] | Issue #15 precheck | verdict |
| [[issue15_step1_curvature_fix_verify]] | issue #15 step1 verify | JSON |
| [[issue15_step2_precheck_rejudge]] | issue #15 step2 rejudge | JSON |
| [[issue15_step4_gate3_nogo_verdict]] | issue #15 step4 NO-GO | verdict |

### C.2 Issue #17-23 (Stage 3 防线 / 一致率 / mixing variant)

| 文件 | 主题 |
|---|---|
| [[issue17_step1_2_3_evidence]] / [[issue17_final_step1_to_6_verdict]] | Issue #17 系列 |
| [[issue17_step5_nogo_verdict]] | Issue #17 step5 NO-GO |
| [[issue17_step6_canary]] | Issue #17 step6 canary |
| [[issue18_option_a_nogo_verdict]] / [[issue18_option_b_nogo_verdict]] / [[issue18_final_no_go_verdict]] | Issue #18 全 NO-GO |
| [[issue19_r18_nogo_verdict]] | Issue #19 |
| [[issue20_option_c_nogo_verdict]] | Issue #20 |
| [[issue21_r18_nogo_verdict]] | Issue #21 |
| [[issue22_r23_nogo_verdict]] | Issue #22 |
| [[issue23_option_d_r23_verdict]] | Issue #23 (R23 触发) |

### C.3 Issue #188-#193 (canary 收口 + 长程运行)

| 文件 | 主题 |
|---|---|
| [[issue188_gate3_fix_a_result]] | Issue #188 |
| [[issue189_gate3_fix_b_result]] | Issue #189 |
| [[issue190_canary_autoregressive_a_result]] | Issue #190 |
| [[issue191_canary_autoregressive_b_result]] | Issue #191 |
| [[issue192_long_run_result]] | Issue #192 |
| [[issue193_long_run_result]] | Issue #193 |

### C.4 Issue #24-29 (precheck / protocol)

| 文件 | 主题 |
|---|---|
| [[issue24_precheck_v3_verdict]] | Issue #24 |
| [[issue25_protocol_ab_verdict]] | Issue #25 |
| [[issue26_precheck_v4_verdict]] / [[issue26_t1c_canary_verdict]] | Issue #26 |
| [[issue27_gate3_close_verdict]] | Issue #27 |
| [[issue28_alpha131_fulltest]] / [[issue28_natural_fullvalid]] | Issue #28 (logs) |
| [[issue29_natural_fullvalid]] | Issue #29 (log) |

### C.5 Issue #40, #55-57, #76

| 文件 | 主题 | 状态 |
|---|---|---|
| [[issue40_final_evidence]] | Issue #40 协议换 source | (历史) |
| [[issue76_curvprior_end2end_verdict]] | Issue #76 curvprior 端到端 (FAIL, 0.047) | ❌ |
| [[task157_issue55_mode_collapse_no_go]] | Issue #55 mode collapse NO-GO | ❌ |
| [[task158_issue57_gate0_stage4_nogo]] | Issue #57 stage4 NO-GO | ❌ |

---

## D. T5 容量 / 训练 (Task #157-#161)

| 文件 | 主题 | 状态 | 关键数据 |
|---|---|---|---|
| [[task157_t5_base_capacity_unlock_result]] | T5-base 220M 容量解锁 | ✅ | R@10 > 0.1020 |
| [[task157_issue55_mode_collapse_no_go]] | #55 mode collapse | ❌ | n/a |
| [[task158_armA_beam20_metrics]] | armA beam20 实测 | JSON | |
| [[task158_issue57_gate0_stage4_nogo]] | #57 Gate0 NO-GO | verdict | |
| [[task175_orc_locked_result]] / [[task175_orc_locked_metrics]] | Task #175 orc 锁定 | (混合) | |
| [[task176_broken_baseline_terminated]] / [[task176_posdep_sigmoid_t5small_metrics]] | Task #176 pos-dep sigmoid | (终止) | |
| [[task177_broken_baseline_terminated]] / [[task177_posdep_conformal_t5small_metrics]] | Task #177 pos-dep conformal | (终止) | |
| [[task181_phase0.6_result]] / [[task181_phase0.6_metrics]] | Task #181 phase0.6 | verdict | |
| [[task186_fresh_1000epoch_result]] | Task #186 fresh 1000 ep | (历史) | |
| [[task187_encoder4_result]] | Task #187 encoder4 | (历史) | |

---

## E. 复现审计

| 文件 | 主题 | 关键 finding |
|---|---|---|
| [[task55_opq_result]] | OPQ (ITQ) 变体 — 负结果 | 闭式旋转无效 |
| [[task56_curvature_result]] / [[task56_curvature]] | κ 扫描 (10 个值) | c=1 健康区 |
| [[phonism_v5_5defense_legacy_verdict]] | phonism Sinkhorn+vanilla | R@10=0.1058 |
| [[paper_aligned_defaults_v2_verdict]] | paper-aligned defaults v2 | (paper 契合) |
| [[stage2_collapse_fix_verdict]] | Stage2 量化器塌缩修复 | poincare recon + κ clamp |
| [[phase0_mode_collapse_verdict]] | phase0 mode collapse | (诊断) |
| [[gate_m_cond5_w_upper_bound_result]] | Gate M cond5 w upper bound | (诊断) |

---

## F. 几何 NO-GO 系列

7 方向几何路线 + Entailment Cones + Latent Radius Live 全部 NO-GO/NOPE（详见 [[index|旧 index §B]] 共同根因 `√c·ρ 张力`）:

> 官方 HG-Rec baseline 的 `√c·ρ ≈ 0.54` (Task #189/#191 实测: ‖p‖=0.262/0.100/0.072, λ≈2.01-2.15) 远低于激活阈值 2, 几何不参与分配. 把 √c·ρ 推入激活区 (>2) 时, 几何激活与量化可分性**直接冲突**.

- **距离饱和** (Task #209 A3): dyn_range 2.14 → 1.27, 球面单层薄壳, collision 99.97%
- **死码本螺旋** (Task #211 C1): L0 利用率 23.4% (15/64), 钉半径→纯余弦分配→49/64 死码

详细参见 [[north_star_ceiling_status]] 收线总结 + paper.md §6.7.

---

## G. 方向 C 适配器 (Task #448-#475, issue157-187)

> **方向 C** = κ learning 编码到 T5 adapter 旁路（不修改 stage2，stage3 T5 加 wrapper）。本节记录该方向的逐步裁定。

| Task | Issue | 主题 | 状态 | 关键数据 |
|---|---|---|---|---|
| #448 | #157 | Gate2 κ-sync 重新校准 | ✅ PASS | (rec) |
| #449 | #158 | Gate2 weighted-mixed curvature | ✅ PASS | (rec) |
| #468 | #175 | Stage2 κ-vq-loss forward path fix | (系列) | |
| #469 | #176 | Stage2 mixing κ-vq-loss fix | (系列) | |
| #470 | #177 | Gate3a recontinue | (系列) | |
| #471 | #178 | Gate3b recontinue | (系列) | |
| #472 | #179 | 方向A Gate4 200ep (BoundedKappaScaleConditioner) | ❌ NO-GO | Stage4 R@10=**0.0000** (R23) |
| #473 | #181 | 方向B Gate4 200ep | verdict | |
| #474 | #186 | canary stage4 argmax | (系列) | |
| #475 | #187 | canary stage4 argmax | (系列) | |

**关键教训**: 训练仿真 `val_R@10_sim=0.1150` ≠ Stage4 真实 R@K → 协议 split，必须双跑实测。

---

## H. Issue #30 α-sweep / beam20 reeval

> 大文件集 (60+ json), Issue #30 α-sweep / beam20 全量复评结果。`alphasweep_a*_logit_*.json` 共 26 个 logit 扫描点 + `taskA_*_v5_sweep*` 共 11 个 α sweep 变体 + `taskB_*_v5a1_sweep*` 共 5 个 β sweep 变体。

| 关键文件 | 主题 |
|---|---|
| [[issue30_beam20_reeval_verdict]] | Issue #30 beam20 总览 |
| [[issue30_taskA_pure_t5_test_beam20_reeval]] | taskA pure-t5 beam20 baseline |
| [[issue30_taskA_v8_fulltest_beam20_reeval_beam20_reeval]] | taskA v8 fulltest |
| [[issue30_taskA_v7c_fulltest_beam20_reeval_beam20_reeval]] | taskA v7c fulltest |
| [[issue30_taskB_v5_beam20_reeval]] | taskB v5 beam20 baseline |
| [[issue30_taskB_v6_stage2inj_test_beam20_reeval]] | taskB v6 stage2-inj test |
| [[issue30_taskA_v6_reeval_beam20_reeval]] / [[issue30_taskA_v6_asweep_beam20_reeval]] | taskA v6 系列 |
| [[issue30_taskA_fine_logit_-2.3626_beam20_reeval]] / [[issue30_taskA_fine_logit_-2.4855_beam20_reeval]] | taskA fine logit scan |
| [[issue30_taskA_fine_logit_-2.0597_beam20_reeval]] / [[issue30_taskA_fine_logit_-1.8953_beam20_reeval]] / [[issue30_taskA_fine_logit_-2.1518_beam20_reeval]] / [[issue30_taskA_fine_logit_-2.7833_beam20_reeval]] | taskA fine logit scan |
| [[issue30_taskA_fine_logit_-2.3626_beam20_reeval]] | taskA fine logit scan |
| [[issue30_taskA_v5_sweep_a1.04_beam20_reeval]] / [[issue30_taskA_v5_sweep_a1.486_beam20_reeval]] / [[issue30_taskA_v5_sweep_a1.82_beam20_reeval]] | taskA v5 α sweep |
| [[issue30_taskA_v5_sweep2_a1.685_beam20_reeval]] / [[issue30_taskA_v5_sweep2_a1.751_beam20_reeval]] / [[issue30_taskA_v5_sweep2_a1.977_beam20_reeval]] / [[issue30_taskA_v5_sweep2_a2.06_beam20_reeval]] | taskA v5 sweep2 |
| [[issue30_taskA_alphasweep_v2_logit_-1.0502_beam20_reeval]] / [[..._-2.2522_...]] / [[..._-3.4915_...]] / [[..._-4.6002_...]] / [[..._-5.2958_...]] | taskA v2 logit scan |
| [[issue30_taskA_alphasweep_0.005_beam20_reeval]] / [[..._0.01_...]] / [[..._0.03_...]] | taskA α 0.005/0.01/0.03 |
| [[issue30_taskA_v5b_beam20_reeval]] / [[issue30_taskA_v5_prior_beam20_reeval]] / [[issue30_taskA_v5_prior_rerun_beam20_reeval]] / [[issue30_taskA_v5_prior_valid_beam20_reeval]] | taskA v5 系列 |
| [[issue30_taskB_v5a1_alpha0_beam20_reeval]] / [[issue30_taskB_v5a1_alpha0_rerun_beam20_reeval]] | taskB v5a1 α=0 |
| [[issue30_taskB_v5a1_sweep_b2.485_beam20_reeval]] / [[..._b2.97_...]] / [[..._b3.49_...]] / [[..._b3.9_...]] / [[..._b4.6_...]] | taskB v5a1 β sweep |
| [[issue30_taskB_v5_alpha01_beam20_reeval]] / [[issue30_taskB_v6_stage2inj_test_beam20_reeval]] | taskB 系列 |
| [[issue30_v8_stdSID_contrast_beam20_reeval]] / [[issue30_v10bsid_pre_beam20_reeval]] | baseline 对照 |
| [[issue30_entry_smoke_beam20_reeval]] / [[issue30_smoke_beam20_reeval]] / [[issue30_post_cleanup_smoke_beam20_reeval]] | 冒烟 |
| [[issue30_sanity_inputids_fullvalid_beam20_reeval]] / [[issue30_fullvalid_beam20_reeval]] | sanity |
| [[issue30_taskA_alphasweep_verify1_logit_-2.2522_beam20_reeval]] / [[..._verify2_...]] / [[..._verify3_...]] | taskA α verify |
| [[issue30_taskA_alphasweep_valid_logit_-2.2522_beam20_reeval]] | taskA α valid |
| [[issue30_issue28_alpha131_fulltest_beam20_reeval]] / [[issue30_issue28_diag_beam20_reeval]] | issue #28 |
| [[issue30_issue28_natural_fullvalid_beam20_reeval]] / [[issue30_issue29_natural_fullvalid_beam20_reeval]] | issue #28-29 |
| [[issue30_alphaboost_v0_fulltest_beam20_reeval]] / [[..._fullvalid_...]] | α-boost |
| [[issue30_alphaboost_v2_eos_fix_fulltest_beam20_reeval]] / [[issue30_alphaboost_v2_fulltest_beam20_reeval]] | α-boost v2 |
| [[issue30_alphasweep_a0_beam20_reeval]] / [[issue30_alphasweep_a-1_beam20_reeval]] / [[issue30_alphasweep_a1_beam20_reeval]] / [[issue30_alphasweep_a-2_beam20_reeval]] / [[issue30_alphasweep_a2_beam20_reeval]] / [[issue30_alphasweep_a-3_beam20_reeval]] | α-sweep a0..2 |
| [[issue30_alphaboost_v0_fullvalid_beam20_reeval]] / [[issue30_alphaboost_v0_test_beam20_reeval]] / [[issue30_alphaboost_v0_test_a_beam20_reeval]] / [[issue30_alphaboost_v0_natural_fulltest_beam20_reeval]] | α-boost logs |
| [[issue30_taskA_eos0_pure_t5_beam20_reeval]] | taskA EOS=0 pure-t5 |
| [[sanity_inputids_fullvalid.log]] / [[alphaboost_v0_natural_fulltest.log]] / [[alphaboost_v0_test_a.log]] | raw logs |

> **Obsidian 折叠**: 默认已折叠, 详细列表见 [[issue30_beam20_reeval_verdict]] 或上面所有 `[[issue30_*]]` 链接（用 Ctrl+Click 跳转）。

---

## I. 诊断 / 分析报告

| 文件 | 主题 |
|---|---|
| [[_analysis_why_no_baseline_exceeded]] | **根因分析 (Task A/B 未超 baseline)** — eval 协议不匹配 (greedy argmax vs beam20) |
| [[d0_round1_result]] | D0 轮 1 结果 |
| [[codebook_cnorm_distribution_verdict]] | codebook ‖·‖ 分布 |
| [[codebook_hypnorm_diagnostic_verdict]] | hypnorm 诊断 |
| [[c_k_distribution_diagnostic]] / [[c_k_distribution_diagnostic.json]] | κ 分布 JSON |
| [[cross_epoch_drift_diagnostic]] | cross-epoch drift |
| [[gate1_evidence]] | gate1 证据 |
| [[_fullsplit_baseline_correction]] | full-split baseline 校正 |
| [[_fullsplit_baseline_probe]] | full-split probe |
| [[_phase0_kappa_scan]] / [[_phase0_kappa_scan_rescale]] / [[_phase0_kappa_scan_no_go]] / [[_phase0_kappa_scan_task220_no_go]] | phase0 κ scan |
| [[_phase0_radius_pin]] / [[_phase0_radius_pin.json]] | phase0 radius pin |
| [[_phase0j_data_audit]] / [[_phase0j_data_audit.json]] | phase0j 数据审计 |
| [[_jplan_phase1a]] / [[_jplan_phase1_anchor_sweep_no_go]] | phase1 计划 |
| [[_phase0b_anti_collapse]] | phase0b anti-collapse |
| [[_protocol_probe_deciding]] | protocol probe |
| [[gate_m_cond5_w_upper_bound_result]] | Gate M cond5 上界 |
| [[task_gate_skip_history_v6]] | gate skip 历史 |
| [[north_star_ceiling_status]] | north-star ceiling 收线 |

---

## J. Idle / Loop tick log

| 文件 | 时间 |
|---|---|
| [[idle_status_2026-08-02]] | 2026-08-02 |
| [[idle_status_2026-08-02_1735]] | 17:35 |
| [[idle_status_2026-08-02_1800]] | 18:00 |

---

## K. Paper / 文档收尾

| 文件 | 主题 |
|---|---|
| [[task157_t5_base_capacity_unlock_result]] | T5-base 容量解锁 paper §5.x |
| [[paper_aligned_defaults_v2_verdict]] | paper-aligned defaults v2 |
| [[phonism_v5_5defense_legacy_verdict]] | phonism 5 防御 legacy |
| [[north_star_ceiling_status]] | ceiling 状态 |
| [[stage2_collapse_fix_verdict]] | Stage2 collapse fix |
| [[task_gate_skip_history_v6]] | gate skip v6 历史 |

---

## L. 关键数字速查

| 指标 | 值 | 来源 |
|---|---|---|
| HG-Rec baseline valid R@10 | **0.1267** | Task #84 |
| HG-Rec baseline test R@10 | **0.1024** | Task #84 |
| HG-Rec paper test R@10 | 0.1315 | HG-Rec paper Table 1 |
| phonism Sinkhorn+vanilla R@10 | 0.1058 | phonism legacy verdict |
| **hyp v2 端到端 test R@10 (历史最佳)** | **0.1048** (+0.0024) | [[taskA_hyp_v5_capmatch_relstruct_bf16_bs1024_1000ep_verdict]] |
| hyp v2 valid R@10 | 0.1269 (+0.0002) | 同上 |
| hyp v2 NDCG@20 | 0.0860 (+0.0105) | 同上 |
| Issue #41 anchored κ test | 0.0993 (-0.003) | [[taskA_hyp_v6_issue41_anchored_kappa_verdict]] |
| Issue #43 distance-bucket neg test | 0.0968 (-0.0056) | [[taskA_hyp_v7_issue43_distance_bucket_neg_verdict]] |
| Issue #37 KMeans cluster neg test | 0.1044 | [[taskA_hyp_v6_issue37_layered_semantic_neg_verdict]] |
| Issue #39 κ EMA + TR | 0.1002 (-0.0022) | [[taskA_hyp_v6_issue39_kappa_ema_tr_verdict]] |
| Task #472 方向A Gate4 | R@10=0.0000 (R23 触发) | [[task472_issue179_direction_a_gate4_200ep_result]] |
| 5% 超基线 (R@10=0.1075) 不可达确认次数 | **9** | v22/v26/v15/v20/v25/#39/#41/#43 + 历史 |

---

## 决策原则 (R10/R11 v2)

> 按 CLAUDE.md R11 自主决策 + R19 激进 owner + R22 OPEN 立即闭环:
> - 5% 超基线 (R@10=0.1075) 在 taskA hyp 路线**物理不可达**已 **9 次确认** → 收线
> - 唯一 operative mechanism: **hyp v2 端到端** (per-item radius + capmatch REL_STRUCT + 自由 κ)
> - 方向 C 适配器 (Task #448-#475) Gate4 多次 R@10=0 → wrapper broken，禁止继续
> - 任何 OPEN issue 出现 → 立即 R16+R17+R18+R20+R21 闭环

---

## Vault 拓扑结构

```
verdicts/                      ← Obsidian vault 根
├── index.md                   ← 本 MOC (入口)
├── README.md                  ← vault 使用说明
├── *.md (67 个)               ← 已加 YAML frontmatter + 类型识别
├── *.json (148 个)            ← 机器 verdict, Obsidian 不展示但可 grep
└── *.log (10+ 个)             ← raw 输出, 同上
```

### Obsidian 打开方式

```bash
# macOS
open -a Obsidian /home/wlia0047/ar57/wenyu/GeneRec/verdicts/

# Linux
obsidian /home/wlia0047/ar57/wenyu/GeneRec/verdicts/   # 如果 obsidian-cli 已装

# 手动: Obsidian → Open Vault → Other → 选 verdicts/ 目录
```

### 链接语法

- `[[index]]` — 指向本 MOC
- `[[task472_issue179_direction_a_gate4_200ep_result]]` — 具体 verdict 文件
- `issue41` (无方括号) — issue 抽象名（注意: vault 内 issue41 没有同名 .md/.json 文件，落地请用 `[[taskA_hyp_v6_issue41_anchored_kappa_verdict]]` 跳到对应 verdict）
- tags: #hyp / #taskA / #direction-a / #kappa / #alpha-sweep / #beam20-reeval / #loop-tick 等

### Frontmatter 字段

所有 `.md` 文件已加:

```yaml
type: verdict | precheck | canary | diagnostic | analysis | status | index
issue: 41           # 可选
task: 448           # 可选
gate: 4             # 可选 (1..4)
status: PASS | FAIL | PARTIAL | NO-GO | NOPE
tags: [taskA, hyp, kappa, ...]
created: 2026-08-02
up: "[[index]]"
```

Obsidian 可用 Dataview plugin 直接查表:
```dataview
TABLE status, type, issue, task
FROM ""
SORT created DESC
```

---

> 本 MOC 由 `add_obsidian_frontmatter.py` + `patch_task_ids.py` 自动整理 (2026-08-05)。备份位于 `$CLAUDE_JOB_DIR/tmp/verdicts_backup_2026-08-05/`。
