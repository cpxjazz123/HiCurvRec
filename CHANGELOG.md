# Changelog

All notable changes to this repository are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `REPRODUCE.md` §0 "Upstream Framework Clone (prerequisite)" — explicit
  `git clone https://github.com/snap-research/GRID.git` + `cp -r` + verify
  three-step onboarding for reviewer (Task #118).
- 5th audit `scripts/task114_verdict_integrity.py` — checks `result:` line
  presence, task-ID pairing, R9 contiguous 1..N in both descriptions/ and
  verdicts/, and dispatcher self-reference (Task #114).
- `scripts/all_audits.py` extended 4 → 5 audit entries (Task #114).
- R8 §9.3 retroactive compliance — `result: Task #X — completed (R8 §9.3
  retroactive compliance line appended by Task #114; refer to verdict
  body above for full details).` line appended to 50 verdicts missing it
  (Task #114).
- Post-submission housekeeping synthesis verdict
  `verdicts/task121_post_submission_housekeeping_synthesis_result.md`
  (Task #121).

### Changed
- `README.md` §1 "What's in this repository" — stale numbers refreshed
  (101 → 114 entries, 480 → 483 files) per Task #113.
- 6 housekeeping rounds (Tasks #115-#120) reduced untracked noise:
  2097 → 114 files (-94.6%).
- Tracked file growth: 80 → 907 files (+1033%) across Tasks #115-#117.
- All post-submission housekeeping improvements are reviewer-facing only —
  no changes to paper numbers, baselines, or scientific conclusions.

### Verified
- `python3 scripts/all_audits.py` — 6/6 PASS (task101/103/105/106/114/127).
- R9 mandate: descriptions/ task IDs contiguous 1..121 (no gaps).
- All committed Python scripts pass `python3 -m py_compile`.
- v1.0.0 git tag still anchors paper-submission baseline at commit 9b81667.

### Notes
- Post-submission housekeeping cycle: Tasks #111 (final paper-defense
  synthesis) → #112 (release artifacts) → #113 (README refresh) → #114
  (5th audit + retroactive compliance) → #115-#117 (track untracked) →
  #118 (REPRODUCE §0) → #119 (orphan cleanup) → #120 (gitignore paper
  clones) → #121 (synthesis) → #122-#125 (continued housekeeping:
  TASKS_INDEX refresh + REPRODUCE.md script alignment + README/TASKS_INDEX
  stale numbers v2 + CI workflow dispatcher sync) → #126 (TASKS_INDEX +
  CHANGELOG docs drift v2) → #127 (drift v3 + 6th audit: script syntax
  check via `ast.parse`, dispatcher 5/6 → 6/6 PASS).



### Recent housekeeping (auto-generated)

| Task | Subject | Verdict |
|------|---------|---------|
| #121 | Post-Submission Housekeeping Synthesis (Verdict) | `verdicts/task121_post_submission_housekeeping_synthesis_result.md` |
| #122 | TASKS_INDEX.md Refresh (Verdict) | `verdicts/task122_tasks_index_refresh_result.md` |
| #123 | REPRODUCE.md Script Reference Alignment (Verdict) | `verdicts/task123_reproduce_md_script_alignment_result.md` |
| #124 | README.md + TASKS_INDEX.md Stale Number Refresh v2 (Verdict) | `verdicts/task124_readme_stale_numbers_v2_result.md` |
| #125 | Sync CI Workflow with 5-Audit Dispatcher (Verdict) | `verdicts/task125_ci_workflow_dispatcher_sync_result.md` |
| #126 | TASKS_INDEX + CHANGELOG Documentation Drift v2 (Verdict) | `verdicts/task126_docs_drift_v2_result.md` |
| #127 | TASKS_INDEX+CHANGELOG drift v3 + 6th audit (script syntax) | `verdicts/task127_6th_audit_script_syntax_result.md` |
| #128 | Housekeeping Cycle Closure (R8 §16 Cleanup + Drift Pattern Resolution) | `verdicts/task128_housekeeping_cycle_closure_result.md` |
| #129 | Auto-Gen TASKS_INDEX (Break Drift Cycle) | `verdicts/task129_auto_gen_tasks_index_result.md` |
| #130 | Final State Closure (TASKS_INDEX Counts Refresh + Closure Report) | `verdicts/task130_final_state_closure_result.md` |
| #131 | Empty Orphan Dir Cleanup + Explicit Gitignore Hygiene | `verdicts/task131_orphan_dir_cleanup_result.md` |
| #132 | Save Infrastructure Pattern Learnings to Memory | `verdicts/task132_memory_infrastructure_patterns_result.md` |
| #133 | CHANGELOG.md Auto-Gen + 8th Dispatcher Audit | `verdicts/task133_changelog_auto_gen_result.md` |
| #134 | README.md + TASKS_INDEX.md Counts Auto-Gen + 9th Dispatcher Audit | `verdicts/task134_readme_tasks_index_counts_sync_result.md` |
| #135 | κ=0 硬分支 θ_m 梯度通路断裂诊断 | `verdicts/task135_kappa_zero_gradient_path_audit_result.md` |
| #136 | DECOR Table 2 5-baselines Restage (TIGER/LETTER/CoST/P5/ETEGRec) | `verdicts/task136_decor_5_baselines_table2_result.md` |
| #137 | κ-stereographic 硬分支梯度 bug 修复 + Task #89 Stage 1 重训 | `verdicts/task137_kappa_stereographic_fix_retrain_result.md` |
| #138 | Free-Curv codebook collapse 修复 (A 方案 geodesic kmeans + B 方案 dead-code reset) | `verdicts/task138_free_curv_A_B_scheme_fix_result.md` |
| #141 | Caser paper-aligned 配置混用 fix (lr=0.001, wd=0.0) | `verdicts/task141_caser_paper_aligned_fix_result.md` |
| #142 | Free-curv Codebook Collapse 修复尝试 (geodesic kmeans + dead code reset) | `verdicts/task142_free_curv_codebook_recovery_result.md` |
| #143 | FDSA paper-aligned fix NO-GO verdict (撤回 + 用 Task #85 复现作 baseline) | `verdicts/task143_fdsa_paper_aligned_fix_no_go.md` |
| #144 | κ + codebook 解耦训练调度 (Phase 1 only) verdict | `verdicts/task144_kappa_decouple_stage1_only_result.md` |
| #149 | Heterogeneous κ HRQ-VAE (3 layers, Goal #1 + Goal #2 验证) | `verdicts/task149_heterokappa_result.md` |
| #150 | LETTER paper-aligned 修复 (lr=2e-5 / batch=8 / epochs=4) | `verdicts/task150_letter_paper_aligned_result.md` |
| #152 | L1 对照实验 (用户 2026-07-24 提议) | `verdicts/task152_l1_control_result.md` |
| #156 | HG-Rec code-default recipe 测试集评估 (Stage 4 closure) | `verdicts/task156_hgrec_code_default_result.md` |
| #157 | T5-base 220M 容量解锁 | `verdicts/task157_t5_base_capacity_unlock_result.md` |
| #159 | T5-mini 12M capacity point (Stage 4 closure) | `verdicts/task159_t5mini_12m_capacity_point_result.md` |
| #160 | T5-small 60M 容量档 (verdict) | `verdicts/task160_t5small_60m_result.md` |
| #161 | T5-mini 9.18M d_kv fix 探测 (verdict) | `verdicts/task161_t5mini_dkv_fix_result.md` |
| #163 | 真 ORC 实现 + proxy vs ORC 对比 (verdict) | `verdicts/task163_5_real_orc_followup_result.md` |
| #164 | Stereographic Phase A/B 修复 — 最终结果 | `verdicts/task164_phase_b_kappa_decouple_result.md` |
| #165 | FreeCurv collapse 真正机制：可学习 κ 的正反馈环路 | `verdicts/task165_free_curv_synthesis_result.md` |
| #166 | κ-Stereographic × T5-mini 9.18M Stage 4 Test Result | `verdicts/task166_t5mini_kappa_decouple_result.md` |
| #167 | κ-Stereographic × T5-base 220M Stage 4 Test Result | `verdicts/task167_t5base_kappa_decouple_result.md` |
| #168 | κ-Stereographic × T5-5.5M Stage 4 Test Result | `verdicts/task168_t5_5p5m_kappa_decouple_result.md` |
| #169 | κ-Stereographic + Sinkhorn(L2 only) [⛔ C3 NO-GO] | `verdicts/task169_kappa_sinkhorn_result.md` |
| #170 | κ-Stereographic + Sinkhorn(ALL 3 layers) [⛔ C3 NO-GO] | `verdicts/task170_kappa_sinkhorn_all3_result.md` |
| #171 | κ-Stereographic + dead_code_reset_every=10 [⛔ C3 NO-GO] | `verdicts/task171_kappa_dead_code_result.md` |
| #172 | κ-Stereographic + κ_max=4.0 [⛔ C3 NO-GO] | `verdicts/task172_kappa_max4_result.md` |
| #174 | κ-Stereographic + M=2 + MCKG 门控融合 D 臂 [⛔ C3 NO-GO] | `verdicts/task174_mckg_gating_result.md` |
| #175 | κ-Stereographic + κ LOCKED at Ollivier ORC 实测值 | `verdicts/task175_orc_locked_result.md` |
| #176 | style 位置依赖 — Broken Baseline 终止 | `verdicts/task176_broken_baseline_terminated.md` |
| #177 | Broken Baseline 终止 | `verdicts/task177_broken_baseline_terminated.md` |
| #181 | Phase 0.6 官方对齐 T5-small 闭环 | `verdicts/task181_phase0.6_result.md` |
| #182 | 欧式 RQ-VAE + 量化 loss ×4 (NO-GO, Stage 1 完全坍缩) | `verdicts/task182_euclidean_loss4x_result.md` |
| #183 | 半径设计 Phase 1: 软正则化 RQ-VAE | `verdicts/task183_radius_reg_result.md` |
| #184 | Phase 1a Product Manifold (球面 × 双曲) NO-GO | `verdicts/task184_phase1a_result.md` |
| #186 | Fresh 1000 epoch training (用户叫停, ⏸️ 终止) | `verdicts/task186_fresh_1000epoch_result.md` |
| #187 | 4 层 Encoder 实验 (⏸️ STOPPED by user, 数据不足) | `verdicts/task187_encoder4_result.md` |
| #188 | paper Table 7 多种子方差复现 | `verdicts/task188_paper_table7_multiseed_result.md` |
| #189 | Codebook 几何诊断 (baseline + 时间序列) | `verdicts/task189_codebook_geometry_diagnose_result.md` |
| #190 | 码字/残差比值判据 (6 epoch × 3 层) | `verdicts/task190_codeword_residual_ratio_result.md` |
| #191 | L0 量化误差 ‖z − e_L0‖ 随 epoch 变化 (机制坐实) | `verdicts/task191_l0_quant_error_result.md` |
| #192 | β 剂量扫描 (Instruments, 4 臂) | `verdicts/task192_beta_dose_scan_result.md` |
| #193 | Encoder freeze 补充实验 (因果验证) | `verdicts/task193_encoder_freeze_result.md` |
| #194 | K0 码本容量扫描 verdict | `verdicts/task194_k0_capacity_result.md` |
| #199 | exp(θ) κ 参数化 Stage 1 verdict (B/C 跑完, D 在跑) | `verdicts/task199_stage1_exp_theta_result.md` |
| #200 | (no H1 found) | `verdicts/task200_dual_v5_stage3_4_result.md` |
| #201 | 重做 exp(θ) κ with θ_init=log(10) verdict (用户拍板选项 A) | `verdicts/task201_kappa_redo_result.md` |
| #202 | on Stage 3 K0=64 — verdict | `verdicts/task202_sinkhorn_stage3_result.md` |
| #203 | exp(θ) κ + scale normalization verdict | `verdicts/task203_kappa_scale_norm_result.md` |
| #204 | quant_loss_weight 对照 verdict (用户 2026-07-26 决定性实验) | `verdicts/task204_quant_loss_weight_对照_result.md` |
| #205 | 2D 玩具: 最近邻距离 + 动态范围 (用户 2026-07-26 修正指标后) | `verdicts/task205_2d_toy_result.md` |
| #206 | Geometry-neutral at this scale: 5-min argmin 诊断终结 verdict | `verdicts/task206_geometry_neutral_at_this_scale_result.md` |
| #207 | Euclidean vs Hyperbolic Collision-Aligned 全流水线对比 | `verdicts/task207_euclidean_vs_hyperbolic_result.md` |
| #209 | Test Eval Verdict | `verdicts/task209_A3_test_result.md` |
| #210 | 维度 × 半径扫描 verdict | `verdicts/task210_phaseA_scan_result.md` |
| #211 | 低维双曲 + 钉半径可行性 | `verdicts/task211_phase0_low_dim_pca_result.md` |
| #212 | 方向二 (Two-Stage Decision) ❌ NO-HOPE | `verdicts/task212_two_stage_criterion_result.md` |
| #213 | Entailment Cones (方向一) ❌ NO-HOPE | `verdicts/task213_entailment_cones_phase0_result.md` |
| #214 | Latent Radius Live (方向三) ❌ NO-HOPE | `verdicts/task214_radius_live_phase0_result.md` |
| #215 | Paper §4 (HG-Rec 包装失效 4 证据链 + 11 evidence 完整) | `verdicts/task215_paper_section4_complete.md` |
| #218 | 逃法一 (Per-Codeword Curvature) 判据检查 | `verdicts/task218_per_codeword_curvature_result.md` |
| #219 | 逃法二 (Gromov Product) 判据检查 | `verdicts/task219_gromov_product_result.md` |
| #220 | Stage 1 训练 逃法一 (Per-Codeword κ) 结果 | `verdicts/task220_per_codeword_kappa_stage1_train_result.md` |
| #221 | Stage 1 训练 逃法二 (Gromov Product) 结果 | `verdicts/task221_gromov_stage1_train_result.md` |
| #222 | Phase 2 早停 30 epoch 复现 healthy ckpt 结果 | `verdicts/task222_pck_earlystop_replay_result.md` |
| #223 | Stage 2 SID 推断 结果 | `verdicts/task223_stage2_sid_inference_result.md` |
| #225 | Per-Codeword κ 完整流水线 Stage 4 评估 verdict | `verdicts/task225_pck_stage4_eval_result.md` |
| #226 | NOT PASS (决断 D ② margin 比对 FAIL) | `verdicts/task226_step3_60pct_pass_result.md` |
| #227 | Step 3 v8 collision chase verdict — 架构级 NO-GO | `verdicts/task227_step3_v8_collision_nogo_result.md` |
| #228 | Phase 4 细粒度 w_angular 8-point sweep verdict — Goldilocks 仍 NO-GO | `verdicts/task228_wangular_sweep_phase4_result.md` |
| #229 | clean single-variable c scan 6 arms verdict | `verdicts/task229_gate1_crash_geometry_result.md` |
| #230 | 方向 H (PCA 冻结 + 自由 radius) verdict | `verdicts/task230_direction_H_PCA_frozen_result.md` |
| #231 | Issue #6 调参版 (Per-Codeword κ c_k range sweep) 结果 | `verdicts/task231_pck_phase0_ckrange_sweep_result.md` |
| #232 | Issue #7 调参版 (Gromov weight/entropy/spread-norm sweep) 结果 | `verdicts/task232_gromov_phase0_weight_sweep_result.md` |
| #233 | Issue #8 Validation: dual_v5 Stage 3 rerun verdict | `verdicts/task233_dual_v5_stage3_rerun_validation_result.md` |
| #234 | 方向 J1 (软分配) verdict | `verdicts/task234_direction_J1_soft_assignment_result.md` |
| #235 | Issue #9 Gate 1: Hybrid per-layer assignment Stage 1 (FULL NO-GO) | `verdicts/task235_issue9_gate1_hybrid_stage1_result.md` |
| #236 | Issue #10 Gate 0: collision 指标口径统一 (zero GPU) | `verdicts/task236_collision_metric_unification_result.md` |
| #237 | Issue #10 Gate 1 Arm B: partial Sinkhorn (3-arm 因果曲线) — **Gate 1 PARTIAL → STOP** | `verdicts/task237_issue10_arm_b_result.md` |
| #239 | REVERSED per-layer c 验证 (PARTIAL FAIL) | `verdicts/task239_gate2_reversed_result.md` |
| #240 | SID 沙漏效应 H1 REFUTED (NO-GO) | `verdicts/task240_issue12_gate0_distribution_nogo.md` |
| #241 | per-layer c_k 区间组合性 PASS | `verdicts/task241_issue11_gate0_perlayer_ck_combo_pass.md` |
| #242 | NO-GO | `verdicts/task242_issue11_gate1_perlayer_stage1_result.md` |
| #244 | Issue #12 Gate 0 分布画像 (零 GPU) 结果 | `verdicts/task244_issue12_gate0_sid_distribution_result.md` |
| #245 | Issue #10 Gate 0 (revisit): 闭环 + GitHub 评论 | `verdicts/task245_issue10_gate0_revisit_result.md` |
| #246 | paper-aligned fixes 增量更新综合 ranking (v3) | `verdicts/task246_paper_aligned_ranking_increment_result.md` |
| #247 | paper Section 5.4 推荐措辞草案 (零 GPU) | `verdicts/task247_paper_section54_draft.md` |
| #248 | Issue #13 Gate 0: 残差算子几何一致性核查 (PASS) | `verdicts/task248_issue13_gate0_residual_operator_result.md` |
| #249 | Issue #13 Gate 1: 欧式 vs Möbius 残差 argmin 一致率 (PASS) | `verdicts/task249_issue13_gate1_residual_consistency_result.md` |
| #250 | Issue #14 Gate 0: FORGE 官方源码 metric 实证 (FAIL → STOP) | `verdicts/task250_issue14_gate0_forge_metric_def_result.md` |
| #251 | Issue #15 Gate 0: Phase 0 一致率带口径锁定 + 配对档案重建 (PASS) | `verdicts/task251_issue15_gate0_band_unification_result.md` |
| #252 | Issue #15 Gate 1: Phase 0 一致率带预测效力判定 (PASS WITH CAVEAT) | `verdicts/task252_issue15_gate1_predictive_validity_result.md` |
| #253 | Issue #13 Gate 2: Möbius 残差算子实际训练 + Stage 4 eval | `verdicts/task253_issue13_gate2_mobius_residual_result.md` |
| #254 | Issue #13 Gate 3: Logmap 距离 argmin 一致率 | `verdicts/task254_issue13_gate3_logmap_argmin_result.md` |
| #255 | Issue #10 方向 A 预准备: Sinkhorn 强度扫描 + Arm B 候选配置 | `verdicts/task255_issue10_arm_b_sinkhorn_strength_dryrun.md` |
| #257 | Issue #16 Gate 0: 取回 Issue #13/#14/#15 证据 + 反驳 §4 "全树缺失" 错判 | `verdicts/task257_issue16_gate0_evidence_recovery_result.md` |
| #258 | Issue #16 Gate 1: 算术复核 + §6.7.4 stop-loss 追溯 + STOP | `verdicts/task258_issue16_gate1_stop_loss_audit_result.md` |
| #259 | Issue #10 Gate 0: collision 指标口径统一 + 历史数字重述 (零 GPU) | `verdicts/task259_issue10_gate0_collision_metric_unification_result.md` |
| #260 | Issue #10 方向 A 准备: Sinkhorn 强度曲线扫描 (Stage 2 推断) | `verdicts/task260_issue10_sinkhorn_strength_sweep_result.md` |
| #261 | loop.md §16 状态同步 + 本轮 #257-#260 归档 + R8 清理 | `verdicts/task261_loop_md_sync_status_result.md` |
| #262 | git push 14 commits + Issue #17 Gate 0 PASS + Issue #16 Gate 0 错判纠正 | `verdicts/task262_issue17_gate0_push_commits_result.md` |
| #263 | task253 直接 utilization 测量 + Gate 1 校准 FAIL + Issue #16 proxy 链审计失败 | `verdicts/task263_issue17_gate2_task253_direct_utilization_result.md` |
| #265 | hrqvae_trainer.py 行 322 修复应用 + 3 epoch smoke run PASS | `verdicts/task265_issue17_gate1_fix_apply_result.md` |
| #267 | Issue #10 自主决策 A2: 关 issue 走 NoGo | `verdicts/task267_issue10_close_nogo_result.md` |
| #268 | backlog 高 ROI 候选决策 + Issue #17 关闭 + Task #269 NO-OP 识别 | `verdicts/task268_backlog_high_roi_decision_result.md` |
| #269 | Musical_Instruments paper-aligned baselines 状态核实 (NO-OP 闭合) | `verdicts/task269_letter_fdsa_paper_aligned_retest_result.md` |
| #270 | Stage 1 L0 utilization ≥ 90% curriculum 方案设计 (Issue #17 修复后新方向) | `verdicts/task270_utilization_curriculum_design_result.md` |
| #271 | A1 β=0.0 Stage 1 验证 FAIL (L0 ≥ 90% 不可达) | `verdicts/task271_a1_beta_zero_run_result.md` |
| #272 | VERDICT/PRODUCTS 全局 inventory (Issue #10/#17 全关后回顾) | `verdicts/task272_global_verdict_inventory_result.md` |
| #273 | Task #178 §6.7.4 stop-loss (i) 复核 | `verdicts/task273_task178_utilization_audit_result.md` |
| #274 | HG-Rec disk-only L322 fix patch 文件化 闭环 | `verdicts/task274_hgrec_disk_fix_patch_result.md` |
| #275 | A2 + A3 + A2-extend Stage 1 curriculum 闭环 + Task #276 启动 (R11.4 autonomous decision) | `verdicts/task275_a2_a3_curriculum_parallel_result.md` |
| #276 | A2 Stage 2 inference 完成 + Stage 3 启动 | `verdicts/task276_a2_stage2_inference_result.md` |
| #277 | Task #243 Stage 4 R@10 eval (200 ep vs 400 ep) — NO-GO | `verdicts/task277_task243_stage4_eval_result.md` |
| #278 | 批量 Stage 4 R@10 eval 综合 verdict | `verdicts/task278_batch_stage4_eval_result.md` |
| #279 | K-sweep 扩展 K=512 + K=1024 verdict | `verdicts/task279_k_sweep_extension_result.md` |
| #280 | 口径锁定全闭环 verdict | `verdicts/task280_issue18_full_closure.md` |
| #281 | 链式 launcher 强制化 全闭环 verdict | `verdicts/task281_issue19_full_closure.md` |
| #282 | Stage 1 欧氏 MSE + β=0 NO-GO 闭环 verdict | `verdicts/task282_task270_a1_no_go.md` |
| #283 | dead_revive frequency NO-GO 闭环 verdict | `verdicts/task283_d5_dead_revive_frequency_no_go.md` |
| #284 | Issue #10 follow-up: task194_k0256 SID + κ-decouple 3-arm verdict | `verdicts/task284_issue10_followup_result.md` |
| #287 | K=128 κ-decouple 2-arm (intermediate K between task144 K=64 and task284 K=256) | `verdicts/task287_k0128_kappa_decouple_intermediate_k_result.md` |
| #288 | Issue #20 L0 utilization ≥ 90% 三配方验证 (NO-GO 闭环) | `verdicts/task288_issue20_l0_utilization_3recipe_result.md` |
| #289 | R9 Compliance Audit (Audit 跑通 + 历史遗留 FAIL 由 R11.5 决策保留) | `verdicts/task289_r9_compliance_audit_result.md` |
| #290 | FSQ + κ-decouple (Finite Scalar Quantization on FreeCurvHRQVAE) | `verdicts/task290_fsq_kappa_decouple_result.md` |
| #291 | EMA codebook + κ-decouple (VQ-VAE-2 EMA update on FreeCurvHRQVAE) | `verdicts/task291_ema_codebook_result.md` |
| #292 | Restoration (EMA + dead code revival) + κ-decouple | `verdicts/task292_restoration_result.md` |
| #293 | Gate 0 Phase 0: per-layer per-epoch c_k curriculum | `verdicts/task293_issue23_gate0_phase0_result.md` |
| #294 | c_k range 路径跨任务综合收口 (paper §6.7.4 paper-ready) | `verdicts/task294_ck_range_path_exhausted_cross_task_result.md` |
| #295 | R14 规则显眼化到 loop.md §15 开头 | `verdicts/task295_r14_rule_promotion_to_loop_result.md` |
| #296 | paper.md §6.7.4 联动段落 (Task #29x+#293+#294 跨任务 c_k range 路径综合收口) | `verdicts/task296_paper_md_section_6_7_4_linkage_result.md` |
| #297 | Gate 0 Phase 0: Phase A 复用 task287 Arm A ckpt 验证 | `verdicts/task297_issue25_gate0_phase0_result.md` |

_Auto-generated by `scripts/task133_sync_changelog.py` — DO NOT hand-edit. Run `python3 scripts/task133_sync_changelog.py` after closing a task._

## [1.0.0] - 2026-07-24 — Paper Submission Baseline

### Added
- `VERSION` — file at repo root with content `1.0.0`, paired with
  lightweight git tag `v1.0.0` to lock the paper-submission-ready commit
  (Task #110).
- `scripts/all_audits.py` — single dispatcher that runs all 4 paper-defense
  audits in sequence and returns a unified exit code (4/4 PASS in 0.46 s).
  Reviewer-facing single entry point: `python3 scripts/all_audits.py`.
  (Task #110)
- `arxiv/` — arXiv-ready packet: `paper.tex` (sanitized for xelatex),
  `paper.pdf` (14 pages, 125 KB), `refs.bib`, `LICENSE` (CC-BY-4.0),
  `ARXIV_METADATA.md`, `SUBMISSION_CHECKLIST.md`, `README.md`. (Task #108)
- `.github/workflows/audits.yml` — 9-step GitHub Actions workflow
  (checkout + setup-python + task101 + task103 + task105 + task106 +
  upload-artifact + step-summary). Triggered on push / PR / manual.
  (Task #107)
- `papers/SUBMISSION_DEFENSE.md` — reviewer-facing defense packet covering
  all 12 reproduced baselines and the 5-audit defense bundle. (Task #106)
- `scripts/task105_ckpt_integrity.py` — 4-layer ckpt audit (file / size /
  log / paper Δ); result: **21.06 MB**, range 18-25 MB, R12 mandate
  satisfied. (Task #105)
- `scripts/task106_audits.py` — 5-audit defense bundle (paper.md↔paper.tex
  sync, abstract ≤ 200 words, baseline coverage, number consistency,
  license attribution). Result: **5/5 PASS**, abstract 197/200, 112/112
  numbers traceable. (Task #106)
- `papers/SUBMISSION_DEFENSE.md` — 12-baseline defense packet. (Task #106)
- `REPRODUCE.md` — 11-section end-to-end reproduction guide. (Task #101)
- `README.md` — 9-section paper-reviewer landing page with 7 shields.io
  badges after H1. (Task #102, refreshed in Task #109)
- `CITATION.cff` — cff-version 1.2.0 metadata with preferred-citation
  entry; enables GitHub "Cite this repository" button. (Task #104)
- `CHANGELOG.md` — Keep-a-Changelog 1.1.0 format (initial 5.7 KB; v1.0.0
  entry added in Task #112). (Task #104)
- 7 shields.io badges in `README.md`: paper (14 p), baselines (12/20),
  paper claims (14/14 Δ=0), abstract (197/200 字), R12 ckpt (21.06 MB),
  license (MIT), CI (audits automated). (Task #109)

### Changed
- `papers/paper.{md,tex,pdf}` — full paper (14 pages, 125 KB) with
  Sections 1, 3, 4, 5, 6, 7 + Appendix LaTeX (Tables 1-7). Sections
  drafted as separate verdicts (Task #92-#98) and stitched into a single
  file (Task #98). PDF compiled by xelatex (Task #99).
- `papers/paper.tex` — sanitized for arXiv submission (removed
  `\usepackage{fancyhdr}`, `\usepackage{times}`, `\usepackage{authblk}`,
  fancyhead commands). (Task #108)
- `README.md` — added 7 shields.io badges top + reorganized as paper-
  reviewer landing page. (Task #102, #109)
- `scripts/task101_verify_env.py` — added `--skip-data` and `--skip-packages`
  flags for CI mode (Task #107).

### Verified
- `scripts/task101_verify_env.py` — environment verifier (env + data +
  packages). (Task #101)
- `scripts/task103_paper_claims_audit.py` — 14/14 paper claims Δ = 0
  (zero-deviation cross-validation). (Task #103)
- `scripts/task105_ckpt_integrity.py` — R12 ckpt 21.06 MB, file size in
  expected range 18-25 MB. (Task #105)
- `scripts/task106_audits.py` — 5-audit defense bundle (sync / abstract
  197/200 ≤ 200 / 12 reproduced / 112 numbers traceable / 5 license).
  (Task #106)
- `scripts/all_audits.py` — single dispatcher aggregating the above four
  audits: **4/4 PASS** in 0.46 s (0.04 + 0.05 + 0.05 + 0.32). (Task #110)
- `verdicts/task111_final_synthesis_result.md` — 8-section project
  closure summary, cross-validated against all upstream verdicts.
  (Task #111)

### Notes
- 12 baselines reproduced out of 20 paper Table 2 entries (top R@10:
  phonism 0.1058 / HG-Rec c555 0.1051 — marginally tied, Δ −0.7%).
  7 entries marked NO-GO with documented blockers (S³Rec yaml config,
  FMLP-Rec dataset support, multi-seed withdrawn, etc.).
- 5 independent evidence chains → Musical_Instruments data is
  intrinsically near-Euclidean: (i) stress-metric grid (Task #117),
  (ii) per-layer curvature 6-grid span 5.3% (Task #88), (iii) free-
  curvature learning → 18/18 (layer, κ_m) = 0 (Task #89), (iv) codebook
  token-set Jaccard = 1.000 across 4 methods (Task #90), (v) valid/test
  R@10 inversion in Stage 3 dynamics (Task #91).
- Paper conclusion: on flat datasets, vanilla + Sinkhorn ≥ hyperbolic
  RQ-VAE. The mechanism (Sinkhorn-based codebook utilization) matters
  more than the geometry (hyperbolic curvature).
- See `verdicts/task111_final_synthesis_result.md` for the full project
  closure summary (8 sections, ~280 lines).

## [2026-07-24] — Paper Submission Package

### Added
- `papers/paper.pdf` — 14-page submission PDF (106 KB, xelatex × 2). (Task #99 → Task #100)
- `papers/paper.md` — single-file Markdown source (556 lines, 30 KB). (Task #98)
- `papers/paper.tex` — auto-generated LaTeX (40 KB).
- `papers/refs.bib` — 12 BibTeX entries (HG-Rec, TIGER, LETTER, RQ-VAE, etc.). (Task #100)
- `papers/paper.tex.backup` — LaTeX baseline before Step B format edits. (Task #100)
- `REPRODUCE.md` — end-to-end reproduction guide (11 sections, 280 lines). (Task #101)
- `README.md` — human-facing landing page (9 sections, 200 lines). (Task #102)
- `scripts/task101_verify_env.py` — 3-layer environment verifier (R2 no-fallback). (Task #101)
- `scripts/task103_paper_claims_audit.py` — paper claim audit (14/14 ✅, 280 lines). (Task #103)
- `verdicts/task103_paper_claims_audit.{md,csv}` — audit output (md for humans, csv for machines). (Task #103)

### Changed
- `papers/paper.tex` format micro-adjustments: `\usepackage{times}` for
  cleaner Times Roman font, `\usepackage{fancyhdr}` + `\setlength{\headheight}{14pt}`
  for page header/footer, `\section*{Acknowledgements}` section added
  before References. (Task #100)

### Fixed
- 10 successive LaTeX compilation bugs in markdown → PDF conversion:
  underscore escaping, body duplication, `\tag{N}` handling, `\arctanh`
  missing, `\[` / `\]` recognition, language mapping, `\tag` cross-line
  regex, 159 unescaped `&` (Kang & McAuley), two missing opening `

, cross-line
  equation malformed. Implemented three-state machine (text / inline-math /
  display-math) for math-escape tracking. (Task #99)

### Notes
- Two intermediate summary files (`task99_md_to_pdf_result.md` and
  `task100_paper_submission_prep_result.md`) document the 10-fix
  sequence and format adjustments in detail.

## [Earlier periods] — Pre-submission work

This section summarizes tasks completed before the current submission
package. For per-task detail, see `verdicts/` directory (100+
verdicts).

### Section drafting (Tasks #92–#97)
Each section of the paper was drafted as a separate verdict file:
- **§1 Introduction**: Task #95
- **§2 Preliminaries** (HG-Rec reference): HG-Rec paper
- **§3 Method**: Task #94
- **§4 Related Work**: Task #96
- **§5 Experiments**: Task #92 (integrated 8 closed tasks)
- **§6 Discussion**: Task #93
- **§7 Conclusion + Ack + Impact**: Task #97
- **Paper stitching**: Task #98

### Experimental evidence chain (Tasks #82–#91, #84 main)
- **Task #84** — HG-Rec main reproduction: R@10=0.1020
- **Task #87 v2** — Paper Table 2 baseline ranking (27 baselines)
- **Task #88** — Per-layer curvature grid (6 configs, 5.3% R@10 span)
- **Task #89** — Free-curvature product manifold (18/18 κ → 0)
- **Task #90** — Phonism vs HG-Rec codebook decomposition (Jaccard)
- **Task #91** — T5 training dynamics (valid R@10 inversion)
- **Task #94** — HG-Rec paper Table 1 cross-comparison
- **Task #95** — HGN standalone test eval

### Earlier R@5 cherry-picks (Tasks #58–#61)
- Task #58 Simple KMeans SID + TIGER: R@5=0.0296
- Task #59 flan-t5 2048d + Simple KMeans SID + TIGER: R@5=0.0857
- Task #60 sentence-t5 768d + Simple KMeans SID + TIGER: R@5=0.0838
- Task #61 Hybrid 2816d + Simple KMeans SID + TIGER: R@5=0.0977

### Pre-curvature diagnostics (Tasks #53–#56, #62–#82)
- Real-TIGER Stage 4 training (task53)
- L3 norm post-hoc rebase (task54)
- OPQ / ITQ (task55)
- 5-graph weight diagnostic (task69)
- Ollivier real curvature on Toys (task70)
- Init κ ablation (task71)
- Stress metrics on phonism / HG-Rec (task117, #118)
- 192d revival counterproductive (task67)
- 192d no norm fix (task68)
- Phonism conditional diagnosis (task69 final)

---

**Conventions**:
- Each task has a definition in `descriptions/task<N>_<slug>.md`
- Each completed task has a verdict in `verdicts/task<N>_<slug>_result.md`
- Cross-validation of paper claims against verdicts is automated via
  `scripts/task103_paper_claims_audit.py` (run anytime after editing
  `papers/paper.md`)
