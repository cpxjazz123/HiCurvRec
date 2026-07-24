# GRID Tasks Index

> **Snapshot**: 2026-07-24 (post Task #130 final state closure)
> **Status**: ✅ v1.0.0 paper-submission baseline closed + 20 post-submission housekeeping rounds (Tasks #111-#130)

## Three artifact categories

- **descriptions/**: task definitions (markdown). 134 files total — 132
  numbered files covering 130 unique task IDs (Task #1 - Task #130
  contiguous per R9; Task #27 and Task #67 have 2 revision files each)
  + 2 reference files (`README.md`, `_general_pipeline.md`).
- **verdicts/**: task result verdicts (markdown + json + csv + png + legacy).
  261 files — verdicts for Task #1 - Task #129 + auxiliary verdicts +
  data files (task62_tc_zador.csv, task103_paper_claims_audit.csv,
  task109_phonism_sid_*, task119_3tokenizer_kNN_*,
  task123_cross_space_correlation_*, task27_neighborhood_quality_*,
  phonism_v5_5defense_legacy_verdict.md, task127_*, task128_*,
  task129_*).
- **products/**: execution artifacts (ckpt/pt/json/log/paper). 46 task
  directories with physical run outputs (gitignored, regeneratable).

## Coverage matrix (current snapshot)

| Range | Status | Notes |
|---|---|---|
| Task #1 - Task #91 | ✓ closed | Baselines + diagnostics + paper section drafts |
| Task #92 - Task #98 | ✓ closed | 7 paper section drafts (Sections 1, 3, 4, 5, 6, 7) |
| Task #99 - Task #100 | ✓ closed | paper.md → paper.pdf + submission prep |
| Task #101 - Task #110 | ✓ closed | 10 paper-defense artifacts (REPRODUCE / README / audits / CI / arXiv / badges / dispatcher / VERSION / tag) |
| Task #111 - Task #112 | ✓ closed | Final synthesis + v1.0.0 release artifacts consolidation |
| Task #113 - Task #114 | ✓ closed | README stale number refresh + 5th audit (verdict integrity + R8 §9.3 retroactive) |
| Task #115 - Task #117 | ✓ closed | Track untracked: 270 verdicts/descriptions + 486 scripts/papers + .gitignore project-specific + CLAUDE.md + requirements.txt |
| Task #118 - Task #120 | ✓ closed | REPRODUCE.md §0 onboarding + orphan file cleanup + gitignore paper clones |
| Task #121 | ✓ closed | Post-submission housekeeping synthesis + CHANGELOG [Unreleased] update |
| Task #122 - Task #123 | ✓ closed | TASKS_INDEX.md refresh + REPRODUCE.md script reference alignment |
| Task #124 - Task #125 | ✓ closed | README + TASKS_INDEX stale number refresh v2 + CI workflow dispatcher sync |
| Task #126 | ✓ closed | TASKS_INDEX + CHANGELOG docs drift v2 |
| Task #127 | ✓ closed | TASKS_INDEX + CHANGELOG docs drift v3 + 6th audit (script syntax) |
| Task #128 - Task #130 | ✓ closed | Housekeeping cycle closure + auto-gen TASKS_INDEX (break drift cycle) + final state closure report |

## Recent closed tasks (Task #101 - Task #134)

| Task | Subject | Status | Verdict |
|------|---------|--------|---------|
| #101 | REPRODUCE.md 复现包 + 环境验证脚本 闭环 | ✅ | `verdicts/task101_reproduce_md_result.md` |
| #102 | README.md 顶层落地页 闭环 | ✅ | `verdicts/task102_readme_md_result.md` |
| #103 | Paper Claim Cross-Validation Audit 闭环 | ✅ | `verdicts/task103_paper_claims_audit_result.md` |
| #104 | CITATION.cff + CHANGELOG.md 闭环 | ✅ | `verdicts/task104_citation_changelog_result.md` |
| #105 | R12 Best Checkpoint Integrity Verification (Verdict) | ✅ | `verdicts/task105_ckpt_integrity_result.md` |
| #106 | Submission Defense Bundle (Verdict) | ✅ | `verdicts/task106_submission_defense_result.md` |
| #107 | GitHub Actions CI for Paper Defense Audits (Verdict) | ✅ | `verdicts/task107_github_actions_ci_result.md` |
| #108 | arXiv Submission Packet (Verdict) | ✅ | `verdicts/task108_arxiv_packet_result.md` |
| #109 | shields.io Badges + README Visual Upgrade (Verdict) | ✅ | `verdicts/task109_shields_badges_result.md` |
| #110 | Audit Dispatcher + VERSION Tag + Paper-Submission Baseline (Verdict) | ✅ | `verdicts/task110_audit_dispatcher_version_result.md` |
| #111 | Final Synthesis Verdict (Project Closure) | ✅ | `verdicts/task111_final_synthesis_result.md` |
| #112 | v1.0.0 Release Artifacts Consolidation (Verdict) | ✅ | `verdicts/task112_release_artifacts_result.md` |
| #113 | README.md Stale Number Refresh (Verdict) | ✅ | `verdicts/task113_readme_stale_numbers_result.md` |
| #114 | Verdict Integrity Audit (5th Audit for Dispatcher) (Verdict) | ✅ | `verdicts/task114_verdict_integrity_result.md` |
| #115 | Track Untracked Verdicts + Descriptions (Verdict) | ✅ | `verdicts/task115_track_untracked_result.md` |
| #116 | HG-Rec per-layer δ_95/diameter (原版 vs 改造版) | ✅ | `verdicts/task116_hgrec_delta_per_layer_result.md` |
| #117 | Project-Specific .gitignore (Verdict) | ✅ | `verdicts/task117_gitignore_project_specific_result.md` |
| #118 | HG-Rec codebook 利用率 / 碰撞率 (6 curvature 网格) | ✅ | `verdicts/task118_hgrec_codebook_utilization_result.md` |
| #119 | 3 tokenizer 简化 kNN 分析 + 3 核心问题回答 | ✅ | `verdicts/task119_3tokenizer_kNN_result.md` |
| #120 | gitignore Paper Reference Framework Clones (Verdict) | ✅ | `verdicts/task120_gitignore_paper_clones_result.md` |
| #121 | Post-Submission Housekeeping Synthesis (Verdict) | ✅ | `verdicts/task121_post_submission_housekeeping_synthesis_result.md` |
| #122 | TASKS_INDEX.md Refresh (Verdict) | ✅ | `verdicts/task122_tasks_index_refresh_result.md` |
| #123 | 跨空间 n=4 完整相关性分析 | ✅ | `verdicts/task123_cross_space_correlation_result.md` |
| #124 | README.md + TASKS_INDEX.md Stale Number Refresh v2 (Verdict) | ✅ | `verdicts/task124_readme_stale_numbers_v2_result.md` |
| #125 | Sync CI Workflow with 5-Audit Dispatcher (Verdict) | ✅ | `verdicts/task125_ci_workflow_dispatcher_sync_result.md` |
| #126 | TASKS_INDEX + CHANGELOG Documentation Drift v2 (Verdict) | ✅ | `verdicts/task126_docs_drift_v2_result.md` |
| #127 | TASKS_INDEX+CHANGELOG drift v3 + 6th audit (script syntax) | ✅ | `verdicts/task127_6th_audit_script_syntax_result.md` |
| #128 | Housekeeping Cycle Closure (R8 §16 Cleanup + Drift Pattern Resolution) | ✅ | `verdicts/task128_housekeeping_cycle_closure_result.md` |
| #129 | Auto-Gen TASKS_INDEX (Break Drift Cycle) | ✅ | `verdicts/task129_auto_gen_tasks_index_result.md` |
| #130 | Final State Closure (TASKS_INDEX Counts Refresh + Closure Report) | ✅ | `verdicts/task130_final_state_closure_result.md` |
| #131 | Empty Orphan Dir Cleanup + Explicit Gitignore Hygiene | ✅ | `verdicts/task131_orphan_dir_cleanup_result.md` |
| #132 | Save Infrastructure Pattern Learnings to Memory | ✅ | `verdicts/task132_memory_infrastructure_patterns_result.md` |
| #134 | CLAUDE.md R9-Enforce + audit 脚本 verdict | ✅ | `verdicts/task134_r9_enforce_audit_result.md` |

## Paper-defense artifacts (10 件套, Task #101 - #110)

| File | Purpose | Verifier |
|------|---------|----------|
| `REPRODUCE.md` (with §0 upstream clone) | End-to-end reproduction guide | `scripts/task101_verify_env.py` |
| `README.md` | Paper-reviewer landing page (with 7 shields.io badges) | (manual review) |
| `scripts/task103_paper_claims_audit.py` | Paper claim audit (14/14 baseline numbers) | `python3 scripts/task103_paper_claims_audit.py` |
| `CITATION.cff` + `CHANGELOG.md` (with [Unreleased] + [1.0.0]) | Citation + changelog | (yaml.safe_load) |
| `scripts/task105_ckpt_integrity.py` | R12 ckpt integrity | `python3 scripts/task105_ckpt_integrity.py` |
| `scripts/task106_audits.py` | 5-audit defense bundle | `python3 scripts/task106_audits.py` |
| `.github/workflows/audits.yml` | CI automation | (GitHub Actions runner) |
| `arxiv/` | arXiv packet (paper.pdf 14 pages) | `pdfinfo arxiv/paper.pdf` |
| `README.md` (badges) | shields.io visual status | (browser rendering) |
| `scripts/all_audits.py` + `VERSION` + `v1.0.0` | Single dispatcher + version pin (5 audits) | `python3 scripts/all_audits.py` |

## Post-submission housekeeping artifacts (Tasks #111-#121)

| File | Purpose | Verdict |
|------|---------|---------|
| `verdicts/task111_final_synthesis_result.md` | Project closure summary (8 sections) | #111 |
| `VERSION` + `CHANGELOG.md` + `TASKS_INDEX.md` | v1.0.0 release artifacts | #112 |
| `README.md` (refreshed) | Stale numbers fixed (114/483/24) | #113 |
| `scripts/task114_verdict_integrity.py` | 5th audit + dispatcher self-check | #114 |
| 270 verdicts/descriptions + 486 scripts/papers tracked | Phase 1 housekeeping | #115-#116 |
| `.gitignore` (extended) + `CLAUDE.md` + `requirements.txt` | Project-specific gitignore | #117 |
| `REPRODUCE.md` §0 | Reviewer upstream framework onboarding | #118 |
| 11 task data files in `verdicts/` | Move from top-level | #119 |
| 13 gitignore rules for paper clones | ~28 GB foreign git repos hidden | #120 |
| `CHANGELOG.md` [Unreleased] section | 24-commit post-submission summary | #121 |

## Conventions

- Each task has a definition in `descriptions/task<N>_<slug>.md`.
- Each completed task has a verdict in `verdicts/task<N>_<slug>_result.md`.
- Cross-validation of paper claims against verdicts is automated via
  `scripts/task103_paper_claims_audit.py` (run anytime after editing
  `papers/paper.md`).
- Reviewer-facing single entry point: `python3 scripts/all_audits.py`
  returns 0 iff all 5 paper-defense audits pass (task101 env + task103
  claims + task105 ckpt + task106 defense + task114 verdict integrity).
- R9 mandate: `descriptions/` task IDs contiguous 1..N (no gaps); verified
  by `scripts/task114_verdict_integrity.py` A3 sub-audit.
- R8 §9.3 mandate: every verdict ends with `result: Task #X — ...` line;
  verified by `scripts/task114_verdict_integrity.py` A1 sub-audit.
- Project closure: see `verdicts/task121_post_submission_housekeeping_synthesis_result.md`
  for the full post-submission housekeeping cycle summary.