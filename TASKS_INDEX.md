# GRID Tasks Index

> **Snapshot**: 2026-07-24 (post Task #121 post-submission housekeeping synthesis)
> **Status**: ✅ v1.0.0 paper-submission baseline closed + 11 post-submission housekeeping rounds (Tasks #111-#121)

## Three artifact categories

- **descriptions/**: task definitions (markdown). 127 files total — 125
  numbered files covering 123 unique task IDs (Task #1 - Task #123
  contiguous per R9; Task #27 and Task #67 have 2 revision files each)
  + 2 reference files (`README.md`, `_general_pipeline.md`).
- **verdicts/**: task result verdicts (markdown + json + csv + png + legacy).
  255 files — verdicts for Task #1 - Task #123 + auxiliary verdicts +
  data files (task62_tc_zador.csv, task103_paper_claims_audit.csv,
  task109_phonism_sid_*, task119_3tokenizer_kNN_*,
  task123_cross_space_correlation_*, task27_neighborhood_quality_*,
  phonism_v5_5defense_legacy_verdict.md).
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

## Recent closed tasks (Task #101 - Task #125)

| Task | Subject | Status | Verdict |
|------|---------|--------|---------|
| #101 | REPRODUCE.md + env verifier | ✅ | `verdicts/task101_reproduce_md_result.md` |
| #102 | README.md paper-reviewer landing | ✅ | `verdicts/task102_readme_md_result.md` |
| #103 | Paper claims cross-validation audit | ✅ | `verdicts/task103_paper_claims_audit_result.md` |
| #104 | CITATION.cff + CHANGELOG.md | ✅ | `verdicts/task104_citation_changelog_result.md` |
| #105 | R12 ckpt integrity 4-layer audit | ✅ | `verdicts/task105_ckpt_integrity_result.md` |
| #106 | 5-audit defense bundle | ✅ | `verdicts/task106_submission_defense_result.md` |
| #107 | GitHub Actions CI (9-step workflow) | ✅ | `verdicts/task107_github_actions_ci_result.md` |
| #108 | arXiv submission packet (7 files) | ✅ | `verdicts/task108_arxiv_packet_result.md` |
| #109 | 7 shields.io badges in README.md | ✅ | `verdicts/task109_shields_badges_result.md` |
| #110 | Audit dispatcher + VERSION 1.0.0 + tag | ✅ | `verdicts/task110_audit_dispatcher_version_result.md` |
| #111 | Final synthesis verdict (project closure) | ✅ | `verdicts/task111_final_synthesis_result.md` |
| #112 | v1.0.0 release artifacts consolidation | ✅ | `verdicts/task112_release_artifacts_result.md` |
| #113 | README stale number refresh | ✅ | `verdicts/task113_readme_stale_numbers_result.md` |
| #114 | 5th audit (verdict integrity) + R8 §9.3 retroactive | ✅ | `verdicts/task114_verdict_integrity_result.md` |
| #115 | Track 166 verdicts + 104 descriptions | ✅ | `verdicts/task115_track_untracked_result.md` |
| #116 | Track 466 utility scripts + 20 reference papers | ✅ | `verdicts/task116_track_scripts_papers_result.md` |
| #117 | Project-specific .gitignore + track CLAUDE.md + requirements.txt | ✅ | `verdicts/task117_gitignore_project_specific_result.md` |
| #118 | REPRODUCE.md §0 Upstream Framework Clone | ✅ | `verdicts/task118_reproduce_setup_section_result.md` |
| #119 | Orphan file cleanup (descriptions + task data + scripts) | ✅ | `verdicts/task119_orphan_file_cleanup_result.md` |
| #120 | gitignore 15 paper reference clones + reports/result cleanup | ✅ | `verdicts/task120_gitignore_paper_clones_result.md` |
| #121 | Post-submission housekeeping synthesis + CHANGELOG [Unreleased] | ✅ | `verdicts/task121_post_submission_housekeeping_synthesis_result.md` |
| #122 | TASKS_INDEX.md refresh (Task #113-#121 coverage) | ✅ | `verdicts/task122_tasks_index_refresh_result.md` |
| #123 | REPRODUCE.md script reference alignment (7 nonexistent → 8 real) | ✅ | `verdicts/task123_reproduce_md_script_alignment_result.md` |
| #124 | README + TASKS_INDEX stale number refresh v2 (114→125, 253→255) | ✅ | `verdicts/task124_readme_stale_numbers_v2_result.md` |
| #125 | CI workflow sync with 5-audit dispatcher (4 separate → 1 dispatcher) | ✅ | `verdicts/task125_ci_workflow_dispatcher_sync_result.md` |
| #122 | TASKS_INDEX.md refresh (Task #113-#121 coverage) | ✅ | `verdicts/task122_tasks_index_refresh_result.md` |
| #123 | REPRODUCE.md script reference alignment | ✅ | `verdicts/task123_reproduce_md_script_alignment_result.md` |

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