# GRID Tasks Index

> **Snapshot**: 2026-07-24 (post Task #112 v1.0.0 release artifacts consolidation)
> **Status**: ✅ v1.0.0 paper-submission baseline closed

## Three artifact categories

- **descriptions/**: task definitions (markdown). 116 files total — 114
  numbered tasks (Task #1 - Task #112 contiguous per R9) + 2 reference
  files (`README.md`, `_general_pipeline.md`).
- **verdicts/**: task result verdicts (markdown + json + csv + png). 232
  files — verdicts for Task #1 - Task #111 + auxiliary verdicts for
  Task #116, #117, #118, #119, #120, #123, #125, #126, #131, #132, #134
  (cleanup / diagnostic / R9-audit tasks).
- **products/**: execution artifacts (ckpt/pt/json/log/paper). 46 task
  directories with physical run outputs.

## Coverage matrix (current snapshot)

| Range | Status | Notes |
|---|---|---|
| Task #1 - Task #91 | ✓ closed | Baselines + diagnostics + paper section drafts |
| Task #92 - Task #98 | ✓ closed | 7 paper section drafts (Sections 1, 3, 4, 5, 6, 7) |
| Task #99 - Task #100 | ✓ closed | paper.md → paper.pdf + submission prep |
| Task #101 - Task #110 | ✓ closed | 10 paper-defense artifacts (REPRODUCE / README / audits / CI / arXiv / badges / dispatcher / VERSION / tag) |
| Task #111 | ✓ closed | Final synthesis verdict (project closure summary) |
| Task #112 | ✓ closed | v1.0.0 release artifacts consolidation (this task) |
| Auxiliary (Task #116 - #134) | ✓ closed | R9 audits / diagnostic / cleanup verdicts |

## Recent closed tasks (Task #101 - Task #112)

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

## Paper-defense artifacts (10 件套, Task #101 - #110)

| File | Purpose | Verifier |
|------|---------|----------|
| `REPRODUCE.md` | End-to-end reproduction guide | `scripts/task101_verify_env.py` |
| `README.md` | Paper-reviewer landing page | (manual review) |
| `scripts/task103_paper_claims_audit.py` | Paper claim audit | `python3 scripts/task103_paper_claims_audit.py` |
| `CITATION.cff` + `CHANGELOG.md` | Citation + changelog | (yaml.safe_load) |
| `scripts/task105_ckpt_integrity.py` | R12 ckpt integrity | `python3 scripts/task105_ckpt_integrity.py` |
| `scripts/task106_audits.py` | 5-audit defense bundle | `python3 scripts/task106_audits.py` |
| `.github/workflows/audits.yml` | CI automation | (GitHub Actions runner) |
| `arxiv/` | arXiv packet | `pdfinfo arxiv/paper.pdf` |
| `README.md` (badges) | shields.io visual status | (browser rendering) |
| `scripts/all_audits.py` + `VERSION` + `v1.0.0` | Single dispatcher + version pin | `python3 scripts/all_audits.py` |

## Conventions

- Each task has a definition in `descriptions/task<N>_<slug>.md`.
- Each completed task has a verdict in `verdicts/task<N>_<slug>_result.md`.
- Cross-validation of paper claims against verdicts is automated via
  `scripts/task103_paper_claims_audit.py` (run anytime after editing
  `papers/paper.md`).
- Reviewer-facing single entry point: `python3 scripts/all_audits.py`
  returns 0 iff all 4 paper-defense audits pass.
- Project closure: see `verdicts/task111_final_synthesis_result.md`.