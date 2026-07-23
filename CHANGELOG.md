# Changelog

All notable changes to this repository are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

### Changed

### Removed

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
