# Changelog

All notable changes to this repository are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `CITATION.cff` — GitHub-native citation metadata (cff-version 1.2.0) for
  the reproduction paper, including preferred citation entry and the
  original HG-Rec reference. (Task #104)

### Notes
- This is an independent reproduction of HG-Rec (Zhang et al., ICML 2026)
  on Amazon Musical_Instruments (a dataset not featured in the original
  headline experiments). The reproduction package consists of:

  - `papers/paper.{md,tex,bib,pdf}` — submission-ready paper (14 pages, 106 KB)
  - `papers/refs.bib` — 12 BibTeX entries
  - `README.md` — human-facing landing page (TL;DR + Headline Findings)
  - `REPRODUCE.md` — end-to-end reproduction guide
  - `scripts/task101_verify_env.py` — environment verifier
  - `scripts/task103_paper_claims_audit.py` — paper claim audit (14/14 ✅)
  - `src/`, `configs/`, `data/` — unchanged GRID framework (Apache 2.0)

- **Headline numbers** (test R@10 on Musical_Instruments, seed=42):
  - Vanilla RQ-VAE + Sinkhorn ("phonism"): **0.1058** ⭐
  - HG-Rec (κ = 0.5 per layer): **0.1051** (marginally tied, Δ −0.7%)
  - HG-Rec (κ = 1.0, paper default): 0.1020
  - HG-Rec free-curvature (κ → 0 learned): 0.1015
  - LETTER: 0.0997
  - TIGER: 0.0591

- **Five-step reproduction protocol** (paper §5): stress-metric
  diagnostics, per-layer curvature grid, free-curvature learning,
  token-set Jaccard decomposition, training-dynamics comparison.

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
