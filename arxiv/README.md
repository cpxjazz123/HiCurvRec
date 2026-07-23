# arXiv Submission Bundle

> Auto-generated 2026-07-24 by Task #108 (Submission Defense Bundle, §17).
> This directory contains the arXiv submission-ready bundle for this
> reproduction paper.

---

## Files

| File | Purpose |
|------|---------|
| `paper.tex` | LaTeX source (sanitized: removed `fancyhdr`/`times`/`authblk` for arXiv auto-build) |
| `paper.pdf` | Final compiled PDF (14 pages, 106 KB) |
| `refs.bib` | Bibliography (12 BibTeX entries) |
| `fig/` | Empty (paper has no embedded figures) |
| `LICENSE` | CC-BY-4.0 short form |
| `ARXIV_METADATA.md` | Submission form fields (title / categories / license) |
| `SUBMISSION_CHECKLIST.md` | Pre-upload verification |
| `README.md` | This file |

---

## Quick submission

```bash
cd arxiv
zip -r ../arxiv_submission_v1.zip paper.tex refs.bib fig/ LICENSE

# Then go to https://arxiv.org/submit and upload:
#   1. Source ZIP: arxiv_submission_v1.zip
#   2. PDF preview: paper.pdf
#   3. Metadata: see ARXIV_METADATA.md
```

---

## Sanitization notes (vs `papers/paper.tex`)

| Element | papers/paper.tex | arxiv/paper.tex | Why |
|---------|-----------------|-----------------|-----|
| `\usepackage{fancyhdr}` | yes | **removed** | arXiv auto-build sometimes chokes on fancyhdr |
| `\usepackage{times}` | yes | **removed** | default font is fine; speeds up arXiv build |
| `\fancyhead[L]{\ldots}` | yes | **removed** | depends on fancyhdr |
| `\usepackage{authblk}` | commented | **removed** | TeX Live default OK |
| `\author{Reproduction Paper --- 2026}` | placeholder | **Reproduction of HG-Rec ... \\texttt{wlia0047@github.com}** | arXiv wants identifiable submitter |
| `\date{\today}` | dynamic | **July 24, 2026** | reproducible date |

---

## See also

- `../papers/SUBMISSION_DEFENSE.md` — 5-audit defense packet (Task #106)
- `../papers/refs.bib` — source BibTeX
- `../papers/paper.tex` — paper-conference-style version (for any future
  venue submission)
