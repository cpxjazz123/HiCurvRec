# arXiv Submission Pre-Upload Checklist

> Run each check before submitting to arXiv. If any item fails, fix the bundle
> and re-run. Items marked ⚠️ are warning-only (block submission if
> critical issue; otherwise proceed).

---

## Bundle completeness

- [ ] `arxiv/paper.tex` exists and is the sanitized version
- [ ] `arxiv/paper.pdf` exists and ≤ 10 MB (actual: 106 KB)
- [ ] `arxiv/refs.bib` exists with 12 BibTeX entries
- [ ] `arxiv/fig/` exists (even if empty)
- [ ] `arxiv/LICENSE` exists (CC-BY-4.0)

## LaTeX compiles clean

```bash
cd arxiv
xelatex paper.tex
bibtex paper
xelatex paper.tex
xelatex paper.tex
# Verify no "Error" / "Overfull \hbox" / "Undefined reference" warnings
# Allowed: minor spacing warnings on long lines
```

- [ ] xelatex exit 0
- [ ] bibtex exit 0
- [ ] xelatex #2 exit 0 (resolves refs)
- [ ] xelatex #3 exit 0 (final version)
- [ ] No "Undefined reference" / "Undefined citation" warnings
- [ ] No `?` placeholders in compiled PDF (all refs resolved)

## References completeness

- [ ] Every `\cite{X}` in `paper.tex` has matching entry in `refs.bib`
- [ ] No orphan `@entry` in `refs.bib` (warnings only, but cleaner bundle)
- [ ] `bibtex` accepts the .bib file (no malformed entries)

```bash
grep -oE '\\cite\{[^}]+\}' paper.tex | sort -u > /tmp/cited.txt
grep -oE '@[a-z]+\{[^,]+,' refs.bib | sort -u > /tmp/defined.txt
# diff: every cite should have a defined key (strip prefix etc)
```

## File size constraints

- [ ] `paper.pdf` < 10 MB (arXiv limit)
- [ ] Source ZIP < 50 MB (arXiv limit per submission)
- [ ] No embedded `.eps` files (use `.pdf`/`.png`/`.jpg`)

## License compliance

- [ ] `LICENSE` file matches submitted license form (CC-BY-4.0)
- [ ] Abstract or footer mentions license if required
- [ ] All third-party content (figures, code snippets) is cited

## Metadata consistency

- [ ] Title in `ARXIV_METADATA.md` matches paper.tex `\title{}` (verbatim)
- [ ] Abstract in submission form matches paper.tex abstract section
- [ ] Authors list matches paper.tex `\author{}` (verbatim, including
  any LaTeX special chars like `\\`)
- [ ] Categories correct: `cs.IR` (primary) + `cs.LG` (cross)
- [ ] Comments field populated (optional but recommended)

## Final sanity

- [ ] Open `paper.pdf` and check: 14 pages, header reads correctly
- [ ] Open `paper.tex` and check: no `fancyhdr`, no `\usepackage{times}`,
      no `authblk` mentioned
- [ ] Cross-check with `papers/SUBMISSION_DEFENSE.md` — 5/5 audits still
      pass on the paper.md source
- [ ] Run `git log -- papers/paper.tex | head -5` to see last commit
      author & timestamp

---

## Upload steps

1. ✅ Run checks above
2. Bundle: `cd arxiv && zip -r ../arxiv_submission_v1.zip paper.tex refs.bib fig/ LICENSE`
3. Go to https://arxiv.org/submit
4. Login (or register); first submission may need endorser
5. Step 1: Begin submission → choose "Scientific Papers"
6. Step 2: Upload files
   - Source files: `arxiv_submission_v1.zip`
   - Optional: Preview PDF: `arxiv/paper.pdf`
7. Step 3: Process files (auto-build verification) — verify "Success"
8. Step 4: Metadata — paste from `ARXIV_METADATA.md`
9. Step 5: License — select CC-BY-4.0
10. Step 6: Verify submission summary — confirm
11. Submit — wait for moderation (1-3 business days)
12. After approval: copy arXiv ID (e.g., `2607.12345`) and update
    `README.md` and `papers/SUBMISSION_DEFENSE.md` with the URL

---

**Re-run any failing check before submitting. arXiv submissions cannot be
edited after acceptance (only version-overwritten after 24h).**
