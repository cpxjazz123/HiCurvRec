# arXiv Submission Metadata

> Submission configuration for arXiv preprint server (https://arxiv.org/submit).
> Use this exact metadata when filing through the arXiv submission form.

---

## Required Metadata

| Field | Value |
|-------|-------|
| **Title** | Hyperbolic RQ-VAE enhanced Generative Recommendation with Differential-Length Codebook: An Independent Reproduction on Amazon Musical_Instruments |
| **Authors** | Reproduction of HG-Rec (Zhang et al., ICML 2026) |
| **Abstract** | (verbatim from paper.tex, 197 words, ≤ ICML 200 word limit) |
| **Categories (1 primary required, 0+ cross optional)** | cs.IR (Information Retrieval) [primary]; cs.LG (Machine Learning) [cross] |
| **Comments** | Reproduction of HG-Rec (Zhang et al., ICML 2026), 14 pages, 7 figures, 7 tables |
| **License** | CC-BY-4.0 (Creative Commons Attribution 4.0 International) |
| **Version** | 1 (initial submission) |

---

## Source Files (upload as ZIP)

| File | Origin |
|------|--------|
| `paper.tex` | `arxiv/paper.tex` (sanitized: removed `fancyhdr`/`times`/`authblk`) |
| `refs.bib` | `arxiv/refs.bib` (12 BibTeX entries from papers/refs.bib) |
| `fig/` | empty placeholder (paper has no embedded figures) |
| `LICENSE` | `arxiv/LICENSE` (CC-BY-4.0 short text) |

---

## ZIP creation

```bash
# From project root
cd arxiv
zip -r arxiv_submission_v1.zip paper.tex refs.bib fig/ LICENSE
# Upload at https://arxiv.org/submit
```

---

## Categories explained

- **cs.IR (Information Retrieval)** — primary, since this is fundamentally an
  information-retrieval evaluation (Recall@K / NDCG@K on Amazon dataset).
- **cs.LG (Machine Learning)** — cross-list, since the reproduction compares
  Euclidean vs hyperbolic RQ-VAE (geometric deep learning subfield).

If cs.IR is unavailable as primary (rare arXiv form issue), use **cs.AI**
(Artificial Intelligence) as primary with both cs.IR and cs.LG as cross.

---

## License rationale

**CC-BY-4.0** chosen because:
1. arXiv requires license declaration per 2024 policy; CC-BY is the
   most permissive standard license and the most common for reproducibility
   packages.
2. Aligns with the upstream HG-Rec paper authors' use of research-use
   without redistribution restriction.
3. Compatible with the repo's MIT license (MIT for code, CC-BY-4.0 for
   paper text per arXiv norms).

Alternatives if CC-BY-4.0 is undesired:
- `CC-BY-SA-4.0` (viral share-alike)
- `CC-BY-NC-4.0` (non-commercial restriction)
- `CC0-1.0` (public domain dedication)
- `arXiv perpetual non-exclusive` (default, restrictive — not recommended)

---

## Submission form mapping

| arXiv form field | Our value |
|------------------|-----------|
| Identifier (skip — auto-assigned) | (e.g., `2607.12345` after moderation) |
| Submitter | `wlia0047@github.com` (GitHub noreply) |
| Title | (verbatim from metadata table above) |
| Authors | (single author; affiliation: "Independent Reproduction") |
| Abstract | (verbatim from paper.tex abstract section) |
| Comments to the editors | "Reproduction of HG-Rec (Zhang et al., ICML 2026). Code: https://github.com/wlia0047/GeneRec" |
| Categories | `cs.IR` (primary) + `cs.LG` (cross) |
| License | `CC-BY-4.0` |
| Source files | upload ZIP |
| Submitter's permission | "I authorize posting of this submission" |
| Verify your submission | verify → submit |

---

## Post-submission expectations

- **Moderation delay**: arXiv submission usually takes 1-3 business days
  for first-time submitters; ≤ 24h once established.
- **Endorsement**: arXiv may require endorsement for the first submission
  by a new author in cs.IR. Have an endorser ready.
- **Version updates**: future versions require separate submission
  (overwrites after 24h unless withdrawn).

---

**Last updated**: 2026-07-24 by Task #108.
