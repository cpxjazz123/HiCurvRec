# GeneRec — Independent Reproduction of HG-Rec on Amazon Musical\_Instruments

<p align="left">
<a href="papers/paper.pdf"><img src="https://img.shields.io/badge/paper-14%20pages-blue?logo=adobeacrobatreader&logoColor=white" alt="paper pdf"></a>
<a href="papers/SUBMISSION_DEFENSE.md"><img src="https://img.shields.io/badge/baselines-12%2F20%20reproduced-brightgreen" alt="baselines reproduced"></a>
<a href="verdicts/task103_paper_claims_audit.md"><img src="https://img.shields.io/badge/paper%20claims-14%2F14%20%CE%94%3D0-brightgreen" alt="paper claims"></a>
<a href="verdicts/task106_audits.md"><img src="https://img.shields.io/badge/abstract-197%2F200%20words-green" alt="abstract compliance"></a>
<a href="verdicts/task105_ckpt_integrity.md"><img src="https://img.shields.io/badge/R12%20ckpt-21.06%20MB-green" alt="R12 ckpt integrity"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="license"></a>
<a href=".github/workflows/audits.yml"><img src="https://img.shields.io/badge/CI-audits%20automated-success?logo=githubactions&logoColor=white" alt="CI"></a>
</p>

> **Paper**: *Hyperbolic RQ-VAE enhanced Generative Recommendation with Differential-Length Codebook: An Independent Reproduction on Amazon Musical\_Instruments*
> **Original work**: Zhang et al., ICML 2026 (see `HG-Rec/`)
> **Reproduction date**: 2026-07-24
> **Status**: ✅ Submission-Ready

This repository contains an independent reproduction of the HG-Rec paper
(Zhang et al., ICML 2026) on the **Amazon Musical\_Instruments** dataset
(5-core filter, leave-one-out protocol), a setting not featured in the
original headline experiments.

---

## TL;DR

| | Vanilla RQ-VAE + Sinkhorn ("phonism") | HG-Rec (κ = 0.5 per layer) |
|---|---|---|
| **R@10** | **0.1058** | 0.1051 (marginally tied, Δ −0.7%) |
| **R@5** | 0.0831 | 0.0816 |

Full results, ablation grids, and curvature learning analysis are in
`papers/paper.pdf` (14 pages, 106 KB).

---

## 1. What's in this repository

```
GeneRec/
├── papers/                       # ← Submission package
│   ├── paper.pdf                  #   Submission-ready PDF (14 pages)
│   ├── paper.md                   #   Single-file Markdown source
│   ├── paper.tex                  #   Auto-generated LaTeX
│   └── refs.bib                   #   12 BibTeX entries
├── REPRODUCE.md                   # ← End-to-end reproduction steps
├── README.md                      # ← (you are here) human-facing landing
├── CLAUDE.md                      # ← AI-agent instructions (not for humans)
│
├── src/                           #   Pipeline (cloned from snap-research/GRID)
│   ├── train.py                   #     Stage 1 / 2 / 3 entry
│   └── inference.py               #     Stage 4 entry
├── configs/                       #   Hydra configs (8 stage-specific YAMLs)
├── data/                          #   Dataset directory
│   └── amazon_data/
│       └── musical_instruments/   #     Amazon_Musical_Instruments_5core.csv.gz
│
├── descriptions/                  # Task definitions (297 entries, R9 continuous)
├── verdicts/                      # Closure reports (one per task + data files)
├── products/                      # Task execution artifacts (checkpoints, runs)
└── scripts/                       # Diagnostic & analysis scripts (~904 files)
```

The directory **BLOGER/**, **DECOR/**, **ETEGRec/**, **HG-Rec/**, **LETTER/**,
**RecBole/**, **RippleNet/**, **knowledge_graph_attention_network/**, **genrec/**
are reference upstream frameworks and papers consulted during the
reproduction project.

---

## 2. Headline Findings

The HG-Rec paper introduces two components: **Hyperbolic RQ-VAE
quantization** and a **Differential-Length Codebook**. Our reproduction
surfaces four findings:

1. **Hyperbolic geometry marginally helps** on Musical\_Instruments — the
   optimal per-layer curvature κ ≈ 0.5 yields R@10=0.1051, only +0.7%
   over vanilla RQ-VAE + Sinkhorn balance (R@10=0.1058). Over a 6-grid
   curvature sweep the R@10 span is **5.3%**; free-curvature learning
   converges all 18 (layer, component) curvatures to **exactly zero**
   from any initialization.

2. **Sinkhorn cascade beats hyperbolic** — a vanilla RQ-VAE with balanced
   Sinkhorn codebook initialization matches or exceeds HG-Rec variants
   on test Recall@10 and generalizes better (smaller valid-test gap).
   We call this configuration *"phonism"*.

3. **Token sets are shared, assignments are not** — vanilla, κ=1, and
   κ=0.5 codebooks share **100%** of codeword token sets (Jaccard = 1.0
   at every layer), yet produce **disjoint** item-to-token assignments.
   The downstream R@10 difference thus depends on the *initialization
   distribution* of codewords, not on intrinsic curvature.

4. **Dataset/protocol gap is systematic** — our absolute numbers are
   18–61% lower than paper-reported baselines across 8 methods (e.g.
   HG-Rec paper R@10 = 0.1315 on Instruments vs ours 0.1020, Δ −22.4%).
   **Relative ranking is preserved** (HG-Rec > Letter > TIGER > SASRec),
   but absolute numbers may transfer imperfectly across datasets.

These claims are documented in **§5–6 of `papers/paper.pdf`** with full
experimental detail.

---

## 3. Reproducing the Results

To reproduce all numbers in `papers/paper.pdf` from a clean L40S machine:

```bash
# 1. Activate the GRID environment (see REPRODUCE.md §1.3)
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /path/to/GeneRec

# 2. Verify environment
python3 scripts/task101_verify_env.py

# 3. Run all 4 stages (≈ 8 hours on L40S, ≤ 32 GB peak)
#    See REPRODUCE.md §3–§6 for verbatim stage commands.
```

Detailed instructions: **[REPRODUCE.md](REPRODUCE.md)**.

Each section of the paper corresponds to one or more closed tasks in
`descriptions/` (R9-numbered, no gaps) and `verdicts/` (closure reports).
The numeric chain of evidence is reproducible from these task definitions.

---

## 4. Citation

If you use this reproduction, please cite the original paper:

```bibtex
@inproceedings{zhang2026hgrec,
  title  = {Hyperbolic RQ-VAE enhanced Generative Recommendation with
            Differential-Length Codebook Strategy},
  author = {Zhang, A. and Yang, Y.-B. and Yu, Y.},
  booktitle = {Proceedings of the 43rd International Conference on
               Machine Learning (ICML)},
  year   = {2026}
}
```

Full reference list in `papers/refs.bib`.

---

## 5. Acknowledgements

- The HG-Rec authors (Zhang, Yang, Yu) for the original paper and methodology.
- The **snap-research/GRID** open-source framework, which provides the
  Stage 1–4 pipeline (Apache 2.0 licensed). The `src/`, `configs/`, and
  upstream `data/` directories in this repository are unmodified
  clones of GRID.
- The institutional GPU cluster (4× NVIDIA L40S) which hosted all
  experiments.
- The RecBole team (for baseline frameworks sampled during reproduction).

This is an **independent reproduction**. No code or unpublished results
were shared with the original HG-Rec authors.

---

## 6. License

- **Paper content** in `papers/` and reproduction artifacts in
  `descriptions/`, `verdicts/`, `products/`, `scripts/`: MIT License
  (see below).
- **Upstream framework code** in `src/`, `configs/`, `data/`: Apache 2.0
  (inherited from snap-research/GRID).
- **Reference papers** in `BLOGER/`, `DECOR/`, `ETEGRec/`, `HG-Rec/`,
  `LETTER/`, `RecBole/`, `RippleNet/`: original publication licenses.

```
MIT License

Copyright (c) 2026 the authors

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

---

## 7. Contact

This reproduction was prepared as a single-author project. Questions
about the reproduction, methodology, or reproduction anomalies can be
raised via the issue tracker of this repository.

---

**Reading order** for newcomers:
1. `papers/paper.pdf` (the deliverable)
2. `REPRODUCE.md` (if you want to run it)
3. `CLAUDE.md` (AI-agent protocol, not for humans)
4. `descriptions/task<NNN>_*.md` (per-task definition chain)
5. `verdicts/task<NNN>_result.md` (per-task closure reports)
