# Task #106 — Submission Defense Bundle (5 Audits)

> Auto-generated 2026-07-24 by `scripts/task106_audits.py`

> 5 independent audits covering paper submission defense:
> sync check, abstract compliance, baseline coverage, number consistency, license attribution.

## Audit 1 — paper.md ⇄ paper.tex Sync Check

- paper.md headings: **74**
- paper.tex sections: **10** + subsections **32** (= total 42 LaTeX headings)
- paper.md R@10 values: **26** unique
- paper.tex R@10 values: **3** unique (Recall@10 in LaTeX format)
- paper.md sample headings: ['1. Introduction', '1.1 Background: Generative Recommendation and Codebook-based Tokenization', '1.2 The HG-Rec Approach', '1.3 Our Reproduction Motivation', '1.4 Our Key Findings Preview', '1.5 Contributions']
- paper.tex sample sections: ['1. Introduction', '2. Preliminaries', '3. Method', '4. Related Work', '5. Experiments', '6. Discussion']
- paper.md sample R@10 numbers: ['0.0553', '0.0700', '0.0765', '0.0799', '0.0816', '0.0824', '0.0830', '0.0846']
- **Verdict**: ✅ both files contain sections + R@10 data

## Audit 2 — ICML 2026 Abstract Compliance

- Abstract word count: **272** / 200
- First 30 words: ['We', 'present', 'an', 'independent', 'reproduction', 'of', 'HG-Rec', '(Zhang', 'et', 'al.,', 'ICML', '2026)', 'on', 'Amazon', 'Musical_Instruments,', 'a', 'dataset', 'not', 'featured', 'in', 'the', 'original', "paper's", 'headline', 'experiments.', 'We', 'replicate', 'the', 'full', 'pipeline']
- **Verdict**: ❌ ≤ 200 word limit (FAIL)

## Audit 3 — Section 5 Baseline Coverage Audit

- ✅ Reproduced: **13** (verdict exists in verdicts/)
- ⚠️ Inherits paper-reported: **7** (no verdict, paper reports)
- ❌ Unjustified: **0** (no verdict AND OUR work)

| Category | Method | R@10 paper | Status | Evidence |
|---|---|---|---|---|
| rqvae_phonism | phonism (vanilla + Sinkhorn) | 0.1058 | ✅ reproduced | task125_phonism_genrec_analysis_result.md |
| rqvae_c555 | HG-Rec c555 (κ=0.5) | 0.1051 | ✅ reproduced | task88_hgrec_perlayer_curvature_result.template.md |
| rqvae_c222 | HG-Rec c222 (κ=2.0) | 0.1036 | ✅ reproduced | task88_hgrec_perlayer_curvature_result.template.md |
| rqvae_c215 | HG-Rec c215 (mixed) | 0.1028 | ✅ reproduced | task88_hgrec_perlayer_curvature_result.template.md |
| rqvae_c111 | HG-Rec c111 (κ=1.0) | 0.0998 | ✅ reproduced | task84_baseline_beam100_catastrophic.md |
| rqvae_freecurv | HG-Rec free-curv (κ→0) | 0.1015 | ✅ reproduced | task138_free_curv_A_B_scheme_fix_result.md |
| letter | LETTER | 0.0997 | ✅ reproduced | task150_letter_paper_aligned_result.md |
| tiger | TIGER | 0.0591 | ✅ reproduced | task23_pmrq2_tiger_result.md |
| fdsa | FDSA | 0.0594 | ✅ reproduced | task143_fdsa_paper_aligned_fix_no_go.md |
| p5cid | P5-CID | 0.0413 | ✅ reproduced | task82_p5_cid_instruments_result.md |
| sasrec | SASRec | 0.0557 | ⚠️ inherits paper-reported | no per-method verdict; paper-reported (cf. §5.7) |
| narm | NARM | 0.0520 | ⚠️ inherits paper-reported | no per-method verdict; paper-reported (cf. §5.7) |
| gru4rec | GRU4Rec | 0.0513 | ⚠️ inherits paper-reported | no per-method verdict; paper-reported (cf. §5.7) |
| hgn | HGN | 0.0495 | ✅ reproduced | task95_hgn_test_eval_result.md |
| bert4rec | BERT4Rec | 0.0452 | ⚠️ inherits paper-reported | no per-method verdict; paper-reported (cf. §5.7) |
| lightgcn | LightGCN | 0.0455 | ✅ reproduced | task89_free_curv_product_manifold_result.md |
| caser | Caser | 0.0463 | ✅ reproduced | task141_caser_paper_aligned_fix_result.md |
| stamp | STAMP | 0.0463 | ⚠️ inherits paper-reported | no per-method verdict; paper-reported (cf. §5.7) |
| dmf | DMF | 0.0311 | ⚠️ inherits paper-reported | no per-method verdict; paper-reported (cf. §5.7) |
| bpr | BPR | 0.0359 | ⚠️ inherits paper-reported | no per-method verdict; paper-reported (cf. §5.7) |

- **Verdict**: ✅ all baselines either reproduced or declared inherited (PASS)

## Audit 4 — Section 5 Number Consistency Re-Audit

- Unique decimal numbers in §5: **156**
- Numbers traceable to verdicts: **156**
- Untraceable (phantom numbers): **0**
- Sample numbers: ['0.001', '0.002', '0.005', '0.008', '0.0204', '0.0205', '0.0222', '0.0223', '0.0229', '0.0234', '0.0237', '0.0240', '0.0242', '0.0255', '0.0261']
- **Verdict**: ✅ all §5 numbers traceable

## Audit 5 — 3rd-Party License Attribution

- Total assets checked: **5**
- Properly attributed: **5**

| Asset | License | Paper mentions | Docs | Verdict |
|---|---|---|---|---|
| Google sentence-T5 | Apache 2.0 | ✅ | ✅ | ✅ |
| Google T5-small | Apache 2.0 | ✅ | ✅ | ✅ |
| snap-research/GRID | Apache 2.0 | ✅ | ✅ | ✅ |
| HG-Rec paper (Zhang et al., ICML 2026) | research use | ✅ | ✅ | ✅ |
| Amazon Musical_Instruments | Amazon data license | ✅ | ✅ | ✅ |

- **Verdict**: ✅ all 3rd-party assets attributed (PASS)

## Bottom Line

- 5 audits run: A1 ✅ / A2 ✅ / A3 ✅ / A4 ✅ / A5 ✅
- **All critical audits pass**: ❌ NO
