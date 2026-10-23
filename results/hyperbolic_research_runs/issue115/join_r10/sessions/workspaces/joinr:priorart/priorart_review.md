# Prior-art and novelty review

## Scope reviewed

This review treats `candidate.md` as the sole, complete declaration of one mechanism. The claimed operator is the **unsigned 2-simplex area**

`A_k = 1/2 sqrt(||u||² ||v_k||² − <u,v_k>²)`

used in the code-selection score `S_k = A_k / (1 + d_c(p,z_k))`, with `argmin` choosing a quantiser code. The task is specifically whether this kind of unsigned triangle-area quantity is used to choose a code in a recommender SID. It is not a review of an oriented/signed area, a second score term, or any other proposal: none is declared in `candidate.md`.

The frozen evaluation context is 32-D, three codebooks of 256 entries, three integer SID tokens, and the original data, splits and recipes. The conjunctive strict thresholds are Hard-L1 ER > 16.38953942232631, Recall@10 > 0.06406796775709883 and NDCG@10 > 0.03530651701940677. These are experimental gates, not evidence of novelty.

## Evidence ledger

### Verified in the supplied campaign record

- The campaign scan identifies **HiHPQ**, DOI [10.1609/aaai.v38i5.28261](https://doi.org/10.1609/aaai.v38i5.28261), arXiv [2401.07212](https://arxiv.org/abs/2401.07212), as its only verified match, with evidence depth explicitly marked **ABSTRACT_ONLY**.
- The scan therefore verifies a relevant neighboring work, but the recorded evidence depth does **not** verify that HiHPQ uses the candidate's Gram-determinant unsigned area to select a quantiser code in a recommender SID. A match at title/abstract level is not proof of that exact operator or use.
- The candidate itself explicitly specifies an unsigned, basepoint-invariant area term in code selection and distinguishes it from distance-only scoring. No claim of signed area or an extra score term is supported by the candidate text.

### Recollection / contextual understanding (not verified evidence)

HiHPQ appears to concern hyperbolic product quantization and is consequently close in broad subject matter: hyperbolic geometry and quantisation. That is a reason to compare methods carefully, not a basis for asserting that its score is the same. Any recollection about its internal equations, assignment rule, downstream recommender use, or SID construction is **not** treated here as a verified fact.

### Unknown from the available record

- Whether HiHPQ's full text, supplementary material, or implementation contains an unsigned triangle/2-simplex-area term.
- Whether any such term, if present, actually chooses among quantiser codes rather than appearing in an auxiliary objective, regularizer, or analysis.
- Whether that assignment is used in a recommender SID and whether the relevant details are disclosed in a cited or earlier work.
- Whether the exact combination of basepoint tangent logs, Gram area, distance tempering, and `argmin` assignment is disclosed elsewhere.

The scan is explicitly abstract-only; absent full-text equation and assignment-path inspection, these questions remain open. This review does not claim an exhaustive global prior-art search.

## Novelty assessment

**Classification: ADJACENT, not established as NOVEL and not established as OCCUPIED.** HiHPQ is verified adjacent prior art, but the evidence supplied does not establish that the specific unsigned area operator is already used to choose a recommender-SID quantiser code. Conversely, abstract-only evidence is insufficient to support a positive novelty claim against the closest known match.

**Can a novel claim be made now? No—not responsibly on this record.** The exact claim is neither shown occupied nor cleared. Phrase any current statement as a candidate distinction / unverified novelty hypothesis, not as “the first” or as established novelty. To resolve the question, inspect the full HiHPQ paper and cited method lineage for the actual code-assignment equation and its role; document the comparison. If those sources disclose the same area-based code choice in the stated setting, classify the claim OCCUPIED; if a scoped search finds no such disclosure, a narrowly qualified novelty claim may then be considered.

## Verdict

**APPROVE_FOR_JOIN_E — ADJACENT.** This is approval to proceed to the next experimental join/evaluation gate, not approval of a novelty claim and not a finding that prior art is clear. The operator is sufficiently explicit to test as written, but novelty remains unresolved. Keep the candidate's frozen protocol and conjunctive thresholds intact; the result of those experiments cannot substitute for the missing full-text prior-art check.

## Boundary of this review

Five families are treated as already refuted per the supplied review brief: radius-dominated product assignment, horospheric level sets, transported-residual frame, role-assigned layout (measured vestigial), and Euclidean reformulation. They are not revived or reviewed as alternatives here. The sole reviewed mechanism remains the unsigned area score declared in `candidate.md`.
