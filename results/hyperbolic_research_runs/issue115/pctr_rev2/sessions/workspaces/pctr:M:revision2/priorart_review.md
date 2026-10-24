# Prior-art / novelty review — candidate.md

## Verdict

**REVISE — ADJACENT; do not make a novelty claim yet.** The available campaign record identifies HiHPQ (DOI [10.1609/aaai.v38i5.28261](https://doi.org/10.1609/aaai.v38i5.28261), arXiv:2401.07212) as the only verified match, but only at **ABSTRACT_ONLY** depth. That is not enough to establish that HiHPQ discloses the particular operator, nor enough to establish that it does not. A claim that an unsigned triangle-area quantiser-selection term is novel is therefore **not presently supportable**.

## Scope and declaration check

I treated `candidate.md` as the complete declaration, as requested. There is an important mismatch between the review prompt and that file: the file does **not** declare an unsigned 2-simplex/triangle-area operator. Its stated candidate is PCTR: a signed tangent-space residual–group-mean inner-product term in the score (candidate.md, §§OPERATOR, INVARIANTS THAT FAIL FOR THE IDENTITY). It explicitly distinguishes itself from an unsigned tangent-space wedge area (candidate.md, §WHY NOT THE SIX CLOSED FAMILIES). I found no second candidate or second score term in the file.

Accordingly, the answer to “is an unsigned triangle-area term used to choose a quantiser code already occupied?” is **not established by this candidate or the verified scan record**. I will not attribute that area operator to candidate.md or silently review it as if declared. If the intended operator really is unsigned triangle area, the complete declaration needs to be corrected to say so before that mechanism can receive a substantive review. If candidate.md is authoritative, the mechanism to review is the stated signed inner-product reranking instead.

## Evidence status

- **Verified from the supplied materials:** `candidate.md` describes the PCTR signed inner-product score and says the unsigned tangent-space wedge-area family is excluded. The supplied `frozen_artifacts.json` records control statistics and shapes, not prior-art evidence. No other local prior-art report or source was present in the workspace.
- **Verified campaign-scan result as supplied in the request:** HiHPQ is the sole match, with the scan explicitly limited to **ABSTRACT_ONLY**. This supports treating HiHPQ as relevant adjacent art to check; it does not verify a triangle-area code-selection operator in HiHPQ.
- **Unknown:** Whether HiHPQ's full paper, supplementary material, code, or cited work uses an unsigned triangle-area term to choose a quantiser code in a recommender SID. The supplied abstract-only status cannot settle the operator-level comparison.
- **No recollection used as evidence:** I have not relied on memory of HiHPQ or any other paper to fill in the missing operator details.

## Novelty assessment and required next check

No “novel” claim should be made on this record. This is a limitation of the evidence, not a finding that the area operator is already occupied. Nor does the record justify a confident “not occupied” claim. The defensible status is **ADJACENT / unresolved**, pending (1) reconciliation of the unsigned-area prompt with the signed-inner-product declaration in `candidate.md`, and (2) full-text and, where available, code-level review of HiHPQ and relevant references for the exact operator and its use in quantiser-code selection for recommender SIDs.

This is a scoped technical prior-art screen, not a legal patentability opinion.
