# Prior-art and novelty review — PCTR

## Scope reviewed

I reviewed the single mechanism declared in `candidate.md`: prefix-internal tangent residual ordering (PCTR). I did not review or infer a second candidate or proposal. In particular, the candidate does **not** define an unsigned triangle-area / 2-simplex-area score. It explicitly excludes an unsigned tangent wedge/2-simplex area and instead uses a **signed, normalized tangent inner product** between a sample residual and a batch-group mean residual to alter within-prefix code ranking. A “signed area” or “second score term” would be text absent from the declared candidate and is not reviewed here.

The supplied frozen protocol is taken as fixed: 32-dimensional latent, three 256-entry codebooks, three integer SID tokens, original data/splits/recipes, and strict conjunctive thresholds of Hard-L1 ER > 16.38953942232631, Recall@10 > 0.06406796775709883, and NDCG@10 > 0.03530651701940677. These performance gates do not establish novelty. The five stated refuted families (radius-dominated product assignment, horospheric level sets, transported-residual frame, role-assigned layout, and Euclidean reformulation) are treated as closed; they do not resolve the external prior-art question.

## Evidence status

- **Verified in the supplied campaign record:** HiHPQ, DOI [10.1609/aaai.v38i5.28261](https://doi.org/10.1609/aaai.v38i5.28261), arXiv:2401.07212, with evidence explicitly limited to **ABSTRACT_ONLY**. This verifies that HiHPQ is a relevant prior-art lead, not that it contains the specific operator at issue.
- **Recollection:** none relied upon. No remembered paper or technique is treated as evidence.
- **Unknown / not verified:** whether HiHPQ or another work uses an **unsigned triangle-area term to choose a quantiser code in a recommender SID**. The supplied abstract-only record does not establish its scoring operator, code-selection role, recommender/SID setting at the necessary level of detail, or an exact match. No full-text or code inspection evidence is supplied.

## Assessment of the question

**Is an unsigned triangle-area term used to choose a quantiser code in a recommender SID already occupied?** The supplied evidence does not establish that it is occupied. It also does not establish that it is unoccupied. HiHPQ is adjacent and unresolved on this record; an abstract-only match cannot support a categorical anticipation or non-overlap conclusion.

Separately, that unsigned-area question is not the operator declared by `candidate.md`. PCTR uses a signed inner-product correction based on batch-group residuals, not an unsigned area. Its operator-level difference from an area term is clear from the candidate, but this distinction alone is not a prior-art search and does not prove novelty for PCTR either.

## Novelty disposition

**A novelty claim may not presently be made.** The record is too thin to claim either that the unsigned-area code-selection idea is novel or that it is already occupied. Nor does it establish novelty of PCTR: the only supplied external lead remains abstract-only, and no full-text/code comparison is documented. Any paper or disclosure should describe the operator precisely and qualify novelty pending examination of the HiHPQ full text/code and a targeted search for the exact combination of area-based code choice, recommender quantisation, and SID generation.

## Verdict

- **Prior-art classification:** **ADJACENT** (HiHPQ is a verified lead on an abstract-only basis; exact overlap is unresolved, not demonstrated).
- **Process verdict:** **APPROVE_FOR_JOIN_E**, as a scoped prior-art review / experimental handoff only—not as clearance, a novelty finding, or authorization to claim non-overlap. Join E must preserve the distinction between the declared signed PCTR operator and the unsigned-area question, and must not convert the abstract-only lead into a verified match.
- **Novelty status:** **NOT ESTABLISHED**. A categorical **NOVEL** claim is not supported; **OCCUPIED** is also not supported by the supplied evidence.
