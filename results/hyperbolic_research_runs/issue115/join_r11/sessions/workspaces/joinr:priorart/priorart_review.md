# Prior-art and novelty review

## Scope

Reviewed the single declared mechanism in `candidate.md`: choose a code for the current host point `p` using the **unsigned 2-simplex area** spanned by `log_p(q)` and `log_p(z_k)`, tempered by `1 + d_c(p,z_k)`, and assign by `argmin`. The frozen 32-D/three-256-codebook/three-token protocol and the strict conjunctive ER, Recall@10 and NDCG@10 gates are treated as fixed. The five families stated in the review brief are treated as already refuted, not as alternative candidates.

## Prior-art finding

**Verified in the supplied campaign record:** HiHPQ (DOI `10.1609/aaai.v38i5.28261`; arXiv `2401.07212`), and only at **ABSTRACT_ONLY** evidence level. That establishes a relevant hyperbolic product-quantization/recommender-SID neighbor worth distinguishing. It does not establish from the available evidence that HiHPQ uses an unsigned triangle-area term to choose a quantizer code.

**Not verified:** No inspected full text, methods, equations, or code in this record demonstrates the particular unsigned Gram-determinant area operator, its use in code assignment, or its use in a recommender SID. Nor does the abstract-only match establish that those features are absent.

**Recollection / inference (not evidence):** Area- or angle-based geometric objectives are plausible neighboring ideas in quantization and hyperbolic representation learning. That general resemblance is not enough to identify an anticipating disclosure or to conclude that the exact operator is occupied. No specific additional reference is asserted here.

**Unknown:** Whether HiHPQ or another prior publication discloses this exact area-based code-selection rule in an SID; whether the combination is novel over the full literature and patent record; and whether the claimed assignment behavior is empirically distinct under the frozen protocol.

## Novelty judgment

The question “is an unsigned triangle-area term used to choose a quantiser code, in a recommender SID, already occupied?” cannot be answered **yes** on the verified evidence provided. It also cannot be answered **no**: an abstract-only scan is not sufficient to establish novelty. The defensible classification is **ADJACENT**, not OCCUPIED and not established NOVEL.

**A NOVEL claim may not yet be made.** A carefully qualified candidate/mechanism claim may be investigated, but any assertion of priority or literature-wide novelty requires checking the HiHPQ full text and relevant references, then targeted searches for area/wedge/Gram-determinant code assignment in recommender quantization and SID systems. Distinguish the exact operator and assignment role, not merely hyperbolic quantization or geometric scoring in general.

## Kill points / review disposition

1. **The declared degeneracy rule is inconsistent with the score.** When `A_k = 0`, `S_k = 0/(1+d_c(p,z_k)) = 0` for every collinear candidate. Dividing by distance does not break this tie. In particular, `q=p` makes every code score zero, so the stated “distance factor alone” tie-break does not follow from `S_k`; only the lowest-index fallback actually resolves it. The same issue applies to `z_k=p` versus any other collinear code. Revise the operator or state and implement an explicit lexicographic tie-break, then ensure the audit reflects it.
2. The review packet contains no emitted assignment audit or experimental results. Accordingly, no conclusion is possible about the 5% flip gate, tie share, partition gate, or the frozen strict metric thresholds. The proposed evidence schema is a plan, not evidence.
3. The prior-art basis is deliberately narrow: the sole verified match is abstract-only HiHPQ. It supports adjacency, not occupation or novelty.

**Verdict: REVISE. Classification: ADJACENT.** This is a disposition for joining the experiment queue, not an experimental pass or a novelty clearance. Resolve the tie-rule contradiction and obtain full-text prior-art coverage before making a novelty claim.
