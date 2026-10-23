# Prior-art review — candidate v7

## Scope and verdict

This review concerns the one mechanism declared in `candidate.md`: choosing a quantiser code by minimizing

\[
S_k = \frac{\tfrac12\sqrt{\|u\|^2\|v_k\|^2-\langle u,v_k\rangle^2}}{1+d_c(p,z_k)},
\]

where the numerator is the **unsigned** 2-simplex (Gram/wedge) area at the current residual host. The question is whether this area term is used to choose a code in a recommender SID.

**Verdict: REVISE. Classification: ADJACENT (not established as occupied).**

The available verified record does not establish that the exact operator is already occupied. It also does not support an affirmative novelty claim yet. Treat novelty as **unverified**, not as proven. A bounded claim of “not found in the one verified match” is supportable; a claim that the mechanism is novel or first is not.

## Evidence ledger

| Status | Record | What it supports | What it does not support |
|---|---|---|---|
| **Verified in the supplied campaign record** | HiHPQ, DOI `10.1609/aaai.v38i5.28261`, arXiv `2401.07212`; the scan labels the record **ABSTRACT_ONLY**. | HiHPQ is the one identified nearby prior-art match and is relevant enough to warrant comparison. | An abstract-only match cannot verify whether HiHPQ uses an unsigned triangle-area term to select quantiser codes, or whether it uses this score in a recommender SID. No such exact overlap is established by the evidence available here. |
| **Recollection / unverified** | Hyperbolic product quantisation is adjacent in subject matter: hyperbolic representations and quantisation/code assignment are in the same broad design space. | There is a credible reason to inspect HiHPQ and related full texts. | This is not evidence that the specific unsigned area operator is present. It must not be upgraded into a verified match. |
| **Unknown** | Full text and implementation-level details for HiHPQ; broader search coverage beyond the single campaign match; whether other recommender SID or quantisation work uses this precise score. | These are the checks needed to close the novelty question. | Absence from the current scan is not evidence of absence from the literature. |

## Attempt to kill the claim

The closest identified work is HiHPQ, but the record available for this review is explicitly abstract-only. That evidence is inadequate to distinguish among (a) hyperbolic distance/product quantisation alone, (b) an area-like geometric quantity used elsewhere in the method, and (c) the specific unsigned area being used to choose a code. Only (c), in the relevant recommender SID setting, would establish the overlap posed in the review question. Do not infer it from the title, topic, or a recollection of the paper.

Conversely, the candidate's operator is quite specific: an orientation-free Gram-determinant area between the target and code log vectors based at the current residual host, divided by `1 +` the host-to-code geodesic distance, with `argmin` selecting among codes. Similarity at the level of “hyperbolic quantisation” is adjacency, not proof that this operator is occupied. The candidate itself makes no signed-area, plane, frame, or orientation claim; those are outside the declared mechanism and this review.

## Novelty decision and required follow-up

**Is the exact unsigned triangle-area term already verified as being used to choose a quantiser code in a recommender SID? No.** The supplied verified match does not answer that question at the necessary level of detail. This is not equivalent to “no.”

Accordingly, the candidate may state that the exact overlap is **not established by the current scan**, but may **not** make an unqualified NOVEL / first-of-its-kind claim on this record. Before such a claim, obtain and inspect the full HiHPQ paper (and, if available, code/supplement), checking its assignment objective and recommender/tokenization setup; then search for the specific combination of unsigned triangle/Gram area, code-selection use, and recommender SID. Record the sources and the exact passages or equations supporting the comparison.

The requested category is **ADJACENT**, not **OCCUPIED**: HiHPQ is a relevant adjacent lead, while exact occupancy remains unknown. The review is **REVISE** because the novelty assertion must remain qualified pending that verification—not because an exact prior-art collision has been demonstrated.
