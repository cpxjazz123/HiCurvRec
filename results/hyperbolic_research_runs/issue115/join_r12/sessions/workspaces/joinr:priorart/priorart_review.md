# Prior-art review — unsigned 2-simplex area code selection

## Scope

This review covers exactly the mechanism declared in `candidate.md`: for a residual host point `p`, target `q`, and candidate code `z_k` in the Poincaré ball, compute the **unsigned** Gram-determinant area

`A_k = 1/2 sqrt(||log_p(q)||² ||log_p(z_k)||² − <log_p(q), log_p(z_k)>²)`,

temper it as `S_k = A_k / (1 + d_c(p,z_k))`, and choose a code by `argmin`. It is the 32-D / three-256-codebook / three-integer-SID recommender setting under the frozen protocol. No signed area, orientation, second score term, alternate candidate, or withdrawn proposal is reviewed; none is declared in the supplied candidate.

## Finding

**Verdict: REVISE. Classification: ADJACENT, not established as OCCUPIED.**

The available verified match is **HiHPQ** (DOI `10.1609/aaai.v38i5.28261`; arXiv `2401.07212`), and the campaign record characterizes its evidence as **ABSTRACT_ONLY**. That makes it a relevant prior-art lead in hyperbolic product quantization for recommendation, but the available evidence does not establish that HiHPQ uses an unsigned triangle/2-simplex area to choose a quantizer code, nor that it uses this area-over-distance score, this base point/target construction, or this SID setup. An abstract-only match cannot settle those distinctions.

Accordingly, the specific question—whether unsigned triangle area is already used to choose a code in a recommender SID—is **unknown on the evidence supplied**. The match is adjacent in application and quantization context; it is not a verified occupancy of the operator.

## Evidence status

- **Verified:** The candidate explicitly defines an unsigned Gram-determinant 2-simplex area and uses the resulting score in code assignment. The campaign scan reports HiHPQ as its only verified match, with DOI/arXiv identifiers and `ABSTRACT_ONLY` evidence status.
- **Not verified:** Whether HiHPQ, or any other cited item, uses unsigned triangle area as a code-selection objective. No full text, equations, algorithm details, or independent search results are present in the materials supplied for this review.
- **Recollection:** No recollection is treated as evidence here.
- **Unknown:** Whether the same operator appears in HiHPQ's full text, its supplementary material, or other prior art not represented by the campaign scan.

## Novelty position

**A novelty claim may not be made as an established fact on this record.** It is fair to describe the proposed operator as a *candidate distinction* from the currently verified abstract-level match, but not to claim that the mechanism is novel or that prior art does not occupy it. The evidence is insufficient for either a positive novelty conclusion or an occupancy conclusion.

Before upgrading the verdict, inspect HiHPQ's full text and supplements for its assignment objective, geometric quantities, and code-selection rule; then search the relevant terminology and citations for unsigned triangle area, Gram determinant/wedge norm, and area-based quantizer assignment in recommender SID/RQ systems. Record sources and exact passages. If that review confirms no such use, the novelty statement can be reconsidered with an appropriately narrow scope. If it finds the area used to select codes in the relevant setting, classify the mechanism as **OCCUPIED** and identify the matching features.

## Protocol boundary

This is a prior-art/novelty screen only. It does not assess the frozen conjunctive metric thresholds, assignment-audit artifacts, or whether the experiment is ready to join. It does not alter the candidate or protocol. The candidate's own declared mechanism and falsifiers remain the sole subject; no alternative signed-area or second-term mechanism is inferred.
