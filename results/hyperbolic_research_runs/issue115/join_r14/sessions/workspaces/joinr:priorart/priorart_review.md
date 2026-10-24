# Prior-art review — candidate v7

## Scope and verdict

**Verdict: APPROVE_FOR_JOIN_E**  
**Prior-art classification: ADJACENT (not established as occupied)**

The reviewed mechanism is the single mechanism declared in `candidate.md`: use the **unsigned 2-simplex area** formed by `log_p(q)` and `log_p(z_k)`, tempered by `1 + d_c(p,z_k)`, to choose a quantiser code by `argmin`. The question is whether an unsigned triangle-area term is already used to choose a code in a recommender SID.

On the evidence available, I cannot say that this exact use is already occupied. The one campaign match, HiHPQ (DOI `10.1609/aaai.v38i5.28261`; arXiv `2401.07212`), is recorded as **ABSTRACT_ONLY**. That is a verified bibliographic match and a verified limit on the evidence reviewed; it is **not** verification that HiHPQ uses this operator, nor that it does not. The abstract-only record does not resolve whether HiHPQ uses unsigned triangle area to select a quantiser code in a recommender SID.

Accordingly, a **qualified novelty claim may be made**, framed as a distinction from the verified adjacent work and explicitly subject to full-text comparison. An unqualified claim that this is the first such method, or that HiHPQ definitely does not use it, is not supported. This is a prior-art triage approval to proceed to JOIN_E, not a finding of priority or legal novelty.

## Evidence labels

| Status | Finding |
|---|---|
| **Verified from the supplied record** | The candidate defines one code-selection rule using the non-negative Gram-determinant area `A_k = ½√(||u||²||v_k||² − ⟨u,v_k⟩²)` and score `S_k = A_k/(1+d_c(p,z_k))`, with `argmin` selection. It explicitly has no orientation or signed-area term. |
| **Verified campaign match, limited scope** | HiHPQ is the only verified match reported by the campaign scan; its identifier is DOI `10.1609/aaai.v38i5.28261` / arXiv `2401.07212`, and the inspected evidence is **ABSTRACT_ONLY**. |
| **Not verified** | Whether HiHPQ, or another unexamined work, uses an unsigned geodesic triangle/Gram area as a term that chooses quantiser codes specifically for recommender SIDs. The abstract-only evidence cannot settle the operator-level comparison. |
| **Recollection / intuition, not evidence** | Hyperbolic quantisation and geometric or angular code-selection criteria are plausible neighboring ideas. This is only a search intuition, not a verified citation or a basis for asserting that this particular area operator is known. |

## Novelty assessment

The available record supports **adjacency**, not occupation: HiHPQ is a relevant hyperbolic-quantisation match to inspect, but there is no reviewed evidence tying it to this exact unsigned-area code-selection rule and recommender-SID setting. The candidate's own distinctions—unsigned rather than oriented area, area divided by a distance tempering factor, and code selection within the frozen three-token RQ SID protocol—are descriptions of the proposal, not independent proof of novelty.

Suggested defensible wording: *“To our knowledge, the use of an unsigned geodesic 2-simplex-area term to select quantiser codes in a recommender SID is distinct from the adjacent hyperbolic quantisation work identified in our scan; a full-text operator-level comparison remains necessary.”* Do not state that HiHPQ lacks the mechanism until its full text and relevant equations/algorithm are checked.

## JOIN_E condition

Proceed to JOIN_E with the above qualification. Before any publication-grade or priority claim, retrieve and inspect the full text of HiHPQ (including its assignment objective and quantisation procedure), then record the exact comparison. Also expand the search beyond the one reported match if the claim is intended to be broad (“first” or “no prior work”). This review does not revisit the frozen protocol, thresholds, or the five refuted families listed in the request.
