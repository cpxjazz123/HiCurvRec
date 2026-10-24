# Prior-art review — candidate v7

## Decision

**Verdict: REVISE. Classification: ADJACENT (not established as occupied, and not cleared as novel).**

Do **not** make a novelty claim yet that an unsigned triangle-area term for choosing a recommender SID quantiser code is new. The campaign's only verified match is HiHPQ, but its record is **ABSTRACT_ONLY**. That establishes a relevant prior-work lead, not whether HiHPQ uses this operator. The decisive occupied-vs-novel question remains unknown on the evidence available here.

This is not a rejection of the experimental mechanism. It is a hold on novelty wording pending a source-level comparison. Proceeding to JOIN_E is defensible only if JOIN_E is explicitly an experiment/feasibility gate and does not treat novelty as established; otherwise revise the gate to require the comparison below.

## Scope reviewed

I reviewed the single declared mechanism in `candidate.md`: for each candidate code `z_k`, form the tangent vectors at the current residual host `p`, `u = log_p^c(q)` and `v_k = log_p^c(z_k)`, compute the **unsigned** tangent-space wedge/Gram area `A_k = ½√(‖u‖²‖v_k‖² − ⟨u,v_k⟩²)`, and choose a code by minimizing `A_k/(1+d_c(p,z_k))`, with the declared tie rule. This is an area-dependent code assignment in a quantiser, not a signed area, orientation, frame, or separate score term. The frozen 32-D / three 256-codebook / three-token SID protocol is part of the evaluation context, not evidence of prior-art novelty.

The five stipulated refutations are treated as closed and are not reopened here: radius-dominated product assignment, horospheric level sets, transported-residual frame, role-assigned layout (vestigial), and Euclidean reformulation. I found no basis in the supplied record to introduce a second candidate or a withdrawn proposal.

## Evidence ledger

| Status | Finding | What it supports |
|---|---|---|
| **Verified in the supplied campaign record** | HiHPQ, DOI `10.1609/aaai.v38i5.28261`, arXiv `2401.07212`; source coverage is explicitly **ABSTRACT_ONLY**. | HiHPQ is a directly relevant lead that must be compared. The bibliographic identification is verified; the abstract-only limitation is material. |
| **Not verified** | That HiHPQ computes unsigned triangle/wedge area, a Gram-determinant equivalent, or any monotone equivalent as a term in the score that selects a quantiser code. | No claim that the exact mechanism is occupied can be made from this record. |
| **Not verified** | That HiHPQ's selected code is an item/recommender SID token under the same or a materially equivalent quantisation setup. | Recommender/SID overlap at the exact operator-and-role level is unresolved. |
| **Unknown** | Whether another paper, patent, or implementation uses the exact area-based code-selection rule. | The scan's single verified match is not proof that the wider literature is clear. |
| **Recollection only / not evidence** | Any remembered similarity between HiHPQ or other hyperbolic quantisation work and geometric/angular assignment. | Recollection can guide searching but cannot establish either anticipation or novelty. No such recollection is used as a finding here. |

## Novelty assessment — attempt to kill

The potentially distinguishing proposition is narrow: **an unsigned tangent-space 2-simplex/wedge area involving the target direction and each candidate-code direction is used to choose a quantiser code in a recommender SID**. It is not enough to show that a paper uses hyperbolic quantisation, residual quantisation, angular similarity, a triangle-area visualization, or a geometric regularizer. To occupy this claim, prior work must disclose the operative area (or a mathematically equivalent quantity) in the code-selection rule, in the relevant recommender/SID quantisation role. Conversely, changing the name, multiplying by a constant, or writing the same Gram determinant in another form would not rescue novelty.

The documented HiHPQ lead is close enough that this exact comparison is a blocking issue, but the abstract-only record does not expose the assignment equation or establish whether area selects a code. Therefore:

* **Is the exact claim already occupied?** Unknown on verified evidence; neither “yes” nor “no” is justified.
* **May a NOVEL claim be made now?** **No.** The available evidence is insufficient to clear the closest identified lead, and a single-match abstract-only scan cannot support a broad priority claim.
* **Is OCCUPIED established?** **No.** Do not assert that HiHPQ uses this mechanism without inspecting a source that actually shows it.

This is an **ADJACENT / unresolved** result, not a positive novelty finding and not a finding of anticipation.

## Required revision before a novelty claim

Obtain and inspect the HiHPQ full text and its supplementary material, implementation, or other primary source if available. Record the exact equation and the role of each quantity. In particular answer: (1) what score chooses each quantiser code; (2) whether an unsigned 2-simplex/tangent wedge area or algebraic equivalent appears in that score; (3) whether it is computed from target-to-host and code-to-host directions, or equivalent operands; and (4) whether that assignment emits recommender SID tokens. Cite page, equation, or code lines. If the primary source is unavailable, retain **unknown / no novelty claim** rather than converting absence of evidence into evidence of absence.

Then extend the search beyond this one record before using an unqualified “novel” statement. Keep the claim scoped to the actual operator and assignment role; do not claim that area itself, hyperbolic quantisation, or geometric recommender codes are new.

## Bottom line

The candidate declares one unsigned area-based code-selection mechanism. The only verified campaign match is a relevant but abstract-only HiHPQ record. That is enough to block a novelty assertion, but not enough to say the mechanism is occupied. **REVISE; ADJACENT / unresolved; no NOVEL claim.**
