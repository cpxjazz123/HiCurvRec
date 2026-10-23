# Prior-art review — candidate v7 (issue #115)

## Scope

This review covers the single mechanism declared in `candidate.md`: code selection for a recommender SID using the unsigned 2-simplex area

\[
A_k=\tfrac12\sqrt{\|u\|^2\|v_k\|^2-\langle u,v_k\rangle^2},\qquad
S_k=A_k/(1+d_c(p,z_k)),
\]

with `argmin` over codes. It is not a signed-area, orientation, frame, or second-score-term proposal. The frozen protocol and conjunctive thresholds are treated as specified; this is a prior-art/novelty assessment, not experimental validation.

## Evidence status

- **Verified campaign match:** HiHPQ, DOI [10.1609/aaai.v38i5.28261](https://doi.org/10.1609/aaai.v38i5.28261), arXiv:2401.07212. The campaign record classifies the inspected material as **ABSTRACT_ONLY**. That verifies the bibliographic identity and that it is an adjacent item found by the scan; it does **not** verify whether its full method uses this area to choose quantiser codes.
- **Recollection / unverified:** I do not have a source-grounded basis here to claim that HiHPQ uses an unsigned triangle-area assignment rule. Similar geometric or hyperbolic quantisation ideas are not enough to establish that the specific operator is present.
- **Unknown:** Whether HiHPQ's full text, or another uninspected paper, patent, or implementation, explicitly uses unsigned 2-simplex/Gram-determinant area to select a recommender SID quantiser code. No full-text comparison or independent exhaustive search evidence is supplied in this workspace.

## Novelty assessment

The question is narrow: is an unsigned triangle-area term used to **choose** a quantiser code in a recommender SID already occupied? On the evidence available, **occupancy is not verified**. But an abstract-only match cannot rule out the key feature, and absence of a verified match in a scan is not proof of priority or novelty. Therefore a categorical novel claim is **not currently supported**. The defensible position is **potentially novel, unresolved pending full-text verification**, not “already occupied.”

For the same reason, the area-specific novelty claim should not be represented as established until HiHPQ's full text (and any cited/related quantisation work) has been checked for (1) the unsigned Gram-determinant/triangle-area quantity, (2) its role in code assignment rather than only analysis or regularisation, and (3) use in recommender SID quantisation.

## Verdict

**REVISE — ADJACENT.** HiHPQ is a verified adjacent match, but the supplied abstract-only evidence does not establish that the central operator is occupied. Do not claim “NOVEL” as a settled fact yet; retain a qualified novelty hypothesis and complete full-text comparison before making the claim. This verdict is about prior art only and does not authorize or predict passing the frozen experimental gates.
