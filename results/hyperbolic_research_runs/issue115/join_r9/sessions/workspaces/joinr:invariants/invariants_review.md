# Measurability review: candidate v7

## Verdict: REVISE

The declared operator is the unsigned 2-simplex area

\[
A_k=\tfrac12\sqrt{\|u\|^2\|v_k\|^2-\langle u,v_k\rangle^2},\qquad
S_k=A_k/(1+d_c(p,z_k)),
\]

with minimum-score assignment. I reviewed that operator and its stated area-zero ablation. I did **not** review a signed area or a second score term: neither is in `candidate.md`.

The protocol constraints and strict conjunctive ER/Recall@10/NDCG@10 thresholds are clear in the declaration. The stated audit call, timing, artifact contents, and thresholds are a useful start, but the artifact specification is not sufficient to make the claimed falsifiers decisive from the artifact alone. A call signature and a list of fields do not define an independently auditable record.

## What is not yet verifiable from the described artifact

1. **Exact counterfactual being compared.** “Area-zero score” needs an explicit formula. Is it `0/(1+d)` for every code, hence effectively nearest distance with the same tie rule, or does the ablation retain any other part of the score? Record the full and ablated score formulas, constants, curvature, scale, epsilon, and tie-breaking rule in the artifact. The prose does not define an ablation implementation precisely enough to distinguish a causal area effect from an implementation difference.
2. **Reconstructible per-example evidence.** Winning indices, margins, and “per-example components” do not say whether the artifact contains the example identity, every candidate's component and score, or only the winning candidates. To verify each flip and margin without rerunning, include stable example IDs and, for all 256 codes per example, the distances, raw and smoothed areas, full and ablated scores, and selected index. If storage constraints preclude this, the artifact cannot support the requested from-artifact-alone audit; a digest is not a substitute for the underlying values.
3. **Unambiguous denominators and definitions.** Define flip rate as the number of examples whose selected code differs divided by which eligible-example count; define tie share (what constitutes a tie, at what numeric tolerance, and under which score); and give numerator and denominator counts, not only rounded rates. State how NaNs/errors and excluded examples enter each denominator. Apply the ≥5% and ≤10% tests to unrounded counts/rates, with the exact boundary behavior stated.
4. **Clamp accounting.** “Clamp count” is not enough to verify a clamp *rate*. Emit the number of clamp-eligible evaluations and the denominator for each level/stage, identify whether counts refer to log-map calls, points, or coordinates, and define the aggregation. Include raw counts and unrounded rate so the >0.1% kill gate can be checked.
5. **Numerical and tie auditability.** Emit precision/dtype, score-comparison tolerance, the exact deterministic tie-break, and non-finite/out-of-ball counts with their denominators. A zero reported error count without a definition and scope is not independently interpretable.
6. **Artifact integrity and run identity.** Include schema/version, stage and level, timestamp or run ID, code/config/data/split/recipe identifiers (preferably content hashes), protocol/threshold version, and a checksum or equivalent integrity mechanism. Otherwise a reviewer cannot establish that the JSON belongs to the frozen run or was not altered, nor associate per-stage files with one run.
7. **Other stated falsifiers are not covered by this audit description.** The `c→0` assignment comparison and the partition gate (distinct-code counts and maximum cell share) are separate kill criteria, but no corresponding artifact fields or exact calculation rules are specified here. Include the control assignments/comparison and per-level partition counts/shares, with denominators and definitions, or provide named, integrity-linked artifacts for them. The ER and Stage3 metric decisions also need their frozen evaluation outputs and protocol identity; the assignment audit alone cannot establish those outcomes.

## Decision

Do not approve for JOIN_E on the present measurability declaration. Amend the artifact contract so each threshold can be recomputed from emitted counts/values and each falsifier has a linked record, without rerunning the model. In particular, the currently described artifact could report flip/tie/clamp rates, but does not specify enough to independently verify their numerators, denominators, or the assignments that produced them. Once those omissions are closed, the artifact can make the proposed mechanism's effect auditable; this verdict is about measurability, not an empirical finding that the operator succeeds or fails.
