# R02 mathematical self-review

Single-agent review, not an invented external reviewer. No new paper reproduction or independent authorship is claimed. Frozen definitions: CONTRACT.md; independent numerical oracle: reference.py (SciPy LP and arithmetic-progression schedule), not production imports.

For E=3000 and the existing I=50, anchors are e_k=1+50k, k=0..59. Last anchor2951; for every training epoch t, e=1+50 floor((t-1)/50), hence0<=t-e<=49. The production modulo predicate and independent enumeration must agree on all3000 epochs, including1/50/51/100/101/2951/3000. Initial calibration is separate, so epoch1 must not be repeated.

At an anchor, averaging parameter VJP vectors before group norms defines a_g=||o_g|| and b_g=||c_g||. The one-variable feasible interval is the intersection of [0,a_g/b_g] over active groups; its maximal endpoint is their minimum. The independent HiGHS LP answers the same constrained problem by another mathematical path. Every parameter element belongs to one natural group, including disjoint query rows. Decoder is curvature-independent, not a missing active constraint.

The bound o_g·(o_g+lambda*c_g)>=a_g(a_g-lambda*b_g)>=0 applies to the actual selected-cost surrogate VJP vectors at that state/sample. It is not an exact derivative of the discontinuous hard-assignment map, an intervening-step bound, an AdamW/momentum/weight-decay descent theorem, or a Recall guarantee. Float32 collective/buffer rounding is admitted only within the frozen2e-6 budget tolerance; no epsilon makes a zero-original active group feasible.

Native-block refresh addresses stale calibration age only. R01 final global13/17 violations are direct evidence of drift, while local Key/Query0 sign claims have global counterexamples. Euclidean gate concentration, mixed geometric metrics, sample weighting and optimizer dynamics remain alternatives; the causal reason for low Recall is unknown.

The actual epoch51 diagnostic must use current DistributedSampler epoch50 and structural step300, not initial items/step0. It must independently solve/check the current global budget without parameter/optimizer updates or RNG/coverage advancement. The stale-coefficient counterexample modifies a diagnostic buffer temporarily and restores it before training resumes; it is not an ablation training arm. Formal training receives no diagnostic observer or mutation.

The inherited floor repair is greedy, not a global minimum solver. The exact2x2 counterexample in VERIFIER_REPAIR.md disproves the old optimal-value perturbation assertion. Actual native assignments must replay exactly on their own original costs; continuous shared-GNN loss/VJP equivalence is conditional on the same recorded discrete branch. Native branch code differences are recorded, not called bitwise equivalence. This correction changes no production rule or tolerance, and exact global SID export remains mandatory.

Execution outcomes belong to their real source-bound receipts; this review does not predeclare any check PASS.