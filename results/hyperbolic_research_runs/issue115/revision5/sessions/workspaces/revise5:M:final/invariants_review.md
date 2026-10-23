# Adversarial invariants review

## Scope and decision rule

The frozen 32-D latent, three 256-entry codebooks, three integer SID tokens, data/splits, and Stage-2/Stage-3 recipes are constraints, not evidence for either mechanism. A candidate only passes the outcome gate if the *same* run has Hard-L1 ER `> 16.38953942232631`, Recall@10 `> 0.06406796775709883`, and NDCG@10 `> 0.03530651701940677`. No unit identity or invariant test substitutes for that gate.

Available evidence is uneven. Partition statistics can expose code usage/entropy and assignment changes; CGV G1–G6 checks declared implementation/gradient contracts; H1–H7 checks the declared hyperbolic mathematics on fixtures; frozen ER and (only after ER qualifies) Stage-3 provide outcome evidence. CGV/H gates establish implementation correctness, not useful partitions or recommendation benefit. The proposed area/gyration contribution, conditional on actual data, is **not automatically present** in these artifacts: it must be explicitly instrumented in the candidate's assignment trace and reported. The numerical isometry/curvature-limit fixture checks are not the same as data-batch measurements.

## Candidate 1 — Signed geodesic-triangle-area assignment

**Verdict: FALSIFIABLE, but the proposal's current invariant list is mostly mathematical smoke tests, not discriminating evidence.** Its assignment-ablation falsifier is meaningful if the run records paired assignments for the full data under the declared score and the zero-area score. Partition statistics then show whether a changed assignment actually yields noncollapsed, host-conditional partitions; frozen ER and gated Stage-3 decide utility. H/CGV can check the formula and gradient path, but neither implies that area affects real assignments or improves ranking.

### Invariants: what is vacuous and what to require instead

The following are correct implementation checks but do not distinguish this mechanism from a useless operator: exactly 256 codes and three tokens; Gauss–Bonnet agreement; zero area for collinear points; common-isometry invariance; and vanishing area at `c→0`. An implementation can satisfy every one while its area is zero/negligible on all training examples, or while the distance term determines every argmin. The sign-reversal check is likewise insufficient if the resulting argmin never changes.

Replace the effectiveness invariant with paired, real-data tests (report per stage and overall):

1. **Nontrivial score contribution:** report `Var(A)` and `Var(βd²)` and the area share of score variance using a fixed, declared convention; also report the fraction of examples whose winning code changes under `A=0`. The registered “at least half the variance” condition must be calculated on validation examples, not selected post hoc. Variance share alone is not causal, so it cannot replace the ablation below.
2. **Discriminating assignment effect:** compare full-score and `A=0` assignments on the identical examples/codebooks/hosts. Report exact agreement, per-code occupancy, conditional entropy by host, and the assignment-change rate broken down by residual-radius bins. Pre-register a minimum change and non-collapse criterion; otherwise an arbitrarily tiny nonzero area passes. Include a shuffled-orientation/sign control: it must destroy the claimed signed-direction advantage rather than merely leave assignments unchanged.
3. **Useful distinction rather than arbitrary churn:** report the same frozen ER and, if ER passes, Recall@10/NDCG@10 for the full candidate. A changed partition is not a win; all three strict thresholds are necessary.
4. Keep the geometric fixture checks as H-gate/unit checks: collinearity, sign reversal, Gauss–Bonnet, isometry, and flat limit. They establish the operator is implemented as declared, not its empirical relevance.

### Simplest non-cosmetic control

Run a paired **area-off control**: identical initialization, data order, recipe and seed, with only `A_c` set to zero in assignment (retain the same distance term and all other code). Freeze the full arm's learned codebooks/hosts and recompute both assignments on the same held-out examples for a clean direct ablation; report assignment agreement/occupancy/conditional entropy and ER. If claiming a *trained* benefit, additionally train that single area-off arm under the unchanged recipe and compare frozen ER; Stage-3 only if its own ER clears the gate. The geometry's sign is oriented by a frame: specify and freeze that frame and include a reflected/orientation-reversed control, or the purported signed effect is not isolated.

**Kill conditions:** area-off assignment agreement effectively 100% (with uncertainty/tolerance declared before seeing results); area contribution negligible or validation variance criterion fails; pathological collapse; or failure of any strict frozen metric. Area-off matching is a direct falsifier; metric failure kills utility even if the geometric identities pass.

## Candidate 2 — Non-associative gyro-composition quantization

**Verdict: FALSIFIABLE only after the score and transport are made operationally unambiguous; as written, one central invariant is not yet measurable from the text.** The proposed “remove gyration term” and `c→0` assignment comparisons can use paired assignment traces plus partition statistics. CGV/H checks can validate the implemented formula and gradients, but cannot prove actual data pairs exhibit meaningful gyration or that the ranking benefits. Report measured gyration magnitude, score components, assignment ablations, partitions, and frozen metrics; do not infer these from toy non-collinear fixtures.

### Specification defects and vacuous invariants

- `u⊕0=0⊕u=u`, code/token counts, finite in-ball points, and norm-preserving gyration/transport are algebraic/safety checks. They can all pass when `γ` has no effect and the assignments are unchanged.
- “Two association orders differ for non-collinear nonzero inputs” on a hand-picked fixture does not show they differ on actual residual/code pairs. “Transport preserves tangent norms” does not show transport contributes to a decision.
- The score's first term composes code then residual and logs the result. Its second term compares `gyr[p, Exp_o(T_s Log_o(z))] Exp_p(r)` with `Exp_p(r)` using `g_p`; the points `Exp_p(r)` are not tangent vectors at `p`, while `g_p` is a tangent metric. This subtraction/norm needs an explicit valid tangent-space definition (and matching transport) before independent H-oracle verification. Also define whether `γ` is a penalty or reward: with fixed `γ>0` and a minimized score it is a penalty, not a reward.
- Formula (35) defines `Exp_o(T_s Log_o(z))` but separately defines `v_k`, `u_k`; the latter are unused. Specify which code point enters composition and ensure the same operator is used by training, hard assignment, and oracle. Define exactly what the reported “gyration contribution” means.

Rewritten discriminating invariants, after fixing those points:

1. On actual held-out assignment examples, report gyration angle/norm distribution and the change in each score component, stratified by stage and residual/code norms. Report clamp/boundary rate. A nonzero toy gyration is only a fixture, not evidence.
2. Paired hard assignments with full score versus `γ=0` (same hosts/codewords), and versus the flat `c→0` operator (same inputs/parameters where mathematically well-defined). Report per-stage exact agreement, code occupancy, host-conditional entropy and change rates. Pre-register a minimum decision-change threshold; a tiny numerical deviation is not evidence of a meaningful operation.
3. Check an order-swap control (code-after-residual versus residual-after-code) using the identical examples. If order is the asserted semantic distinction, it must change assignments beyond numerical tolerance and the effect must not be reproduced by a common coordinate reparameterization.
4. Then require noncollapse and the frozen ER; only ER-qualified arms get Stage-3, and final acceptance requires all three strict thresholds. Gyration magnitude or partition churn alone does not establish recommendation utility.

### Simplest non-cosmetic control

Use a paired **gyration-off control**: replace only the explicit gyration penalty by zero, holding the trained codebooks, hosts, residuals, and all other score terms fixed; recompute held-out assignments and report agreement and partition statistics. This is the cheapest causal check that the named term participates in decisions. To establish a learned performance effect rather than a post-hoc assignment effect, train one otherwise identical `γ=0` arm. Keep a flat-limit assignment control as a second, distinct check of curvature dependence; it does not replace `γ=0`, since other curved terms may remain. Apply the original ER gate before any Stage-3 control run.

**Kill conditions:** corrected score remains undefined or cannot be independently reproduced; actual-batch gyration is negligible; gyration-off/order/flat controls leave assignments effectively unchanged; clamp rate is material; partitions collapse; or any of the three frozen metrics fails. Algebraic gyro identities passing is never sufficient.

## Bottom line

Both candidates have potentially measurable, operator-specific ablations, so neither is inherently untestable. Their current invariant lists overemphasize truths about the mathematics that can hold while the operator is decision-irrelevant. Candidate 1 is closer to falsifiable as written. Candidate 2 must first repair its tangent-space score definition and unused/ambiguous quantities; until then the declared H1–H7 oracle cannot unambiguously certify the proposed forward. In both cases, report implementation gates, real-batch assignment effects, partition health, and frozen outcome gates as separate evidence layers. Do not treat CGV/H completion as a performance verdict.
