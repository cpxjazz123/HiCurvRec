# Adversarial invariants review

Scope: the two mechanisms in `proposals.md`; frozen 32-D latent, three 256-way codebooks, three integer SID tokens, unchanged data/splits/recipes, and strict conjunctive thresholds (Hard-L1 ER > 16.38953942232631, Recall@10 > 0.06406796775709883, NDCG@10 > 0.03530651701940677). The five already-refuted families are not reconsidered.

## Executive verdicts

| Candidate | Verdict | Main reason |
|---|---|---|
| 1. Signed geodesic-triangle-area assignment | **FALSIFIABLE** (after tightening) | Its assignment ablation and score variance can be measured on actual host/code pairs. The listed geometric identities mostly validate an implementation, not a useful assignment effect; moreover the orientation convention needs to be made well-defined in 32 dimensions. |
| 2. Non-associative gyro-composition quantization | **FALSIFIABLE** (after fixing an ambiguity) | Gyration/assignment ablations and frozen metrics are measurable. Several axioms are true even when the operator is irrelevant; the proposed gyration penalty also subtracts points rather than explicitly comparable tangent vectors. |

Neither proposal is validated by passing CGV or H1-H7 alone. Those establish the declared computation/gradient and mathematical behavior, not that the mechanism changes a useful partition. The hard assignment, partition statistics, and frozen evaluation supply the empirical falsifiers. If the existing artifacts do not retain per-example assignments and score components, the proposal must explicitly add these to the already-produced partition-statistics report; aggregate ER alone cannot diagnose its mechanism-specific claims.

## Candidate 1 — Signed geodesic-triangle-area assignment

### What is measurable

- **Directly measurable from partition/run artifacts:** per-stage hard assignments, assignment agreement under an area-zero rescoring pass, code counts/perplexity/Gini/largest cell, host-conditional assignment entropy, clamp fraction, Hard-L1 ER, and (only after ER admission) Recall@10 and NDCG@10.
- **Measurable in CGV/H1-H7 fixtures:** formula parity, finite/nonzero gradients, degenerate-triangle behavior, curvature-limit behavior, and numerical agreement with an independent area oracle. These are implementation checks, not evidence of utility.
- **Not established by the frozen metrics alone:** “splits residuals with similar radii but different directions.” Save the real-batch residual/code score matrix and report a conditional test: among pairs with matched `d(p,q)` (pre-registered tolerance/bins), do area-based scores/assignments distinguish residual directions, and does that distinction survive the area-zero control? Aggregate perplexity cannot answer this.

### Invariants: vacuity and rewrite

1. **Collinear area = 0 / sign reversal / Gauss–Bonnet agreement:** not vacuous as formula/unit tests, but they can all pass while area is ignored by the argmin. Keep as correctness checks, not mechanism evidence. Define the orientation convention concretely: choose a fixed oriented 2-plane/frame in each host tangent space and specify how it is transported/constructed. In dimension 32, an arbitrary triangle spans a 2-plane, but there is no canonical sign on that plane from the three points alone; a global frame-dependent sign is not invariant under arbitrary isometries/reflections. State whether the frame is transported with the isometry and test only orientation-preserving isometries, or supply a deterministic host-frame convention and weaken the invariance claim accordingly.
2. **Common-isometry score invariance:** as written it can be true for every score built from invariant distances (and is not discriminating). Retain it as an implementation property, but rewrite the mechanism test as: on a fixed nondegenerate real-batch fixture, transform all points by a sampled orientation-preserving isometry; verify the signed area and the complete 256-code score vector transform consistently and the hard argmin is identical, while a reflection flips the signed term and produces the predicted score/assignment changes where the area margin is decisive.
3. **Flat-limit area vanishes:** a useful mathematical limit test, but it does not prove nonzero curvature matters at the declared fixed `c`. Rewrite: at frozen trained parameters and actual examples, report the fixed-`c` area variance share and assignment disagreement against the exact same score with `A=0`; require a pre-registered nontrivial disagreement/margin (not merely floating-point differences), and show the effect is specifically due to area rather than a changed distance scale. Report bootstrap uncertainty over examples.
4. **256 codes / three tokens:** protocol compliance only; an identity or ordinary Euclidean quantizer passes. Keep as a hard constraint, not a scientific invariant.
5. **Area contributes at least half the variance:** potentially discriminating, but `Var(S)` can be dominated by arbitrary scale/covariance and cancellation. Pre-register the precise statistic (e.g. `Var(A)/Var(S)` is not additive when terms covary); report `Var(A)`, `Var(βd²)`, covariance, and the score variance share, plus per-example argmin margins. Set a minimum area-induced assignment-change rate and paired metric comparison before training.

### Simplest non-cosmetic control arm

Run a **paired area-zero ablation**: identical initialization, seed, batches, optimizer, all frozen losses/recipes, codebook sizes, and evaluation; replace only `-A_c(p,q,z)` by zero while retaining the same `β d²` and all numerical handling. Compare assignments on the same examples (per-stage agreement and changed-code fraction), partition statistics, ER, and—if ER clears admission—Stage-3 metrics. This is simpler and more decisive than a Euclidean model or a new baseline. A post-hoc rescoring ablation diagnoses the assignment operator cheaply, but does not replace the trained paired arm for causal performance claims.

**Kill criteria:** negligible area contribution or no material assignment changes kills the claimed mechanism even if metrics pass; changed partitions with any frozen threshold missed kills the candidate; passing all three thresholds without a material paired area effect does not establish an area mechanism.

## Candidate 2 — Non-associative gyro-composition quantization

### What is measurable

- **Directly measurable:** actual-pair gyration magnitude, component-wise score variance, hard assignment agreement with the gyration term removed, per-stage partition statistics, clamp/boundary rate, frozen ER and gated Stage-3 metrics.
- **CGV/H1-H7 can check:** Möbius/gyration formula against an independent oracle, limits/identities, gradient route and finite values. They cannot show that actual residual/code pairs exhibit consequential non-associativity or useful downstream structure.
- Record actual-pair quantities, not synthetic generic non-collinear examples alone. A synthetic witness that associativity fails is compatible with gyration being effectively identity on every training pair.

### Invariants: vacuity and rewrite

1. **Identity (`u⊕0=0⊕u=u`):** implementation sanity only; every candidate with an unused gyro operator passes. Keep as unit test, not evidence.
2. **Euclidean limit and metric preservation:** correctness checks only. Rewrite as a fixed-curvature real-data test: report gyration angle/norm on actual `(host, code, residual)` tuples and the fraction whose gyration contribution changes the top-1 code beyond a declared score-margin tolerance. Verify the same quantities approach zero under a controlled `c→0` diagnostic with parameters/inputs held fixed.
3. **Association orders differ on generic inputs:** vacuous for the learned assignment if the actual inputs are zero/collinear or if the score ignores the difference. Replace “generic inputs” with actual training/validation tuples; report the distribution of both association-order discrepancy and assignment effect. Require a pre-registered non-negligible fraction above numerical tolerance.
4. **Transport preserves tangent norms:** an isometry/implementation test that can pass while transported vectors never affect choices. Keep as an oracle check; additionally report norm error and transport's contribution to actual score/assignment differences.
5. **Strictly inside ball / clamp rate:** protocol/numerical safety, not evidence of non-associative utility. Report max radius margin and clamp rate by stage; define “material” numerically before run (ideally zero clamps in forward scoring).
6. **256 codes / three tokens:** protocol compliance only, as for candidate 1.
7. **Gyration penalty term:** the proposal writes `gyr[...] Exp_p(r) - Exp_p(r)` and measures it with `g_p`; these are points in the ball, not tangent vectors at `p`. Specify a valid comparison, e.g. `Log_p(gyr[...] Exp_p(r)) - Log_p(Exp_p(r))` in `T_p` (with the gyration action's basepoint/frame stated), or a geodesic distance between the two points. Otherwise the claimed invariant/score is ambiguous and CGV cannot have a unique independent oracle.
8. **Assignment sensitivity:** “remove gyration leaves assignments effectively unchanged” is falsifiable but “effectively” is undefined. Pre-register changed-code fraction and top-1 score-margin criteria, and report per-stage agreement, not only a global average. Also ablate gyration from the full trained scoring rule, preserving the first `||G||²` component.

### Simplest non-cosmetic control arm

Run a **paired `γ=0` ablation** with identical initialization, training schedule, data order, optimizer and all frozen losses, retaining the same Möbius composition/first score term but removing only the gyration penalty. This isolates the proposal's explicit non-associativity contribution better than a Euclidean control. Report actual-pair gyration statistics, per-stage assignment agreement, partition statistics and the three frozen metrics. A post-hoc `γ=0` rescoring on a fixed checkpoint is a cheap diagnostic; use the paired trained arm to support a causal metric claim.

**Kill criteria:** if actual-pair gyration is negligible or the `γ=0` arm has effectively identical assignments, the claimed gyration mechanism is killed regardless of generic synthetic tests. If assignments change but any frozen threshold fails, reject. If the ball-boundary/clamp rate is material, the result is numerically uninterpretable and fails regardless of metrics.

## Cross-candidate guardrails

- Freeze the single curvature, `β`/`γ`, tolerances, and minimum materiality thresholds before looking at test metrics. Do not choose them to manufacture assignment disagreement or metric gains.
- Use validation for mechanism diagnostics and keep test evaluation protocol untouched. Report assignment changes separately from metric outcomes; an assignment change is not itself a win.
- The actual control should differ in exactly one operator term. Do not call mathematical identities, code/token counts, CGV passage, or H1-H7 passage evidence that the mechanism is non-cosmetic.
- Final success remains conjunctive and strict: ER, Recall@10, and NDCG@10 must all exceed the frozen thresholds under the unchanged protocol.
