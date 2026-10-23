# Adversarial invariants review

## Scope and decision rule

The frozen 32-D latent, three 256-entry codebooks, three integer SID tokens, data/splits, Stage-2/Stage-3 recipes, and strict conjunctive gates remain fixed. Neither a passing geometry unit test nor changed partitions establish utility: a candidate must demonstrate a non-cosmetic mechanism effect and beat **all** frozen thresholds (ER > 16.38953942232631, Recall@10 > 0.06406796775709883, NDCG@10 > 0.03530651701940677).

The listed repository artifacts can measure downstream gates (including frozen ER), partition/code-usage statistics, CGV gates, and H1–H7 oracle outcomes. They do **not**, as described, record per-example triangle areas, score components, gyration angles, tangent transport errors, or clamp fractions. Thus the proposed numerical/geometric falsifiers are not measurable from those artifacts alone. They require adding run-time instrumentation; do not infer them from ER, CGV, or H1–H7. Partition assignment comparisons are measurable only if per-stage assignments (or sufficient contingency statistics) are retained. The H1–H7/CGV/partition outcomes can test behavioral consequences, but cannot certify that curvature caused them.

## Candidate 1 — Signed geodesic-triangle-area assignment

**Verdict: UNMEASURABLE as currently instrumented; conditionally falsifiable after adding score/assignment diagnostics.**

### Falsifier audit

- “Area negligible versus distance” and the validation variance-share requirement need per-example, per-code area and distance score components. These are not recoverable from partition statistics, CGV gates, H1–H7, or frozen ER.
- “Assignment unchanged when area is zero” needs paired argmin indices from the actual candidate and an otherwise identical area-zero ablation. Partition summaries alone cannot establish per-item agreement unless they include a joint assignment contingency table.
- Code collapse/usage and ER are available from the stated artifacts; they test collapse and the frozen gate, not whether area is the cause. CGV and H1–H7 are downstream behavioral checks, not geometry diagnostics.
- The geometric identities (collinearity, Gauss–Bonnet, isometry invariance, flat limit) need explicit geometry test inputs and numeric outputs. None is established by the listed run artifacts.

### Invariant audit and discriminating replacements

The original (i)–(iii) are useful implementation/unit tests, but can pass while the operator has no effect on learned assignments. (iv) is also too weak: an area term vanishing as curvature tends to zero says nothing about whether it matters at the declared nonzero curvature. (v) is a protocol constraint, not evidence for this mechanism.

Replace/add these operational invariants, evaluated on held-out examples with fixed trained parameters and candidate sets:

1. **Nontrivial score contribution:** report the distribution of `A_c`, `β d²`, and their contribution to score variance; require a preregistered non-negligible area share (the proposal's ≥ half variance rule is measurable only with these logs). Report near-tie margins too, since large score variance can still leave every argmin unchanged.
2. **Assignment sensitivity:** compare the actual assignments against paired `A_c=0` assignments; report per-stage agreement, changed fraction, and code-usage/entropy changes. Require a predeclared minimum changed fraction (rather than merely “not 100% agreement”), with tie-breaking held identical. Also report whether changes are concentrated in examples where the area term changes the winning margin.
3. **Discriminating geometry:** for matched host/residual cases with comparable distance/radius, verify that reversing orientation reverses the area contribution and can reverse a winner; verify collinear controls yield zero. Report these alongside the numerical Gauss–Bonnet check. A test that only checks formula identities is not a learned-mechanism test.
4. **Frozen outcome:** report all three gates, plus CGV and H1–H7, against the frozen baseline on the same protocol. Passing a single metric or an oracle subset is not survival.

### Simplest anti-cosmetic control

Run a paired **area-zero ablation** (`A_c := 0`) with the same initialization, seeds, data, codebook size, optimizer, and frozen recipes; change no other term. Compare per-stage assignments and all frozen outcomes. This is the direct control already implied by the proposal. A stronger interpretation requires that the area arm both changes assignments in the diagnosed area-sensitive cases and clears the same strict triple gate while the ablation does not; if both arms pass (or neither does), the claimed causal advantage is not shown. Retain a distance-only baseline assignment as the simple reference, not a substitute recipe.

## Candidate 2 — Non-associative gyro-composition quantization

**Verdict: UNMEASURABLE as currently instrumented; conditionally falsifiable after adding operation-level logs.**

### Falsifier audit

- Actual-pair gyration magnitude, score-component contribution, Euclidean-limit assignment comparison, tangent-norm preservation, and boundary/clamp rate are not present in the listed partition/CGV/H1–H7/frozen-ER artifacts. They require instrumentation and geometry tests.
- “Removing the gyration term leaves assignments effectively unchanged” is testable with paired assignment outputs, but not from aggregate metrics alone. Record per-stage joint assignments, changed fraction, and margins.
- Code usage, CGV/H1–H7, and the strict triple gate can falsify collapse or lack of useful downstream behavior, but cannot establish that non-associativity caused a difference.
- The original algebraic invariants (identity, flat limit, metric preservation, non-associativity, transport norm, ball membership) are mostly generic operator correctness tests. In particular identity elements and norm preservation can hold even if the gyration term is zero or never affects an argmin.

### Invariant audit and discriminating replacements

Replace/add operational invariants evaluated on actual held-out residual/code pairs:

1. **Observed non-associativity:** report the distribution of the two association-order discrepancies and gyration angles on actual inputs, versus matched collinear/zero-input controls. Require a preregistered non-negligible fraction above numerical tolerance; a synthetic non-collinear unit test alone is insufficient.
2. **Score and winner sensitivity:** report both score components and their scale/margins, then paired assignments with `γ=0` (gyration penalty removed), holding all else fixed. State an a priori changed-assignment threshold and report per-stage agreement, usage, and entropy. “Gyration is non-identity” is not enough if it never changes a winning code.
3. **Curvature-specific check:** compare actual assignments/scores at declared `c` with a numerically stable flat-limit implementation, with all parameters/inputs held fixed. Report changed fraction and whether changes track measured gyration contribution; merely observing the algebraic `c→0` limit is vacuous for effect.
4. **Numerical validity:** report maximum/quantile transport norm error, ball-boundary distance, clamp fraction, and component-wise finite-value/error checks. “Points inside the ball” alone passes trivially for many operators and can conceal pervasive clamping.
5. **Frozen outcome:** report all three strict gates plus CGV and H1–H7 outcomes on identical splits/protocol; changed codes without downstream improvement do not validate the recommendation hypothesis.

### Simplest anti-cosmetic control

Use the paired **`γ=0` ablation**, preserving the same Möbius composition, transport, learned parameters/initialization, data, seeds, and training recipe. This isolates the explicitly claimed gyration-score contribution without changing token/code constraints. Compare per-stage assignments and frozen metrics. To claim the gyration mechanism is useful, require a preregistered nontrivial assignment effect tied to measured gyration contribution and strict success on all three gates; if `γ=0` performs equivalently, the gyration claim is killed even if generic hyperbolic operations remain in the model. A flat-limit assignment control is a useful secondary diagnostic, not the simplest causal ablation.

## Required reporting before either candidate is reviewable

Add small, deterministic diagnostic exports to the run: paired per-example/stage winning indices (or joint contingency tables), winning and runner-up margins, named score components, operation magnitudes, and clamp/error summaries. Freeze diagnostic thresholds before looking at test metrics. Keep the existing data/splits/recipes and report baseline and candidate results together. Without these additions, the proposed mechanism-specific falsifiers are **unmeasurable**, and the available ER/CGV/oracle evidence can only judge outcomes—not validate the proposed invariants or attribution.
