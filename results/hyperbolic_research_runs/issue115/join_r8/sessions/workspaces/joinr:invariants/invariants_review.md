# Adversarial invariants review

## Scope and decision rule

Keep fixed the 32-D latent, three 256-entry codebooks, three integer SID tokens, original data/splits and Stage-2/Stage-3 recipes. A candidate survives only if the *same run* strictly clears all three frozen gates: Hard-L1 ER > 16.38953942232631, Recall@10 > 0.06406796775709883, and NDCG@10 > 0.03530651701940677. Geometry identities, CGV passes, or a changed partition are not evidence of ranking utility.

The requested existing artifacts have different scopes. Partition statistics can show changed code occupancy/entropy and (if assignments are retained) assignment agreement. CGV and H1–H7 oracle outcomes establish declared-forward/gradient/numerical contract correctness; they do not establish recommendation benefit or that curvature is causally responsible. Frozen ER is an outcome gate, not a mechanism diagnostic; Stage-3 metrics are required to test ranking. Neither listed artifact family intrinsically provides the proposals' per-example score components, gyration/area magnitudes, or clamp rate. Consequently, those numerical falsifiers are **not measurable as written** without logging or a paired ablation. Do not substitute a passing oracle for the missing mechanism evidence.

## Candidate 1 — Signed geodesic-triangle-area assignment

**Verdict: UNMEASURABLE (as currently instrumented).** It becomes falsifiable with explicit score logging and paired counterfactual assignments.

### Falsifier audit

- “Area negligible versus distance” and “area contributes at least half the score variance” require per-example/per-code area and distance components on a declared validation set. Existing partition statistics, CGV, H1–H7, and ER cannot reconstruct these.
- “Assignment unchanged when replacing area with zero” requires paired argmin assignments under the trained model, same data and codebooks. Per-code counts alone are insufficient: different assignments can have identical marginals.
- “Host-conditional code usage collapses” is measurable only if the artifact records host×code contingency counts. Marginal perplexity/usage cannot test host-conditional collapse. ER failure is measurable, but cannot isolate area as the cause.
- Collinearity, sign reversal, Gauss–Bonnet agreement, isometry covariance, and flat-limit behavior are useful implementation/unit tests, but are not falsifiers of learned utility; several can pass for a zero-area/identity implementation. The proposed frame-based signed area also needs a precise orientation convention: an arbitrary fixed 2-frame in 32D does not define a canonical orientation for every varying 2-plane. A valid oriented 2-simplex area should derive its sign from the ordered tangent pair (and hence reverse when q,z swap), not silently depend on an unrelated frame.

### Rewrite invariants to discriminate

1. **Forward involvement:** On actual held-out assignments, log for every item and all 256 candidates `A`, `βd²`, and total score; require finite values and a preregistered nontrivial area contribution on the distribution (report variance and score-scale ratio, not only a variance fraction susceptible to rescaling). Reject if area is numerically zero or contributes no decision margin for essentially all items.
2. **Decision ablation:** With the *same trained parameters/codebooks*, recompute assignments with `A=0`; report per-stage exact agreement, changed-item rate, and a 256×256 transition matrix. Require changes above deterministic tie/numerical noise and no systematic single-code collapse. Then separately compare full retraining arms; inference-time ablation alone does not establish training effect.
3. **Nontrivial geometry correctness:** On nondegenerate triples, verify signed area is antisymmetric under q↔z, zero for collinear triples, and agrees with an independent Gauss–Bonnet implementation; apply random isometries and verify area magnitude/score invariance with orientation transformed consistently. Include adversarial nonzero-curvature cases where area is nonzero. These tests rule out zero/incorrect implementations, not establish usefulness.
4. **Behavioral outcome:** Record host×code conditional occupancy/entropy and all frozen metrics; compare against the simplest matched control below. Require all three thresholds, with exact frozen protocol. A threshold pass alone is weak evidence because the frozen ER gate is not a causal effect estimate.

### Simplest non-cosmetic control

Run a matched **area-zero ablation**: same model, initialization/seed, training recipe, codeword parameterization, and `βd²` term, but set `A≡0` in both training score and assignment. Compare it to full signed-area arm using paired per-stage assignments/partition stats and final ER plus Stage-3 Recall@10/NDCG@10. Keep a no-auxiliary frozen baseline as the reference if budget permits. A post-hoc assignment-only ablation is the cheapest mechanism check, but cannot replace this matched retraining control because the soft area loss may have changed learned codewords.

## Candidate 2 — Non-associative gyro-composition quantization

**Verdict: UNMEASURABLE (as currently instrumented).** The declared removal/flat-limit falsifiers need score-level counterfactuals not contained in the current artifact set. The listed invariants are largely algebraic sanity checks and can pass while the operator has no effect on SID decisions.

### Falsifier audit

- “Gyration near identity on actual pairs,” component contribution, clamp rate, and `c→0` assignment comparison require per-example gyration/score/point diagnostics or recomputed assignments. CGV/H1–H7 may validate the implemented formula and gradient but do not report its empirical decision leverage. Partition marginals and frozen ER are insufficient.
- “Removing gyration leaves assignments effectively unchanged” is in principle measurable from paired assignments, but only if the ablation is evaluated with the same learned parameters and the assignment artifacts preserve item-level IDs. A retrained ablation is additionally needed to test whether training with gyration matters.
- Identity `u⊕0=0⊕u=u`, Euclidean-limit behavior, metric-preservation, invertibility of `T_s`, and points-inside-ball can all hold for an implementation whose learned `T_s` is identity, whose actual residual/code pairs are collinear/near-origin, or whose gyration term is too weak to change any argmin. They do not prove non-associativity participates in real SID decisions. Norm preservation of transport also does not show that transport is the declared transport or that it changes the forward result.
- There is a specification ambiguity worth blocking on: `v_k` is first defined via parallel transport, then the score uses `Exp_p(T_s Log_o(z_k))`; clarify whether `T_s` acts before or after transport and specify domains/metrics. Also ensure the stated gyration convention and the score’s subtraction compare vectors in the same tangent space. CGV should oracle-check the exact frozen formula, not a narrowed substitute.

### Rewrite invariants to discriminate

1. **Actual non-associativity exposure:** On real validation residual/code pairs, log the gyration map and its metric displacement/angle, plus both score components and their scales. Report distributions and the fraction of examples for which the gyration term changes the winning code, not merely a synthetic non-collinear witness.
2. **Paired decision effect:** For the same trained parameters, recompute exact hard assignments with the gyration contribution zeroed and report per-stage agreement, changed-item rate, transition matrices, and host×code conditional occupancy. Reject if the full term is decision-inert (within tie/numerical tolerance).
3. **Curvature-specific counterfactual:** Recompute the assignment on the same saved inputs/parameters under a mathematically consistent flat-limit implementation (or a preregistered small-curvature sequence) and report convergence of gyration to identity and assignment agreement. A generic “c→0” check at one arbitrary epsilon is not enough; control numerical scaling and keep the Euclidean-limit score semantics explicit. This is a diagnostic, not permission to alter the final frozen run.
4. **Correctness and bounds:** Independently oracle-check the exact composition order, gyration convention, tangent spaces, and gradients (CGV/H1–H7); report maximum ball norm, boundary margin, and clamp fraction on actual training/validation inputs. Algebraic identities must be tested on nontrivial random and observed inputs, not only zero/identity cases.
5. **Behavioral outcome:** Compare partition and all three frozen metrics to the matched control; require strict conjunctive threshold success. Changed assignments alone are not a success criterion.

### Simplest non-cosmetic control

Run a matched **gyration-off ablation** with the same codebooks/transport/composition and same seed/recipe, setting only the explicit `γ` gyration contribution to zero (and logging the resulting composition score unchanged). First do same-checkpoint assignment recomputation to establish direct decision leverage; then, if that is nonzero, compare full versus off retraining for partition statistics, ER, Recall@10, and NDCG@10. Also retain the ordinary frozen baseline. This isolates the claimed non-associative term more cleanly than replacing the entire operator with Euclidean addition, which changes multiple things at once.

## Bottom line

Neither proposal is presently falsifiable from the named artifacts alone: both require mechanism-specific telemetry and paired ablations. Candidate 1 has crisp counterfactuals once area and distance scores are logged. Candidate 2 has a broader specification ambiguity and several vacuous algebraic invariants; it must first resolve tangent-space/operator details, then demonstrate that gyration changes real hard assignments. A CGV/H1–H7 pass is necessary implementation evidence, never a substitute for those tests or for the three frozen outcome gates.
