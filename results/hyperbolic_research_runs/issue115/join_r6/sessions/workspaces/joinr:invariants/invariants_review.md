# Adversarial invariants review

## Scope and decision rule

The frozen protocol stays fixed: 32-dimensional latent, three 256-entry codebooks, exactly three integer SID tokens, the original data/splits/recipes, and strict conjunctive gates: Hard-L1 ER `> 16.38953942232631`, Recall@10 `> 0.06406796775709883`, and NDCG@10 `> 0.03530651701940677`. These are candidate reviews, not permission to change the protocol. The five already-refuted families remain excluded.

The proposals name useful diagnostics (assignment agreement, code usage, entropy, score contribution, clamp rate, and the three outcome metrics), but those are not automatically available merely because the repository emits ordinary partition statistics, CGV gates, H1-H7 oracle results, and frozen ER. In particular, none of those generic artifacts proves that a claimed geometric term was active in the actual hard assignment. The operation-specific quantities below must be logged on actual training/evaluation inputs, not only on synthetic unit-test points. A failed numerical identity is an implementation defect; passing one is not evidence of recommendation benefit.

## Candidate 1 — Signed geodesic-triangle-area assignment

**Verdict: FALSIFIABLE (with operation-level logging; not falsified by the generic gates alone).**

### Can the stated falsifier be measured?

- The frozen ER and Stage-3 metrics are measurable from the existing evaluation artifacts. Existing partition statistics can establish code occupancy/perplexity and, if host-conditioned assignments are retained, conditional entropy. The proposal's “assignment unchanged when area is zero” requires both the candidate hard assignments and a rerun/replay of the same inputs and codewords with the area term ablated; ordinary partition summaries cannot reconstruct that counterfactual.
- “Area numerically negligible versus distance” is measurable only if per-example area and distance score components (or their validation distributions/variance contribution) are explicitly recorded. The proposed half-variance condition is measurable, but it is not itself causal evidence: a high-variance term can be constant in argmin-relevant comparisons, and an area contribution can be large while never changing a winning code.
- CGV gates and H1-H7 oracle can test wiring, gradient reachability, and declared numerical properties only if this candidate is actually registered in those probes. A generic pass does not establish that the area term reaches the production assignment or that it changes partitions. The ordinary frozen ER gate also cannot isolate why ER changed.
- There is a serious identifiability issue in the definition: a signed area in dimension 32 needs a specified orientation convention for each triangle plane. A single ordered 2-frame at `p` does not generally define the orientation of every 2-plane in a 32-D tangent space. The “sign taken in the plane spanned by” two vectors needs a deterministic, frame-independent rule or an explicit chosen 2-form. Without that, the operator is under-specified and its sign test is not reproducible.

### Invariants: vacuity and discriminating rewrite

The collinearity, Gauss–Bonnet, and isometry checks are implementation/unit tests, not efficacy invariants. The code-count/token-count condition is frozen bookkeeping and is true regardless of mechanism. The flat-limit condition is potentially vacuous as written: an area tends to zero as curvature tends to zero, but that says nothing about whether the proposed nonzero-curvature area affects actual assignments.

Replace the invariant set with these **measured, discriminating** requirements:

1. On held-out real assignment inputs, log `A_k`, `βd_k²`, and total `S_k` for every candidate code; report signed-area range, argmin margins, area variance share, and clamp/degeneracy rate. Pre-register a nontrivial minimum rate of examples whose winning code changes under area ablation, with uncertainty across frozen seeds. A variance-share threshold alone does not pass this test.
2. For each stage, compare hard assignments from the full score against the matched `A=0` counterfactual with identical latents/codewords. Report agreement, per-code occupancy deltas, and host-conditional entropy deltas. Require a pre-registered nonzero assignment-change effect, not merely “some numerical difference.”
3. Numerically verify collinearity-zero, signed swap, and Gauss–Bonnet on nondegenerate test triangles; verify common-isometry invariance on randomly transformed *same* real triples with orientation transported by the declared orientation structure. These tests validate implementation only; do not count them as evidence of a useful partition.
4. The curvature-specific claim passes only if actual-data assignments differ from a matched flat/area-ablated control beyond numerical tolerance, and the frozen three-metric conjunction passes. Do not require only that the area scalar is nonzero.
5. Keep the 256-way/three-token constraints as protocol assertions, not as evidence for the mechanism.

### Simplest non-cosmetic control

Train the identical arm with `A_c` set to zero in both hard and soft assignment, retaining the same distance term, `β`, initialization, seed, codebook updates, optimizer schedule, and all frozen recipes. This is the direct mechanism ablation. Also replay the trained candidate's fixed validation latents/codebooks through both scorers to separate “operator changes assignments” from training-mediated effects. A flat-limit run is a secondary geometry check, not a substitute for the zero-area ablation because it changes more of the score geometry.

**Kill conditions:** no assignment changes under area ablation (within numerical tolerance); area contribution is negligible in actual score comparisons; ill-defined orientation; material clamp/degeneracy; partition collapse; or failure of any strict frozen metric gate. Merely passing geometric identities is not survival.

## Candidate 2 — Non-associative gyro-composition quantization

**Verdict: FALSIFIABLE, but the current formula is not yet operationally falsifiable as written.**

### Can the stated falsifier be measured?

- The frozen ER and Stage-3 metrics are measurable as above; partition statistics can show occupancy/entropy changes. The proposed gyration-near-identity test and “remove gyration leaves assignments unchanged” require per-example score-component and hard-assignment counterfactual logs. They are not derivable from aggregate ER, CGV, or H1-H7 outputs.
- CGV/oracle can verify origin identities, Euclidean limits, metric preservation, and gradient/parameter wiring only if the exact production operator is included. They do not prove useful non-associativity on real item/code pairs or downstream relevance. Report actual-data gyration angle/norm and argmin changes, not just synthetic non-collinear examples.
- The displayed score has potentially unused quantities: `v_k`/`u_k` are introduced but not used in `S_k`; `G_k` composes a transformed code point with the residual, whereas the verbal definition emphasizes parallel-transported code tangents. This may be intended, but the implemented operator must be made unambiguous before a falsifier can be interpreted.
- The stated “transport preserves tangent norms” is not automatically true for an arbitrary learned invertible `T_s`; invertibility does not imply isometry. Distinguish geometric parallel transport (which should preserve Riemannian norm) from learned `T_s` (which generally will not). Also specify the tangent space and action of `gyr` in the second term so the subtraction is well-typed and geometrically meaningful.
- A further confound: `γ>0` adds an explicit penalty that may dominate assignment changes even when the gyration is tiny. The proposal must report the two score components separately and quantify score/assignment influence of each; the total-score variance is insufficient.

### Invariants: vacuity and discriminating rewrite

Identity, Euclidean-limit, and metric-preservation properties are operator correctness tests, not evidence that the operator matters on these data. “All points remain inside the ball” can pass trivially if inputs are clamped; therefore pair it with clamp rate and pre-clamp boundary proximity. “Association orders differ on some non-collinear inputs” is existential and can pass on an irrelevant synthetic case. The count/token constraint is protocol bookkeeping.

Rewrite as:

1. Freeze a precise, dimensionally valid definition of both score components and all tangent spaces. Log per-example `||gyr(w)-w||`, gyration angle, `||G_k||²`, the signed/weighted gyration term, and hard argmins on real held-out pairs. Report distributions stratified by host norm and residual/code angle.
2. On real validation inputs, compare full assignment to a matched `γ=0` arm using identical trained parameters/codewords and inputs. Require a pre-registered nontrivial assignment-change rate and report margin changes. Separately compare the full composition score to its explicitly declared associative/flat counterpart; neither control may silently change the data, latent size, recipes, or token budget.
3. Verify identity, associativity defect, Euclidean limit, and isometry numerically on both synthetic tests and sampled real inputs. Replace “transport preserves tangent norms” with: parallel transport preserves the specified Riemannian norm within tolerance; report learned `T_s` norm distortion separately and do not call it isometric unless constrained and tested as such.
4. Report pre-clamp violations, post-map boundary margin, and clamp fraction per operation and stage. A nonzero clamp rate must not be hidden by post-clamp “inside-ball” checks.
5. Demonstrate that the non-associative component changes the actual partition (relative to the matched ablation), does not cause code collapse, and clears all three strict frozen metric thresholds. A changed assignment alone is not a success criterion.
6. Keep the 256-way/three-token constraints as protocol assertions only.

### Simplest non-cosmetic control

Use a paired `γ=0` ablation: same model, curvature, `T_s`, initialization, data order, optimizer, and recipes, with only the gyration penalty removed. Evaluate both (a) fixed-parameter replay on the same validation residual/code pairs and (b) the matched full training arm with `γ=0`. This isolates whether gyration affects the selected code and whether any downstream change survives retraining. To support the broader “non-associative composition” claim, add an associative/flat composition control with all other scoring terms held fixed. A generic Euclidean baseline is not sufficiently controlled by itself.

**Kill conditions:** production score is ambiguous or dimensionally inconsistent; actual-data gyration is negligible; `γ=0` yields effectively identical assignments; material clamping; partition collapse; or any one of the three strict gates fails. Non-associativity on a hand-picked synthetic triple does not rescue it.

## Cross-candidate ruling on available evidence

Neither proposal is validated by the literature scan: the cited HiHPQ abstract is an image-retrieval overlap, not evidence for these exact operators, the frozen recommendation setup, or threshold attainment. Neither should be approved from style, abstract invariants, or a generic CGV/H1-H7 pass. The minimum acceptable evidence packet for either is: resolved production configuration; operation-level per-example validation logs; the specified one-factor control; per-stage partition statistics; numerical oracle results for the exact implementation; and the frozen three-metric result with all strict inequalities satisfied. If operation-specific logs or matched controls are not artifacts the current pipeline produces, classify that particular falsifier as **unmeasurable until instrumentation/control runs are added**, rather than treating a proxy metric as a pass.
