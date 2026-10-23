# Adversarial prior-art review

## Scope and evidence standard

This review uses the supplied `literature.md` as the verified local search record. Its only verified direct match is **HiHPQ**, DOI **10.1609/aaai.v38i5.28261**, arXiv:**2401.07212**, whose status in that file is explicitly **ABSTRACT_ONLY**. The literature scan says it verified an abstract describing hyperbolic product quantisation, hyperbolic codebook attention, and quantized contrastive learning on a hyperbolic product manifold; it did not inspect the full paper. I therefore do not attribute a particular assignment formula or residual-stage design to HiHPQ.

No recommendation-specific hyperbolic quantizer is verified in the supplied scan. The scan is targeted, not exhaustive; absence is not evidence of absence. Statements below about general differential/hyperbolic geometry are mathematical recollection, not a verified bibliographic claim. I cannot honestly supply a verified paper identifier for signed-area code assignment or gyration-scored SID assignment from the evidence at hand.

## Candidate 1 — Signed geodesic-triangle-area assignment

### Prior-art status

**Closest verified work: HiHPQ (DOI 10.1609/aaai.v38i5.28261; arXiv:2401.07212), ABSTRACT_ONLY.** It occupies the broad territory of putting hyperbolic structure inside product quantization for retrieval. Its abstract does not establish signed geodesic triangle area, host-relative oriented simplex scoring, residual-stage SID assignment, or recommendation. The precise formula in this proposal is therefore **not shown occupied by the supplied literature**, but neither is novelty established: the nearby full text and further literature would need checking.

General geometric recollection (not a verified paper citation): geodesic triangle area, angle defect/Gauss–Bonnet, and oriented simplex area are standard hyperbolic geometry, not new operators. Their use as a learned vector-quantizer assignment score may be a new *application*, but “the formula is a geometric invariant” alone does not establish a novel quantization method.

### Where the operator already appears in this pipeline

The proposal places area directly in the **assignment** score and also adds a soft-assignment area term to training. It does not use area in the encoder, decoder, or frozen reconstruction/quantization objectives. The locality term is the ordinary geodesic-distance assignment component. Reusing the same area score in hard and soft assignment is one mechanism in two optimization paths, not independent novelty evidence.

Moving a standard geometric quantity from a geometric loss or descriptor into code assignment can be a useful application, but is not automatically non-obvious. The proposal needs to establish that the area term induces a partition not reducible to an existing hyperbolic metric/distance assignment or a known angular/wedge quantizer, and that the gain is specifically from signed curvature area rather than a generic directional feature.

### Distinguishability and kill risks

On a constant-curvature surface, the area is determined by the triangle's angle defect. It is not simply a curvature sweep or a reparameterized distance: it depends on all three vertices and changes under reflection. However, the proposal's score combines it with squared distance, and the actual implementation may collapse to a function of host-conditioned angle and side lengths. It must report ablations against distance-only, unsigned area, and an angle/wedge control at the same fixed curvature. A `c→0` area disappearance is necessary but does not by itself establish that the useful effect is uniquely hyperbolic rather than a directional partition.

There is a definition gap worth treating as a blocker: in 32 dimensions, the ordered triple defines an oriented 2-simplex, but the stated sign convention “an ordered orthonormal 2-frame at p” does not specify a canonical frame for every candidate plane. A global orientation of the ambient manifold does not automatically supply an orientation for each 2-plane. Define the sign consistently (and show it is independent of arbitrary frame choices); otherwise the score may be frame-dependent or discontinuous. The stated swap-sign invariant and isometry invariant need tests against the actual implementation, not just the unsigned Gauss–Bonnet angle formula.

Also, “the unique geodesic 2-simplex” needs the usual nondegenerate/minimal-simplex domain made explicit. Collinear and near-collinear cases, numerical branch behavior, and points near the ball boundary can make the signed term unstable. The strict variance condition is a useful anti-degeneracy guard, but does not show that assignment changes improve retrieval.

### Verdict: **ADJACENT**

Closest work is HiHPQ (verified abstract only): hyperbolic product quantization for image retrieval. The proposed distinction is a **host-relative signed simplex-area term in residual-stage code assignment**, rather than the abstract-level hyperbolic product quantizer described by HiHPQ. That distinction is plausible but unverified against HiHPQ full text and broader quantization literature. Do not claim “first hyperbolic quantization” or a recommendation novelty claim on this record. Novelty would require a broader search plus the controls above; empirical survival still requires all three frozen thresholds.

## Candidate 2 — Non-associative gyro-composition quantization

### Prior-art status

**Closest verified work: HiHPQ (DOI 10.1609/aaai.v38i5.28261; arXiv:2401.07212), ABSTRACT_ONLY.** It makes hyperbolic structure part of retrieval quantization at a broad level. The supplied abstract record does not say it uses Möbius addition, gyration, order-sensitive residual composition, or SID codes. No verified identifier for that exact combination is present in `literature.md`; the exact prior-art status is therefore **UNKNOWN**, not demonstrated novel.

General mathematical recollection (not verified as a specific quantization paper): Möbius addition and gyration are standard gyrovector-space tools. Their use to express non-associativity in hyperbolic geometry is established mathematics. The possibly differentiating step is putting a gyration-dependent quantity into the selected-code score, not the existence of gyration itself.

### Where the operator already appears in this pipeline

The proposal uses Möbius operations in the **assignment score**: code and residual are composed in an order, then mapped back with `Log`; gyration is explicitly included as a second score component. The same operations also occur in the softmin training path through the assignment score and in the gradient path. They are not specified in the encoder or decoder, and the frozen Stage-2 losses remain otherwise unchanged. The origin-log/parallel-transport construction gives codewords a canonical tangent semantic before host-conditioned scoring.

Using the same operator for scoring, relaxation, and differentiation does not create three separate contributions. More importantly, hyperbolic representation or ranking papers that use Möbius operations upstream would be adjacent precedent, but would not by themselves establish that this particular use in quantizer assignment is occupied. The supplied scan did not verify those papers, so no identifiers are asserted here.

### Distinguishability and kill risks

This is not merely a curvature sweep on paper: at fixed nonzero curvature, composition order and gyration can alter the score, while the Euclidean limit makes gyration the identity. It is also more than a metric swap if the order-dependent term materially changes assignments. But the first score component is a norm after composing code and residual; it may behave like a reparameterized distance/metric comparison, and a nonzero gyration value alone is not evidence that the gyration penalty contributes meaningfully. Require matched-curvature ablations: ordinary geodesic-distance assignment, composition without the gyration term, reversed composition order, and the flat limit. Report assignment agreement and score variance attributable to each component.

There are mathematical/implementation ambiguities that could kill the proposal before novelty is considered:

* In `G_k`, the formula uses `Exp_o(T_s Log_o(z_k))` in the actual composition, while the preceding definition introduces `v_k` by parallel transport and `u_k=Exp_p(v_k)`. Those are not obviously the same construction. Specify which code point is composed and whether `T_s` acts on origin or host tangent vectors; otherwise the transport definition is redundant or inconsistent.
* The gyration expression must use one convention consistently. The displayed definition is a point-level gyrovector action, while the penalty subtracts `Exp_p(r)` as though both terms were vectors in the same tangent space. If `gyr[...]` acts on a point, this subtraction is not intrinsically defined; if it acts on a tangent vector, specify the tangent-space identification and parallel transport. A coordinate subtraction can be chart-dependent and may fail the claimed metric-preserving invariant.
* The code-plus-residual composition score and the gyration penalty may double-count related order effects. Establish that the second term changes assignment beyond the first term, rather than being an arbitrary auxiliary regularizer.
* “Fixed signed γ” is contradicted by the requirement `γ>0`; choose and freeze its value by a non-test rule. The prose says it can “penalize or reward” but the specified positive squared penalty only penalizes.
* The formula in the proposal defines `gyr[a,b]w` as a gyrovector operation; verify its sign/order against the adopted Möbius convention. A convention error could turn the claimed invariant into a test of an incorrect operator.

The flat-limit checks are necessary but weak: any non-flat chart parameterization can differ from Euclidean arithmetic without yielding a novel or useful assignment principle. The decisive question is whether gyration changes assignments and downstream ranking at fixed `c`, beyond composition-only and distance/metric controls.

### Verdict: **ADJACENT**

Closest work is HiHPQ (verified abstract only), which occupies hyperbolic quantization for retrieval broadly. The candidate's proposed distinction is **order-sensitive Möbius composition with an explicit gyration-dependent term at residual SID assignment**. Exact prior-art status for that mechanism is **UNKNOWN** on the supplied evidence. Standard gyrovector operations are not themselves novel. The candidate is distinguishable in principle from a curvature sweep, but not yet securely distinguishable from a composition-based metric reparameterization, and its tangent/point typing and transport definitions need repair before the mechanism is testable.

## Overall disposition

Neither proposal earns **NOVEL** on the evidence supplied. Neither can be called **OCCUPIED** by an exact verified match either. Both are **ADJACENT** to verified hyperbolic product quantization (HiHPQ, ABSTRACT_ONLY), with their claimed distinctions requiring full-text and broader prior-art checks. Candidate 1 has the clearer geometric distinction but a serious orientation-definition gap. Candidate 2 has a potentially sharper order-sensitive distinction but more severe operator/type ambiguities. These are novelty assessments only; neither is approved on style or presumed to pass the frozen strict ER, Recall@10, and NDCG@10 thresholds.
