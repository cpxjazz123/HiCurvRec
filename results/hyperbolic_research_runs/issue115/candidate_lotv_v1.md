# CANDIDATE -- Lorentzian oriented tetrahedral volume (LOTV), issue #115 JOIN R round 1

One mechanism. The full declaration by Role M follows verbatim, together with the frozen protocol and
the requirement that it be numerically well defined in FINITE PRECISION as part of the declaration.

```
# Mechanism proposal: Lorentzian oriented tetrahedral volume (LOTV)

## OPERATOR

This proposal changes only the Stage-2 **candidate assignment score**; it does not change the encoder, latent dimension, number or size of codebooks, SID serialization, dataset/splits, or Stage-2/Stage-3 recipes. It is a testable proposal, not a claim that the frozen performance thresholds will be met.

Let curvature be `-1` and let `x,c_0,c_1,c_2` be four points on the 32-dimensional hyperboloid

`H^32 = {p in R^33 : <p,p>_L = -1, p_0 > 0}`,

where `<u,v>_L = -u_0 v_0 + sum(i=1..32) u_i v_i`. Here `x` is the item and `c_j` are the three codewords selected by the **ordinary frozen residual-quantization path** (one from each of the three existing codebooks). The origin/basepoint is `o=(1,0,...,0)`. No tangent frame is selected or transported.

Define the signed Lorentzian simplex determinant

`D(x,c0,c1,c2) = det([x c0 c1 c2]_{rows 0..3})`.

The determinant uses the first four ambient coordinates in their existing fixed coordinate order. Define the candidate interaction

`A = D / (1 + |D|)`.

Use the sign of `A` as a tie-break/interaction term in the existing assignment objective: among candidate SID triples whose ordinary frozen objective differs by at most `tau_score`, prefer larger `A`; otherwise retain the ordinary objective winner. This bounded interaction has no tunable coefficient. Its semantics are ambient oriented 4-volume, not a tangent residual, codeword radius, level set, or group-composition score. The basepoint states the hyperboloid chart/orientation convention; the determinant itself is evaluated on the actual hyperboloid coordinates and does not use `log_o` or a transported frame.

## FINITE-PRECISION DEFINITION

All values and operations are IEEE-754 binary64 for this operator, with round-to-nearest, ties-to-even; determinant evaluation is the explicitly specified sum over the 24 permutations, in lexicographic permutation order, with each product accumulated left-to-right using binary64. Inputs must be finite binary64 values. This contract is independent of the model's storage dtype: coordinates are converted to binary64 before the operator.

* Hyperboloid validity: require `p_0 > 0` and `abs(<p,p>_L + 1) <= eps_H`, where `eps_H = 1e-10`. No projection or renormalization is performed. Violation makes the operator invalid for that example; it is not silently repaired.
* Determinant degeneracy: `eps_D = 1e-12`. If `abs(D) <= eps_D`, define `A=0` and mark the interaction degenerate. Otherwise compute `D/(1+abs(D))`. The denominator is at least one; no division-by-zero branch exists.
* Non-finite intermediate or result: invalidate the example and fail the run; no clamp, replacement, or alternate operator is allowed.
* Score comparison tolerance: `tau_score = 1e-12` in the units of the frozen ordinary assignment objective. For scores `s1,s2`, treat them as tied iff `abs(s1-s2) <= tau_score * max(1,abs(s1),abs(s2))`. In a tie, compare `A`; compare A values with the same relative tolerance. If still tied, choose the lexicographically smaller integer SID triple. Outside a tie, the ordinary objective decides.
* Boundary cases: a zero determinant, repeated points, linearly dependent first-four-coordinate columns, or any orientation-reversing configuration with negative determinant are all defined directly: zero/near-zero maps to `A=0`; a negative determinant retains its negative sign. Exact comparison ties resolve lexicographically. Invalid hyperboloid points and non-finite arithmetic fail closed rather than falling back.

These tolerances are part of the proposal and must not be tuned after observing results. The prescribed determinant evaluation is not interchangeable with a library `det` unless that implementation is demonstrated to produce the same binary64 value on the audit corpus.

## GRADIENT PATH

The hard argmin/SID selection remains the existing non-differentiable assignment path. `A` is used only to resolve the declared near-tie set; it contributes **no training loss** and is not inserted as a differentiable surrogate. The existing Stage-2 commitment/quantization gradient and straight-through estimator are unchanged. Thus there is no new gradient claim: the operator changes discrete assignment decisions only. In an exact/near-degenerate tie it can choose a different SID triple; away from ties it is exactly inert.

## INVARIANTS THAT FAIL FOR THE IDENTITY

The identity mechanism is the frozen ordinary assignment with no LOTV tie-break. The following are falsifiable non-identity invariants:

1. On a fixture with a valid triple, `D > eps_D` and two ordinary candidate scores tied within `tau_score`, the selected triple must be the one with greater `A` (unless the final `A` comparison also ties).
2. Under an odd permutation of the four determinant columns, the sign of nondegenerate `A` reverses; under an even permutation it is preserved. The ordinary assignment score is unchanged by this operator-specific transformation, so this orientation response cannot be supplied by the identity.
3. On a constructed nondegenerate tied fixture with opposite-sign determinants, the chosen SID triple changes when the ambient orientation is reversed, while the frozen ordinary score and its gradient are byte-for-byte unchanged.

These are operator-level invariants, not evidence of improved retrieval. The fixed-coordinate orientation convention is intentionally explicit; the mechanism does not claim full rotational invariance.

## PRE-REGISTERED FALSIFIERS WITH NUMBERS

Run all checks before any full training. Thresholds below are fixed in advance.

1. **Numerical oracle:** 10,000 deterministic finite hyperboloid quadruples, including 1,000 degeneracy fixtures. Fail if any output differs from the independently implemented lexicographic 24-term binary64 oracle by more than `2e-15` absolute, if a finite valid input yields a non-finite output, or if any invalid input is silently accepted.
2. **Sign/degeneracy:** 1,000 random valid quadruples and their odd column permutations. Fail if any nondegenerate sign does not reverse, or any even permutation changes sign. For `abs(D)<=1e-12`, fail unless output is exactly `0` and the degenerate flag is set.
3. **Identity/non-vacuity on training assignment stream:** log every ordinary candidate comparison. Fail as a useful candidate if fewer than `0.1%` of item assignments encounter a distinct candidate within the declared score tolerance, or if fewer than `100` assignments change in the full training set. These are screening falsifiers, not permission to alter the tolerance.
4. **Capacity/protocol:** fail immediately unless the emitted SID array has exactly three integer tokens per item, all tokens in `[0,255]`, exactly three codebooks of 256 entries, and latent dimension 32. No extra codebook, learned scalar, or post-hoc token is permitted.
5. **Frozen outcome:** the mechanism is successful only if the same original data/splits and unmodified Stage-2/Stage-3 recipes produce all three strict inequalities: Hard-L1 ER `> 16.38953942232631`, Stage3 Recall@10 `> 0.06406796775709883`, and NDCG@10 `> 0.03530651701940677`. Equality or failure on any one is a falsification; do not round before comparison.

## CAPACITY STATEMENT

No new learned parameters. No change to the 32-dimensional latent, the three 256-entry codebooks, or the three-integer SID. The operator is a deterministic, bounded, parameter-free tie-break on existing candidate triples. It is not a token layout and performs no frame transport. It may alter assignments only within the predeclared ordinary-score tolerance.

## WHY NOT THE SIX CLOSED FAMILIES

1. Not product-geodesic level-0 radius assignment: it uses a four-point oriented determinant, not a codeword radius, and does not collapse an item to one code.
2. Not horospheric/Busemann levels: no Busemann function or level gap is computed.
3. Not transported residual: no residual is transported and no codeword frame is constructed.
4. Not role-assigned layout: token roles and serialization are untouched; the tie-break is evaluated on already formed candidate triples.
5. Not gyro/Möbius associator: no Möbius addition, gyrooperation, or associator appears.
6. Not the parked unsigned tangent wedge: the operator is a signed ambient Lorentzian 4-by-4 determinant of four hyperboloid points, not an unsigned tangent-space 2-area. It does share the broad idea that orientation can matter; that is a mathematical relation, not a claim that the prior candidate has been rehabilitated. If the review classifies this as a disguised wedge family, withdraw rather than relabel it.

## WHY NOT HiHPQ (arXiv:2401.07212, abstract-only)

This distinction is provisional because only the abstract is available here; no claim is made about the paper's complete method. LOTV is specified as a deterministic, signed Lorentzian simplex-determinant tie-break over existing Stage-2 candidate triples, with a fixed finite-precision contract. The proposal does not assert that HiHPQ lacks geometric ranking or hyperbolic quantization, and abstract-only evidence cannot establish novelty or non-overlap. Before implementation/admission, an auditor must compare the full paper and supplementary material against the exact determinant, assignment scope, and tie rule above. If that comparison reveals the same operator or a direct equivalent, withdraw this proposal.

## AUDIT: REQUIRED EMITTED OPERANDS

For every evaluated item and every candidate triple considered within `tau_score`, emit enough raw operands to recompute the decision independently:

* item ID; three integer code indices; codebook IDs; candidate enumeration order;
* the binary64 `x,c0,c1,c2` vectors (all 33 ambient coordinates), plus original source dtype and conversion rule;
* each hyperboloid norm check `<p,p>_L`, `p_0`, validity flag, and `eps_H`;
* the frozen ordinary assignment score before tie-break, score units, winning/runner-up values, computed comparison scale, and whether the pair passed the exact tie test;
* the 24 determinant permutation terms in prescribed order, accumulated `D`, `eps_D`, degeneracy flag, `A`, and selected/rejected reason;
* final SID triple, serialized token dtype/range/shape checks, codebook cardinalities, latent dimension;
* hashes/identifiers for dataset split manifests, Stage-2 and Stage-3 recipes/configurations, source revision, and evaluator outputs; final unrounded Hard-L1 ER, Recall@10, and NDCG@10.

An independent checker must be able to reconstruct the tie set, determinant, tie resolution, and final SID from these operands without importing the implementation under test. Missing operands make the result unauditable and therefore not a pass.

```
