# LOTV declaration — revision 2 (JOIN R)

## Decision

**WITHDRAWN.** The required pre-registered occurrence rate cannot be substantiated from the staged material: no assignment-stream measurements or complete candidate-score stream are supplied. The declared intervention acts only on near-ties, so without evidence that ties occur often enough and are actually broken, it is a no-op in practice. No frequency is invented or inferred from the mathematical definition. The mechanism is withdrawn rather than advanced on the basis of a merely possible tie fixture. The thresholds and audit contract below are retained as the criteria a future, separately authorized preregistration would have had to meet; they are not post-hoc tuning permission.

This declaration reviews exactly one mechanism: signed Lorentzian 4-by-4 minor (LOTV) tie-breaking. The parked unsigned-area candidate is not under review and is not part of this declaration.

## OPERATOR

This proposal changes only Stage-2 candidate assignment. It does not change the encoder, latent dimension, codebooks, SID serialization, data/splits, or Stage-2/Stage-3 recipes. Curvature is fixed at -1. For item `x` and three codewords `c0,c1,c2` selected by the ordinary frozen residual-quantization path, each represented in the 33-coordinate hyperboloid model

`H^32 = {p in R^33 : <p,p>_L=-1, p0>0}`, with `<u,v>_L=-u0*v0 + sum(i=1..32) ui*vi`,

define `D = det([x c0 c1 c2]_{rows 0..3})`, with columns in the stated order and rows 0 through 3 in the existing fixed ambient coordinate order. Define `A=0` when `|D|<=eps_D`, otherwise `A=D/(1+|D|)`, where `eps_D=1e-12`.

`D` is a coordinate-dependent 4-by-4 minor, **not** the full ambient Lorentz-invariant simplex volume. Its fixed-coordinate dependence and orientation convention are part of the operator; it is not invariant under general spatial rotations. It is not a tangent area or transported-frame quantity.

The frozen ordinary objective is the primary objective. For each assignment level, choose its ordinary winner by minimum objective (or the frozen objective's existing direction, if not a minimization); candidates within the declared score tolerance of that winner form the tie set. If the tie set has more than one distinct SID triple, select the member with greatest `A`. If `A` values tie under the same tolerance, select the lexicographically smallest integer SID triple. Outside that set, retain the ordinary winner. No weighted secondary score is added.

## FINITE-PRECISION DEFINITION AND ADMISSIBLE DOMAIN

All operator inputs are converted to IEEE-754 binary64 with round-to-nearest, ties-to-even before evaluation. The admissible coordinate domain is finite binary64 coordinates with `|p_i|<=1e100` for every coordinate of each of the four points, `p0>0`, and a finite binary64 Lorentz norm check satisfying `abs(<p,p>_L+1)<=eps_H`, `eps_H=1e-10`. The norm check is evaluated in the fixed order: products `p_i*p_i` in increasing `i`, accumulated left-to-right from `-p0*p0`, then add 1 for the validity test. This bound keeps each product and the at-most-32-term sum far below binary64 overflow. It does not eliminate cancellation; validity is defined by this specified binary64 calculation, not by an ideal real-valued norm. Inputs outside the domain are invalid, not repaired or projected.

Evaluate the determinant by the explicit 24-term Leibniz sum, permutations in lexicographic order, each four-factor product left-to-right in binary64, and signed terms accumulated left-to-right in that order. Under the coordinate bound each product and the 24-term accumulation is finite (each product is at most `1e400`, and their sum at most `24e400`, below binary64's approximately `1.8e308` limit). If any specified intermediate is nevertheless non-finite, the example/run fails closed; there is no clamp, replacement, alternate determinant, or fallback. The near-zero branch is a hard threshold, not a smooth regularization.

The comparison tolerance is fixed as `tau_score=1e-12`. Scores `s1,s2` are tied iff `abs(s1-s2) <= tau_score*max(1,abs(s1),abs(s2))`; the same formula and tolerance apply to `A`. Comparisons use binary64 values. Exact remaining ties resolve by lexicographically smallest SID triple, independent of candidate enumeration order. Candidate triples must be unique before this final comparison. Invalid inputs fail the run.

## PRE-REGISTERED EFFECT / OCCURRENCE GATES

Rates are defined on the complete frozen assignment stream, with assignment as the unit and levels attributed to the codebook decision being made. An occurrence is an assignment where the ordinary winner's tie set contains at least one distinct alternative under the exact rule above. A break is an occurrence where LOTV selects a different triple from the ordinary winner. The occurrence-rate denominator is all attempted assignments at that level; the break/flip-rate denominator is also all attempted assignments at that level. Invalid or failed assignments remain in the denominator and cause run failure; they are never dropped. Degenerate `A=0` cases count as occurrences if their score tie test passes, but count as breaks only if the declared rule selects a different triple.

Pre-registered gates: (a) at least **5% flips per level**, inclusively (`breaks / all attempted assignments >= 0.05` at each of the three levels); (b) the tie share is at most **10%**, inclusively (`occurrences / all attempted assignments <= 0.10` per level); and (c) clamp rate greater than **0.1%** fails. LOTV specifies no clamp: clamp event means any value clipped, saturated, or substituted by the implementation, including an attempted non-finite recovery; the permitted count is zero, and any such event fails. The 5% per-level flip gate is the pre-registered required occurrence/effect rate: because a break cannot occur without an occurrence, a lower occurrence rate makes the required effect impossible. The 10% upper tie-share gate is applied separately and cannot excuse failure of the 5% flip gate. These are screening criteria, not tuning directions.

No qualifying measured rate is present in the staged evidence. Therefore the mechanism is withdrawn now; these gates do not establish that LOTV meets them.

## GRADIENT AND PROTOCOL

Selection remains a hard, non-differentiable assignment. `A` contributes no training loss and is not a surrogate gradient; the existing commitment/quantization gradient and STE remain unchanged. No smoothness, flat-limit, curvature-discrimination, or gradient-improvement claim is made.

Frozen capacity/protocol: latent dimension 32; exactly three codebooks of 256 entries; exactly three integer SID tokens per item, each in `[0,255]`; same original data/splits and unmodified Stage-2/Stage-3 recipes. No learned parameters, extra codebook, scalar, token, or post-hoc token. Success outcome thresholds remain strict and conjunctive, unrounded: Hard-L1 ER `>16.38953942232631`, Stage3 Recall@10 `>0.06406796775709883`, and NDCG@10 `>0.03530651701940677`. Equality or failure of any one fails. This withdrawal does not alter or claim results for the frozen protocol.

## AUDIT / REPLAY CONTRACT

Before any training, retain a reproducible numerical-oracle corpus of 10,000 deterministic finite fixtures (including 1,000 degeneracy fixtures), its generator and seed or exact inputs, independent oracle implementation/version and outputs. Verify the prescribed binary64 operation sequence, sign reversal under odd determinant-column permutations, sign preservation under even permutations, and exact zero/degenerate flag at `|D|<=1e-12`. Retain the fixture-level results.

For assignment-stream replay, use a versioned machine-readable schema with canonical encoding (UTF-8 JSON Lines; integers decimal; binary64 values serialized as exact hexadecimal representations), stable item IDs, level, monotonically increasing sequence numbers, expected item and candidate counts, and a manifest binding every row to split and run. Emit every attempted item, every candidate triple and ordinary score, candidate enumeration order, exact ordinary winner and its deterministic tie handling, all tie-set members, each member's vectors after conversion, norm operands/results/validity, 24 determinant terms, accumulated `D`, degeneracy status, `A`, comparisons, and selected triple before and after LOTV. Include explicit failure rows so an invalid item cannot disappear. Include per-level totals for attempted assignments, occurrences, breaks, flips, tie share, clamp events, invalids, and codebook decisions; an independent checker must recompute all rates and selection from rows.

Bind the finalized stream, final SID artifact, source revision/build, dataset/split manifests, Stage-2/Stage-3 configs, evaluator inputs and outputs, and checker version/results with row counts and cryptographic file hashes in a run manifest. Finalize the manifest after emission and protect/sign it using a recorded key identity or immutable artifact-store receipt. Report exact evaluator population and aggregation, unrounded metrics, and their strict comparisons. Hashes alone do not prove metric provenance: retain evaluator inputs/outputs or a signed evaluator receipt binding metrics to the run, split, final SID and recipe hashes. Retain gradient capture scope plus before/after digests if asserting unchanged gradients; no such claim is made here. An independent checker must be able to detect missing rows and replay the decision. Any missing operand, count mismatch, failed item, unverifiable artifact, or clamp makes the run non-passing.

## PRIOR ART AND SCOPE

HiHPQ (DOI `10.1609/aaai.v38i5.28261`, arXiv `2401.07212`) is known here only from an abstract-only record. It establishes adjacency at a broad hyperbolic/geometric-quantization level, not occupation or novelty of this operator. Full paper and supplement inspection and a broader search for geometric quantization in recommender/SID systems are required before any novelty or non-overlap claim. Defensible wording only: “Potentially distinct operator; novelty and non-overlap remain unverified.” If direct equivalence is found, withdraw. The parked unsigned tangent-space area is neither this mechanism nor rehabilitated by this declaration.

## STATUS: WITHDRAWN — no complete assignment-stream evidence establishes the pre-registered >=5% per-level tie-breaking rate; without frequent ties LOTV is an operational no-op, and no rate is fabricated.
