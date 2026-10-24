# The nine items of JOIN R round 12 (the last mile)

Both reviews now say the same thing at the top: they reviewed ONE mechanism, the unsigned
2-simplex area, and found no signed term and no second score. The nine items below are therefore
specification work, and every one is determinable.

## Mathematics (4)

1. **Rename the operator.** It is the **unsigned tangent-space wedge area**
   `A_k = ½·sqrt(‖u‖²‖v_k‖² − ⟨u,v_k⟩²)` with `u = log_p(q)`, `v_k = log_p(z_k)` and the Riemannian
   inner product at `p`. It is **not** the curved geodesic-simplex area, and calling it that is the
   defect. State the stable numerical treatment of `D = ‖u‖²‖v_k‖² − ⟨u,v_k⟩²` (it is a difference of
   large products and must be computed as a Gram determinant of the two vectors, not expanded).
2. **One tie rule, total, everywhere.** §2 and §8 currently disagree (`(A_k, d_c, k)` versus
   `(margin, lowest index)`). Adopt ONE rule: compare `A_k`, then `d_c(p, z_k)`, then the code index,
   ascending; report the compared keys and the selected winner per row. A non-degenerate equal-score
   minimum follows the same rule.
3. **The flat control.** Specify the coordinate/scaling convention of the `c→0` arm (how `p, q, z_k`
   are held or scaled) instead of "everything else identical", and remove the internal contradiction:
   agreement between the flat and curved arms is a NECESSARY check, and the candidate is rejected when
   the area-zero ablation fails the flip gate OR when the flat arm reproduces the same decisions.
4. **ε and the metric-swap claim.** State that training requires `ε > 0` (at `ε = 0` the surrogate does
   not smooth) and give `ε`'s units (it sits inside a squared length). Narrow the claim: the score is
   not a distance-only replacement because it depends JOINTLY on the target direction and the candidate
   direction; no experiment can separate it from every conceivable metric reformulation.

## Measurability (5)

5. **Emit the operands.** The file must carry, per example and code, enough to RECOMPUTE the area and
   the score: the Gram terms (`‖u‖²`, `‖v_k‖²`, `⟨u,v_k⟩`) plus `d_c`, with units and precision, and a
   normative verifier that recomputes `A_k`, `S_k` and both winners from them.
6. **The tie rule in the schema** must be the same rule as item 2, with the actual compared keys
   emitted and exact-versus-serialization ties distinguished.
7. **Denominators, operationally.** Report counts by operation type and level, the exact rate formula,
   and keep the failure boundary STRICT: clamp rate **> 0.1%** fails (not ">= 0.1%").
8. **Label what this file does not decide.** It supports the area-zero flip, tie and clamp checks; the
   flat-control comparison and the partition gate (distinct codes, largest-cell share) are SEPARATE
   artifacts and must be named as such, not implied.
9. **Binding.** Define the manifest's required fields, its canonical serialisation and digest, and bind
   every level's file to one run: frozen data, splits, recipes, checkpoint, `c`, `s`, `ε` and the
   candidate hash.

## Where each lands

| item | implementation |
|---|---|
| 1, 3, 4 | `candidate_v13.md` (the declaration) |
| 2, 6 | the declaration AND `emit_v2`'s tie block |
| 5, 7, 9 | `assignment_audit.emit_v3` (operands, per-operation counts, manifest) |
| 8 | the declaration's evidence section, split into three labelled artifacts |

---
## The reviews' own words

### Mathematics
```
# Mathematical review of `candidate.md`

## Verdict: REVISE

The single declared operator is reviewed here; there is no signed-area term or second score term in `candidate.md`. The frozen protocol and conjunctive thresholds do not affect the mathematical findings below.

The formula is a valid scalar **at each valid input**, but it is not the area of a geodesic 2-simplex in the curved host. Its main mathematical defect is therefore a mismatch between the named geometric object and the quantity actually computed. There are also inconsistencies in the tie-breaking specification and in the stated interpretation of the flat control. These are minimal, document-level revisions; they do not require a different candidate mechanism.

## Findings

1. **Scalar and degeneracies.** For `c>0`, valid points strictly inside the Poincaré ball, and the Riemannian inner product on `T_p`, `u=log_p(q)` and `v=log_p(z_k)` are finite tangent vectors. Cauchy–Schwarz gives
   `D=||u||²||v||²-<u,v>² >= 0`, so `A=½√D` is a well-defined nonnegative scalar. It vanishes exactly for linearly dependent vectors, including `u=0`, `v=0`, and the general collinear case. Thus the listed zero/collinear degeneracies are algebraically handled without division. This assumes the inner product is the host's tangent-space metric; the document should say so explicitly. In floating-point arithmetic, the Gram subtraction can produce a small negative value despite the exact inequality, so a numerical clamp or stable wedge-norm evaluation is needed to guarantee no NaN.

   **But:** `½√D` is the area of the Euclidean triangle formed by the two tangent vectors in `T_p` (equivalently, half their wedge norm). In nonzero curvature it is not, in general, the area of the geodesic triangle with vertices `p,q,z_k`. The logarithm map does not preserve areas. Calling it the “unsigned 2-simplex area the two geodesics ... span” is therefore false as written. Either rename it consistently as *tangent-space wedge area* (the minimal fix, matching the flat-limit formula), or replace the formula with the actual curved geodesic-triangle area and redo the limit/gradient analysis.

2. **Regularization and gradient.** With `ε>0`, `Ã=½√(D+ε²)` is well-defined. Its difference from `A` is exactly `ε²/[4(Ã+A)]`; it is bounded by `ε/2`, converges pointwise to zero, and for `D>0` is at most `ε²/(4√D)`. Hence the stated convergence and off-degenerate `O(ε²)` claim are correct (with fixed positive `D`). `D` is smooth in its vector arguments and `Ã` is smooth for positive epsilon. The unregularized wedge norm is locally Lipschitz and generally nondifferentiable where the wedge vanishes; this supports the stated Lipschitz/nondifferentiability claim. Add that training requires `ε>0`; at `ε=0` the surrogate does not smooth the degeneracy. Also specify the units of epsilon: since `D` has units of tangent-length to the fourth power, `ε` has units of tangent-length squared.

3. **Flat limit and control.** Under a finite-point, fixed-coordinate flat limit, the logarithms tend to `q-p` and `z_k-p`, and the distance tends to Euclidean distance. The displayed limiting expression is algebraically correct as a tangent wedge-area score. It is not a limit to curved geodesic-simplex area (which is the naming issue above). The control can test whether curvature changes assignments, provided its implementation specifies how `p,q,z_k` are held/scaled as `c→0`; that is not fully specified by merely saying “everything else identical.” The prose is internally contradictory: §3 calls agreement between flat and curved arms a necessary condition for non-triviality, while the decisive criterion and §6 reject when the flat arm reproduces assignments. Agreement is not the needed condition; **non-reproduction** is the stated discriminating criterion. Correct the sentence to say that the control tests whether curvature changes decisions, and that the mechanism passes this discriminator only if the flat arm does not reproduce them.


```

### Measurability
```
# Invariants review — measurability

## Verdict: REVISE

The candidate declares one mechanism: the unsigned 2-simplex area, tempered by geodesic distance, used to score assignments. I reviewed that operator and its `assignment_audit.emit_v2` evidence declaration only. I did not review a signed area or a second score term; neither is part of this candidate.

The schema is a substantial improvement over a summary-only audit. If the file is complete and truthful, its per-example winners, all 256 per-code full/ablated scores, margins, area/distance components, tie counts and coverage make the *reported assignment effect* inspectable without rerunning: a reviewer can recompute winners, flips and exact full-score tie share from the stored scores. The declared ablation is understandable (set area to zero, retain distance). The candidate also correctly acknowledges that the artifact alone does not prove the declared rule produced those scores.

That limitation is decisive for the stated question. As declared, the artifact does not carry enough independent evidence to verify the mechanism's effect rather than merely verify internally consistent reported scores:

1. **No score-reconstruction inputs.** Per-code area and distance are named, but the schema does not require the underlying `u`, `v_k` (or equivalent Gram terms and `d_c` values), nor a normative score-reconstruction/check procedure. Thus an auditor cannot independently check that the reported area is the Gram-determinant area, that the distance is the declared geodesic distance, or that `S_k = A_k/(1+d_c)` and its ablation generated the stored scores. Require sufficient per-example/per-code operands (or a separately specified independently checkable witness) and a verifier that recomputes scores and winners from them. Include units/conventions and numeric precision.
2. **Tie rule is inconsistent/underspecified.** §2 defines tie-breaking by `(A_k, d_c(p,z_k), k)`, whereas schema 2 says the applied key is `(margin, lowest index)`. Those are not equivalent, and the latter does not implement the declared operator in degenerate cases. Define one deterministic rule and record the actual compared keys and selected winner. Also distinguish an exact minimum-score tie from a tie after serialization/rounding; specify how equality is determined on the stored precision.
3. **Denominators need operational definitions.** `evaluation_count` is not explicitly tied to the number of log-map operations, score evaluations, or rows; clamp rate's numerator/denominator and aggregation scope are therefore not fully fixed. Report operation counts by operation type and level, with the exact rate formula. Reconcile `clamp_count`, `out_of_ball`, and `nonfinite_rows` with the clamp rule and ensure their index sets/counts are auditable. The stated failure is **clamp rate > 0.1%** (not ≥); preserve that strict boundary.
4. **The artifact does not cover all cited falsifiers.** It supports the area-zero flip/tie/clamp checks, but does not, by this schema alone, establish the flat-control decision comparison or the partition gate (distinct-code count and largest-cell share). Those require corresponding bound artifacts or fields, and explicit acceptance computations. If the requested claim is limited to the local assignment effect, label these as separate evidence rather than implying this one file decides them.
5. **Binding is not yet independently verifiable.** `candidate_sha256` and `run_manifest` are present, but the manifest's required fields, canonicalization/hash, and linkage to the frozen data/splits/recipes/checkpoint and per-level files are not specified. Define these, include a schema/versioned deterministic serialization and artifact digest, and bind all levels/stages to one run. Otherwise a hash-shaped field does not establish which exact run or protocol the rows represent.

The flip-rate threshold (≥5% **per level**) and exact-tie share threshold (≤10% per level) are stated. The audit should emit the numerator and denominator, not just rates, and a machine-checkable pass/fail for each level; this prevents ambiguity at boundary values and makes the conjunctive gate directly auditable. The current declaration provides a good basis, but does not yet make the falsifiers decisive from the artifact alone.

## Scope and frozen protocol

This review accepts the stated frozen 32-D latent, three 256-entry codebooks, three integer SID tokens, and unchanged data, splits and recipes as protocol constraints. The strict conjunctive downstream thresholds remain Hard-L1 ER > 16.38953942232631, Recall@10 > 0.06406796775709883 and NDCG@10 > 0.03530651701940677. They are not established by the assignment audit and still require their prescribed evaluation evidence. The five previously refuted families remain out of scope; this review does not revive them.

**Disposition:** revise the audit contract and align the tie-breaking definition before treating the evidence as sufficient for JOIN E. No exper
```
