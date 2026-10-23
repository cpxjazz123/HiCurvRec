# Subspace / product-manifold SID research: feasibility decision memo (issue #111)

**Decision in one paragraph.** The subspace family survives as a research direction, but not in the
form the issue's prompt suggested. Two of three candidates are mathematically new and survive the
geometry review; one is **provably inert** and is rejected. The review also produced a finding that
governs the whole hyperbolic programme: **in the Poincaré-ball model the curvature is a pure
rescaling**, so any "curvature sweep" is a scale change and cannot be the mechanism. Nothing here is
executable today: no implementation exists, no gate has been run, and #111 forbids GPU work. The
next arm should be **`hyp_transported_residual`** (cheapest, strongest boundary, answers a question
the cohort never asked); the family's flagship **`hyp_product_l1_quantiser`** needs a v2
specification plus one verifier extension before it can be built.

## 1. What was produced

| deliverable | file | status |
|:---|:---|:---|
| verified literature matrix | `results/hyperbolic_research_runs/subspace_111/subspace_literature_matrix.json` | 10 entries, all identity-verified; SAHL and Gu et al. covered |
| provenance + negative results | `.../source_search_log.jsonl` | 30 provider rows, including the 10 arXiv failures (HTTP 429 / timeout) |
| paper ↔ mechanism ↔ operator map | `.../prior_art_map.md` | every proposed operator mapped to its nearest paper and old arm |
| evidence-specific diagnosis | `.../subspace_failure_analysis.md` | 5 case studies over the #109 cohort, derived statistics, 5 pre-registration tests |
| mechanism proposals | `.../subspace_mechanism_proposals.json` | 3 frozen proposals (campaign schema, digests, seven-group contracts) |
| geometry review | `.../subspace_geometry_review.json` | H1–H7 per proposal + 4 cross-cutting findings |
| this memo | `.../subspace_feasibility_decision.md` | — |
| sandbox (reproducible) | `.../sandbox/h1_h7_feasibility.py`, `results.json` | CPU, synthetic, verifier-owned operators |

The shared #110 literature store was extended rather than duplicated: an OpenAlex provider was added
to `tools/harness2_genrec/literature.py` (the only provider that resolves arXiv-only identifiers
here), the paper store was fixed to accumulate across searches instead of overwriting itself, a
regression test was added, and `tools/harness2_genrec/tests/test_literature.py` is green (10 tests).

## 2. The three findings that change the picture

### F1 — the Poincaré curvature is a pure rescaling (`high`)

`x ↦ x/√c` is an isometry from the curvature-`−c` ball onto the unit ball. Therefore any intrinsic
quantity computed on points whose **normalised radius** `√c‖x‖` is held fixed is independent of `c`.
Measured in the sandbox: the relative deviation of the parallel transport from the identity is
`0.5143924039356781` at `c = 0.25`, `1` and `4`, and `0.5143924039356785` at `c = 0.001` — identical
to fifteen significant digits.

Consequences, stated plainly:

* a curvature sweep on a Poincaré-ball arm is a **rescaling of the latent space**, which H3 itself
  classifies as a scale-only cosmetic change;
* an arm whose hyperbolic content is "we used `c ≠ 1`" has no geometric claim to make;
* a genuine curvature effect needs a **relative** structure — different curvatures per factor of a
  product with a *common* tangent scale, a fixed ambient scale imposed by the host, or a
  non-conformal metric;
* the R10 arm's Busemann construction is unaffected (its content is the *shape* of the level sets,
  not the value of `c`), but no arm in this repository may claim that "curvature matters" on the
  strength of a `c` sweep.

### F2 — a conformal metric cannot change a conflict projection (`high`, refutes one candidate)

With `g_x = λ(x)² I`, both the dot product and the squared norm carry the same `λ²`, so

```
⟨u, v⟩_g / ‖v‖²_g  =  λ²⟨u, v⟩ / (λ²‖v‖²)  =  ⟨u, v⟩ / ‖v‖²
```

and the PCGrad step is **exactly** the Euclidean one. Measured: conflict-set disagreement `0.0` over
256 codewords while the metric ranges over `5.31 … 324.22`. The candidate
`hyp_factor_conflict_projection` is **invalid**, not merely derivative. A metric-aware projection
would need a metric that is not a scalar multiple of the identity within the projected coordinates.

### F3 — a domain violation is silent (`medium`)

A point outside the ball does not raise: at `c‖x‖² = 2.38` the transport stayed `finite=True` while
its value reached `2.4e11`. A finiteness check is therefore not a stability check. The verifier and
any implementation must assert `c‖x‖² ≤ 1 − margin` and declare what happens when it trips.

## 3. Ranking (the issue's four criteria, applied)

| rank | candidate | transfer hypothesis | novelty | verification risk | implementation cost | verdict |
|:--:|:---|:---|:---|:---|:---|:---|
| 1 | `hyp_transported_residual` | **medium**: targets level-2/3 quantisation quality, the one lever the cohort never tested (no arm moved Stage3 *without* moving L1). Caveat: the control's L2→L3 branching is already only `1.465`, so the L3 token is nearly determined by L2 and the ceiling may be low | **high**: no verified paper found on a codeword-frame residual; PT itself is textbook | **medium-low**: the operators are sandbox-verified (isometry 8.7e-09, roundtrip 6.7e-09); the L1 bit-identity test is a hard boundary; but H4/H6 are unverified and G5 still cannot pool the codebook | **medium**: one forward change + one decode inverse, inside the existing trainer | **recommended first arm** |
| 2 | `hyp_product_l1_quantiser` | **high**: the only candidate that can change the L1 partition without coarsening it, and the measured failure (largest cell `374`/`382` against the control's `237`) is an L1-tail failure | **high**: the product-geodesic assignment with factor-masked supervision has no verified prior art | **high**: the declared tangent scale is degenerate (boundary margin `9.46e-06`), F1 removes the curvature claim as declared, and the L1 partition is precisely what failed to transfer 14 times | **high**: a new codebook module, the residual decode, the factor mask | **flagship, blocked on a v2 + the G5 extension** |
| 3 | `hyp_factor_conflict_projection` | none | none (r4_08 family) | — | low | **rejected (invalid, F2)** |

**Why the recommendation is not the top-ranked one.** P1 has the higher ceiling and is the family's
real contribution; P2 is the one that can be built, sealed and run without first rewriting its
specification and without extending the verifier. Given #110's rule of one cold 3000-epoch Stage2 per
mechanism and no repeats, the cheaper, more falsifiable arm should spend the next slot.

## 4. What must happen before either arm is launched

1. **v2 of P1 and P2** (Agent M, new proposal version, new digest):
   * the tangent scale must come from the **measured** latent-norm distribution of the frozen host
     (`s = s₀/(√c · ‖z‖_p95)` with `s₀` dimensionless and both numbers frozen) — a pre-specification
     measurement, never a post-hoc tuning;
   * the curvature declaration must be **relative** (a per-factor curvature ratio with a common
     tangent scale) or dropped, because of F1;
   * P2 must declare the **domain guard** (F3).
2. **The CGV extension**: G5's non-trace branch must pool the **codebook** gradient. All three
   proposals, and the R10 arm before them, declare the codebook as the target while the gate sets
   `codebook_deviation = float("nan")` in the non-trace branch. This is now the single most valuable
   verifier extension for #110: without it, no codebook-targeting mechanism can pass G5 except by
   declaring a target it does not have.
3. **Implementation** (Agent I) only after 1 and 2, with the frozen digest, and with the
   `forbidden_substitutions` regexes enforced (no `cdist` assignment, no PCA/whitening, no ambient
   residual, no capacity inflation, no zeroed curvature).
4. **H4 and H6** then become runnable; until then they are `unknown`, and `unknown` blocks.

## 5. Unanswered questions this issue leaves open

* **Can a subspace design be coherent under 3 tokens?** Every verified subspace-SID paper (SAHL and
  the 2026 RecSys papers) uses 16–64 digits. P1 sidesteps the question by putting the factorisation
  *inside* one 256-way level-0 codebook instead of across tokens. Whether that preserves the
  inductive-bias benefit SAHL reports is unknown, and it is the family's central risk.
* **Is the L0 tail the mediator of the transfer failure?** The cohort's `ρ(largest cell, Recall)`
  is only `-0.18` (n = 14) — far too weak to call it the mediator. P1's falsifier is built on it, so
  a P1 run is also a test of that premise.
* **Why is L2→L3 branching so low (≈1.47 for the control and most arms)?** If the L3 token is nearly
  redundant, P2's ceiling is small and the real capacity problem is at L1/L2. This is a diagnostic
  question with no answer in the current evidence.
* **Does G_H1 mean anything at matched occupancy?** Case C shows the probe rewards degeneracy; no arm
  in the cohort reports G_H1 at matched L0 occupancy, so the `+0.46` association remains
  uninterpretable.

## 6. What would make the *next* hypothesis different

The issue asks for a mechanism that is not a renamed Euclidean loss. From this review, a genuinely
different next direction would have to involve a **non-conformal** metric (F2) or a **relative**
curvature structure (F1) — the two places where the geometry can actually change a computation. A
candidate that merely places Euclidean machinery inside a ball will fail H2/H3 exactly as P3 did, and
the review can now reject it in minutes with the sandbox in this directory.

## 7. Limitations of this memo

* No candidate was trained and no metric was claimed. Every expected effect in the proposals is a
  **prediction**, and the falsifiers are stated so that a single run can refute them.
* The literature search could not reach arXiv (HTTP 429 on all ten queries); an arXiv-only paper on
  Riemannian or product-manifold SID construction could exist that this review would have found.
  `subspace_literature_matrix.json` records that limit rather than hiding it.
* The sandbox's operators are the *verifier's* implementation of the declared mathematics, which is
  what H7 checks; they are not the candidates' implementations, and no claim is made that an
  implementation would be numerically identical.
* The #109 cohort is retrospective and already inspected; the failure analysis's correlations are
  exploratory and are labelled as such in that document.
* #111 authorises no GPU work. Nothing in this directory starts a run, and no registry outside
  #110's exists.
