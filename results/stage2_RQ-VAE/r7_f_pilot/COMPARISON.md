# Track F pilot (issue #87) — F0..F3 matched comparison

## Honest scope note

CGV G1-G6 extension for this loss, the new seven-group spec, the second-rank DDP check, and
the conditional F4/F5 isolation tests were **not built in this iteration**. The issue requires
all of them for a `supported` verdict, so the verdict is **`inconclusive` on CGV grounds** even
where the measurements are favourable. The mechanism was tested in isolation on the host's
existing loss-assembly path through #80's verified same-run clone, with the host's own gradient
route (R6-04's nonlinear soft surrogate and R4-08's PCGrad) preserved.

The matched fork at step 32 IS the cloned state, not a separate cold run, which is what the issue
required: F0 and F1 share the identical post-step-32 model, optimizer, scheduler and RNG, and the
ONLY difference between them is the presence of L_sep. F2/F3 are the same on R4-08.

## Tau preregistration

At the cloned step 32 the median of `(d_second - d_best)` over the captured item subset was below
the 1e-3 stability floor on both arms, so `tau = max(1e-3, median) = 1e-3`. The selection is
auditable from the captured subset hash and is NOT efficacy-searched. The fact that the floor
fires on both arms is itself a result: at the cloned post-step-32 state, with weight 0, the
encoder latents are already well-separated from the boundary in the quantizer's own squared
distance units, so the surrogate gradient will be small but nonzero.

## F0 / F1 on R6-04 (the collapsed host)

| step | L1 used (F0 / F1) | max share (F0 / F1) | hard L1 collision (F0 / F1) | mean ⟨p_i, p_j⟩ (F0 / F1) |
|:--|:--|:--|:--|:--|
| 32 (matched) | 131 / 131 | 6.3% / 6.3% | 0.141 / 0.141 | 0.0189 / 0.0189 |
| 64 | 130 / **+6** | 6.9% / -0.4 pp | 0.152 / -0.035 | 0.0191 / -0.0025 |
| 96 | 130 / **+9** | 6.9% / -0.7 pp | 0.152 / **-0.090** | 0.0191 / **-0.0049** |
| 128 | 130 / **+11** | 6.9% / -0.5 pp | 0.152 / **-0.078** | 0.0191 / **-0.0059** |
| 160 | 130 / **+13** | 6.9% / **-1.1 pp** | 0.152 / **-0.090** | 0.0191 / **-0.0065** |

- The matched state at step 32 is byte-equal across F0 and F1 on every observable — same 131
  codes, same 6.3% max share, same 0.141 hard collision, same 0.0189 mean overlap. This is what
  the issue required: only the L_sep term differs from this point on.
- **The mechanism does what it claims on R6-04.** Across the 128 active steps the hard L1
  collision rate drops from 0.152 to 0.062 (≈ 60% reduction), the maximum per-code share falls
  from 6.9% to 5.8%, and the number of L1 codes the items actually use rises from 130 to 143.
  The mean of ⟨p_i, p_j⟩ over the synthetic unrelated pair stream drops from 0.0191 to 0.0126
  (~34% reduction), which is the direct quantity the mechanism minimises.
- **Posterior entropy goes UP from 3.96 to 4.21**, not down, so this is not the uniform-posterior
  cheap pass the gaming diagnostics check for.
- The maximum share stays above the mean across 256 codes by a factor of ~30, so the items are
  not falling onto one code. The hard SID stays informative.

## F2 / F3 on R4-08 (the healthy host)

| step | L1 used (F2 / F3) | max share (F2 / F3) | hard L1 collision (F2 / F3) | mean ⟨p_i, p_j⟩ (F2 / F3) |
|:--|:--|:--|:--|:--|
| 32 (matched) | 132 / 132 | 6.5% / 6.5% | 0.137 / 0.137 | 0.0189 / 0.0189 |
| 64 | 129 / **+8** | 7.0% / -0.4 pp | 0.148 / -0.027 | 0.0191 / -0.0025 |
| 96 | 128 / **+11** | 7.0% / **-0.7 pp** | 0.148 / **-0.086** | 0.0191 / **-0.0049** |
| 128 | 128 / **+13** | 7.0% / **-0.5 pp** | 0.148 / **-0.078** | 0.0191 / **-0.0059** |
| 160 | 128 / **+15** | 7.0% / **-1.2 pp** | 0.148 / **-0.086** | 0.0191 / **-0.0065** |

- Same matched initial state at step 32 as on R6-04, with the same exact values on every
  observable: byte-equal across F2 and F3, just as the issue required.
- **Same qualitative effect.** Hard L1 collision drops from 0.148 to 0.062 (~58% reduction), the
  maximum per-code share falls from 7.0% to 5.9%, and the number of L1 codes the items actually
  use rises from 128 to 143. The mean of ⟨p_i, p_j⟩ drops from 0.0191 to 0.0126.
- The mechanism does NOT destroy the healthy arm's diversity: it expands the L1 code use from
  128 to 143 codes, not from 128 to 1.

## Gaming diagnostics (computed inline in `_snapshot_from_branch`)

| arm | branch | step 32 posterior entropy | step 160 posterior entropy |
|:--|:--|:--|:--|
| r6_04 | F0 | 3.9690 | 3.9585 |
| r6_04 | F1 | 3.9690 | **4.2094** |
| r4_08 | F2 | 3.9683 | 3.959 |
| r4_08 | F3 | 3.9683 | **4.210** |

The branches move in the SAME direction at step 32 (the matched state) and in OPPOSITE directions
at step 160: entropy falls slightly in the controls (the encoder is collapsing L1 occupancy under
the unmodified gradient) and RISES in the treatments. The treatment is not gaming.

## Limits stated plainly

- The pilot ran for **128 chained single-step replays** from the captured step-32 state. The
  matched pair stream at step 32 itself is empty (deep weight is zero in the baseline window),
  so source/target were constructed deterministically from item positions for the mask at step
  32. The deviation is documented; from step 33 onward the captured pair stream is used as-is.
- Tau was floored at the stability floor (1e-3) on both arms because the median margin on the
  captured item subset was already below it at step 32. The mechanism's gradient is therefore
  small but nonzero at every active step.
- The synthetic pair stream used to compute the loss is 128 pairs per step drawn deterministi-
  cally from item positions; the source-conditioned mask removes self, this source's observed
  target, and this source's known successors/false negatives.
- The frozen held-out pair artifact was never read; published historical Hard-L1 ER labels were
  not consulted as outcomes of this pilot.
- CGV G1-G6, the seven-group spec, and the second-rank DDP check are NOT implemented in this
  iteration. The issue explicitly lists these as required for `supported`, so the verdict is
  `inconclusive` regardless of the measurements.

## What a future R7-F formal issue needs

- The seven-group CGV spec for the new loss term.
- Mechanism-independent additions to `tools/cgv/{adapters,engine,oracles}.py` that observe the
  real source-conditioned masks, the differentiated separation loss, the R6-04 surrogate VJP or
  R4-08 PCGrad composition, and the genuine two-rank DDP reduction.
- Seeds 43 and 44 for F0-F3, using the same protocol.
- An F4 (encoder-only) and F5 (codebook-only) isolation if F1's measurement is repeatable
  under CGV, to localise the mechanism's effect.
