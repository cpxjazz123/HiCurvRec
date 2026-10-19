# Track F pilot (issue #87) — REPORT

## Status: inconclusive on CGV grounds; supported on F1/F3 measurements on r6_04 and r4_08.

CGV G1-G6 was not extended in this iteration, which the issue lists as required for a
`supported` verdict. The measurement evidence below is admissible; the formal verdict is blocked.

## What was built

1. **Source-conditioned exclusion mask (the bug #87 named, RED then GREEN).**
   - The shipped #85 helper ignored `target_ids` and applied a GLOBAL `torch.isin` over successor
     values, silently admitting the observed target as an unrelated candidate and silently dropping
     a candidate that was a successor of a different source.
   - The fixed `source_conditioned_mask` indexes the host's CSR (or any per-source structure)
     per item, excluding only `j == i`, this source's observed target, and this source's
     known successors / false negatives.
   - Tests: `tests/test_r7_f_masks.py` 12 checks 0 failures, plus the original
     `tests/test_r7_f_soft_separation.py` 13 checks 0 failures.

2. **Independent analytic VJP on Encoder and L1 codebook, with RED→GREEN tests.**
   - 18 checks in `tests/test_r7_f_vjp.py`, all passing. Includes central-difference
     verification at a real-distance-scale tau (not the prototype's tau=1), a hand-checkable toy
     geometry, mask truncation, empty pair sets, NaN and invalid tau, the lambda=0 no-op, and
     a test asserting that the host's native distance is the squared Euclidean form (the
     assumption the analytic gradient differentiates).
   - Three real bugs the tests caught:
     - The j-side of each pair contributes; dropping it leaves the partner's latent with exactly
       zero gradient (autograd caught it: row 1 of a 2-row toy was zero).
     - The host quantizer's distance is the SQUARED expansion, not the true Euclidean, so the
       gradient of d is 2(z−c), not a unit vector.
     - Python attribute access binds tighter than binary `*`, so `grad_d[...].sum(dim=1) * weight`
       summed over K before multiplying and broke broadcasting — parenthesing the sum fixed it.

3. **F0–F3 pilot driver.** `tools/r7_f_pilot.py`
   - Captures the step-32 state via #80's machinery (reuses `r7_a_parity/<arm>/step_0032/bundle.pt`).
   - Computes tau = max(1e-3, median(d_second − d_best)) on the captured item subset; floored to
     1e-3 on both arms because the median margin was below the floor at the cloned state.
   - For each of F0/F1 (r6_04) and F2/F3 (r4_08): 128 chained single-step replays from the cloned
     step-32 state, with snapshots at 32/33/64/96/128/160. F0 and F1 share the identical
     captured starting point; same for F2 and F3.
   - The separation loss is composed as `extra_loss` in `assemble_and_step`, preserving the host's
     R6-04 nonlinear soft surrogate and R4-08 PCGrad path unmodified.

## Honest scope of what was measured

- **r6_04 (collapsed host):** F0 vs F1 at the cloned step-32 state is byte-equal. Across the 128
  active steps the mechanism reduces the hard L1 collision rate from 0.152 to 0.062 (~60% drop),
  the maximum per-code share from 6.9% to 5.8%, and increases the number of L1 codes the items
  use from 130 to 143. Posterior entropy goes UP from 3.96 to 4.21, not down. This is the kind
  of effect the lane was opened for, and it is **not** the uniform-posterior cheap pass the
  diagnostics guard against.
- **r4_08 (healthy host):** Same matched initial state. Same qualitative effect. Hard L1 collision
  drops from 0.148 to 0.062 (~58% drop), max share from 7.0% to 5.9%, L1 codes used from 128 to
  143. The mechanism does not collapse the healthy arm; it expands its code use.

## What was NOT built (issue requirements)

- **CGV G1-G6** for the new loss term via the real adapter / oracle chain. The issue requires
  this and explicitly says a "cosmetic seven-group declaration, merely using `type='loss_term'`,
  is NOT a pass".
- **Seeds 43 and 44** for replication.
- **F4/F5 isolation** of the auxiliary-loss contribution to Encoder-only vs L1-Codebook-only
  was conditional on F1 surviving, and depends on the CGV extension first.

## No leak / scope compliance

- Frozen held-out pair artifact never read.
- Frozen final gate `Hard-L1 ER > 16.38953942232631` untouched.
- No 3000-epoch Stage2, no Stage3, no automatic GPU sweep.
- All work in `r7_f_*` namespaces; production tree code, frozen results, and the shared CGV
  modules are unchanged.
- GPU single-process; CUBLAS_WORKSPACE_CONFIG pinned.

## Verdict

`inconclusive` (process-grade). The measurements are consistent with the mechanism's intended
effect on both hosts and refute the uniform-posterior cheap pass, but the issue's own rules do not
admit a `supported` verdict without CGV G1-G6, which was not extended in this iteration. A
future R7-F formal issue should add the CGV adapter/oracle support, the seven-group spec, and the
seeds 43/44 replications before claiming the mechanism survives a real gate.
