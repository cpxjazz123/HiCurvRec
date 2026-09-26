# Iter26 Failure Attribution

## Mechanism executed

Iter26 was a one-factor replication of Iter16's closed-form fixed-curvature
implementation with the residual input vector replaced by raw residual medians
`[1.0, 0.10941, 0.09331]`. No other mechanism changes; the `c_base=0.5`,
`alpha=0.2`, `m_min=0.09331`, `s_l = log1p(B_l) / log1p(m_l/m_min)`, and
`c_l = clip(c_base * exp(alpha * z_l), 0.05, 1.5)` mapping is byte-identical to
Iter16. The derived `c_l = [0.6145357379232853, 0.5333020920777128,
0.3814078098431606]` is locked into `scripts/computed_behavior_branching.json`
and registered as a non-trainable buffer in every `Quantize` layer.

## Five-label attribution

| Label | Applies? | Evidence |
|---|---|---|
| IMPLEMENTATION_FAIL | NO | MVG four-layer PASS; the runtime `_check_fixed_curvature_invariant` logged every step in `{0, 25000, 50000, 100000}` without raising; the on-disk `rqvae_best.pth` matches the locked `c_l` to `1e-6`. |
| MECHANISM_NOT_ACTIVATED | NO | The fixed `c_l` is registered as a non-trainable buffer; `get_c()` returns the locked values at every checked step; behavior loss gradient nonzero (MVG layer 2); 5-step optimizer updates change every trainable parameter by L2-norm relative magnitude > 1e-7. |
| GEOMETRY_MISMATCH | partial | Stage2 SID geometry is slightly less balanced than Iter18 (full_gini 0.0681 vs 0.0642, collision 7.16% vs 6.65%, unique 22827 vs 22951). H(L1\|L0) 5.4343 vs 5.5907 is a small drop in the expected direction. Proxy hypothesis is partial, not satisfied, but not failed. |
| PIPELINE_FAIL | NO | Stage3 ran 150 epochs on 4-GPU DDP; `test_final.json` was produced and contains `n_eval=57439`; Stage2 SID export wrote the 4-token extension npy + JSON and the `SID_WIRING_PASS` was printed. |
| TRUE_MECHANISM_FAIL | YES | All four upstream conditions PASS; downstream `test_R@10 = 0.057017` is below every completed baseline (Iter11/18/25) except Iter24 and is far below the 0.065 adoption threshold. |

## Counterfactual

The bridge hypothesis was that the Iter25 regression was caused by the wrong
residual input vector. Iter26 corrected exactly that vector while holding all
other Iter16 components fixed, and the hard contract that `c_l` must remain
identical for the entire 100,000-step run was enforced and met. The corrected
input did not improve downstream recall; it regressed further than Iter25. The
counterfactual rollback test (turn the corrected input back to the Iter1
calibration and re-run Stage2→Stage3) is the next experiment if Iter27 chooses
to revisit the family, but the bridge recommends retiring the family instead.

## Ledger decision

No new entry is appended to `references/failed_mechanism_ledger.md`. The ledger
rule allows entries only when Agent D records `MECHANISM_FAIL` for the direct
effects, which is not the case here (direct effects all PASS). Iter26 is logged
as a continuation of the class-3 ceiling pattern (now 9 instances across
iter7-iter13 + iter25 + iter26) in `gate_decision_iter26.md` and
`test_final_iter26.json` for next-iteration reference.

The `mechanism_pool.md` should mark the closed-form fixed-curvature family as
retired per the skill's (c) stopping condition. The three one-factor
perturbations attempted (Iter16 itself, Iter25's residual input + learnable
scale, Iter26's corrected residual input + fixed scale) have all stalled under
the same Stage3 trainer.

## Recommended next steps

- (a) Stop adding per-layer curvature family mechanisms in Stage2: this
  family now has nine class-3 outcomes (iter7-13 + iter25 + iter26) and
  the bridge-conditioned (c) stopping condition is satisfied.
- (b) Move the next novelty outside Stage2 RQ-VAE mechanics (Stage1 embedding
  model, Stage3 trainer/protocol, or both).
- (c) If a Stage2 mechanism is retried despite (a), it must propose a
  structural change to the binding assumption behind the prior class-3
  ceiling (e.g., a different input tokenization or a different loss family),
  not another reweighting of the existing mechanism.