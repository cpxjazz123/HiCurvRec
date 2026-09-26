# Iter25 Failure Attribution

## Mechanism executed

Pre-registered bounded branching-plus-raw-residual layer-scale prior
`u_l_prior = [0.990000, 0.126233, 0.058186]` anchoring the existing
learnable layer scales. No new loss, parameter, code path, or
training hyperparameter change. Iter11 cyclic formula, optimizer,
warm-start, dimensions, and Stage3 trainer preserved.

## Five-label attribution

| Label | Applies? | Evidence |
|---|---|---|
| IMPLEMENTATION_FAIL | NO | MVG four-layer PASS; prior, scales, gradients, five-step updates, 200-step ON/OFF all finite and active. |
| MECHANISM_NOT_ACTIVATED | NO | `c_layer_scale` initialized to `[0.99, 0.126, 0.058]`; curvature ordering `c_0 > c_1 > c_2` holds at fixed phase; total loss `requires_grad` and `grad_fn` present; behavior loss gradient non-zero; curvature regularization gradient non-zero after perturbation. |
| PIPELINE_FAIL | NO | Stage2 completed 100k global steps, ckpt and 3-token SID saved at every 10k step, SID export to 4-token extension PASS; Stage3 completed 150 epochs and produced `test_final.json` with `n_eval=57439`. |
| METRIC_MISMATCH | partial | Stage2 descriptive (Gini, per-layer, collision, H(L1|L0)) all move in the predicted direction relative to Iter18 but downstream `test_R@10` regressed by 0.00106 vs Iter18 and 0.00094 vs canonical Iter11. |
| TRUE_MECHANISM_FAIL | YES | All four upstream conditions PASS; downstream recall regressed and remains below the 0.065 adoption threshold. This is the class-3 ceiling pattern predicted by the iteration bridge. |

## Counterfactual

The bridge explicitly framed this iteration as a "prior-mapping
experiment, not a causal isolation"; an explicit ON/OFF counterfactual
Stage3 run with the Iter11 fallback prior is not performed (would
duplicate Iter11's run and Stage3 trainer is read-only). The 200-step
ON/OFF arm in MVG shows the prior change produces a finite total
loss delta and produces the expected layer-scale/curvature
reordering.

## Ledger decision

No new entry is appended to `references/failed_mechanism_ledger.md`.
The ledger rule allows entries only when Agent D records
`MECHANISM_FAIL` for the direct effects, which is not the case
here (direct effects PASS). Iter25 is logged as a class-3 ceiling
outcome in `gate_decision_iter25.md` and `test_final_iter25.json`
for next-iteration reference.

## Recommended next steps

- (a) Stop adding per-layer curvature family mechanisms in
  Stage2: this family now has eight class-3 outcomes (iter7-13
  plus iter25) and the bridge-conditioned (c) stopping condition
  is satisfied.
- (b) Move the next novelty outside Stage2 RQ-VAE mechanics
  (Stage1 embedding model, Stage3 trainer/protocol, or both).
- (c) If a Stage2 mechanism is retried despite (a), it must
  propose a structural change to the binding assumption behind
  the prior class-3 ceiling (e.g., a different input tokenization
  or a different loss family), not another reweighting of the
  existing mechanism.