# What is done, what is still open (handoff, not an admission claim)

Done and tested locally (unit level only):
- squared-distance stress `(d^2/(s^2 h^2) - 1)^2`, smooth at a collision;
- `phi(t) = asinh(sqrt(t))^2` as the distance, with `phi(0)=0`, `phi'(0)=1`,
  series `t - t^2/3 + 8t^3/45` for `t < 1e-3`, computed in float64; opposite-point
  identity `d_H = 4 atanh(r)` pinned by a regression;
- Mobius primitives (`mobius_add`, `mobius_neg`, `log0` inverse of the radial map),
  `residual_step` (E subtract, H Mobius left inverse) and `nested_compose`
  (`c1 (+) (c2 (+) c3)` folded from the last level);
- calibration: block gradient budget `min_b ||g_base||/||g_struct||` over encoder
  and all codebooks, on a real reconstruction batch, all ranks run the same
  forward and only two scalars are broadcast;
- degenerate calibration raises `CalibrationDegenerate` (status, null lambda) and
  the trainer exits before the loop, so no `optimizer.step` and no export;
- collision probe reports coverage honestly (item-pair based, NOT_COVERED
  recorded) and separates reconstruction / quantisation / structure gradients.

Still open, in order:
1. wire `residual_step` / `nested_compose` / `log0` into `RQLayer.forward` and the
   decoder readback behind a switch, with an equivalence test against the current
   tangent path on small normals and a divergence test at large normals;
2. bind `reference_contract.json` to the production primitives and cover full
   32-d inputs, non-collinear H directions, real `h = 1` and `2`, positive scale,
   diagonal and near-diagonal, with double-precision value/gradient comparison;
3. DDP negative test for the "one rank empty, one rank non-empty" structure
   sample (collective must still be entered), sampling centres over the full
   catalogue with acceptance denominators, per-level `use_sk` report;
4. move logging/paths to the framework's `_runs/<run_id>/<arm>/result` contract and
   add a smoke regression that checks entered/completed epochs and optimizer steps
   (calibration has zero steps) instead of a timeout;
5. four independent suites, one-epoch real entry, verification id, formal run,
   Stage 3, R2 archive.

Evidence: the unadmitted 1074-epoch attempt is archived byte-for-byte under
`results/research/issue117/runs/run_9f65a8558d314391/evidence/unadmitted_1074epoch/`
with full SHA-256 in its manifest; the copy under `results/research/issue144/evidence/`
carries a misfiled note and an index of digests.
