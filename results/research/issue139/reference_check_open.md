# Reference agreement: open defect, recorded as FAIL

`tools/research/stage2/checks.py::check_reference_agreement` on
`stage2_RQ-VAE/TIGER_mixed_rq` returns **FAIL**. It is not a pass and nothing in
this issue treats it as one.

Measured on the contract's own sampling (seed 17, radius 0.2, c = 1):

| weight | `model.mixed_rq.point_cost` | statement-derived reference |
|---|---|---|
| alpha = 1 (Euclidean half only) | 0.648652 | 0.063389 |
| alpha = 0 (curved half only) | 0.000836 | 0.022784 |
| alpha = 0.5 | 0.324744 | 0.043086 |

The two halves therefore disagree in *both* directions, which rules out a
formula error in one term and points at the binding: the reference maps the
declared `symbols` list positionally onto the `args`, and the contract's symbol
order and its statements' first-use order do not coincide. The arms' own
verification does not depend on this file -- the mechanism is checked by
`tests/test_mixed_rq.py` against an arc-cosh closed form that is
implementation-independent -- but **the admission gate does**, so no arm may be
dispatched until this is reconciled and the check returns OK. The reconciliation
is a small, isolated change to the statements' variable introduction order; it
is listed as the next action rather than patched blind at the end of a session.
