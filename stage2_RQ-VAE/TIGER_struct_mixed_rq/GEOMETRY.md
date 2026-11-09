# The exact formulas this candidate trains on

Written down because "sum three codes" is not a specification.

## Representation
A latent `z` is 32 numbers. `z_E = z[0:16]` is Euclidean. `z_H = z[16:32]` is a
**tangent vector at the origin** of the Poincare ball of curvature 1.

Ball point: `b(h) = h * tanh(||h||) / ||h|| * (1 - 1e-6)`. The factor keeps every
point strictly inside the ball, so no point can reach the boundary where the
geodesic distance is infinite.

Squared product distance:
`d_M(x, y)^2 = ||x_E - y_E||^2 + d_H(b(x_H), b(y_H))^2 / 4`
with `d_H(a, c) = arcosh(1 + 2||a - c||^2 / ((1-||a||^2)(1-||c||^2)))`, the
argument clamped at `1 + 1e-6` and the two denominators clamped at `1e-3` so a
near-boundary point cannot produce an infinite value or a NaN gradient.

## Residual recursion, code choice and reconstruction
Levels `l = 1, 2, 3`, 256 codewords each. With `r_1 = z`:

1. cost to every codeword `k` at level `l`: the same `d_M^2(r_l, e_{l,k})` (the
   geometry switch replaces it with the Euclidean distance over all 32 numbers);
2. assignment: the baseline Sinkhorn normalisation of the centred cost matrix,
   then `argmax` of the transport plan (training and inference use the same
   function; the switch only changes the cost);
3. code vector `c_l = e_{l,k*}`;
4. residual: `r_{l+1} = r_l - c_l`, componentwise in the tangent/ambient
   coordinates of the product representation (E and H halves both subtract
   directly; the H half stays a tangent vector until a distance is computed);
5. readout position: `q = c_1 + c_2 + c_3` (the cumulative tangent vector);
   `d_M` is evaluated on `q`, mapping the H half with `b(.)` as above;
6. reconstruction: the decoder reads `q`; the straight-through path keeps
   `x + (q - x).detach()` so the encoder gets gradients;
7. export: the same cost function, the same assignment, and the same three
   tokens; `q` is only used by the loss, never by the token export.

## Structure objective (squared stress)
For a triple (centre, partner A, partner B): pairs `(0,1)`, `(0,2)`, `(1,2)`,
targets `h = 1, 1, 1` on a closed triangle and `h = 1, 1, 2` on an open wedge.

`L_structure = mean over pairs of ( d_M(q_a, q_b)^2 / (s^2 h^2) - 1 )^2`

At a collision (`q_a = q_b`) this is exactly 1 and its gradient is exactly 0 —
the loss is smooth there, and it does **not** push a shared codeword apart by
itself. The encoder still receives gradient through the straight-through path and
the codebook through the gather term; the reconstruction objectives are unchanged.

## Calibration (TRAIN only, once, frozen)
- `s`: after codebook initialisation, sample `4096` structures with a fixed seed,
  take the **median** of `sqrt(d_M^2)` over the pairs with `h = 1`. If the sample
  is empty or the median is not finite / `<= 1e-6`, set `s = 1.0` and record
  `degenerate_scale = true` (never silently).
- `lambda`: same sample; measure `||grad_theta L_base||` on one real
  reconstruction batch and `||grad_theta L_structure||`; solve
  `lambda = 0.1 * ||grad_base|| / ||grad_structure||`. If the structure norm is
  `<= 1e-12`, set `lambda = 0` and record `degenerate_gradient = true`.
- DDP: the sample seed is rank-independent, rank 0 computes both scalars and
  broadcasts them, so all four ranks train with one `s` and one `lambda`.

## Switches
- `BEHAVIOUR_ON = False`: the graph is not built, the structure term is not
  computed and adds nothing; only reconstruction and quantisation losses remain.
- `GEOMETRY_ON = False`: same 32 numbers, same codebooks, same capacity, but the
  assignment cost and the structure distances are the Euclidean distance over all
  32 coordinates (the H half is still used, just measured flat); this is the
  control arm, not a different architecture.

## Assignment branches (baseline protocol, unchanged)

`VQLayer._indices` chooses: Sinkhorn normalisation then `argmax` when
`(self.training and self.use_sk) or (self.use_sk and infer_use_sk)`, otherwise
`argmin` of the cost matrix.

Per level, from the frozen baseline source (`RQVAE(T._tokenizer_config())`):
levels 1 and 2 have `use_sk = False` (Sinkhorn is configured on the last level
only), so they always take `argmin`; level 3 has `use_sk = True`, takes Sinkhorn
in training and `argmin` at export where `infer_use_sk=False` and the module is in
eval mode.  `tests/test_assignment_branches.py` pins this pattern, because an
earlier note here claimed "every level" and the source contradicted it.

Consequences recorded here rather than papered over:

- the structure term reads the tokens produced by the training branch (Sinkhorn
  + argmax); the exported SID can differ from those tokens;
- sharing `_distances` proves the two branches use one **cost function**, not that
  they select the same codes.  The trainer records the export-vs-training token
  mismatch rate as a diagnostic, and nothing in this candidate removes Sinkhorn or
  widens it to more levels.
