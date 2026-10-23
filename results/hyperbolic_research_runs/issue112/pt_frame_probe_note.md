# The path-B candidate: refuted by a frame-invariance argument (and the earlier probe's own check was wrong)

## 1. The proposal as written is not well posed

`hyp_transported_residual` transports the RESIDUAL into the level-0 codeword's tangent frame and then
quantises it with the **shared** level-1 codebook. The transported frame depends on the item, so the
shared codebook has no fixed coordinates in that frame. The only well-posed variant keeps the codebook
canonical and transports it INTO the item's frame — the dual.

## 2. The dual is inert, and that is a theorem, not a measurement

Transporting from the origin into the codeword's frame is a **common change of coordinates**: the
residual and every codebook entry are moved by the SAME isometry. The level-1 assignment (an argmin
over distances) and the quantisation error (a norm) are both invariant under a common isometry.

Measured on 4000 frozen items (the matched control checkpoint, epoch 3000), with the verifier's
regression-tested transport (`tools/hyperbolic_verifier/transport.py`, isometry deviation 2.2e-16):

| quantity | value |
|:---|---:|
| level-1 assignments changed by the frame change | **0.0000** |
| mean level-1 residual norm, ambient frame | 0.273690 |
| mean level-1 residual norm, transported frame | 0.273690 (identical) |

⇒ **There is nothing to gain.** The path-B family is closed: the proposal's own form is ill posed, and
its only well-posed form is an identity.

## 3. The earlier probe's "failed isometry check" was a bug in the probe, not evidence

The first probe reported an isometry deviation of 0.0679 and the note read it as "the transport is not
yet the verified one". That was wrong twice over: the probe compared the transported norm at the
TARGET with the original's norm at the ORIGIN (the transport was applied "from x"), and — more
importantly — **the isometry property cannot detect that misuse at all**, because the transport is a
scaled isometry and preserves the Riemannian norm of ANY input. The verifier's regression test now
states this explicitly (`test_transporting_a_vector_from_the_wrong_point_gives_a_different_vector`):
the misuse changes the VECTOR, not the norm.

The corrected probe uses the verifier's transport and the correct frame change (from the origin), and
its conclusion is the one above.

## 4. Where the campaign stands

* the horosphere family: refuted twice with verified measurements (a near-radial operator, and a
  level-matching constraint that fights the host's assignment);
* the path-B transported residual: refuted by the frame-invariance argument, measured with the
  verifier's transport;
* the remaining approved candidate is the **product-geodesic level-0 quantiser** (`hyp_product_l1_
  quantiser` v2, frozen with the measured tangent scale) — a different family, and the next thing to
  pre-check on the frozen data before any implementation.
