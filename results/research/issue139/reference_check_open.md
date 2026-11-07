# Reference agreement: resolved, and the two defects it found

The check on `stage2_RQ-VAE/TIGER_mixed_rq` returns **PASS**: values agree to
1.4e-17, gradients to 1.1e-16, `torch.autograd.gradcheck` passes on both sides,
and the coupling check reports that the weight moves the computation by 0.115.

It failed first, and the failure was real. Two defects, both mine:

1. **The scalar cost reduced over the batch.** `point_cost` summed the squared
   distance over the coordinates *and* the batch, so it returned one number for
   sixteen sampled rows while the reference returns one per row. It also squared
   `squared_distance`, which is already the squared geodesic distance, so the
   curved half was a fourth power. The symptom looked like a formula error in
   both halves at once: the Euclidean half was ten times too large and the
   curved half twenty-seven times too small, in opposite directions.
2. **The Möbius inner product lost its sign.** The frozen statement wrote
   `xy = h0*y0 + h1*y1` where the canonical addition needs the inner product of
   the *negated* first argument, `xy = -(h0*y0 + h1*y1)`. The repository's own
   example contract writes it with the minus sign; this one did not.

The first defect was found by comparing the two halves separately: a formula
error in one term cannot push them opposite ways, so the binding had to be
wrong. The second was found by evaluating the statement by hand on the sampled
row and noticing that the hand value agreed with the reference and not with the
candidate.

What did not change: the mechanism's own verification never depended on this
file. `tests/test_mixed_rq.py` compares the cost against an arc-cosh closed form
written independently of the implementation, and the selector's increment against
a full recomputation. The admission gate did depend on it, and it is the reason
no arm was dispatched while it failed.
