# TOLLCAP-SID -- round four, implementation state

Selected by `round_20261031_091611_264` with three first places (mean rank 1.0, a
unanimous selection). Shape A of the round: a capacity-frozen, behaviour-funded
relabelling of the FROZEN baseline SID (`results/stage2_RQ-VAE/TIGER_RQ-VAE`), so
the codebooks, the cell set and every cell's size are supposed to stay
bit-identical and only boundary items may change cell.

## Implemented

`relabel.py`:

* the geometry is taken in the QUANTIZER's input space -- the frozen text
  embedding is pushed through the baseline checkpoint's encoder
  (`encoder.mlp.1/4/7/10`, 768-512-256-128-32), because the codebooks live there
  and a 768-dimensional text point cannot be compared with a 32-dimensional
  codeword;
* `phi(x) = tanh(alpha ||x||) x/||x||` with `alpha = 1 / max ||z||` on those
  latents; `m_s = phi(c1_a + c2_b + c3_d)` from the frozen codebooks;
  `d_P` in float64;
* `eps_rec = 0.1754` (the quantizer's own noise floor), `sigma_A = 2.3636`,
  `lambda = eps_rec / sigma_A = 0.0742`;
* `A(i,s) = log((p(s|i) + eps_p) / (pi(s) + eps_p))` from the training-only
  directed transitions, aggregated to cells;
* one `linear_sum_assignment` per level-1 group over the (items x cell slots)
  cost `toll - lambda * (A(i,s) - A(i,t(i)))`, with the cells replicated to their
  baseline sizes, then the 5% cycle cap.

## What does not hold yet

The diagnostic says **1,229 items moved** (the 5% cap) but the prefix counts are
NOT preserved: L1 247 (identical, good), L2 7,414 -> 7,314, L3 20,223 -> 19,412.
The specification requires the cell capacities to stay exactly the baseline's, so
this implementation is not yet the mechanism it claims to be, and a Stage 3 run on
it would measure something else. The likely cause is in how the replicated slots
are mapped back to cells (`unique_keys[slots[column]]`), which allows a cell's
slots to be re-used by a key that is not its own.

**No Stage 2 SID of this mechanism has been produced, Stage 3 has not been run,
and there is no recall value for round four.**

## Context: the three completed rounds

| round | mechanism | prefixes | test_recall@10 |
| --- | --- | --- | --- |
| one | curved chain on the frozen text geometry | 251/1769/2201 | 0.05439 |
| two | geodesic step to the successor barycentre | 230/1251/1298 | 0.05165 |
| three | fourth level from the PPMI successor profile | 208/2766/6235/6302 | 0.04823 |
| gate | -- | -- | **> 0.065** |
| frozen baseline | plain three-level RQ-VAE | 247/7414/20223 | about 0.0679 |
