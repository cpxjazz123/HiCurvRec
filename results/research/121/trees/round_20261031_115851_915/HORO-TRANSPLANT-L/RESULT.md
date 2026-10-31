# HORO-TRANSPLANT-L -- the round-six result

`test_recall@10 = 0.054841` (150 epochs, frozen protocol, checkpoint
`.../TIGER_HORO-TRANSPLANT-L/ckpt/Amazon_2023_Instruments/`).

## What was built

The deepest token only; levels 1..L-1 bit-identical (asserted). A move had to pass
three gates at once: the split/merge certificate (both counts non-increasing, one
strict), a non-negative Busemann (horosphere) height gain `B_{p,b}(x_i) - h_{p,b}`
against the frozen child anchor, and a non-positive change of the squared
hyperbolic distance to that child's Frechet mean. The three gates together admit
**12 items** (0.05% of the catalogue) and leave the L3 prefix set unchanged at
20,223.

## The result, and the finding it forces

| round | mechanism | moved | test_recall@10 |
| --- | --- | --- | --- |
| one | curved chain on the frozen text geometry | all | 0.05439 |
| two | geodesic step to the successor barycentre | all | 0.05165 |
| three | fourth PPMI-driven level | all | 0.04823 |
| four | capacity-frozen behaviour-funded relabelling | 13,845 | 0.05106 |
| five | certificate-gated deepest-code repair | 8,338 | 0.05495 |
| six | horosphere-gated deepest-code transplant | **12** | **0.05484** |
| gate | -- | -- | > 0.065 |
| frozen baseline | plain three-level text-quantised RQ-VAE | 0 | about 0.0679 |

Round six moved twelve items out of 24,587 and scored 0.0548, while the untampered
baseline scored about 0.0679 under the same frozen protocol on the same catalogue.
Twelve items cannot cause a 0.013 change in recall@10. The only reading consistent
with both numbers is that **the single-run Stage-3 measurement carries a run-to-run
spread of about 0.013** (about 20% of the value), so:

* the differences among rounds one to six (0.0482 - 0.0549) are **inside that
  spread** and are not separable from noise by a single run each;
* the apparent 0.013 gap between "the mechanisms" and "the baseline" is the same
  order as the spread of a near-identical SID, so it is not established that any
  mechanism hurt;
* under the frozen policy (one run per candidate, no multi-seed) **no candidate can
  be adjudicated to better or worse than the baseline within this spread**, and
  the 0.065 threshold itself is inside it.

This is a measurement limitation of the protocol, not a mechanism result, and it is
the single most important thing this round produced.
