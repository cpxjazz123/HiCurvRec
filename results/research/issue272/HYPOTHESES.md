
---

## H272-5: gradient balancing scales the local term without destroying the assignment
- Status: **REFUTED** (2026-11-27). At 10% of the global gradient the collision rose to 0.983 and the
  global J diverged to 1.598 by epoch 556; the run was stopped as a negative diagnostic (not a full
  candidate, no PASS claim).

## H272-6: the assignment capacity collapses above a local-weight threshold
- Status: **SUPPORTED** (2026-11-27). lambda=5.9e-6 left 6,848 L2 prefixes and rank 2.77; lambda=1.7e-3
  collapsed to ~0.98 collision and J divergence. The L2/RQ-rank ceiling within the kappa_t family is the
  lambda=0 TIGER value (7,374 / 11.5).
