## Round 2 (negative): the gradient balance is destructive

Calibrated the local curvature weight to **10% of the active global gradient** at `mu = 0.1`
(batch: local 6.295e-5, global 0.184435; `lambda` 5.884e-6 → 0.001724, 293x). The 1-epoch smoke
passed (`local/global = 0.10` by construction), but the formal run diverged:

- epoch 550: L3 collision **0.9832**, global J **1.598** (bound mean ≤ 0.32), `mu` 1.054,
  reconstruction 2.01e-4 (bound ≤ 8.59e-5).
- The run was stopped at epoch 556 as a **negative diagnostic**; it is not a full 3000-epoch candidate
  and no PASS is claimed.

Root cause: at that weight the local A.1 gradient through the SID assignment destroys the RQ-VAE
assignment (collision 0.12 → 0.98) and drives J above its start; the trade-off the round was designed to
measure does not exist at this intervention point.

The observed family now spans `lambda` = 0 (TIGER: L2 7,374, rank 11.5, MAE 7.31), 5.9e-6 (round 1:
L2 6,848, rank 2.77, MAE 2.34), 1.7e-3 (collapse). L2 ≥ 10,918 with RQ rank ≥ 8 has never been observed
together; its only L2 record is the excluded Euclidean #265 objective.

**Issue #272 remains OPEN.** The fixed-objective L2 ≥ 10,918 + rank ≥ 8 conjunction has no executable
`kappa_t`/`lambda` setting; the real goal needs an objective/capacity intervention, which the Issue text
and the skill's transfer rule retain with the user.
