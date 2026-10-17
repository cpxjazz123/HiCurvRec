# HBPE-R3-09 (#45) — Within-run L1 assignment trust region

## Stage 2 Results and Conditional Stage 3 Gate

- **Arm**: `hbpe_r3_09`
- **SID SHA256**: `999e69bc2f5f9fbe4166e5a7d1c36f0f182ef78dcc9baa93c2f348fbf93e20f4`
- **L1 distinct codes**: 171 (effective `exp(H)`: 159.87)
- **L1+L2 distinct prefixes**: 13555
- **Full 3-token collision**: 0.0589

### Held-Out Behavioral Prefix Enrichment

| Level | pos_share | neg_share | gap | **ER** | 95% CI | Baseline ER | Δ vs Resample (Paired CI) |
|:---:|---:|---:|---:|---:|:---:|---:|:---:|
| L1 | 0.09520 | 0.00632 | +0.08889 | **15.0704** | [14.46, 15.72] | 16.3895 | -1.3359 [-2.12, -0.58] |
| L1+L2 | 0.00708 | 0.00013 | +0.00695 | **55.8209** | [44.12, 75.41] | 58.3871 | -3.3991 [-36.28, +21.14] |

### Stage 3 Gate Verdict

- **Verdict**: `stage3_allowed = false`
- **Reason**: Stage 2 Hard L1 ER=15.070380 (baseline=16.389539), Hard L1+L2 ER=55.820896 (baseline=58.387097). Failed dual threshold gate -> Stage 3 refused.

**Gate failed**: At least one level did not strictly exceed the canonical Resample baseline. Stage 3 will NOT be started or queued. Closing issue as completed based on Stage 2 evidence.
