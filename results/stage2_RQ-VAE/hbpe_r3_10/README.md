# HBPE-R3-10 (#46) — Degree-corrected graph modularity

## Stage 2 Results and Conditional Stage 3 Gate

- **Arm**: `hbpe_r3_10`
- **SID SHA256**: `77a448e29837e010a404925dccb2e9f490050f9171d6b5157559f41b8978d987`
- **L1 distinct codes**: 175 (effective `exp(H)`: 159.37)
- **L1+L2 distinct prefixes**: 13519
- **Full 3-token collision**: 0.0607

### Held-Out Behavioral Prefix Enrichment

| Level | pos_share | neg_share | gap | **ER** | 95% CI | Baseline ER | Δ vs Resample (Paired CI) |
|:---:|---:|---:|---:|---:|:---:|---:|:---:|
| L1 | 0.09861 | 0.00644 | +0.09217 | **15.3069** | [14.66, 15.96] | 16.3895 | -1.1049 [-1.94, -0.30] |
| L1+L2 | 0.00744 | 0.00013 | +0.00730 | **55.3521** | [42.77, 72.08] | 58.3871 | -4.0366 [-32.93, +19.84] |

### Stage 3 Gate Verdict

- **Verdict**: `stage3_allowed = false`
- **Reason**: Stage 2 Hard L1 ER=15.306902 (baseline=16.389539), Hard L1+L2 ER=55.352113 (baseline=58.387097). Failed dual threshold gate -> Stage 3 refused.

**Gate failed**: At least one level did not strictly exceed the canonical Resample baseline. Stage 3 will NOT be started or queued. Closing issue as completed based on Stage 2 evidence.
