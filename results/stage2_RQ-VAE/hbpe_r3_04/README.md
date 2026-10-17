# HBPE-R3-04 (#40) — Target-popularity matched negative sampling

## Stage 2 Results and Conditional Stage 3 Gate

- **Arm**: `hbpe_r3_04`
- **SID SHA256**: `7ebeed66a235bea5a312cee653614e6bc231bf66ab9646f354e8052a85c3fd93`
- **L1 distinct codes**: 173 (effective `exp(H)`: 158.79)
- **L1+L2 distinct prefixes**: 13481
- **Full 3-token collision**: 0.0627

### Held-Out Behavioral Prefix Enrichment

| Level | pos_share | neg_share | gap | **ER** | 95% CI | Baseline ER | Δ vs Resample (Paired CI) |
|:---:|---:|---:|---:|---:|:---:|---:|:---:|
| L1 | 0.09660 | 0.00608 | +0.09052 | **15.8867** | [15.24, 16.61] | 16.3895 | -0.5149 [-1.33, +0.28] |
| L1+L2 | 0.00770 | 0.00012 | +0.00758 | **62.6154** | [49.76, 81.20] | 58.3871 | +3.2439 [-28.49, +27.94] |

### Stage 3 Gate Verdict

- **Verdict**: `stage3_allowed = false`
- **Reason**: Stage 2 Hard L1 ER=15.886745 (baseline=16.389539), Hard L1+L2 ER=62.615385 (baseline=58.387097). Failed dual threshold gate -> Stage 3 refused.

**Gate failed**: At least one level did not strictly exceed the canonical Resample baseline. Stage 3 will NOT be started or queued. Closing issue as completed based on Stage 2 evidence.
