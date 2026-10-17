# HBPE-R3-05 (#41) — Same-prefix semi-hard negatives

## Stage 2 Results and Conditional Stage 3 Gate

- **Arm**: `hbpe_r3_05`
- **SID SHA256**: `72db701da59ffdebf0dabaac252815fc484ede08de6cf0b68aafe0a7a7513e81`
- **L1 distinct codes**: 133 (effective `exp(H)`: 122.65)
- **L1+L2 distinct prefixes**: 12473
- **Full 3-token collision**: 0.0630

### Held-Out Behavioral Prefix Enrichment

| Level | pos_share | neg_share | gap | **ER** | 95% CI | Baseline ER | Δ vs Resample (Paired CI) |
|:---:|---:|---:|---:|---:|:---:|---:|:---:|
| L1 | 0.11503 | 0.00863 | +0.10640 | **13.3333** | [12.90, 13.83] | 16.3895 | -3.0802 [-3.79, -2.30] |
| L1+L2 | 0.00791 | 0.00015 | +0.00776 | **52.2500** | [42.05, 68.06] | 58.3871 | -7.3947 [-38.12, +15.21] |

### Stage 3 Gate Verdict

- **Verdict**: `stage3_allowed = false`
- **Reason**: Stage 2 Hard L1 ER=13.333333 (baseline=16.389539), Hard L1+L2 ER=52.250000 (baseline=58.387097). Failed dual threshold gate -> Stage 3 refused.

**Gate failed**: At least one level did not strictly exceed the canonical Resample baseline. Stage 3 will NOT be started or queued. Closing issue as completed based on Stage 2 evidence.
