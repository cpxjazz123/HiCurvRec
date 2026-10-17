# HBPE-R3-02 (#38) — Top-2 sparse boundary assignment

## Stage 2 Results and Conditional Stage 3 Gate

- **Arm**: `hbpe_r3_02`
- **SID SHA256**: `a0dcc887f017a8d5847fdfd41b0c3d5eb8831d82957fd0fcffac4d5f4c0c20f7`
- **L1 distinct codes**: 180 (effective `exp(H)`: 165.61)
- **L1+L2 distinct prefixes**: 13684
- **Full 3-token collision**: 0.0626

### Held-Out Behavioral Prefix Enrichment

| Level | pos_share | neg_share | gap | **ER** | 95% CI | Baseline ER | Δ vs Resample (Paired CI) |
|:---:|---:|---:|---:|---:|:---:|---:|:---:|
| L1 | 0.09428 | 0.00607 | +0.08821 | **15.5379** | [14.90, 16.23] | 16.3895 | -0.8725 [-1.67, -0.05] |
| L1+L2 | 0.00711 | 0.00014 | +0.00697 | **50.8108** | [39.89, 66.14] | 58.3871 | -8.9853 [-39.70, +13.72] |

### Stage 3 Gate Verdict

- **Verdict**: `stage3_allowed = false`
- **Reason**: Stage 2 Hard L1 ER=15.537886 (baseline=16.389539), Hard L1+L2 ER=50.810811 (baseline=58.387097). Failed dual threshold gate -> Stage 3 refused.

**Gate failed**: At least one level did not strictly exceed the canonical Resample baseline. Stage 3 will NOT be started or queued. Closing issue as completed based on Stage 2 evidence.
