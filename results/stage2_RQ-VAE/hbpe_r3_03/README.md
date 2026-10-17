# HBPE-R3-03 (#39) — Multi-successor supervised prefix contrast

## Stage 2 Results and Conditional Stage 3 Gate

- **Arm**: `hbpe_r3_03`
- **SID SHA256**: `c3fc9c13cd9ce0ba7e8e6218950abb1654c65d9bc9082833bf2617e66e6eb36d`
- **L1 distinct codes**: 189 (effective `exp(H)`: 175.44)
- **L1+L2 distinct prefixes**: 14486
- **Full 3-token collision**: 0.0484

### Held-Out Behavioral Prefix Enrichment

| Level | pos_share | neg_share | gap | **ER** | 95% CI | Baseline ER | Δ vs Resample (Paired CI) |
|:---:|---:|---:|---:|---:|:---:|---:|:---:|
| L1 | 0.08832 | 0.00586 | +0.08246 | **15.0824** | [14.39, 15.78] | 16.3895 | -1.3351 [-2.17, -0.55] |
| L1+L2 | 0.00473 | 0.00010 | +0.00463 | **49.0196** | [37.77, 68.38] | 58.3871 | -10.2533 [-39.63, +15.59] |

### Stage 3 Gate Verdict

- **Verdict**: `stage3_allowed = false`
- **Reason**: Stage 2 Hard L1 ER=15.082391 (baseline=16.389539), Hard L1+L2 ER=49.019608 (baseline=58.387097). Failed dual threshold gate -> Stage 3 refused.

**Gate failed**: At least one level did not strictly exceed the canonical Resample baseline. Stage 3 will NOT be started or queued. Closing issue as completed based on Stage 2 evidence.
