# HBPE-R3-01 (#37) — Hard-ST prefix ranking

## Stage 2 Results and Conditional Stage 3 Gate

- **Arm**: `hbpe_r3_01`
- **SID SHA256**: `8fdb52e0095cd189d3f25e08ea36c664b31a635516419b749ed1cea26ac5b4c9`
- **L1 distinct codes**: 184 (effective `exp(H)`: 171.12)
- **L1+L2 distinct prefixes**: 13996
- **Full 3-token collision**: 0.0587

### Held-Out Behavioral Prefix Enrichment

| Level | pos_share | neg_share | gap | **ER** | 95% CI | Baseline ER | Δ vs Resample (Paired CI) |
|:---:|---:|---:|---:|---:|:---:|---:|:---:|
| L1 | 0.09242 | 0.00588 | +0.08654 | **15.7124** | [15.02, 16.43] | 16.3895 | -0.7045 [-1.56, +0.12] |
| L1+L2 | 0.00605 | 0.00009 | +0.00596 | **64.0000** | [48.71, 90.28] | 58.3871 | +4.9795 [-27.27, +33.86] |

### Stage 3 Gate Verdict

- **Verdict**: `stage3_allowed = false`
- **Reason**: Stage 2 Hard L1 ER=15.712448 (baseline=16.389539), Hard L1+L2 ER=64.000000 (baseline=58.387097). Failed dual threshold gate -> Stage 3 refused.

**Gate failed**: At least one level did not strictly exceed the canonical Resample baseline. Stage 3 will NOT be started or queued. Closing issue as completed based on Stage 2 evidence.
