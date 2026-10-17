# HBPE-R3-07 (#43) — Text-semantic neighborhood preservation

## Stage 2 Results and Conditional Stage 3 Gate

- **Arm**: `hbpe_r3_07`
- **SID SHA256**: `d51de92d70882fbb46203db3534c49292cbeb6f1bfb7b8e53fdaf210938015fb`
- **L1 distinct codes**: 172 (effective `exp(H)`: 159.00)
- **L1+L2 distinct prefixes**: 13395
- **Full 3-token collision**: 0.0612

### Held-Out Behavioral Prefix Enrichment

| Level | pos_share | neg_share | gap | **ER** | 95% CI | Baseline ER | Δ vs Resample (Paired CI) |
|:---:|---:|---:|---:|---:|:---:|---:|:---:|
| L1 | 0.09677 | 0.00646 | +0.09031 | **14.9780** | [14.41, 15.61] | 16.3895 | -1.4330 [-2.24, -0.65] |
| L1+L2 | 0.00674 | 0.00011 | +0.00663 | **62.4561** | [48.47, 85.24] | 58.3871 | +3.2387 [-27.88, +29.66] |

### Stage 3 Gate Verdict

- **Verdict**: `stage3_allowed = false`
- **Reason**: Stage 2 Hard L1 ER=14.978038 (baseline=16.389539), Hard L1+L2 ER=62.456140 (baseline=58.387097). Failed dual threshold gate -> Stage 3 refused.

**Gate failed**: At least one level did not strictly exceed the canonical Resample baseline. Stage 3 will NOT be started or queued. Closing issue as completed based on Stage 2 evidence.
