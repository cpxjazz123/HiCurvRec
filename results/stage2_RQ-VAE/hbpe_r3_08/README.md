# HBPE-R3-08 (#44) — L1-conditional L2 enrichment

## Stage 2 Results and Conditional Stage 3 Gate

- **Arm**: `hbpe_r3_08`
- **SID SHA256**: `fdb99ba522a9821516bc21298335d342e54ace2aa57839a97809c405162fbc67`
- **L1 distinct codes**: 175 (effective `exp(H)`: 161.71)
- **L1+L2 distinct prefixes**: 13544
- **Full 3-token collision**: 0.0609

### Held-Out Behavioral Prefix Enrichment

| Level | pos_share | neg_share | gap | **ER** | 95% CI | Baseline ER | Δ vs Resample (Paired CI) |
|:---:|---:|---:|---:|---:|:---:|---:|:---:|
| L1 | 0.09615 | 0.00628 | +0.08987 | **15.3072** | [14.66, 16.08] | 16.3895 | -1.0901 [-1.90, -0.27] |
| L1+L2 | 0.00749 | 0.00011 | +0.00738 | **67.1186** | [51.94, 89.78] | 58.3871 | +7.7447 [-20.80, +32.09] |

### Stage 3 Gate Verdict

- **Verdict**: `stage3_allowed = false`
- **Reason**: Stage 2 Hard L1 ER=15.307229 (baseline=16.389539), Hard L1+L2 ER=67.118644 (baseline=58.387097). Failed dual threshold gate -> Stage 3 refused.

**Gate failed**: At least one level did not strictly exceed the canonical Resample baseline. Stage 3 will NOT be started or queued. Closing issue as completed based on Stage 2 evidence.
