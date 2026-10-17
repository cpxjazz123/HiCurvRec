# HBPE-R3-06 (#42) — Encoder gradient budget

## Stage 2 Results and Conditional Stage 3 Gate

- **Arm**: `hbpe_r3_06`
- **SID SHA256**: `5c3f247ac03dd12ac7d46d484c44f9c4ae1d619a5c5c3ffd829eed85a1fdf34e`
- **L1 distinct codes**: 241 (effective `exp(H)`: 222.92)
- **L1+L2 distinct prefixes**: 15780
- **Full 3-token collision**: 0.0377

### Held-Out Behavioral Prefix Enrichment

| Level | pos_share | neg_share | gap | **ER** | 95% CI | Baseline ER | Δ vs Resample (Paired CI) |
|:---:|---:|---:|---:|---:|:---:|---:|:---:|
| L1 | 0.07948 | 0.00484 | +0.07464 | **16.4294** | [15.62, 17.27] | 16.3895 | +0.0316 [-0.73, +0.77] |
| L1+L2 | 0.00435 | 0.00006 | +0.00429 | **67.6471** | [48.26, 102.38] | 58.3871 | +9.4890 [-24.48, +44.66] |

### Stage 3 Gate Verdict

- **Verdict**: `stage3_allowed = true`
- **Reason**: Stage 2 Hard L1 ER=16.429409 (baseline=16.389539), Hard L1+L2 ER=67.647059 (baseline=58.387097). BOTH exceed baseline strictly -> Stage 3 allowed.

**Gate passed**: Both levels strictly exceeded the baseline. Stage 3 may proceed.
