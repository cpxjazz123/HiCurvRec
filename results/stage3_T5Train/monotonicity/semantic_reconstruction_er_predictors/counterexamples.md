# Counterexamples & Case Forensics: ER vs Semantic Reconstruction Trade-offs

This document accompanies Issue #50 and analyzes empirical cases across the 38 frozen models (32 with verified checkpoints and decoder reconstruction, all 38 with verified prefix neighborhood structure).

---

## 1. Quadrant Analysis (Median Split on 32 Reconstruction-Available Models)

Using empirical medians on the 32 reconstruction-available models:
- **Median L1 Hard ER**: $10.0636$
- **Median $R_{\text{cos}}$**: $0.8771$

| Quadrant | N | Representative Models | ER Range | $R_{\text{cos}}$ Range | Recall@10 Range |
|:---|:---:|:---|:---:|:---:|:---:|
| **High ER, High Recon** | 10 | `resample` (4 seeds), `fork_radial_corr`, `fork_more_sib`, `fix_det_euclid_raw` | 10.37 – 16.43 | 0.8842 – 0.8874 | 0.0573 – 0.0636 |
| **High ER, Low Recon** | 6 | `smoothness_arms/A_euclid_smooth_s42`, `curvature_RQ-VAE/euclid_l2_256_72k_s43`, `smooth_fair_arms/A_euclid_smoothfair_s42` | 10.09 – 12.61 | 0.1311 – 0.8739 | 0.0584 – 0.0608 |
| **Low ER, High Recon** | 6 | `TIGER_RQ-VAE`, `prefix_sib`, `nce_euclid_radius_control`, `nce_euclid_group`, `fix_det_euclid_ball` | 3.51 – 8.70 | 0.8778 – 0.9298 | 0.0537 – 0.0616 |
| **Low ER, Low Recon** | 10 | `curvature_RQ-VAE`, `smoothness_arms/B_poincare_smooth_s42`, `smooth_fair_arms/B_poincare_smoothfair_s42`, `nce_hyp_group` | 7.17 – 10.04 | 0.1876 – 0.8767 | 0.0577 – 0.0595 |

---

## 2. Salient Counterexamples

### A. The Pure-Semantic Failure: `TIGER_RQ-VAE`
- **Metrics**: L1 Hard ER = $6.47$ (32nd / 38), $R_{\text{cos}} = 0.9298$ (**#1 highest across all 32 models**), $N_{\text{sem}} = 0.0439$ (38th / 38), Stage 3 Recall@10 = $0.053674$ (**36th / 38**).
- **Forensic**: `TIGER_RQ-VAE` trains a standard autoencoder without behavioral alignment, achieving near-perfect reconstruction of Sentence-T5 text embeddings ($R_{\text{cos}} = 0.93$). However, its codebook collapses into severe behavioral fragmentation (low ER). Despite having the cleanest semantic reconstruction in the entire dataset, its downstream recommendation performance is near the absolute bottom. High semantic fidelity alone is utterly insufficient for Stage 3 recommendation.

### B. High Recommendation with Low Semantic Reconstruction: `curvature_RQ-VAE` Family
- **Models**:
  - `smoothness_arms/A_euclid_smooth_s42`: ER = $10.22$, $R_{\text{cos}} = 0.1311$, Recall@10 = $0.060499$ (Rank 7 / 38).
  - `curvature_RQ-VAE/euclid_l2_256_72k_s43`: ER = $10.09$, $R_{\text{cos}} = 0.1506$, Recall@10 = $0.060429$ (Rank 8 / 38).
  - `smooth_fair_arms/A_euclid_smoothfair_s42`: ER = $10.37$, $R_{\text{cos}} = 0.2310$, Recall@10 = $0.059541$ (Rank 11 / 38).
- **Forensic**: Curvature models incorporate Poincaré ball constraints, working radius pinning, and behavioral context channels. In terms of unconstrained Euclidean text reconstruction, their decoders emit low cosine similarity ($R_{\text{cos}} \approx 0.13 - 0.27$). Yet downstream Stage 3 recommendation is strong ($0.0595 - 0.0605$, exceeding `TIGER_RQ-VAE` by $+12.7\%$). Semantic reconstruction in original text space is uncoupled from recommendation efficacy.

### C. The Low-ER High-Recall Anomaly: `TIGER_FORK_RQ-VAE/prefix_sib`
- **Metrics**: L1 Hard ER = $3.51$ (38th / 38), $R_{\text{cos}} = 0.8778$, $N_{\text{sem}} = 0.0714$ (2nd / 38), Recall@10 = $0.061613$ (Rank 5 / 38).
- **Forensic**: As established in Issue #47 and #48, `prefix_sib` explicitly optimizes sibling prefix sharing in Stage 2, destroying L1 pairwise discrimination (hence ER drops to 3.51) while inflating prefix retention. Its high Recall@10 is driven by code-space prefix clustering rather than standard behavioral ER. However, its high $R_{\text{cos}}$ ($0.878$) is typical of Euclidean TIGER models and does not distinguish it from worse TIGER models like `nce_hyp_group` (Recall@10 = $0.058828$, $R_{\text{cos}} = 0.8767$).

---

## 3. Why the "Both High" Interaction Fails Out-of-Family

1. **Family Confounding**:
   - `TIGER_BEHAVIOR_SAMPLE_RQ-VAE` arms have both high ER ($15.6 - 16.4$) and high $R_{\text{cos}}$ ($0.887$).
   - `curvature_RQ-VAE` arms have intermediate ER ($9.3 - 10.4$) and low $R_{\text{cos}}$ ($0.13 - 0.34$).
   - `TIGER_RQ-VAE` has low ER ($6.5$) and very high $R_{\text{cos}}$ ($0.93$).
2. **Out-of-Family Overfitting**:
   - In LOFO validation, when training folds see that the best TIGER models have high $R_{\text{cos}}$ while low-performing TIGER models have lower $R_{\text{cos}}$, the model fits a positive weight on $R_{\text{cos}}$ and the interaction `log_ER * R_cos`.
   - When this trained model is evaluated on the held-out `curvature_RQ-VAE` family, the low $R_{\text{cos}}$ of curvature models ($0.13 - 0.34$) causes the model to severely underpredict their Recall@10 (predicting $\approx 0.052 - 0.054$ when actual is $\approx 0.059 - 0.060$).
   - As a result, M8 increases MAE by $+2.8\%$ ($\Delta\text{MAE} = -0.000122$), and M10 increases MAE by $+12.3\%$ ($\Delta\text{MAE} = -0.000543$).
3. **Scientific Conclusion**:
   The "both high" hypothesis (that semantic reconstruction complements behavioral ER to predict Stage 3 recommendation) is an artifact of within-family co-occurrence, and fails the test of out-of-family generalization. L1 Hard ER remains the sole robust predictor.
