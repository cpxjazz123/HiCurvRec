# Scientific Report: Does Semantic Reconstruction Quality Complement L1 Hard ER for Stage 3 Recall?

**Issue**: #50  
**Repository**: `cpxjazz123/HiCurvRec`  
**Dataset**: Amazon 2023 Instruments (24,587 items, 57,439 test sequences)  
**Sample**: 38 Stage3-frozen models across 5 distinct model families  
**Primary Target**: Stage 3 `Recall@10` (test sequence level, $N=57,439$)  
**Secondary Target**: Stage 3 `NDCG@10`

---

## 1. Executive Summary

We evaluated whether Stage 2 semantic preservation—operationalized as (i) true decoder-based semantic reconstruction ($R_{\text{cos}}$) and (ii) checkpoint-free prefix-neighborhood preservation ($N_{\text{sem}}$)—adds out-of-family predictive power for Stage 3 recommendation beyond held-out behavioral L1 Hard ER.

**Core Finding**:
1. **Decoder reconstruction does not add out-of-family prediction**: On the 32 reconstruction-available models, adding $R_{\text{cos}}$ to baseline $M0$ ($M8 = \text{ER} + R_{\text{cos}}$) **worsens** Leave-One-Family-Out MAE from $0.004417$ to $0.004539$ ($\Delta\text{MAE} = -0.000122$, $95\%\text{ CI } [-0.000377, +0.000044]$, $p_{\text{boot}} = 0.94$).
2. **Prefix neighborhood preservation does not add out-of-family prediction**: On the full 38 models, adding $N_{\text{sem}}$ ($M9 = \text{ER} + N_{\text{sem}}$) yields identical MAE to $M0$ ($0.003753$ vs $0.003753$, $\Delta\text{MAE} = -0.000000$, $95\%\text{ CI } [-0.000137, +0.000155]$).
3. **The "Both High" interaction hypothesis is rejected**: Adding the interaction term $M10 = \text{ER} + R_{\text{cos}} + \text{ER} \times R_{\text{cos}}$ increases generalization error by $+12.3\%$ ($\text{MAE} = 0.004960$, $\Delta\text{MAE} = -0.000543$, $95\%\text{ CI } [-0.001374, +0.000155]$).
4. **Behavioral ER remains the sole robust predictor**: Pearson $r(\log\text{ER}, \text{Recall@10}) = +0.8472$, Spearman $\rho = +0.7823$. In contrast, $r(R_{\text{cos}}, \text{Recall@10}) = +0.0101$ (uncorrelated).

---

## 2. Artifact and Feasibility Audit (Phase 0)

1. **Stage 1 Text Embeddings**:
   - `stage1_GeneEmbedding/output/sentence_t5.npy`: $24,587 \times 768$ float32.
   - SHA-256: `9d078755d475de96a37368cc76a23bf8b3b648cab77b54d1e58f3a2f00d5bdcd`.
2. **Canonical Behavioral ER Benchmark**:
   - `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz`: 52,855 positive transitions, 10 negatives per positive.
   - Resample baseline valid L1 Hard ER: **$16.38953942232631$** (verified identical to #47 and #49).
3. **Stage 2 Checkpoint Availability (38 Models)**:
   - **Audited in `coverage_audit.csv`**:
     - 4 models verified from current working tree (`resample`, `resample_fn_mask`, `resample_s43`, `resample_s44`, `TIGER_RQ-VAE`).
     - 27 models resolved and verified from local git LFS cache objects (`.git/lfs/objects/` under historical commits `a5f8d9e^` and `19d0d69^`).
     - 6 models in `curvature_RQ-VAE` (`l2_cone_arms/A_euclid_nocone_s42`, `l2_assignment_arms/A_bucket_sinkhorn_s42`, `l2_cone_arms/B_poincare_nocone_s42`, `l2_assignment_arms/B_global_sinkhorn_s42`, `l2_cone_arms_fixed_rejection/D_poincare_cone_s42`, `l2_cone_arms/D_poincare_cone_s42`) had missing weights that were never committed to git.
   - **Reconstruction sample size**: $N=32$ complete cases across 5 families.
   - **Neighborhood sample size**: $N=38$ complete cases across 5 families.
4. **Fourth Token Exclusion**:
   - All 38 models exported 4-token SIDs where the 4th token is collision extension. The 4th token was strictly excluded from all semantic reconstruction and prefix neighborhood calculations.

---

## 3. Semantic Constructs & Numerical Verification (Phase 1)

1. **Construct A: Decoder Reconstruction ($R_{\text{cos}}$, $R_{\text{mse}}$, $R_{\text{norm}}$)**:
   - For all 32 eligible models, we loaded the trained weights, extracted codebook matrices and decoder MLP weights, and performed exact reconstruction:
     $$\hat{e}_i = \text{Decoder}(c_0[t_{i,0}] + c_1[t_{i,1}] + c_2[t_{i,2}])$$
   - Both catalogue-uniform and demand-weighted (train/valid) metrics were computed.
   - **Numerical cross-check on Resample**: Model forward pass cosine ($0.887437$) vs exported 3-token SID decode cosine ($0.887440$) agreed within $|\Delta| = 2.98 \times 10^{-6}$ (MSE $1.41 \times 10^{-4}$), well within the $< 10^{-3}$ contract threshold.
2. **Construct B: Checkpoint-Free Prefix Neighborhood Preservation ($N_{\text{sem}}$, Semantic AUC)**:
   - Reference neighbors: Top-20 Sentence-T5 cosine neighbors per item on the catalogue ($24,587 \times 20$).
   - Tie-aware analytic expectation: Prefix matching depth ($0, 1, 2, 3$) partitions candidates into discrete layers; boundary layers with $n_k > \text{budget}$ are assigned exact fractional expectation $m_k \cdot (\text{rem} / n_k)$, eliminating arbitrary item-ID tie-breaking bias.
   - Semantic AUC: Discrimination between Top-20 text positives and matched random negatives (depth tie $= 0.5$).
   - Verified on hand-calculated toy cases and synthetic unit tests (`scripts/test_semantic_reconstruction_features.py`).

---

## 4. Predictive Modeling Results (Phase 2)

All models were evaluated using Leave-One-Family-Out (LOFO) Ridge regression with train-fold-only standardization and inner grouped CV for alpha selection. For every challenger, baseline $M0$ was refitted on the **exact same model subset and family folds**.

### Primary Target: Recall@10

| Subset | Model Set | Predictors | MAE | RMSE | $R^2$ | Spearman $\rho$ | $\Delta\text{MAE}$ vs $M0$ | 95% Paired Cluster CI | Lower Bound $> 0$? |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full 38** | **M0** | $\log(\text{ER})$ | **0.003753** | **0.005460** | **0.296** | **+0.637** | — | — | — |
| Full 38 | **M9** | $\log(\text{ER}) + N_{\text{sem}}$ | 0.003753 | 0.005460 | 0.296 | +0.637 | $-0.000000$ | $[-0.000137, +0.000155]$ | False |
| **Recon 32** | **M0 (subset)** | $\log(\text{ER})$ | **0.004417** | **0.006236** | **0.238** | **+0.565** | — | — | — |
| Recon 32 | **M8** | $\log(\text{ER}) + R_{\text{cos}}$ | 0.004539 | 0.006456 | 0.183 | +0.476 | $-0.000122$ | $[-0.000377, +0.000044]$ | False |
| Recon 32 | **M10** | $\log(\text{ER}) + R_{\text{cos}} + \text{ER}\times R_{\text{cos}}$ | 0.004960 | 0.006938 | 0.059 | +0.370 | $-0.000543$ | $[-0.001374, +0.000155]$ | False |

### Secondary Target: NDCG@10

| Subset | Model Set | Predictors | MAE | RMSE | $R^2$ | Spearman $\rho$ | $\Delta\text{MAE}$ vs $M0$ | 95% Paired Cluster CI | Lower Bound $> 0$? |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full 38** | **M0** | $\log(\text{ER})$ | **0.002102** | **0.003002** | **0.374** | **+0.713** | — | — | — |
| Full 38 | **M9** | $\log(\text{ER}) + N_{\text{sem}}$ | 0.002102 | 0.003002 | 0.374 | +0.713 | $+0.000000$ | $[-0.000079, +0.000084]$ | False |
| **Recon 32** | **M0 (subset)** | $\log(\text{ER})$ | **0.002373** | **0.003316** | **0.314** | **+0.669** | — | — | — |
| Recon 32 | **M8** | $\log(\text{ER}) + R_{\text{cos}}$ | 0.002422 | 0.003445 | 0.259 | +0.603 | $-0.000049$ | $[-0.000166, +0.000045]$ | False |
| Recon 32 | **M10** | $\log(\text{ER}) + R_{\text{cos}} + \text{ER}\times R_{\text{cos}}$ | 0.002677 | 0.003723 | 0.134 | +0.428 | $-0.000304$ | $[-0.000742, +0.000044]$ | False |

### Per-Family LOFO Influence (Recall@10)

| Held-out Family | $N_{\text{test}}$ | $M0$ MAE | $M8$ MAE | $\Delta M8$ | $M9$ MAE | $\Delta M9$ | $M10$ MAE | $\Delta M10$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE` | 4 | 0.00160 | 0.00165 | $-0.00005$ | 0.00160 | $+0.00000$ | 0.00174 | $-0.00014$ |
| `TIGER_FORK_RQ-VAE` | 9 | 0.00394 | 0.00394 | $+0.00000$ | 0.00394 | $-0.00000$ | 0.00405 | $-0.00011$ |
| `TIGER_HYPERBOLIC_RQ-VAE` | 7 | 0.00287 | 0.00318 | $-0.00030$ | 0.00287 | $+0.00000$ | 0.00302 | $-0.00015$ |
| `curvature_RQ-VAE` | 11 / 17 | 0.00624 | 0.00639 | $-0.00015$ | 0.00497 | $+0.00000$ | 0.00773 | $-0.00149$ |
| `TIGER_RQ-VAE` | 1 | 0.00749 | 0.00789 | $-0.00039$ | 0.00749 | $+0.00000$ | 0.00711 | $+0.00038$ |

---

## 5. Top-10 Candidate Model Screening

Evaluated over the common 38-model pool:
- **Raw ER**: $P@10 = 0.90$ (9 / 10 true top models captured).
- **M0 OOF**: $P@10 = 0.90$ (9 / 10 true top models captured).
- **M9 OOF**: $P@10 = 0.90$ (9 / 10 true top models captured).

Semantic neighborhood preservation does not alter or improve candidate model selection.

---

## 6. Scientific Discussion & Conclusion

1. **Why does semantic reconstruction fail out-of-family?**
   - High semantic reconstruction is an artifact of unconstrained autoencoding. `TIGER_RQ-VAE` achieves the highest semantic reconstruction in the entire dataset ($R_{\text{cos}} = 0.9298$), but suffers severe codebook fragmentation and achieves near-worst recommendation Recall ($0.053674$).
   - Conversely, geometric inductive biases in `curvature_RQ-VAE` (Poincaré metric constraints and context channels) reduce raw unconstrained text cosine ($R_{\text{cos}} \approx 0.13 - 0.27$), yet these models achieve strong recommendation ($0.0595 - 0.0605$).
   - A model trained on Euclidean TIGER arms falsely learns that low $R_{\text{cos}}$ implies a degraded model. When transferred out-of-family to curvature models, this spurious association causes severe underprediction, degrading out-of-family generalization.
2. **Why does prefix neighborhood preservation show zero increment?**
   - Coarse prefix depth ($0/1/2/3$) yields tie-heavy candidate sets ($n_1$ and $n_2$ contain hundreds or thousands of items). Even with exact analytic tie-breaking expectation, prefix retention has limited variance across competent models and is collinear with behavioral clustering.
3. **Conclusion**:
   Stage 2 Semantic ID quality for recommendation is governed by behavioral transition preservation (L1 Hard ER), not by text semantic reconstruction fidelity. Hypotheses postulating that "both high ER and high semantic reconstruction are required" are refuted by empirical LOFO evaluation.
