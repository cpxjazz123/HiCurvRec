# SID geometry report — iter17 (post-Stage2, before Stage3)

## Scope and evidence

All geometric fingerprints below use the **three raw quantizer tokens** in `results/stage2_RQ-VAE/curvature_RQ-VAE_iter17/out/rqvae/instruments/sids_raw.npy` and the first three entries of the item SID JSONs for iter11, iter16, and iter4. Each run has 24,587 rows and 256-code quantizers. For the L0 concentration feature, `geometry_features.per_l0_concentration` was applied to each SID matrix with the common iter17 `dataset/Instruments/item_emb.npy` (24,587 × 768); this intentionally holds the embedding fixed while comparing SID assignments. Per-layer utilization, code entropy, and the utility's mean per-code conditional entropy use `sid_metrics_collector/geometry_features.py` (`per_layer_code_usage`, `cross_layer_alignment`). Weighted conditional entropy in bits is also reported: it is calculated directly from the empirical joint distribution and matches the training log's `H_l1_given_l0` convention. Utility conditional-entropy means are in nats and are *unweighted means over L0 codes with at least five items*, so they are not interchangeable with the weighted values.

The table's prefix collision rate is `1 − unique(L0,L1)/N`; three-token collision rows are `N − unique(L0,L1,L2)`, and rate is rows/N. Coarse/fine balance is reported as a directly computed branching fingerprint: mean number of distinct L2 codes per observed (L0,L1) pair divided by mean number of distinct L1 codes per L0. It is descriptive, not a gate or a repository-defined canonical metric.

## Three-token comparison

| Run | L0/L1/L2 utilization (of 256) | Marginal code entropy L0/L1/L2 (nats) | Weighted H(L1\|L0) / H(L2\|L0) (bits) | Utility mean H(L1\|L0) / H(L2\|L0) (nats) | Unique (L0,L1) / prefix collision rate | Unique 3-token / collision rows (rate) | Coarse/fine branching ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| **iter17** | 256/256/256 (1.000/1.000/1.000) | 5.4873 / 5.4067 / 5.2792 | **5.5939 / 5.8641** | 3.7976 / 3.9719 | 14,746 / 40.03% | **23,092 / 1,495 (6.08%)** | 0.02719 |
| iter11 | 256/256/256 (1.000/1.000/1.000) | 5.4859 / 5.4711 / 5.4589 | 5.5844 / 5.9401 | 3.7829 / 3.9988 | 14,749 / 40.01% | 22,948 / 1,639 (6.67%) | 0.02701 |
| iter16 | 256/256/256 (1.000/1.000/1.000) | 5.5257 / 5.4637 / 5.3611 | 5.4549 / 5.8041 | 3.7528 / 3.9856 | 14,096 / 42.67% | 22,827 / 1,760 (7.16%) | 0.02941 |
| iter4 | 256/256/255 (1.000/1.000/0.996) | 5.4811 / 5.1687 / 4.9444 | 5.0162 / 5.3613 | 3.4315 / 3.6413 | 11,297 / 54.05% | 20,813 / 3,774 (15.35%) | 0.04175 |

Additional context: the training SID summary at step 100,000 reports `full_gini=0.0586`, layer Ginis `[0.1904, 0.2873, 0.3977]`, 23,092 unique three-token SIDs, 14,746 unique L0/L1 pairs, and `H_l1_given_l0=5.5939`. This agrees with the raw-array unique/prefix counts and weighted conditional entropy above. Utilization is not a measure of code-frequency balance: despite all 256 codes being used at every iter17 layer, the maximum single-code shares are L0 0.8175%, L1 2.0824%, and L2 1.9400%.

### L0 concentration (common item embeddings)

| Run | Mean within-L0 norm std | Max within-L0 norm std | Std of L0 mean norms | L0 codes with >10 items |
|---|---:|---:|---:|---:|
| iter17 | 0.00021247 | 0.00026488 | 0.00002256 | 256 |
| iter11 | 0.00021256 | 0.00025897 | 0.00002150 | 256 |
| iter16 | 0.00021209 | 0.00025507 | 0.00002054 | 256 |
| iter4 | 0.00021082 | 0.00026185 | 0.00002297 | 256 |

These norm-based L0 concentration measures are effectively similar across the four assignments under the supplied common embeddings; they do not establish geometric separation in the full 768-D space.

## Final training and export checks

`train_migrated.log` records the run completing at global step **100,000** (total time 1,363.6 s) and saving `rqvae_best.pth` at that step. The final SID summary is the one above. The last ordinary training sample before that checkpoint (step 99,652) records loss 3.0602, reconstruction loss (`rl`) 1.2815, vector loss (`vl`) 0.5056, behavior loss 6.3502, curvature regularizer 0.623299, and displayed curvature `c=0.051/0.051/0.051`; its metric display is `2.116/2.116/2.116`. These are the last logged per-step component values, not separately logged step-100,000 loss values.

The final exported four-token NPY has shape (24,587, 4), and its matching JSON reproduces it exactly. The fourth token ranges from 768 to 851 (84 distinct extension values); 23,092 rows use default 768. The four-token rows are unique (24,587/24,587), consistent with the export's `SID_WIRING_PASS`. The 3-token geometry and collision figures above intentionally exclude this collision-extension token.

## Pre-registered direct effects (hypothesis_iter17.md)

- **DE-1 — PASS (cyclic `c`/multiplier schedule).** The final audit is persisted in `mvg_check_iter17.log`: all `c_l` and `m_l` values are finite and positive at steps 0, 50,000, and 100,000; `mean(m_l(0))=1`; the midpoint multipliers `[0.399268, 0.082085, 0.073250]` differ from their step-0 values `[2.151887, 0.448100, 0.400013]` by more than `1e-3`.
- **DE-2 — PASS (per-layer updates after five ON steps).** The checker's corresponding output line is labeled `DE-3 five-step relative updates`; values are layer0 `0.047632`, layer1 `0.103153`, layer2 `0.100836`, each finite and above `1e-7`.
- **DE-3 — PASS (same-seed ON/OFF loss difference after 200 steps).** The checker labels this `DE-4 200-step counterfactual`: `L_on=2.54342532`, `L_off=2.46642065`, `abs(delta)=0.07700 > 1e-6`; `behavior_on=5.38333893`, `behavior_off=5.38952351`.

The MVG's internal labels include an extra gradient gate (`DE-2`), so its five-step update and 200-step ON/OFF lines map to the pre-registered hypothesis DE-2 and DE-3, respectively. All three hypothesis-specific direct effects pass.

## L0 oracle availability

**Unavailable.** `sid_metrics_collector/collect_metrics.py::compute_oracle_topk` returns `None`, and the collector writes `oracle_topk_full: None`. A repository search found no reproducible existing L0-oracle implementation to run for these arrays. The training log's `hitrate@50=0.9971` is not relabeled as an oracle: it is a different statistic and does not fill this gap.

## Descriptive classification

**ALIGNED.** All three registered direct effects pass. The proxy threshold `H(L1|L0) ≥ 5.45` is met by the weighted raw-SID value, 5.5939 bits. Three-token collision rate is 6.08%, lower than iter11 (6.67%), iter16 (7.16%), and iter4 (15.35%). The coarse/fine branching ratio is iter11-like (0.02719 vs. 0.02701), and common-embedding L0 concentration remains similar; neither contradicts a directional prediction in the registered proxy hypothesis. This is geometry alignment only; Stage3 determines downstream performance.

**This label is descriptive only. Under current `CLAUDE.md` §2, Stage2 has no outer or inner hard gate: SID geometry metrics cannot veto or delay iter17 Stage3.**