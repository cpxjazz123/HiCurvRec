# SID-Diag-01 — Three-Level Semantic ID Distribution Audit

## Scope and verdict

Frozen audit of the latest completed Stage2 snapshot. No Stage2 training, checkpoint/SID mutation, or new Stage3 launch occurred. The analysis ran on CPU; no model or training implementation was changed.

The checkpoint and three exported SID files are linked by the native Stage2 `snapshot_saved` and `train_end` records at global step 72,000. `item_sids.json`, `sids_raw.npy`, and `sids_for_hgrec.npy` agree exactly on item order and code columns. Strict loading of the checkpoint into the current `RQVAE` succeeded. The saved SID arrays—not a CPU-generated re-encoding—are authoritative for SID statistics below.

**Re-encoding caveat:** full-corpus CPU re-encoding did not exactly reproduce the saved GPU-generated codes: 382 L1, 2,040 L2, and 6,589 L3 per-level code differences; cumulative prefix differences were 382/24,587 (L1), 2,063/24,587 (L1-L2), and 6,767/24,587 (L1-L2-L3). A device/numerical effect in the iterative Sinkhorn/argmax path is a possible cause, not established. This does not invalidate the mutual consistency of the exported files or their native snapshot provenance. Geometry and residual measurements use the strict-loaded checkpoint's CPU encoder outputs together with the saved GPU SID indices, so treat those measurements as cross-device diagnostics, not bit-exact GPU reconstructions.

## Frozen inputs and provenance

| Input | Value |
|---|---|
| Repository commit | `8ef7b6ed8621cc060d54d8359f970e672804781e` |
| Mechanism | `poincare_behaviour_context_channel` |
| Checkpoint | `results/stage2_RQ-VAE/curvature_RQ-VAE/out/rqvae/instruments/rqvae_best.pth` |
| Stage2 completion | global step 72,000; local optimizer steps 18,000; native run wall time 1,404.549 s |
| Geometry | Poincaré fixed curvature; `c=[1,1,1]`, working radii `[0.2,0.2,0.2]`, effective epsilons `[0.003,0.003,0.003]` |
| Item embeddings | `sentence_t5.npy`, shape `(24,587, 768)` |
| Training data | 396,958 rows; 339,519 valid last-seen-item → target transitions |
| Item/SID rows | 24,587 items; raw SID `(24,587,3)`; HG-Rec SID and JSON `(24,587,4)` |
| SID extension | one fixed extension value, `768`; all 24,587 raw three-level SIDs are unique |

SHA-256 hashes are recorded in `summary.json` for the checkpoint, all SID exports, embedding matrix, training parquet, native training metrics, and source files.

## 1–2. Codeword use and balance

Each level has 256 codewords; all are used. Entropy is in bits; normalized entropy divides by `log2(256)=8`. Top-K entries are fractions of all items assigned to the K most-used codes.

| Level | Used / 256 | Entropy (bits) | Normalized entropy | Gini | Top-1 | Top-5 | Top-10 | Top-20 | Top-50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 256 | 7.9830 | 0.9979 | 0.0851 | 0.525% | 2.550% | 5.003% | 9.741% | 23.301% |
| L2 | 256 | 7.9798 | 0.9975 | 0.0943 | 0.634% | 2.880% | 5.531% | 10.562% | 24.582% |
| L3 | 256 | 7.9846 | 0.9981 | 0.0805 | 0.553% | 2.619% | 5.084% | 9.843% | 23.455% |

Codeword-level counts and shares: [`codeword_usage.csv`](codeword_usage.csv). Frequency histograms: [`codeword_frequency_histogram.csv`](codeword_frequency_histogram.csv) and [chart](codeword_frequency_histogram.svg).

## 3. Prefix occupancy, cluster sizes, branching

| Prefix | Unique prefixes | Cluster size min / median / max | Singleton nodes | Collision pairs |
|---|---:|---:|---:|---:|
| L1 | 256 | 57 / 97 / 129 | 0 | 1,195,286 |
| L1-L2 | 24,587 | 1 / 1 / 1 | 24,587 | 0 |
| L1-L2-L3 | 24,587 | 1 / 1 / 1 | 24,587 | 0 |

| Parent → child | Children per parent: min / median / max | Conditional entropy |
|---|---:|---:|
| L1 → L2 | 57 / 97 / 129 | `H(L2|L1)=6.6026` bits |
| L1-L2 → L3 | 1 / 1 / 1 | `H(L3|L1,L2)=0` bits |

Thus L1-L2 already uniquely identifies every item; L3 increases no prefix uniqueness in this snapshot. Per-node sizes and branch factors: [`prefix_nodes.csv`](prefix_nodes.csv), [`branching.csv`](branching.csv). Cluster-size chart: [prefix_cluster_size_distribution.svg](prefix_cluster_size_distribution.svg).

## 4. Popularity-stratified SID hierarchy

Popularity is a deterministic rank split over all 24,587 items, sorted by target interaction count descending and `item_id` ascending for ties: first 20% head, next 30% mid, remaining 50% tail. Equal-frequency items can straddle a boundary due to that tie-break. Counts use target occurrences in `train.parquet`.

| Group | Items | Target frequency range | Interaction share | L1 entropy (bits) | Unique L1-L2 prefixes |
|---|---:|---:|---:|---:|---:|
| Head | 4,918 | 17–1,914 | 63.90% | 7.7346 | 4,918 |
| Mid | 7,376 | 8–17 | 21.02% | 7.9553 | 7,376 |
| Tail | 12,293 | 0–8 | 15.08% | 7.9685 | 12,293 |

Every group's L1-L2 prefixes are unique within that group. Full distributions and counts: [`popularity_groups.csv`](popularity_groups.csv).

## 5. Behavior-pair SID sharing

The training sequence rule is the current Stage2 `_transition_pairs`: `seen_history[-1] → target`. There are 339,519 transition events and 323,633 unique directed pairs. A deterministic seed-42 uniform sample without replacement selects 200,000 unique positive pairs; event multiplicity is retained per row. For each positive, the popularity-matched negative uses the same source and the positive target's rank decile, excluding source and positive target. The random negative uses the same source and a uniform item, also excluding source and positive target.

| Pair class | Sampled pairs | Share LCP ≥ 1 | Share LCP ≥ 2 | Share LCP ≥ 3 |
|---|---:|---:|---:|---:|
| Positive transition | 200,000 | 6.987% | 0% | 0% |
| Rank-decile-matched negative | 200,000 | 0.507% | 0% | 0% |
| Uniform random negative | 200,000 | 0.401% | 0% | 0% |

The positive L1-sharing rate is 13.8× the matched-negative rate and 17.4× the random rate in this sample. No pair in any class shares L1-L2 because every L1-L2 prefix is a singleton. Raw sampled pairs: [`behavior_pair_sample.csv`](behavior_pair_sample.csv); summary: [`behavior_prefix_sharing.csv`](behavior_prefix_sharing.csv); [chart](behavior_prefix_sharing.svg).

## 6. Encoder geometry versus SID LCP

A seed-42 sample of 100,000 non-self item pairs was measured using the frozen CPU encoder output. Hyperbolic distance uses `_poincare_distance_tangent_pairs` at `c=1`; Euclidean distance is tangent-space L2 distance, multiplied by 10.7421 to match the sample median hyperbolic distance. The two distance rankings have Spearman `ρ=0.9003`.

| Distance ranking bin | Hyperbolic P(LCP≥1) | Median-scaled Euclidean P(LCP≥1) |
|---|---:|---:|
| Lowest decile | 3.770% | 3.620% |
| Second decile | 0.260% | 0.240% |
| Third decile | 0.050% | 0.050% |

No sampled distance decile has LCP≥2 or LCP≥3. Hyperbolic distance provides stronger L1-prefix enrichment in the lowest decile here, but the differences are small and observational. This is not a curvature ablation: both metrics use the same trained encoder vectors, and there is no Stage3 comparison in this audit. All deciles and pairwise distances: [`geometry_distance_quantiles.csv`](geometry_distance_quantiles.csv), [`geometry_pair_sample.csv`](geometry_pair_sample.csv); [chart](distance_quantile_prefix_sharing.svg).

## 7. Residual and quantization error

For each saved SID level, residual updates follow the model's three-level Möbius residual order, including configured radius pinning/restoration. The measurement pairs saved GPU token IDs with the frozen CPU encoder output; see the cross-device caveat above. Errors are hyperbolic distances from the pinned residual input to its saved codeword.

| Level | Mean error | Median error | 90th percentile error | Mean residual tangent norm after level |
|---|---:|---:|---:|---:|
| L1 | 0.02719 | 0.02698 | 0.03358 | 0.14389 |
| L2 | 0.35486 | 0.35444 | 0.38766 | 0.12757 |
| L3 | 0.35384 | 0.35342 | 0.38903 | 0.11281 |

Dense-L1-cluster (`Q4_dense`) mean error by popularity group:

| Level | Head | Mid | Tail |
|---|---:|---:|---:|
| L1 | 0.02642 | 0.02793 | 0.02969 |
| L2 | 0.35347 | 0.35320 | 0.35247 |
| L3 | 0.35416 | 0.35278 | 0.35314 |

The four L1 size quartiles rank codeword cluster sizes ascending, breaking ties by code ID. Full per-item residuals and all popularity × cluster-quartile × level summaries: [`item_residual_diagnostics.csv`](item_residual_diagnostics.csv), [`residual_summary.csv`](residual_summary.csv); [dense-prefix chart](residual_error_dense_prefixes.svg).

## 8. Anomaly examples

- **Dense L1 prefixes:** top 10 by item count, with example item IDs.
- **Sparse L1 prefixes:** none exist; every L1 cluster has 57–129 items.
- **Sparse L1-L2 prefixes:** all 24,587 are singleton nodes; 100 examples are exported.
- **Behavior-positive but SID-separated:** 100 sampled positive transitions with LCP 0, ordered by observed event count.
- **Geometry-near but SID-separated:** 100 closest LCP-0 pairs within the random geometry sample; this is not an exhaustive all-pairs nearest-neighbor search.

All cases, IDs, SIDs, distances, and counts: [`anomaly_examples.csv`](anomaly_examples.csv).

## 9–10. Evidence-backed findings

| Finding | Observed fact | Possible cause (not established) | Testable hypothesis (not validated here) |
|---|---|---|---|
| **Uniqueness arrives at L2; L3 adds no new identifiers** | L1-L2 has 24,587/24,587 unique prefixes; `H(L3|L1,L2)=0`; L3 adds zero unique prefixes. | Per-bucket residual assignments may already separate every item at L2; global codeword balance does not imply that later levels carry additional identity. | The current Stage3 consumer may receive little incremental item-disambiguation from L3; only a controlled recommendation evaluation can establish whether L3 contributes to Recall@10. |
| **Behavior affinity is concentrated at L1 and disappears by L2** | Positive transitions share L1 in 6.987% of sampled pairs vs 0.507% popularity-matched and 0.401% random; none share L1-L2. | The coarse first-level code may capture some successor-context affinity, while the second-level assignment refines it into item-unique prefixes. | A curvature-aware mechanism that preserves behavior relations across the L1→L2 residual transition could improve recommendation quality; it must be judged by Stage3 Recall@10, not these diagnostic rates. |
| **Hyperbolic and scale-matched Euclidean distances mostly order pairs similarly** | Spearman `ρ=0.9003`; lowest-decile L1 sharing is 3.77% under hyperbolic distance and 3.62% under median-scaled Euclidean distance. | Much of the pairwise ordering may come from the learned encoder's direction/norm structure; the measured curvature-specific reordering is comparatively limited in this sample. | Curvature-specific ordering changes may be too small to explain large behavior-separation gains without a geometry-aware representation mechanism; a matched counterfactual or downstream test is needed. |
| **The item interaction distribution is head-heavy, but all popularity groups remain unique by L1-L2** | The top-ranked 20% of items account for 63.90% of target events; the tail 50% account for 15.08%; each group's L1-L2 prefixes are unique. | Frequency concentration and identifier uniqueness are distinct properties; the hierarchy can preserve item uniqueness without allocating more prefix depth to popular items. | Popularity-aware behavior diagnostics should separate relevance from frequency; whether any reallocation helps Recall@10 remains open. |

These are descriptive findings, not acceptance gates or evidence of a Stage3 improvement. No Stage3 metric was produced by this audit.

## Output inventory

- Reproducible hardcoded CPU audit script: [`stage2_RQ-VAE/curvature_RQ-VAE/scripts/sid_diag.py`](../../../../stage2_RQ-VAE/curvature_RQ-VAE/scripts/sid_diag.py).
- Machine-readable complete summary and hashes: [`summary.json`](summary.json).
- Raw tables: `codeword_usage.csv`, `codeword_frequency_histogram.csv`, `prefix_nodes.csv`, `branching.csv`, `popularity_groups.csv`, `behavior_pair_sample.csv`, `behavior_prefix_sharing.csv`, `geometry_pair_sample.csv`, `geometry_distance_quantiles.csv`, `item_residual_diagnostics.csv`, `residual_summary.csv`, `anomaly_examples.csv`.
- Charts: `codeword_frequency_histogram.svg`, `prefix_cluster_size_distribution.svg`, `behavior_prefix_sharing.svg`, `distance_quantile_prefix_sharing.svg`, `residual_error_dense_prefixes.svg`.

During the audit, four pre-existing `run_stage3_curvature.py` workers (PIDs 2499901–2499904) were observed running. This audit started no Stage3 process and did not signal or modify those workers.
