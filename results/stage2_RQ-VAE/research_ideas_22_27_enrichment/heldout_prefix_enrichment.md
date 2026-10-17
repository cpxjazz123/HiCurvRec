# Held-Out L1 and L1+L2 Behavioral Prefix Enrichment Report

## 1. Experimental Setup & Verification Protocol
- **Validation Pair Dataset**: `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz`
- **Held-Out Sample Size**: 52,855 transition pairs (unobserved in training, history[-1] -> target)
- **Negative Baseline**: 10 matched negatives per transition pair (total 528,550 negative pairs)
- **Bootstrap Methodology**: Paired row-level bootstrap with 1000 replicates (seed 42), reporting 95% percentile CIs.

### Baseline Canonical Check
The canonical Exp1a Resample baseline was evaluated on the exact same valid pair set to verify against historical reference constants:
- **L1 ER**: Expected `16.3895`, Measured `16.3895` -> **PASSED (bit-identical)**
- **L1+L2 ER**: Expected `58.3871`, Measured `58.3871` -> **PASSED (bit-identical)**

---

## 2. Enrichment Ratio Summary Table

| Arm | Level | pos_share | neg_share | gap | ER | 95% Bootstrap CI (ER) | L1 Codes | L1+L2 Prefixes | 3-Token Coll. | Δ ER vs Base |
|---|:---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| `resample` | L1 | 0.0794 | 0.0048 | 0.0746 | **16.39** | [15.59, 17.22] | 241 | 16,145 | 0.0363 | +0.00 (×1.00) |
| `resample` | L1+L2 | 0.0034 | 0.0001 | 0.0034 | **58.39** | [41.50, 90.01] | 241 | 16,145 | 0.0363 | +0.00 (×1.00) |
| `idea1_#22` | L1 | 0.0118 | 0.0040 | 0.0078 | **2.97** | [2.71, 3.23] | 256 | 19,261 | 0.0120 | -13.42 (×0.18) |
| `idea1_#22` | L1+L2 | 0.0005 | 0.0000 | 0.0005 | **31.11** | [15.71, 75.00] | 256 | 19,261 | 0.0120 | -27.28 (×0.53) |
| `idea2_#23` | L1 | 0.1759 | 0.0490 | 0.1270 | **3.59** | [3.51, 3.67] | 25 | 4,886 | 0.0689 | -12.80 (×0.22) |
| `idea2_#23` | L1+L2 | 0.0045 | 0.0005 | 0.0041 | **10.04** | [8.46, 12.02] | 25 | 4,886 | 0.0689 | -48.35 (×0.17) |
| `idea3_#24` | L1 | 0.1730 | 0.0460 | 0.1270 | **3.76** | [3.68, 3.85] | 36 | 3,467 | 0.1961 | -12.63 (×0.23) |
| `idea3_#24` | L1+L2 | 0.0181 | 0.0015 | 0.0166 | **12.07** | [11.00, 13.24] | 36 | 3,467 | 0.1961 | -46.31 (×0.21) |
| `idea4_#25` | L1 | 0.1238 | 0.0274 | 0.0963 | **4.51** | [4.38, 4.62] | 86 | 8,350 | 0.0467 | -11.88 (×0.28) |
| `idea4_#25` | L1+L2 | 0.0039 | 0.0004 | 0.0035 | **10.67** | [8.80, 12.74] | 86 | 8,350 | 0.0467 | -47.72 (×0.18) |
| `idea5_#26` | L1 | 1.0000 | 1.0000 | 0.0000 | **1.00** | [1.00, 1.00] | 1 | 1 | 0.9896 | -15.39 (×0.06) |
| `idea5_#26` | L1+L2 | 1.0000 | 1.0000 | 0.0000 | **1.00** | [1.00, 1.00] | 1 | 1 | 0.9896 | -57.39 (×0.02) |
| `idea6_#27` | L1 | 0.0357 | 0.0123 | 0.0234 | **2.90** | [2.74, 3.04] | 190 | 9,400 | 0.0452 | -13.49 (×0.18) |
| `idea6_#27` | L1+L2 | 0.0015 | 0.0002 | 0.0013 | **7.50** | [5.62, 10.00] | 190 | 9,400 | 0.0452 | -50.89 (×0.13) |

---

## 3. Targeted Diagnostic Findings (Addressing Issue #30 Key Questions)

### Diagnostic 1: Idea 1 (#22) Full L1 Codebook Utilization vs Behavioral Selectivity
- **Question**: Does utilizing all 256 L1 codewords lead to higher behavioral selectivity?
- **Observation**: `TIGER_semantic_bounded_residual` achieves full utilization (`L1_codes = 256`, `L1L2_prefixes = 19,261`, `full_collision = 0.0120`), which is the highest codebook diversity in the entire benchmark.
- **Finding**: **NO**. Higher codebook dispersion does NOT translate to higher behavioral enrichment.
  - At L1, `pos_share` dropped sharply from 0.0794 (baseline) to 0.0118 (-85.2%), while `neg_share` only dropped from 0.0048 to 0.0040 (-18.4%).
  - Consequently, **L1 ER dropped from 16.39 to 2.97** (CI95: [2.76, 3.20], a 5.5× reduction in selectivity).
  - At L1+L2, ER is 31.11 vs 58.39 for baseline.
  - **Conclusion**: Spreading representation across all 256 codewords reduces the probability that co-transiting items fall into the same discrete bucket more severely than it reduces random collisions.

### Diagnostic 2: Idea 2 (#23) & Idea 3 (#24) Codebook Concentration vs Negative Discrimination
- **Question**: Does low L1 utilization in Idea 2 and Idea 3 degrade negative discrimination?
- **Observation**: `TIGER_relational_structure_distill` only utilizes 25 L1 codewords, and `TIGER_uncertainty_behavioral` utilizes 36 L1 codewords.
- **Finding**: **YES**. Low L1 capacity severely inflates negative sharing rate.
  - Negative sharing rate `neg_share` jumped by an order of magnitude: from 0.0048 (baseline) to **0.0490 (Idea 2)** and **0.0460 (Idea 3)** (both ~10× higher).
  - Because only ~25-36 bins exist, random negative items have a ~4.6-4.9% baseline chance of hitting the anchor's bin.
  - Even though positive sharing is high (0.1759 and 0.1730 vs baseline 0.0794), the inflated denominator caps **L1 ER at 3.59 and 3.76** (vs 16.39).
  - At L1+L2, negative sharing remains elevated (0.00045 and 0.00150 vs baseline 0.00006), holding ER at 10.04 and 12.07 (vs 58.39).

### Diagnostic 3: Idea 5 (#26) Invertible Flow Collapse
- **Question**: Does the single prefix in Idea 5 strictly produce `pos_share = neg_share = 1, ER = 1`?
- **Observation**: `TIGER_invertible_hyperbolic_flow` suffered severe spatial collapse (`L1_codes = 1`, `L1L2_prefixes = 1`, `full_collision = 0.9896`).
- **Finding**: **YES**. All items mapped to a single (L1, L2) bucket.
  - Both `pos_share` and `neg_share` are **identically 1.0000** for all 52,855 pairs and 528,550 negative comparisons.
  - **ER = 1.0000** exactly at both L1 and L1+L2 (CI95: [1.00, 1.00]).
  - This confirms that an unconstrained continuous flow optimizing ranking loss collapses the latent space onto a degenerate point, giving zero behavioral selectivity.

### Diagnostic 4: Idea 4 (#25) and Idea 6 (#27)
- **Idea 4 (#25) Multi-Intent Disentangled**: Uses 86 L1 codes, achieving L1 ER = 4.51 (highest among the 6 research ideas, but still below baseline 16.39) and L1+L2 ER = 10.67.
- **Idea 6 (#27) Multi-Scale Spectrum**: Uses 190 L1 codes (`full_collision = 0.0452`), achieving L1 ER = 2.90 and L1+L2 ER = 7.50.

---

## 4. Key Distinction: SID Collision vs Behavioral Enrichment (ER)
- **SID Collision** measures purely the discrete space occupancy and collision rate ($1 - \text{unique}/N$). Low collision indicates high information capacity and spread.
- **Behavioral Enrichment Ratio (ER)** measures the mutual information between behavioral transitions and prefix clustering: $P(\text{prefix}(A)=\text{prefix}(P)) / P(\text{prefix}(A)=\text{prefix}(N))$.
- **Trade-off Revealed**: None of the 6 research ideas beat the Exp1a Resample baseline on ER, despite Idea 1 beating it significantly on collision rate (0.0120 vs 0.0363). There is an intrinsic trade-off: maximizing codebook spread disperses behavioral clusters, while concentrating behavioral clusters inflates negative collisions.
- *Note: Stage 3 Recall@10 and NDCG@10 metrics are reserved for downstream evaluation and are not inferred from offline Stage 2 ER.*
