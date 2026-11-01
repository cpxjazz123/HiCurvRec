# Issue #127 — status: the three losses are implemented, verified, and the matched hyperbolic/Euclidean pair is running

## What was built

`stage2_RQ-VAE/TIGER_HGS-TREE` and its matched control `stage2_RQ-VAE/TIGER_HGS-TREE-EUCLID` carry three mathematically distinct hierarchy terms, each with its own switch, each reaching the encoder and every codebook:

| term | definition | file |
|---|---|---|
| `tree_ordering` | `softplus(d(u[i,L], u[j,L]) − d(u[i,L], u[k,L]) + delta·(T(i,k) − T(i,j)))` over hierarchy-consistent triplets | `model/hierarchy_loss.py` |
| `radius_term` | `softplus(margin_R − (r(u[i,l+1]) − r(u[i,l]))/scale)` over the refinement steps | `model/hierarchy_loss.py` |
| `direction_term` | `softplus(cos(v[i,l], v[k,l]) − cos(v[i,l], v[j,l]) + margin_A)` with a same-cell positive and a sibling-branch negative | `model/hierarchy_loss.py` |
| `prefix_alignment` | the legacy term of the first HGS run, kept separate and switchable | `model/hierarchy_loss.py` |

They act on the **cumulative soft prefix embedding** `s[i,l] = Σ_{t≤l} p[i,t,k]·e[t,k]`, projected into one bounded ball with a **single detached scale shared by all levels**; neither the soft assignment nor the code vectors are detached. `GEOMETRY = "hyperbolic" | "euclidean"` is the only difference between the arms.

The two trees differ in exactly two lines, and a test keeps it that way:

```text
model/hierarchy_loss.py:52   GEOMETRY = "hyperbolic" | "euclidean"
train_rqvae.py:58            ARM_NAME  = "TIGER_HGS-TREE" | "TIGER_HGS-TREE-EUCLID"
```

## The defect the first run exposed, and the fix

The first HGS run supervised a random 128-row slice of the batch. With 13,416 leaf cells, the expected number of same-cell pairs in such a slice is **0.62**, so the finest level was unsupervised in half of the steps while the 17-cell coarse level supplied **978** pairs — the supervision pulled the codebook towards the coarse partition and the first level collapsed.

The new sampler (`model/hierarchy.py::Sampler`) reads cached cell → member tables over the whole catalogue and builds positives that genuinely exist at every level: the measured coverage is now `tree 32, direction {1: 24, 2: 24, 3: 16}, prefix {1: 24, 2: 24, 3: 24}` per step. The legacy prefix term reads the **level-balanced** pairs instead of every same-cell pair of the slice.

## The radius term's degenerate solution, found by the diagnostics and fixed

The first run of the three losses collapsed again, and the logged diagnostics named the cause: the level-0 radius fell to **0.0069** while the finer levels sat on the ball boundary (1.98 / 1.99), and the code usage fell to 380 distinct triples. The inequality `radius(prefix l+1) > radius(prefix l)` is cheapest to satisfy by sending the **coarse** prefix towards the origin, and a coarse prefix at the origin merges every item into one first-level code.

The term now constrains the refinement steps between the later prefixes, which is where the depth is supposed to show. The regression test is `test_the_radius_term_does_not_reward_a_collapsed_coarse_prefix`, and it failed before the fix (0.4747 vs 0.5737) and passes after it.

## Verification

| check | result |
|---|---|
| unit and mechanism tests (both arms) | 19 tests each, green |
| matched-arm test | the arms differ in two declared constants only |
| framework machine admission (`stage2_start`) | `constants`, `geometry_contract`, `entry`, `identifier` (24,587 unique ids of 4 tokens), `four_rank` — all PASS for both arms |
| real entrypoint | one epoch runs and exports the identifier, through `train_rqvae.py` |
| deployed interpreter | the research package imports and its 387 tests pass under Python 3.9 (the interpreter the MCP runs) and 3.10 |

## The behaviour hierarchy is frozen, train-only and reproducible

`results/research/hierarchy_audit/` now carries `report.json` and `manifest.json` (committed) next to the binary table (not committed, per the repository's binary rule). The manifest records the artefact's sha256, the builder and its sha256, the input hashes, the cell counts, the unknown-item counts and `nesting_breaks: 0`.

The audit's **four-point statistic was removed**: a nested partition induces an ultrametric, so that statistic is identically zero for any hierarchy and cannot separate a real one from a rewire. The evidence is the co-membership z-score against degree-preserving random rewires: **z = 181.4 / 76.0 / 14.1** at the three levels, on the independent validation pairs.

## What is not done, and why

- **The ablations of the issue's table were not trained** (tree-only, tree+radius, tree+direction). The repository's experiment policy (§19g) forbids training ablation arms; the three switches exist in the code and are unit-tested, and the previous tree-only run is preserved as the Stage-2 reference.
- The legacy `prefix_alignment` term is **on** in both trained arms, as the issue's objective formula states, and its applied gradient scale is logged with every epoch. Its weight is disclosed there rather than being a silent constant.
- The curvature is **fixed at −1** for the hyperbolic arm. This is fixed-curvature hyperbolic supervision; it is not the separate future idea of a learnable per-level or per-region curvature, and the report says so.

## The numbers

Per-arm evidence: `results/stage2_RQ-VAE/<ARM>/hgs_report.json` (final-epoch terms and applied scales, the usage trajectory, the radius and saturation diagnostics, the angular gap, the sampling coverage, and the SID's agreement with the frozen hierarchy).

### What the frozen hierarchy is worth

Measured on the exported SIDs, with the same definitions:

| arm | L1 codes | L2 prefixes | L3 triples | 4th-token vocabulary | residual identity bits/item | NMI(SID_L3, hierarchy_L3) | tree-triplet satisfaction | prefix contrast |
|---|---|---|---|---|---|---|---|---|
| baseline `TIGER_RQ-VAE` | 247 | 7,414 | 20,223 | 42 | 0.48 | **0.944** | 0.051 | 0.040 |
| tree-only HGS (first run, broken sampler) | 53 | 740 | 3,069 | 1,197 | 6.95 | 0.652 | **0.288** | **0.130** |

The baseline reproduces the frozen hierarchy's finest level with NMI 0.944 **without any hierarchy supervision**: the tree is already encoded. The supervised arm follows the hierarchy *better* on the tree's own terms (triplet satisfaction 0.288 vs 0.051, prefix contrast 0.130 vs 0.040) and ends up with a worse SID (NMI 0.652, 6.95 residual identity bits per item instead of 0.48) and a worse recall (0.045683 vs 0.0553). Following the hierarchy more closely is worth less than the resolution it costs.
