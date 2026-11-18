# QDA: quantized geometry diagnosis on the frozen F0 baseline

Question: **在冻结基线 F0(E16×H16 混合曲率、Sinkhorn 分配、seed 42)上,三级量化到底破坏了哪一段行为结构?**
Answer form: the four-part diagnosis fixed before the run (A layer-wise structural alignment, B layer-wise CKA, C code stability, D capacity dynamics), mapped onto the six pre-declared outcome classes.

## 0. Design (frozen before any result)

| item | value |
|---|---|
| arm | `QDA_F0_seed42` — copy of the frozen F0 tree, five declared edits, nothing else |
| training | 3000 epochs, single card, per-process batch 4096, seed 42 (the F0 recipe) |
| observation epochs | 0, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000 (16) |
| observer | `eval()` + `no_grad()`; records `z`, cumulative quantized vectors, per-level residual, argmin codeword, L3 Sinkhorn assignment, nearest/second-nearest cost margin, quant error, code histogram; restores train mode before returning |
| frozen sample | 315,391 TRAIN behaviour edges, degree-quantile matched non-edges, triangle multiplicity, Louvain communities (37), inside flags, ORCA roles; selection/final split by edge hash, sha256 fixed in `sample_meta.json` **before** any result existed |
| controls | 2 unmonitored runs: the same tree with `QDA_MONITOR = False`, same seed/code/data (only their SIDs and loss traces are kept; the trees differ by that one line) |
| provenance | `arm_meta.json`: 14 of 15 source files byte-identical to `stage2_RQ-VAE/TIGER_mixed_rq_F0` at `c268db1f^`; `train_rqvae.py` carries the five declared edits |

The five edits to `train_rqvae.py`:
1. `ARM_NAME` → `QDA_F0_seed42`;
2. `RESULT_DIR`/`LOG_DIR` → `results/research/qda/runs/<arm>/` (a diagnostic must not write into an arm result tree);
3. `_repository_root()` now requires **both** markers — `results/` also contains `stage2_RQ-VAE`, so the single-marker search stopped one level too high and pointed every data path at `results/stage1_GeneEmbedding`; the only functional change to the arm;
4. `QDA_MONITOR` plus the `observe()` call site and the observation epochs;
5. `QDA_SKIP_READOUT` — the Issue #139 shared readout at epoch 1500 reads an AF3 cache deleted in the November cleanup; only the outer-loop arms (G/L/E/A) consume it, F0 never does, so skipping its write leaves the F0 trajectory untouched.

Final SID: `raw_unique=16772/24587`, `collision=0.317851` — matches the published F0 run, so the fork reproduces the arm.

### Observer inertness (control)

Cross-process bit-identity is not attainable here (cuBLAS is non-deterministic; the export audit already showed two runs rotating 0.4 % of codewords). The monitor is judged against the recipe's own run-to-run noise, not against bit-equality.

| comparison | loss-trajectory \|Δ\| over 3000 epochs | final weight rel-L2 | final SID sha |
|---|---|---|---|
| monitor-on vs off-1 | **2.47e-6** | 1.027 | 07fcb2a41f28 |
| off-1 vs off-2 | **5.39e-6** | 1.101 | eedd3b21741d |
| monitor-on vs off-2 | — | 1.100 | c231d85c8596 |

**The monitored arm differs from an unmonitored one by less than two unmonitored runs differ from each other.** The weight divergence (rel-L2 ≈ 1 in every pair) is the recipe's own chaos: a discrete assignment flip changes the gradient path and the trajectories separate. Verdict: **the observer is inert at the level this experiment can resolve.** `control_compare.json` holds the per-key table.

## A. Layer-wise structural alignment

AUC of a real TRAIN behaviour edge against its degree-matched non-edge, on the held-out final edges. `encoder` = the continuous `z`; `L1..L3` = the cumulative quantized vector at that level (mixed distance, α = 0.5).

| epoch | encoder | L1 | L2 | L3 | lift L1 | lift L2 | lift L3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5959 | 0.5692 | 0.5748 | 0.5786 | +0.031 | +0.005 | +0.001 |
| 50 | 0.6396 | 0.6064 | 0.6311 | 0.6363 | **+0.177** | +0.032 | +0.002 |
| 100 | 0.6556 | 0.6380 | 0.6468 | 0.6518 | +0.161 | +0.028 | +0.002 |
| 200 | 0.6587 | 0.6492 | 0.6546 | 0.6556 | +0.140 | +0.022 | +0.002 |
| 300 | 0.6630 | 0.6529 | 0.6583 | 0.6612 | +0.146 | +0.023 | +0.002 |
| 500 | 0.6638 | 0.6529 | 0.6596 | 0.6625 | +0.143 | +0.023 | +0.002 |
| 1000 | 0.6637 | 0.6527 | 0.6607 | 0.6637 | +0.146 | +0.024 | +0.003 |
| 1500 | 0.6633 | 0.6514 | 0.6593 | 0.6633 | +0.145 | +0.023 | +0.003 |
| 2000 | 0.6624 | 0.6519 | 0.6596 | 0.6621 | +0.144 | +0.023 | +0.003 |
| 2500 | 0.6628 | 0.6524 | 0.6619 | 0.6628 | +0.143 | +0.023 | +0.003 |
| 3000 | 0.6633 | 0.6528 | 0.6606 | 0.6624 | +0.143 | +0.023 | +0.003 |

Layer-wise ΔG at the end: **L1 −1.05 pp, L2 +0.78 pp, L3 +0.18 pp** (positive = the quantized prefix separates behaviour edges better than the level above it).

`lift Lℓ` = P(behaviour edge shares the level-ℓ prefix) − P(matched non-edge shares it). It is discrete, so it is compared only across levels and epochs, never subtracted from an AUC.

Reading:
- Quantization costs **~1 pp** of behaviour discriminability at L1, **recovers most of it at L2** (+0.78 pp over L1) and **finishes level with the encoder at L3** (−0.09 pp).
- The discrete prefix surplus is far larger than any AUC delta: real behaviour edges share the L1 code **14.3 pp** more often than matched non-edges, but only 2.3 pp at L2 and 0.3 pp at L3. **L1 carries nearly all the behavioural content of the SID; L3 carries almost none.**
- The surplus is highest early (+17.7 pp at epoch 50) and settles at +14.3 pp — it decays exactly as L1 usage broadens (see D).

## B. Layer-wise CKA (consecutive observation pairs)

| interval | encoder | L1 | L2 | L3 |
|---|---:|---:|---:|---:|
| 0→50 | 0.755 | 0.649 | 0.724 | 0.738 |
| 50→100 | 0.901 | 0.755 | 0.852 | 0.884 |
| 100→200 | 0.957 | 0.832 | 0.882 | 0.923 |
| 200→300 | 0.975 | 0.907 | 0.908 | 0.936 |
| 500→750 | 0.983 | 0.954 | 0.928 | 0.943 |
| 1000→1250 | 0.992 | 0.981 | 0.951 | 0.956 |
| 2000→2250 | 0.992 | 0.981 | 0.953 | 0.958 |
| 2750→3000 | **0.996** | 0.993 | 0.974 | **0.970** |

- L1's readout moves the most early (0.649 at the first step) — that is the capacity collapse.
- **From epoch ~500 onward the deepest levels are the least settled**: L2 and L3 sit at 0.95–0.97 while the encoder has flat-lined at 0.992+. L3, the level that carries **no** behaviour surplus, is still moving the most at the end.
- The layers are not equally dynamic, and the ordering reverses over training: L1 most volatile early, L2/L3 most volatile late.

## C. Codeword stability

15 consecutive observation intervals × 24,587 items.

| level | mean flips per item (of 15) | max | items never flipped |
|---|---:|---:|---:|
| L1 | 2.39 | 12 | 0 |
| L2 | 7.86 | 15 | **1** |
| L3 | 14.30 | 15 | 0 |

Essentially **no item keeps its code**: one item out of 24,587 holds its L2 code across all 15 intervals, none holds L1 or L3.

Graphlet association with L1 flip count — partial Spearman controlling log-degree and log-community-size, cluster bootstrap over items (300 resamples), BH across the graphlet axes:

| feature | partial ρ | 95 % CI | p (FDR) |
|---|---:|---|---:|
| triangle density | −0.014 | [−0.027, −0.002] | 0.038 |
| **inside-community share** | **−0.035** | **[−0.048, −0.023]** | **1.3e-7** |
| clustering coefficient | −0.004 | [−0.017, +0.008] | 0.527 |

By ORCA role (mean flips; overall 2.387):

| role | items | mean flips | 95 % CI |
|---|---:|---:|---|
| 0 | 3,521 | 2.297 | [2.256, 2.340] |
| 1 | 1,311 | 2.330 | [2.258, 2.401] |
| 2 | 2,643 | **2.219** | [2.173, 2.268] |
| 3 | 3,543 | 2.404 | [2.361, 2.449] |
| 4 | 657 | 2.321 | [2.222, 2.414] |
| 5 | 12,881 | **2.451** | [2.428, 2.476] |

Reading:
- **The robust structural correlate of instability is community position**: items whose behaviour edges stay inside their own community flip L1 codes less (−0.035, FDR 1.3e-7). Triangle counting is a distant second (−0.014) and the clustering coefficient does not survive controlling for degree.
- Role differences are real but small (±0.12 flips); the largest, most "plain" role class flips the most.
- Item-level correlations: flips vs quant error **+0.411**, vs margin **+0.359**, vs degree +0.002, vs triangle density −0.008. The positive margin correlation is the opposite of a "boundary jitter among near ties" story: the items that move are the ones whose codeword cost is *largest*, i.e. the codebook is being rewritten under them, not split between two equidistant codes.
- Margin by degree quintile (Q1→Q5): 1.10e-4, 1.28e-4, 1.26e-4, 1.12e-4, 1.00e-4 — a 28 % spread, **highest for mid-degree items, lowest for the most-linked 20 %**. The most-linked items are the least confidently placed.

## D. Capacity and dynamics

| epoch | L1 used | L1 H | L2 used | L2 H | L3 used | L3 H | rank z | rank L1 | rank L2 | rank L3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 201 | 5.54 | 249 | 6.24 | 251 | 6.82 | 19.27 | 14.27 | 16.85 | 17.99 |
| 50 | **9** | **2.38** | 91 | 5.55 | 256 | 7.67 | **6.63** | **2.68** | **4.72** | **5.16** |
| 100 | 18 | 3.40 | 137 | 6.07 | 256 | 7.86 | 9.78 | 5.10 | 8.16 | 8.50 |
| 200 | 32 | 4.34 | 165 | 6.28 | 256 | 7.75 | 12.22 | 7.80 | 10.26 | 10.61 |
| 300 | 35 | 4.38 | 181 | 6.31 | 255 | 7.87 | 14.00 | 9.16 | 11.74 | 12.10 |
| 500 | 38 | 4.44 | 195 | 6.42 | 256 | 7.86 | 15.43 | 10.44 | 13.02 | 13.40 |
| 1000 | 40 | 4.45 | 210 | 6.42 | 256 | 7.93 | 17.38 | 11.62 | 14.40 | 14.88 |
| 2000 | 43 | 4.55 | 221 | 6.53 | 256 | 7.95 | 18.73 | 12.76 | 15.53 | 16.12 |
| 3000 | **43** | 4.56 | 221 | 6.55 | 256 | 7.97 | 19.04 | 12.98 | 15.75 | 16.37 |

Quant error (mean chosen cost in the mixed metric) and assignment margin (nearest − second-nearest), L1/L2/L3:

| epoch | err L1 | err L2 | err L3 | margin L1 | margin L2 | margin L3 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.10e-3 | 8.18e-4 | 6.39e-4 | 1.64e-4 | 1.18e-4 | 7.43e-5 |
| 50 | 7.49e-4 | 2.39e-4 | 1.26e-4 | 9.14e-4 | 1.24e-4 | 1.88e-5 |
| 500 | 3.02e-3 | 1.44e-3 | 9.12e-4 | 2.46e-3 | 5.41e-4 | 1.09e-4 |
| 3000 | 3.80e-3 | 1.96e-3 | 1.32e-3 | 4.16e-3 | 6.40e-4 | 1.54e-4 |

Reading:
- **L1 never uses more than 43 of its 256 codes (16.8 %)**, and the collapse is violent: 201 → 9 within 50 epochs, then a slow crawl to 43 over 2,950 epochs. Entropy 5.54 → 2.38 → 4.56. **L1's codebook is 83 % dead at the end of training.**
- **L2 loses half its codes in the first 50 epochs** (249 → 91) and re-expands monotonically to 221. Its effective rank passes through the 8–12 band only over epochs 100–300 (8.2, 10.3, 11.7) and reaches **15.8** at the end: the "L2 有效秩 8–12" figure is a **mid-training transient**, not a final state.
- **L3 is saturated** (256/256 from epoch 50) and its rank climbs to 16.4 — the deepest level has the largest effective rank and the *least* behaviour surplus. Ranking capacity is spent on a level that behaviour does not use.
- L1's margin grows **25×** while occupancy stays at 17 %: the surviving codes get confidently assigned, the rest stay dead. Not a boundary problem — a capacity problem.
- Quant error **rises** monotonically after the collapse while the reported reconstruction loss *falls* (4.8e-4 → 1.65e-4): the encoder's output scale grows, so absolute residual cost grows while MSE to the target shrinks. Absolute costs are not comparable across epochs; only within-epoch ratios are.

## Verdict against the six pre-declared classes

| class | holds? | evidence |
|---|---|---|
| 1. L2 effective rank drops **and** structural alignment drops | **no** | L2 rank dips only over epochs 100–300 and L2 AUC *rises* through that window (0.6468 → 0.6583); final rank 15.8 with AUC 0.6606. |
| 2. **Structural alignment improves after quantization** | **partly** | L2/L3 improve over L1 (+0.78 / +0.96 pp) and the discrete prefix lift is large (+14.3 / +2.3 / +0.3 pp). But L1 is **below** the encoder (−1.05 pp), so "quantization helps" is true only for deeper levels, and L3 is level with the encoder rather than above it. |
| 3. Reconstruction loss falls **while** structural alignment falls | **no** | Alignment rises monotonically with the loss fall over the whole run; no window shows the opposition after epoch 50. |
| 4. Structural alignment is fine **but** code stability is poor | **yes** | AUC stays within ~1 pp of the encoder, yet essentially no item holds any code across 15 intervals (1 of 24,587 at L2, 0 at L1 and L3). |
| 5. L1 usage normalizes **but** deeper levels degrade | **no** | L1 does *not* normalize (16.8 % of capacity, 43 codes); the deeper levels do not degrade (L2 +0.78 pp, L3 +0.96 pp over L1). |
| 6. All metrics remain near the encoder | closest single description at 3000 | L3 AUC 0.6624 vs encoder 0.6633 (−0.09 pp) — but "near" is carried by L2, while L1 is 1.05 pp down with 83 % of its codebook dead. |

**Primary classification: class 4 (structure survives, stability does not), with a class-5-shaped capacity pathology concentrated at L1.** The two are one fact seen twice: L1 keeps 43 live codes and those codes are re-cut continuously, so every item is carried by a narrow, churning partition.

What this rules in and out:
- **Not the geometry.** Nothing here says the mixed metric is mis-shaped: L2/L3 preserve — and improve on — the behaviour structure of L1, and L3 lands on the encoder. A curvature pathology would show as alignment loss that grows with depth; the observed depth profile is the opposite.
- **Not the loss.** Reconstruction falls monotonically; no epoch window shows the loss improving while behaviour structure degrades.
- **The assignment/capacity policy is the suspect.** Sinkhorn balancing fills the L1 codebook at init (201 codes at epoch 0); 50 epochs of gradient push kill 192 of them and the balancing term never re-opens them. Dead codes with a 25× margin at 17 % occupancy is the signature of a codebook that specialised and cannot re-expand.
- **Where geometry could still pay**: L2 — the only level with a real behaviour surplus (+2.3 pp prefix lift), full occupancy (221/256) and the slowest CKA. L1 is a capacity problem; L3 has no behaviour content to work with.

## Next experiments this diagnosis licenses (in order)

1. **L1 occupancy during training** — log the Sinkhorn transport plan's row/column mass per epoch (the observer already stores `assignment_l3`; the L1 analogue must be added). If the plan empties, the fix is in the *allocation*, not the geometry.
2. **Freeze-and-probe** — at epoch 300, re-balance L1 only (no gradient step) and re-measure A and C. If AUC holds while flips collapse, stability is an assignment artefact and the mechanism must act on the allocation step.
3. Only then geometry: a per-graphlet curvature bias should be argued and placed on **L2**, not L1. The prior evidence that per-role geometry is the right shape is Stage2-level and produced a degenerate codebook (`TIGER_role_geom_rq`: L1 Hard ER 0.85, L3 collision 0.42 — above the 0.40 Stage-3 gate); this diagnosis says why: it changed geometry while L1 was already starving, so the geometry was never the binding constraint.

## Artefacts

| file | content |
|---|---|
| `trees/F0_seed42/qda_observer.py` | the recorder |
| `trees/F0_seed42/` | the exact 15-file tree that produced the observations (`arm_meta.json` has the per-file hashes) |
| `sample/sample.npz`, `sample/sample_meta.json` | frozen behaviour sample and its hashes |
| `runs/QDA_F0_seed42{,_nomonitor,_nomonitor2}/` | three 3000-epoch runs, observation files, SIDs, checkpoints |
| `analyze.py`, `analysis_QDA_F0_seed42.json` | A/B/C/D tables (regenerable in ~2.5 min) |
| `control_compare.py`, `control_compare.json` | the inertness control |
| `test_qda_metrics.py` | 6 tests: AUC tie handling, prefix lift as a difference of two rates, **monotonicity of prefix agreement in depth**, CKA on a rotation vs an independent sample, scale-freeness of the effective-rank proxy |

Data-quality note: the first observation pass stored per-item margins and quant errors as float16; the margins are ~1e-4 and underflow there, which zeroed the degree-quintile margin table. The monitored arm was re-run with float32 scalars and every number above comes from that pass. Flip counts, CKA and AUCs were never affected.
