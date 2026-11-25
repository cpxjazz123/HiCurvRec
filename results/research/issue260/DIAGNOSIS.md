# Issue #260 — R05 root-cause diagnosis (round 1)

All measurements are read-only. No training was started for this diagnosis. Sources are the frozen
R05 checkpoint/SID, the baseline SID and the persisted validation behavior pairs; the intervention
probe is `diagnosis/r05_probe.json` (same checkpoint, encoder and codebooks; only the declared switch
changes).

## 1. Observed failure phenomena

| observation | value | source |
|---|---:|---|
| R05 test Recall@10 | 0.050349 | `FINAL_STAGE3.json` |
| frozen TIGER baseline test Recall@10 | 0.0557 | frozen record |
| R05 three-code collision | 0.0553 | `FINAL_STAGE2.json` |
| R05 L2 prefixes / triples | 9609 / 23228 of 24587 | layered metrics |
| R05 quantisation error / effective rank | 0.0071 / 4.33 | `FINAL_STAGE2.json` |
| final gate entropy / level-3 mean weights | 0.582 / [0.023, 0.951, 0.026] | training metrics |
| R05 L1 positive / negative prefix hit rate | 0.05354 / 0.00934 | probe |
| R05 L2 positive / negative prefix hit rate | 0.00359 / 0.000316 | probe |
| R05 L3 positive / negative prefix hit rate | 0.000416 / 7.6e-6 | probe |
| baseline L1 / L2 / L3 positive rate | 0.08349 / 0.00804 / 0.001646 | probe |

The phenomenon is not “Recall is low” by itself. It is: **the hard SID is nearly item-unique, the
quantiser reconstructs the encoder almost exactly, and the behaviour/curvature switches barely change
which hard codes are selected.**

## 2. Probe results (decisive measurements)

The probe reproduced the exported SID exactly (`export_match=true`) and then toggled one declared
factor at a time, holding the encoder, codebooks and data fixed.

| intervention | L1 code change | L2 code change | L3 code change | L2-prefix change |
|---|---:|---:|---:|---:|
| behavior off (role mixture → one global mixture) | 0.0000 | **0.00110** | 0.00110 | 0.00110 |
| curvature off (all experts → Euclidean) | 0.01566 | **0.01688** | 0.01672 | 0.01688 |
| gate uniform (topology gate removed) | 0.00500 | 0.00504 | 0.00496 | 0.00504 |

Additional measured facts:

- **Role mixtures are almost uniform.** Entropy 1.508–1.584 bits against a uniform maximum 1.585;
  every role's three weights lie between 0.187 and 0.424. The behaviour-coupling term has almost no
  geometry to apply.
- **The level-3 gate is collapsed**: mean entropy 0.176 bits, 99.59% of items select expert 1, role
  eta² 0.0064. The level-1 gate still has entropy 1.505 bits and 78.7% of items select expert 1.
- **Topology tokens add little beyond degree**: role classification accuracy is 0.566 from token mean,
  0.540 from degree only, 0.570 from both, majority 0.385. The GNN readout carries mostly degree, not a
  separable graphlet signal.
- **Quantisation is near-exact**: residual MSE per level is `1.91e-5 / 5.39e-6 / 2.32e-6`; the encoder
  output is reproduced almost perfectly. Codebooks use all 256 codes; L2 has 9609 prefixes (2.56 items
  per prefix).
- **Margins are not tie-decided** (L1/L2/L3 relative median 1.12 / 0.40 / 0.16), but the cost changes
  are too small relative to the reconstruction-driven ranking to move assignments.

## 3. Root cause

The actionable root cause is **assignment saturation of a cost-only coupling**:

1. The encoder/codebook path already represents each item almost exactly (quantisation error ~1e-5).
2. Role/curvature/gate enter only as weights on the distance cost. Changing those weights moves only
   0.1%–1.7% of hard codes, because the nearest codeword remains the same.
3. Therefore R05's L2 placement cannot coarsen the prefix, and R03/R04's placements cannot either.
   This explains why three consecutive level placements produced the same negative signature.
4. Coarsening by itself is not sufficient either: the frozen baseline has 7,374 L2 prefixes and still
   0.0557; the resample SID has 15,860 and 0.0633. The missing lever is not “fewer prefixes” alone.

Competing explanations remain:

- H4 (null): single-run Stage3 variance cannot be estimated under the frozen no-seed-sweep protocol.
  The 0.0503 vs 0.0557 gap is therefore not claimed as a causal estimate by itself.
- H2 is also supported: the level-3 gate collapse and degree-dominated tokens mean topology is weak
  even before the cost path. A repair that only changes the cost weights cannot fix that.
- H5/H6 are supported as descriptions (L1 carries the behavior signal; capacity is spent on identity)
  but not as sufficient causal repairs.

## 4. Decision for R06

The next mechanism must make the behavior signal enter the **hard assignment path**, not only reweight
distances after the encoder. The chosen candidate will use the frozen graphlet role to select a
role-specific L2 code subset inside the existing curvature cost; the assignment rule and Sinkhorn/floor
remain the baseline's, the role only partitions the cost matrix. This directly targets the measured
inertness while preserving the local GNN, gate, three curvature experts, sectional supervision and
RQ-VAE residual stack.

The preregistered local check must show the partition changes L2 assignments substantially and raises
L2 prefix sharing without changing L1; otherwise the candidate returns to diagnosis instead of burning
formal training. The full contract and six answers are frozen only after that check.

## 5. What remains uncertain

- Whether a role block improves Stage3 Recall; prefix-sharing changes are not a recommendation guarantee.
- Whether the weak token/role signal is sufficient for any legal mechanism to reach >0.065.
- H4 (Stage3 variance) cannot be resolved without a forbidden seed sweep; it stays `INCONCLUSIVE`.
