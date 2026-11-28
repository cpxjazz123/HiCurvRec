**Result: the conditional rank association is supported but is one axis, partly range-restriction, and
gives no predictor increment; #277 retained.**

Analysis-only over #284 `b106d01e`; 129 labels (120/9; 13 high/6 strong), 5 folds, 127 clusters
unchanged; ER = P/N exactly (max deviation 0.0); #284 global bands reproduced (0.907/0.846).

**Conditions are one source:** N–C Spearman 0.864, N–P 0.801, C–P 0.739; PC1 of (N,C,P) explains
**88.8 %**; the low-N and low-C sets overlap with Jaccard 0.667. Fold-local tertiles (training-fold
quantiles): LL 34 (27 strong, 11 high of 4 strong), LN 10 (0 high), LC 7 (unidentifiable, <10),
HH 39 (0 high), other 39. Low N alone contains 11/13 high recalls, low C adds 2, both-high 0; the raw
axes already separate the high tail: **AUC(−N) 0.9025 [0.824, 0.965], AUC(−C) 0.9038 [0.844, 0.955]**.

**Joint grid:** within LL, P ρ=0.951 [0.861, 0.987], ER 0.956, chain ALL 0.926 [0.818, 0.970],
fork/merge 0.926; HH: P 0.240, ALL 0.082. Interaction Δρ(LL−HH): **P +0.711 [0.334, 1.075], ALL
+0.844 [0.429, 1.210]** — supported by the pre-registered cluster-bootstrap rule (both cells ≥10).
But **P ≥ ALL** (0.951 vs 0.926; partial after N,C 0.962 vs 0.925), and chain ALL is not special.
No strict monotonic law: 43/560 (7.7 %) inversions for P, 61/560 (10.9 %) for ALL; counterexamples
listed (`GRID_285.json`). Range restriction explains part: truncating LL to the HH recall interval
leaves n=22 with P 0.878 / ALL 0.815, still far above HH. Robustness: strong-only 0.930/0.890,
without high tail 0.886/0.826, LOO P 0.946–0.973 — no few-SID/high-tail/weak-identity dominance.

**Pre-registered predictors (training-fold gates, same folds):** A 0.0029719/0.6666 (exact);
B (P×G_N) 0.0030144 (worse); C (ALL+ALL·G_N) 0.0027032/0.7648, CI [−7.63e−5, +6.36e−4] — numerically
the #284 raw-ALL arm; D (smooth double) 0.0030102; C-gate alt significantly worse. **All CI lower
bounds <0; no candidate passes.** 13/0/0 at >0.06; 0 rows >0.065 → NOT_EVALUABLE. Power: 383
independent SIDs for the C effect (MDE 4.63e−4 at 129), 847 for the selected; the 34-SID conditional
cell has MDE 2.2e−4 — not validatable at this n.

**Frozen candidate:** C (ALL + ALL·G_N, G_N = 1[N ≤ 0.0089066], alpha 0.01) with the recorded
coefficients and 122 pending predictions side by side with #277/#284; marked exploratory and redundant
with #284. #277, the 124-row prospective file and the #284 candidate are unchanged. No Stage3.
Infra drift 0.
