# Issue #280 — pre-registered contract (frozen before any decoding result)

Frozen 2026-11-28 15:1x.

## 1. Candidate set (identical for every SID)

Unit = directed pair `(a, b)` with `a = history[-1]` and `b = target` of a validation row.
Main set: rows with `c(a,b) = 0` (**unseen** in the training graph). Negatives: one per row, uniform over
the frozen legal population (`history ∪ seen_history ∪ {target}` excluded), seed 42; the reverse pair
`(b,a)` is a separate diagnostic, never a negative. A fixed 10 000-row sample (seed 42) carries five
negatives per row for MRR@5/Hits@5. Every SID is scored on exactly the same candidate set.

## 2. SID-only directional decoder (main model)

Codes: the ordered first three codes of each item in that SID. No continuous vector, item id, text or
graph adjacency enters the score. For a pair `(a,b)`:

```
score(a,b) = Σ_{l=1..3} log [ (P_l(c_l(b) | s_l(a), t_{l-1}(b)) + ε) / (P_l(c_l(b) | t_{l-1}(b)) + ε) ]
```

* `s_l(a)` = source prefix `(a1..a_l)`; `t_{l-1}(b)` = target prefix `(b1..b_{l-1})` (`t_0` = ∅).
* Context counts come from the distinct **training** edges weighted by their counts; Laplace smoothing
  `α` over the 256 codes of the level. A context with total count 0 contributes 0; the marginal
  `P_l(c_l(b))` (Laplace over the training target-code distribution) is used when the denominator
  context is unseen.
* Hyperparameters `(α, ε)` are selected **inside training only**: a source-anchored 80/20 split of the
  training edges (source item hash mod 5 == 0 → calibration) on five fixed SIDs (the canonical
  `TIGER_RQ-VAE` SID plus the four highest-recall strong SIDs by sha order), grid
  `α ∈ {0.1, 1, 10}`, `ε ∈ {1e-3, 1e-2}`, selection by mean calibration AUC. The chosen pair is applied
  to every SID; no validation information enters selection.

## 3. Baselines and reference

`chance` (AUC 0.5), `popularity` = training in-degree of `b` (item identity, reference only),
`prefix` = number of matching leading levels of `(a,b)` (symmetric; reference), and the #279 graph
scores `two_hop` / `common_pred` as the information ceiling (not SID-only).

## 4. Diagnostics

Direction: `score(b,a)` on the same pairs (asymmetry), direction accuracy. Invariance: a global code
permutation per level must leave every score unchanged (checked numerically). Higher-order association:
Spearman between the decoder score and `two_hop` / `common_pred` on the unseen candidates, and the
decoder AUC stratified by the higher-order strength. A low-capacity ordered-code interaction diagnostic
(hashed ordered code pairs, fixed cap) is trained on 10 fixed SIDs only and reported as a diagnostic.

## 5. Frozen SID-level features (main, at most two)

* `dec_auc_unseen` = per-SID AUC of `score(a,b)` on the unseen candidate set.
* `dec_margin_degmatched` = mean `score` difference (positive − negative) inside the
  (anchor out-degree decile × target in-degree decile) cells, weighted by cell counts.

## 6. Models and criterion

`A` = Ridge6; `D1` = A + `dec_auc_unseen`; `D2` = A + `dec_margin_degmatched`; `D3` = A + both;
`G279` = A + the #279 higher-order sharing features (analysis control). Same folds, leak-free scaler,
inner-only alpha. Pass = OOF MAE decrease vs A with the SID-group paired bootstrap 95 % CI **lower
bound > 0**, Spearman/Kendall not worse, >0.06 FP/FN not worse, on the full 129 and the strict 120
subsets; 0.065 NOT_EVALUABLE. Otherwise keep #277; no new predictor or prospective predictions.
