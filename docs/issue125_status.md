# Issue #125 — status: the forensic decomposition is complete, read-only, and it explains the counterexamples

## What was done

A read-only forensic comparison of four Stage3 arms (plus the historical resample reference), on the frozen protocol, with nothing trained, tuned or overwritten. No model, SID, checkpoint, metric or historical decision was modified.

Artefacts:

```text
results/research/sid_generation_forensics/manifest.json          provenance, protocol, hashes, per-arm warnings
results/research/sid_generation_forensics/provenance.json
results/research/sid_generation_forensics/per_model_token_metrics.json   teacher-forced NLL / accuracy / prefixes
results/research/sid_generation_forensics/per_model_beam_metrics.json    real width-20 beam survival and postprocessing
results/research/sid_generation_forensics/per_model_sid_metrics.json     SID structure and the canonical L1 ER
results/research/sid_generation_forensics/paired_comparison.csv          paired differences with bootstrap intervals
results/research/sid_generation_forensics/strata.csv                    per-stratum breakdowns
results/research/sid_generation_forensics/report.md                     the report
```

Module: `tools/research/diagnostics/sid_generation_forensics.py` (read-only by construction). Regression tests: `tools/research/tests/test_issue125_forensics.py`.

## The original scores were verified, not assumed

| arm | recorded recall@10 | re-derived from the checkpoint | verdict |
|---|---|---|---|
| `tiger_fork_hyper_l1_br_euclid_s42` | 0.0625010881 | 0.062066 | CONSISTENT (bf16 batch-order rounding) |
| `tiger_fork_hyper_l1_s42` | 0.0614042724 | 0.062327 | CONSISTENT |
| `tiger_fork_hyper_l1_br_s42` | 0.0406866415 | 0.046710 | **INCONSISTENT** — no protocol variant reaches the recorded value |
| `TIGER_successor_entropy_curvature` | 0.0506276223 | 0.051220 | CONSISTENT |
| `TIGER_BEHAVIOR_SAMPLE_RQ-VAE/resample` | 0.0636327234 | 0.063633 | EXACT |

`n_eval = 57439` for every arm, beam 20, `max_new_tokens = n_digit + 1`, deterministic fallback by sorted item ids.

The three fork arms' Stage2 SID tables were deleted from the working tree by commit `19d0d69f`; they are recovered **read-only** from that commit's parent, and the recovery is recorded in the manifest with the git provenance.

## The answer to the question the issue asks

**Why do near-collapsed-L1 arms reach 0.061–0.063 while `successor_entropy_curvature` reaches 0.0506?**

The single-code arms have exactly one L1 value, so `P(pos) = P(neg) = 1` and ER = 1 *by construction* — that is what the old "L1 HARD ≈ 1" observation measured; it is a property of the SID table, not of a model. And it does **not** explain the gap: the first token is free for them (NLL 0.001 / 0.000 / 0.000, gold-L1 survival 1.000) and their Recall@10 still spans 0.04671 … 0.06233. A quantity that is constant across arms cannot separate them.

What does separate them is **where the SID information sits relative to a width-20 beam**:

| arm | recall@10 | raw beam coverage | NLL T1 | T2 | T3 | T4 | distinct gold T4 tokens |
|---|---|---|---|---|---|---|---|
| `l1_br` | 0.04671 | 0.0678 | 0.00 | 0.00 | 0.00 | 9.03 | 16 462 |
| `l1_br_euclid` | 0.06207 | 0.0847 | 0.00 | 3.22 | 4.36 | 1.23 | 44 |
| `l1_only` | 0.06233 | 0.0855 | 0.00 | 3.23 | 4.38 | 1.19 | 37 |
| `successor_entropy` | 0.05122 | 0.0688 | 3.45 | 2.33 | 2.29 | 0.98 | 91 |
| `resample` | 0.06363 | 0.0884 | 4.55 | 3.82 | 0.73 | 0.05 | 10 |

- `l1_br` pushes all the information into the **fourth** token, where it has 16 462 distinct values at 9.03 nats: twenty beams cover 6.8% of the targets.
- `l1_br_euclid` / `l1_only` push it into T2/T3, where the gold token's median rank is 6/16 and therefore fits inside a width-20 beam: coverage 8.5%.
- `successor_entropy` spreads it over T1–T3 and loses 26.5% of the examples at the very first step: coverage 6.9%.

The first-drop-depth distributions order exactly like the coverages (`{"never_dropped": 3893, …, "4": 53546}` vs `{"never_dropped": 4867, "2": 9835, "3": 38331}` vs `{"never_dropped": 3950, "1": 15197, "2": 19294, "3": 15294, "4": 3704}`).

**Teacher-forced accuracy is not Recall**: the total NLL is nearly flat (8.80 … 9.15) while Recall@10 spans 0.04671 … 0.06363, and `resample` has the *worst* T1 accuracy (0.098) and the *best* Recall.

**Postprocessing and the fallback are minor**: in nine examples in ten the target SID never appears among the twenty beams (`0.9116` … `0.9322`), so duplicate removal, seen-item filtering and the deterministic fallback can only move the rest. Those three mechanisms are replayed exactly and contribute at most the few examples in the middle columns.

## Hypotheses, ranked

1. **A collapsed first level makes the first token easier** — SUPPORTED as a fact, REFUTED as the explanation.
2. **The position that carries the SID information decides the coverage** — SUPPORTED, and it is the dominant factor.
3. **Beam pruning is the mechanism, not a side effect** — SUPPORTED (drop-depth distributions).
4. **The fourth-token collision extension is expensive** — SUPPORTED for `l1_br` only (9.03 nats over 16 462 values vs 0.05 for `resample`); using it as the only discriminating position is the cost, not the extension itself.
5. **Postprocessing and fallback** — MINOR.
6. **Different geometry or training recipe as the cause** — INDETERMINATE; the arms come from different Stage2 families, so the structural differences are observational. Nothing here identifies a causal geometry effect.

## Link back to #124

`successor_entropy_curvature` is the #124 candidate. Its Stage3 result (0.050628, `passed = false`) decomposes as: first token hardest (NLL 3.45, accuracy 0.262, gold-L1 survival 0.735), the gold prefix is dropped at depth 1–3 in 49 785 of 57 439 examples, and the SID never appears in the beams for 93.1% of the targets. Its SID metrics looked better than its Recall because the L1 discrimination was real but the information was not concentrated where a width-20 beam can keep it.

## Remaining unknowns

- `l1_br`'s recorded 0.040687 cannot be reproduced from the surviving checkpoint and SID (no protocol variant reaches it), so that number is reported but never used as a reference.
- The mechanism above identifies *where* each example is lost. It does not identify a causal geometry effect, because the arms differ in more than geometry.
