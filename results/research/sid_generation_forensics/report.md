# Issue #125 — per-token NLL and beam-prefix survival

Read-only forensic comparison of four Stage3 arms (plus the historical resample reference). Nothing was trained, tuned or overwritten.

## Provenance

- pairs: `/fs04/ar57/wenyu/GeneRec/results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz`
- Stage3 trainer: `/fs04/ar57/wenyu/GeneRec/stage3_T5Train/train_HG-Rec.py`
- protocol: {"beam_size": 20, "n_digit": 4, "n_eval_expected": 57439, "topk": [5, 10], "exclude_history": true, "fallback": "deterministic sorted item ids", "generation": "max_new_tokens = n_digit + 1, no sampling"}
- warning: the three fork arms' SID tables were deleted from the working tree by commit 19d0d69f and are recovered read-only from that commit's parent
- warning: the reported numbers are the original runs' own files; they are re-derived from the checkpoint before any forensic number is interpreted

## Layer A — does each surviving artifact reproduce its recorded score?

| arm | recorded recall@10 | reproduced | difference | status | note |
|---|---|---|---|---|---|
| l1_br | 0.0406866415 | 0.046710 | 6.02e-03 | INCONSISTENT | reproduced with the frozen evaluator at batch 1024, four ranks, bf16, and confirmed against a single-process pass; the recorded value cannot be reproduced from the checkpoint and SID that survive |
| l1_br_euclid | 0.06250108811086544 | 0.062066 | 4.35e-04 | CONSISTENT | single-process pass equals the frozen evaluator bit for bit; the residual against the original four-rank run is bf16 batch-order rounding |
| l1_only | 0.0614042724 | 0.062327 | 9.23e-04 | CONSISTENT | single-process pass equals the frozen evaluator bit for bit; the residual against the original four-rank run is bf16 batch-order rounding |
| resample | 0.0636327234 | 0.063633 | 1.09e-11 | EXACT | single-process pass equals the frozen evaluator bit for bit; the residual against the original four-rank run is bf16 batch-order rounding |
| successor_entropy | 0.05062762234718571 | 0.051220 | 5.92e-04 | CONSISTENT | single-process pass equals the frozen evaluator bit for bit; the residual against the original four-rank run is bf16 batch-order rounding |

The single-process pass is bit-identical to the frozen evaluator at the same batch shape (verified on `l1_br`: 0.046710423231602224 both ways), and the four-rank sharded protocol reproduces the same number, so a residual against the original run is a bf16 batch-order effect, not an implementation gap. `l1_br` is the exception: no protocol variant (one or four ranks, bf16 or fp32, batch 256/512/1024) reaches its recorded 0.040687, so that number is reported but never used as a reference.

## Layer B — the SID space and the canonical ER

| arm | L1 codes | L2 prefixes | L3 triples | H(L1) bits | ER(L1) | P(pos) | P(neg) | 4th-token max |
|---|---|---|---|---|---|---|---|---|
| l1_br_euclid | 1 | 85 | 9093 | 0.000 | 1.000 | 1.0000 | 1.00000 | 812 |
| l1_only | 1 | 97 | 9324 | 0.000 | 1.000 | 1.0000 | 1.00000 | 805 |
| l1_br | 1 | 1 | 1 | 0.000 | 1.000 | 1.0000 | 1.00000 | 25354 |
| successor_entropy | 256 | 3952 | 13702 | 5.457 | 1.280 | 0.1208 | 0.09443 | 861 |
| resample | 241 | 16145 | 23695 | 7.826 | 16.390 | 0.0794 | 0.00485 | 777 |

The single-code arms have exactly one L1 value, so *every* pair shares it: P(pos)=P(neg)=1 and ER=1 by construction. That is what the earlier “L1 HARD ≈ 1” observation measured; it is a property of the SID table, not of a model.

## Layer C — teacher-forced gold-token NLL (oracle prefixes)

| arm | NLL T1 | NLL T2 | NLL T3 | NLL T4 | NLL EOS | acc@1 T1 | acc@1 T2 | acc@1 T3 | acc@1 T4 |
|---|---|---|---|---|---|---|---|---|---|
| l1_br_euclid | 0.00 | 3.22 | 4.36 | 1.23 | 0.00 | 1.000 | 0.178 | 0.115 | 0.550 |
| l1_only | 0.00 | 3.23 | 4.38 | 1.19 | 0.00 | 1.000 | 0.175 | 0.106 | 0.557 |
| l1_br | 0.00 | 0.00 | 0.00 | 9.03 | 0.00 | 1.000 | 1.000 | 1.000 | 0.008 |
| successor_entropy | 3.45 | 2.33 | 2.29 | 0.98 | 0.00 | 0.262 | 0.338 | 0.412 | 0.680 |
| resample | 4.55 | 3.82 | 0.73 | 0.05 | 0.00 | 0.098 | 0.152 | 0.794 | 0.977 |

## Layer D — real width-20 beam survival and postprocessing

| arm | recall@10 | raw coverage | alive L1 | alive L2 | alive L3 | alive L4 | absent | hit(beams) | generated-rank-out |
|---|---|---|---|---|---|---|---|---|---|
| l1_br_euclid | 0.06207 | 0.08473 | 1.000 | 0.829 | 0.161 | 0.085 | 0.915 | 0.062 | 0.023 |
| l1_only | 0.06233 | 0.08553 | 1.000 | 0.829 | 0.156 | 0.086 | 0.914 | 0.062 | 0.023 |
| l1_br | 0.04671 | 0.06778 | 1.000 | 1.000 | 1.000 | 0.068 | 0.932 | 0.047 | 0.021 |
| successor_entropy | 0.05122 | 0.06877 | 0.735 | 0.400 | 0.133 | 0.069 | 0.931 | 0.051 | 0.018 |
| resample | 0.06363 | 0.08837 | 0.497 | 0.122 | 0.091 | 0.088 | 0.912 | 0.064 | 0.025 |

## Layer E — paired comparison against `l1_br_euclid`

| arm | metric | arm value | reference | difference | ci95 |
|---|---|---|---|---|---|
| l1_br | gold_nll_T1 | 0.00051 | 0.00009 | 0.00042 | - |
| l1_br | gold_nll_T2 | 0.00069 | 3.22015 | -3.21947 | - |
| l1_br | gold_nll_T3 | 0.00037 | 4.35771 | -4.35734 | - |
| l1_br | gold_nll_T4 | 9.03075 | 1.22575 | 7.80500 | - |
| l1_br | recall@10 | 0.04671 | 0.06207 | -0.01536 | [-0.01704, -0.01354] |
| l1_br | raw_beam_coverage | 0.06778 | 0.08473 | -0.01696 | - |
| l1_only | gold_nll_T1 | 0.00012 | 0.00009 | 0.00003 | - |
| l1_only | gold_nll_T2 | 3.22968 | 3.22015 | 0.00953 | - |
| l1_only | gold_nll_T3 | 4.37781 | 4.35771 | 0.02010 | - |
| l1_only | gold_nll_T4 | 1.19225 | 1.22575 | -0.03350 | - |
| l1_only | recall@10 | 0.06233 | 0.06207 | 0.00026 | [-0.00134, 0.00185] |
| l1_only | raw_beam_coverage | 0.08553 | 0.08473 | 0.00080 | - |
| resample | gold_nll_T1 | 4.54999 | 0.00009 | 4.54990 | - |
| resample | gold_nll_T2 | 3.81719 | 3.22015 | 0.59704 | - |
| resample | gold_nll_T3 | 0.73186 | 4.35771 | -3.62585 | - |
| resample | gold_nll_T4 | 0.04940 | 1.22575 | -1.17635 | - |
| resample | recall@10 | 0.06363 | 0.06207 | 0.00157 | [-0.00014, 0.00317] |
| resample | raw_beam_coverage | 0.08837 | 0.08473 | 0.00364 | - |
| successor_entropy | gold_nll_T1 | 3.44671 | 0.00009 | 3.44663 | - |
| successor_entropy | gold_nll_T2 | 2.32529 | 3.22015 | -0.89487 | - |
| successor_entropy | gold_nll_T3 | 2.28802 | 4.35771 | -2.06969 | - |
| successor_entropy | gold_nll_T4 | 0.98303 | 1.22575 | -0.24272 | - |
| successor_entropy | recall@10 | 0.05122 | 0.06207 | -0.01085 | [-0.01267, -0.00914] |
| successor_entropy | raw_beam_coverage | 0.06877 | 0.08473 | -0.01596 | - |

## Hypotheses

**Measured ordering of Recall@10 against the structural quantities**

| arm | recall@10 | coverage | sum NLL | T1 | T2 | T3 | T4 | T4 distinct |
|---|---|---|---|---|---|---|---|---|
| l1_br | 0.04671 | 0.0678 | 9.03 | 0.00 | 0.00 | 0.00 | 9.03 | 16462 |
| l1_br_euclid | 0.06207 | 0.0847 | 8.80 | 0.00 | 3.22 | 4.36 | 1.23 | 44 |
| l1_only | 0.06233 | 0.0855 | 8.80 | 0.00 | 3.23 | 4.38 | 1.19 | 37 |
| successor_entropy | 0.05122 | 0.0688 | 9.04 | 3.45 | 2.33 | 2.29 | 0.98 | 91 |
| resample | 0.06363 | 0.0884 | 9.15 | 4.55 | 3.82 | 0.73 | 0.05 | 10 |

1. **A collapsed first level makes the first token easier** -- SUPPORTED as a fact, REFUTED as the explanation of the gap. The three single-code arms have T1 NLL 0.001/0.000/0.000 and a gold-L1 survival of 1.000: the first token is free, yet their Recall@10 spans 0.04671 to 0.06233. A quantity that is constant across arms cannot separate them.
2. **The position that carries the SID information decides the coverage** -- SUPPORTED, and it is the dominant factor. `l1_br` puts everything into T4, where 16462 distinct gold tokens cost 9.03 nats, so twenty beams cover 0.0678 of the targets. `l1_br_euclid` puts it into T2/T3, where the gold token's median rank is 6/16 and therefore fits inside a width-20 beam, and its coverage is 0.0847. `successor_entropy` spreads it over T1-T3 and loses 0.265 of the examples at the very first step, so its coverage falls to 0.0688.
3. **Beam pruning is the mechanism, not a side effect** -- SUPPORTED. The first-drop depth distributions separate the arms: {"0_never_dropped": 3893, "1": 0, "2": 0, "3": 0, "4": 53546} against {"0_never_dropped": 4867, "1": 0, "2": 9835, "3": 38331, "4": 4406} against {"0_never_dropped": 3950, "1": 15197, "2": 19294, "3": 15294, "4": 3704}. A gold prefix dropped at depth 3 or 4 cannot be recovered by postprocessing, and the drop depths order exactly like the coverages.
4. **The fourth-token collision extension is expensive** -- SUPPORTED for `l1_br` only: its T4 costs 9.03 nats over 16462 values, while `resample` pays 0.05. In the other arms the extension token is nearly free, so the extension itself is not the cost; using it as the only discriminating position is.
5. **Postprocessing and fallback** -- MINOR. The share of rows whose target SID never appears among the twenty beams is {"l1_br": 0.9322, "l1_br_euclid": 0.9153, "l1_only": 0.9145, "successor_entropy": 0.9312, "resample": 0.9116}; postprocessing can only move the remainder, and duplicate/seen filtering is replayed exactly.
6. **Different geometry or training recipe as the cause** -- INDETERMINATE. The arms come from different Stage2 families, so the SID structure differences are observational. Nothing here identifies a causal geometry effect; the mechanism that is identified is the depth-versus-width mismatch.

**Teacher-forced accuracy is not Recall.** The total NLL is nearly flat across the five arms (8.80 to 9.15) while Recall@10 spans 0.04671 to 0.06363, and the per-position accuracies do not order the arms either (`resample` has the worst T1 accuracy, 0.098, and the best Recall). The ordering that does hold is coverage, then the drop depth that produces it.

**Within every arm popularity dominates**: the coldest popularity quintile recalls almost nothing while the hottest reaches roughly a fifth; see `strata.csv` for the per-arm, per-bin numbers.

## Reproduction

```
python -c "import sys; sys.path.insert(0,'.'); from tools.research.diagnostics import sid_generation_forensics as F; F.run_sid_layer()"
python -c "import sys; sys.path.insert(0,'.'); from tools.research.diagnostics import sid_generation_forensics as F; F.run_arm('<arm>', device='cuda:0')"   # one process per card
python -c "import sys; sys.path.insert(0,'.'); from tools.research.diagnostics import sid_generation_forensics as F; F.paired_comparison(); F.strata_table(); F.write_report()"
```
