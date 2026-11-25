# Issue #258 evaluation consumers (frozen before the formal run)

## 1. What is measured and when

| measurement | artifact | time |
|---|---|---|
| real one-epoch entry + behaviour-graph admission | `ADMISSION.json` | before dispatch |
| inclusive 31–60 epoch four-rank performance | `benchmark/` + `PERFORMANCE.json` | before dispatch |
| real Stage2 3000 epochs, legal re-export, collision | `FORMAL_RUN.json`, `FINAL_STAGE2.json` | one run |
| SID chain + role conditioning (`tools/iteration/analysis.py`) | `ANALYSIS.json` | after Stage2 |
| three-code collision / quantisation error / effective rank (`sid_quality.py`) | `FINAL_STAGE2.json` | after Stage2 |
| frozen Stage3 150 epochs, beam 20, seed 42, 57439 test users | `FINAL_STAGE3.json` | one run |
| test metrics recomputed from predictions (`tools/iteration/replay.py`) | `VERDICT.md` | after Stage3 |
| R2 four-piece archive + ledger | `ARCHIVE.json` | after Stage3 |

## 2. Decision rule (fixed, from the R05 contract)

- The mechanism signature is evaluated on the exported SID: distinct L2 prefixes ≤ 6000, L1 carry inside
  `0.0496..0.0696`; failure line distinct L2 prefixes ≥ 7000 or validation Recall@10 ≤ 0.0576.
- The recommendation prediction is validation Recall@10 ≥ 0.0583.
- Stage3 `test_R@10` is the final exploratory evaluation; `> 0.065` is the adoption threshold, but the round is
  reported as a negative result if the predictions fail. No tuning, no second run, no seed sweep.

## 3. Selection-bias disclosure

The validation decision is frozen before the formal Stage3 test is read. The final test is used once, after the
round's decision, and is reported as exploratory: the test split has been used by all earlier arms, so it cannot
serve as an unbiased model-selection signal.

## 4. Known measurement limitation at lock time

The **recommendation validation** metric (validation Recall@10 on the complete `valid.parquet` population) is
produced in this repository only by `tools/issue256/r04/external_validation.py`, which imports
`tools.issue256.r04.identity` (run id `issue256_r04_formal`) and refuses a Stage3 record whose
`stage2_run_id` differs. No generic, identity-parameterised validation entry exists in `tools/iteration/`,
`tools/research/`, `tools/issue252/`, or the Stage3 MCP. `FROZEN` forbids copying or modifying that evaluator.
Consequence: if no legal path appears, validation Recall@10 is recorded as `NOT_AVAILABLE` with this reason and
is not replaced by the test metric or by a fake value. The SID signature, structural checks, collision and the
final test remain measurable. See `INFRA_FAULTS.md`.
