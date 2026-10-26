# SID tables and their paired Stage 3 recalls

Every SID table reachable in the git history, with the Stage 3 recall of the run that
generated from it. Collected read-only by `collect.py` (re-runnable).

## What is here

| item | count |
|---|---|
| SID files in `sids/` (paired only) | 79 |
| SID contents seen in the history (577 in total) | 577 |
| Stage 3 runs paired to a SID | 92 |
| distinct SIDs that have a recall | 79 |
| recall source files copied to `recalls/` | 92 |

`sid_recall.csv` / `sid_recall.json` hold one row per (SID content, Stage 3 run) pair with
`test_recall@5`, `test_recall@10`, `test_ndcg@5`, `test_ndcg@10` and `n_eval`, taken from the
run's own `train_start` (which names the `code_path` it generated from) and its `test`
event. `sid_inventory.csv` holds one row per **paired** SID content with its prefix structure and
the best recall it ever reached.

The directory keeps **only the paired** SIDs and **only recalls that name a paired SID**:
79 SID files (96 MB) and 92 recall files, and every recall file carries the `paired_sid`
block (the SID's sha256, its file name, its paths and the run's own `code_path`), so the
pairing is verifiable inside the file. The 498 unpaired tables
that the history also contains were pruned; they stay recoverable from the git history, and
`python analysis_sid_recall/collect.py` (without `--paired-only`) writes them again.

## The pairing, and its limits

- A SID is identified by the **sha256 of its content**, so two paths with the same table
  are one entry with both paths listed.
- `%d of %d` pairs point at a SID that only exists in the git history (the fork/hyperbolic
  families were deleted from the working tree by commit 19d0d69f). For those the recorded
  recall comes from the run and the table comes from the history: the two are only
  guaranteed to belong together if the table did not change between the run and the
  deletion. `r8_control` is a known counter-example -- its recovered table and its
  checkpoint disagree on the vocabulary size (784 against 819), so its row must not be
  read as the score of that table.
- The recall values are the runs' own recorded numbers (`n_eval` 57439 in every case).
  Nothing here was re-run or re-derived.

## Highest recalled pairs

| variant | Stage 3 run | R@10 | R@5 | nDCG@10 | SID source |
|---|---|---|---|---|---|
| tiger_sample_crowd_posattract_w3 | Oct-13-2026_09-10-29 | 0.06426 | 0.04239 | 0.03498 | working_tree |
| TIGER_c107_directed_assignment_prefix_information__r8_control | Oct-22-2026_05-01-06 | 0.06375 | 0.04150 | 0.03468 | git_history |
| TIGER_c106_directed_codeword_margin_geometry__r8_control | Oct-22-2026_00-00-42 | 0.06374 | 0.04161 | 0.03467 | git_history |
| tiger_sample_resample | Oct-12-2026_17-28-43 | 0.06374 | 0.04255 | 0.03511 | working_tree |
| tiger_sample_resample | Oct-10-2026_18-16-53 | 0.06363 | 0.04206 | 0.03478 | working_tree |
| tiger_sample_resample_fn_mask | Oct-10-2026_19-03-20 | 0.06356 | 0.04110 | 0.03424 | working_tree |
| tiger_sample_crowd_couser_w3 | Oct-13-2026_08-26-08 | 0.06353 | 0.04189 | 0.03467 | working_tree |
| tiger_sample_crowd_posattract_w3 | Oct-14-2026_22-57-10 | 0.06351 | 0.04116 | 0.03469 | working_tree |
| TIGER_behavior_residual_geometry_transport__r8_control | Oct-21-2026_14-01-57 | 0.06339 | 0.04171 | 0.03454 | git_history |
| TIGER_bootstrap_code_stability__resample | Oct-25-2026_04-19-04 | 0.06335 | 0.04265 | 0.03517 | working_tree |
| tiger_sample_resample_s43 | Oct-10-2026_22-55-05 | 0.06302 | 0.04081 | 0.03410 | working_tree |
| r4_08 | Oct-18-2026_19-46-59 | 0.06295 | 0.04173 | 0.03455 | git_history |
| tiger_sample_resample_l1only | Oct-12-2026_23-59-44 | 0.06283 | 0.04187 | 0.03460 | working_tree |
| TIGER_c107_directed_codeword_tangent_repulsion__r8_control | Oct-22-2026_09-56-09 | 0.06269 | 0.04090 | 0.03394 | git_history |
| tiger_fork_hyper_l1_br_euclid_s42 | Oct-12-2026_09-33-58 | 0.06250 | 0.04069 | 0.03388 | working_tree |
| TIGER_c106_directed_assignment_usage_balance__r8_control | Oct-22-2026_00-56-45 | 0.06241 | 0.04204 | 0.03431 | git_history |
| tiger_sample_resample_s44 | Oct-10-2026_23-43-17 | 0.06238 | 0.04185 | 0.03426 | working_tree |
| TIGER_c1_policy_behavior_directed_soft_code_transition_kl__r8_control | Oct-21-2026_20-54-52 | 0.06226 | 0.04116 | 0.03385 | git_history |
| tiger_fork_fork_radial_corr | Oct-11-2026_09-18-51 | 0.06194 | 0.04077 | 0.03364 | git_history |
| tiger_behavior | Oct-10-2026_16-25-18 | 0.06193 | 0.04072 | 0.03382 | git_history |

## Reproduce

`--paired-only` prunes the output to the paired set (the current state of this directory).

```
python analysis_sid_recall/collect.py
```
