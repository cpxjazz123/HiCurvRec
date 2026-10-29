# SID tables and their paired Stage 3 recalls

Every SID table reachable in the git history, with the Stage 3 recall of the run that
generated from it. Collected read-only by `collect.py` (re-runnable).

## What is here

| item | count |
|---|---|
| **SID + recall pairs (the collection)** | **97** |
| SID files in `sids/` | 97 |
| recall source files copied to `recalls/` | 97 |
| Stage 3 runs considered | 225 |
| runs of a confirmed table that are not the pair | 2 (`excluded/sid_recall_alternatives.csv`) |
| runs with no fingerprint for their table | 101 (`excluded/sid_recall_unverified.csv`) |
| runs whose own fingerprint refutes the table | 10 (`excluded/sid_recall_disputed.csv`) |
| runs whose table exists nowhere | 15 (`excluded/stage3_runs_without_sid.csv`) |
| SID contents seen in the history | 598 |

**The collection is the 97 confirmed pairs and nothing else.** Every pair in
`sid_recall.csv` has the run's own `vocab_size` matching the paired table's token
range, so the table is the one that run read. The runs that cannot be confirmed
are parked in `excluded/`: they are evidence, not conclusions, and they are not
part of the collection.

`sid_recall.csv` / `sid_recall.json` hold one row per (SID content, Stage 3 run) pair with
`test_recall@5`, `test_recall@10`, `test_ndcg@5`, `test_ndcg@10` and `n_eval`, taken from the
run's own `train_start` (which names the `code_path` it generated from) and its `test`
event. `sid_inventory.csv` holds one row per **paired** SID content with its prefix structure and
the best recall it ever reached.

The directory keeps **only the paired** SIDs and **only recalls that name a paired SID**:
178 SID files and 178 recall files — **one SID paired with exactly one recall** — and every
recall file carries the `paired_sid` block
(the SID's sha256, its file name, its paths and the run's own `code_path`), so the pairing
is verifiable inside the file.

When several runs generated from the same table, the **highest recall is kept** as that
table's pair and the other runs are written to `sid_recall_alternatives.csv`; the paired row
records how many runs shared the table (`sid_runs`) and their recall range. A run is listed
in exactly one of the two tables, and a table is stored under exactly one file name: the
SID is its content, so two path names for the same table are one SID, not two.

**A SID path is overwritten between runs.** `results/stage2_RQ-VAE/<arm>/item_sids.json`
holds a different table after every Stage 2 run of that arm, so pairing a run with the
*current* file attributes it to a table it never generated from. Each run is therefore
paired with the version of its path **as of the run's date**, taken from the history
(`sid_resolution = at_run_date`), with two fallbacks: the working tree for a path that
still exists (`working_tree`) and the earliest surviving version for a path committed
after the run (`earliest_surviving`).

That correction is why the numbers are much smaller than a naive path-based pairing:
127 of the 225 Stage 3 runs used a table that exists neither in the working tree nor in the
history, so they cannot be paired at all -- their recalls are listed separately in
`stage3_runs_without_sid.csv` (the best of them is 0.064068) rather than being attributed
to a table they did not use.

144 of the 178 pairs resolve through the `earliest_surviving` fallback and 34 through
`at_run_date`: the table was committed after the run, so the file is the only surviving copy
of it but not provably the exact version the run read. Those rows say so in
`sid_resolution`. The unpaired tables that the history also contains were pruned; they stay
recoverable from the git history, and `python analysis_sid_recall/collect.py` (without
`--paired-only`) writes them again.

**A recorded path may name another root.** The trainer resolves its root from the file it
lives in, so a run records `/home/.../results/stage2_RQ-VAE/<arm>/item_sids.json` while the
repository may sit at `/fs04/...`. `git show` needs a repository-relative path and silently
fails on an absolute one, so every such run used to fall through to the working tree -- and
therefore to a table it never read. The path is now mapped to its repository-relative form
before the history is consulted (`repo_relative`); with that fix the pairs rose from 67 to
181, and the two BRad-RQ runs now pair with the tables they actually read
(`266ce952...` -> 0.044604 and `cfc3f291...` -> 0.030606, both `at_run_date`).

## How strong each pair is

Every pair carries a `pair_confidence`:

| value | count | meaning |
|---|---|---|
| `verified_digest` | 0 | the run recorded the sha256 of the SID file it read; the pair is a fact |
| `verified_vocabulary` | 97 | the run's own configuration recorded the `vocab_size` it built from the SID table, and the paired table reproduces it exactly. Stage 3's vocabulary is a function of the table's highest token, so this is the run's own fingerprint of the table it read. |
| `date_resolved_only` | 0 in the collection | the run left no configuration log; such pairs are parked in `excluded/` and are not counted |
| disputed (in `excluded/sid_recall_disputed.csv`) | 10 | the run's recorded `vocab_size` does NOT match the paired table's token range: the run read a different table, and the pair is refused rather than reported. |

The gate is one-sided: a matching fingerprint does not prove the table is the one
read (a different table with the same highest token would pass), but a *mismatching*
one disproves it, and those runs are listed separately with both numbers. Ten runs
were refused this way, including two that the date filter had called `at_run_date`:
the file had been overwritten by a later Stage 2 run that was never committed.

The recall side is checked for every pair: the paired value is the run's own
recorded test number, read from the run's own metrics (working tree or history),
and all of them match it exactly.

## Q_H of these pairs

`results/research/issue126/correlation_report_confirmed.json` scores every one of
the 97 pairs with the Issue #126 hyperbolic prefix ranking score (frozen
reference, held-out validation probe) and reports the association with its own
sample size. On these 97 pairs: Spearman **+0.634** (p = 3.2e-12), bootstrap 95%
CI [+0.471, +0.765], leave-one-out [+0.622, +0.671]; bucket count and same-bucket
ratio alone reach R^2 0.112 against recall, and adding Q_H reaches 0.472.
`tools/research/science/q_h_correlation.py` recomputes it; each table's record is
in `results/research/issue126/metrics/<sha256>.json`.

The association is observational: the tables come from different families and
settings, Q_H scores the LEVEL-1 prefix only, and it is measured against a frozen
external behaviour reference rather than the candidate's own codebook geometry. It
says the grouping agrees with the behaviour reference, not that the hyperbolic
quantiser is necessary, and not that optimising Q_H would raise recall.

## The pairing, and its limits

- A SID is identified by the **sha256 of its content**, so two paths with the same table
  are one entry with both paths listed.
- 146 of the 181 pairs point at a SID that only exists in the git history (the fork/hyperbolic
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
