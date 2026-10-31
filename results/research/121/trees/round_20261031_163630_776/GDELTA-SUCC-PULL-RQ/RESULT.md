# GDELTA-SUCC-PULL-RQ -- the round-seven result (single-card, NOT comparable)

`test_recall@10 = 0.041017` on one A100, 150 frozen epochs, launched by the
framework in **single-card mode** (`GENREC_SINGLE_GPU=1`, global batch 1024,
launch_mode="single-gpu").

## What ran

| stage | setting | outcome |
| --- | --- | --- |
| admission | machine checks, single-card host | constants PASS, geometry PASS, entry PASS, identifier PASS (24,587 unique 4-token IDs), four_rank PASS *as one rank on one card* |
| Stage 2 | 3000 epochs, 1 x A100, batch/rank 1024 | 1349 s (22.5 min), final_collision 0.356, `item_sids.json` exported |
| Stage 2 metrics | `record_single_sid` before Stage 3 | `stage2_metrics/a3bf77fa....json` |
| Stage 3 | 150 epochs, 1 x A100 | test_recall@5 0.0261, **test_recall@10 0.041017**, test_ndcg@10 0.0235 |

## What this number is and is not

It is a real end-to-end run of the mechanism on the hardware this container has.
It is **not comparable** to the frozen four-card baseline (about 0.0679): the
global batch is 1024 instead of 4096, which is a different optimisation regime,
and the evaluation batch differs with it. The job record carries
`launch_mode: "single-card", comparable: false` for exactly this reason.

So the mechanism's ordering against the other rounds cannot be read from this
number either, on top of the already measured ~0.013 single-run spread.

## The single-card path this round exercised

* `checks.check_four_rank` runs as many ranks as the host has cards and records
  the mode ("single-card host: 1 rank(s) on 1 visible card(s)").
* `research_round_launch` falls back from the ddp4 layout to one card per job when
  four cards are absent, and labels the job `comparable: false`.
* Stage 3 already had the branch (`GENREC_SINGLE_GPU=1` skips the four-rank fork);
  the framework now reaches it instead of refusing.
