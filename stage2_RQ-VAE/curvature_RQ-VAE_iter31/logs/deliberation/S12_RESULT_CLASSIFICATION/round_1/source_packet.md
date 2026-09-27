# Iter31 S12 result classification — frozen source packet

STAGE_ID=S12_RESULT_CLASSIFICATION
ROUND=1
ROUND_TYPE=SCIENTIFIC_CLASSIFICATION
STATUS=FROZEN_SOURCE_PACKET
SOURCE_PACKET=logs/deliberation/S12_RESULT_CLASSIFICATION/round_1/source_packet.md
USER_INPUT_REQUIRED=NO

## Objective

Independently classify the completed Iter31 HRA Step6 mechanism effect and promotion status using primary evidence. Separate whether the registered mechanism was active and what its observed downstream effect was from whether the strict project adoption target was met. Do not classify the mechanism as failed solely because R@10 is below the target. Do not use the invalid first Stage3 run as evidence. Do not launch another run, seed, ablation, sensitivity test, or root-cause study.

## Current authority and locked contract

- Root `CLAUDE.md` §5.1 SHA-256: `5915dd53c810fe650b85131f437f171bd79ee10c8a2fd96d30bc46c1d39a58f2`. It now requires completing configured 150 Stage3 epochs without metric-triggered early stop and a final test.
- Active trainer `stage3_T5Train/train_HG-Rec.py` SHA-256: `638b8bb61403f0497d080c792e63c6d07dab32bfaa6fe8fc903a67c158c51d80`.
- Canonical Stage3 evaluation plan `logs/stage3_evaluation_plan_iter31.md` SHA-256: `c2adb593a829a6e841cd051c3b74ed82ca57410b89c136844cf8490891d1600c`; its round-2 addendum supersedes the erroneous round-1 patience-10 language and authorizes at most one corrected run.
- S11 round-2 Judge `logs/deliberation/S11_STAGE3_EVALUATION/round_2/judge.md` SHA-256: `673aa677692275ec12e43d16227ddc54373f1ad55fd9b613652d73852156ca2e`; verdict `REPAIR_AND_RERUN`, same-iteration replacement authorized. Its operational repair record SHA-256 after post-run verification: `97bc1f275afe2960eab42e1bfa054aa7275c5a3fa801c9cfbb6d65073813c66b`.
- Stage3 preflight/acceptance record SHA-256: `dc5b50a547531b2277d1a91b3c1365a06d8e82e7f8767ba3af2fb4efa85c22ba`.
- Canonical Stage2 analysis `logs/sid_geometry_iter31.md` SHA-256: `4bd7d18b10ea88ed896c7f77dce234bda555b5e7374375dd56c0e9e953685da2`. It reports FCCR-1/Stage2 execution valid, S08 MVG PASS, HRA Step6 direct decoder-facing effect on its registered batch, and consistent SID export. Its SID-quality statistics are descriptive only, not gates or proxies for recommendation effect.

Locked Stage3 route: seed 42; 150 epochs; `NO_EVAL=True`; `SKIP_TEST=False`; train/inference batch 4096/1024; beam 20; top-k `[5,10]`; full seen-history exclusion; one final test over Stage0 Instruments split (`n_eval=57439`); fixed short Iter31 result root. The exact Stage0 test split SHA-256 is `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc`. Iter31 SID JSON SHA-256 is `d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7` and Stage2 best checkpoint SHA-256 is `d07953be968ea02cb8a6183935264c2af6315ea5195bd90c40764f39279961b7`.

## Corrected Iter31 primary Stage3 evidence

Run directory: `results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/Amazon_2023_Instruments/Sep-27-2026_21-27-08/`.

- Training metrics SHA-256: `0f706eb1c7daefd1bf9eac04a64d26424ded1c8b6f237bf06c3cb4009af499dd`.
- `test_final.json` SHA-256: `d446c6b622e51c140cfaab1be5a76e3cb6ee56ec318ecb0dd5c4b3ca8c8f3957`.
- `HG_Rec.log` SHA-256: `96f99a05d2f5ff994b28c5ea8fc0e61793c95e11ec572ec2d9de9a02b58d718b`.
- `HG_Rec_best.pth` SHA-256: `f7ab9ea713434559becea8f5827390849f845260dce3a7b908c16b661ba6796d`.
- Fixed root `_stage3_launcher.log` SHA-256: `b8abdf6036f782de7a32e3104d6dfafa915d72ccb618ac9f061a8214941449cd`; outer `_stage3_run.log` SHA-256 is the empty-file digest `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Active DDP shutdown sentinel is `test_done` (SHA-256 `979fc65999a298b412c1eb4a9769b4aeedec816b0f50c415b5598ccb6204fd72`). Hub process `Iter31Stage3Rerun` exited 0 after 38m32s. Final process/GPU inspection found no trainer process and four GPUs free.
- Deterministic metrics parse: `train_start=1`, `train=150`, `ckpt_saved=122`, `test=1`, `train_end=1`; train epochs are exactly 1–150 once and in order; zero `early_stop` events; one epoch-150 test; `n_eval=57439`. `test_final.json` agrees with the single final test event.
- `train_start` pins `num_epochs=150`, `no_eval=true`, batch size 4096, exact Iter31 `code_path`, new timestamped `log_path`/`ckpt_path`, rank 0/world size 4. Its variant metadata is `unknown_variant`, a limitation disclosed in the canonical S11 plan; the actual SID route is identified by the absolute path and input SHA-256 above.

Observed Iter31 final test:

```text
n_eval=57439
R@5=0.03809258517731855
R@10=0.05659917477671965
NDCG@5=0.025285381077977752
NDCG@10=0.031247955601888432
```

## Primary canonical comparator

Iter29 canonical baseline run: `results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/`.

- `test_final.json` SHA-256: `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`.
- `training_metrics.jsonl` SHA-256: `412860c203e20498e17ffd9c7a87d5e39836fcee563f1835f503e1df65a1b88f`.
- Primary metrics were independently parsed: exactly 150 ordered train events; zero early-stop events; one epoch-150 test with `n_eval=57439`; one train-end event. This matches the corrected Iter31 fixed-epoch route and evaluation sample count.

Iter29 final test:

```text
n_eval=57439
R@5=0.03953759640662268
R@10=0.05921064085377531
NDCG@5=0.026252776900288842
NDCG@10=0.03257647953179143
```

Candidate-minus-comparator deltas, computed directly from the two JSON records:

```text
ΔR@5=-0.0014450112293041342
ΔR@10=-0.0026114660770556603  (relative ΔR@10=-4.410467509556015%)
ΔNDCG@5=-0.0009673958223110901
ΔNDCG@10=-0.0013285239299029965
```

Strict promotion target remains `test_R@10 > 0.065`. Iter31 is `0.008400825223280353` below the target; Iter29 is `0.005789359146224693` below. The invalid Iter31 first attempt at timestamp `Sep-27-2026_20-15-25` has explicit train epochs 1–21 followed by `early_stop/train_loss_patience_exhausted`; its metrics and test are noncanonical and must not be used in S12.

## Classification requirements

Use the skill's S12 rubric and root/skill authorities. Separately choose exactly one mechanism status (`PROTOCOL_INVALID`, `PROVENANCE_INVALID`, `CONTRACT_INVALID`, `IMPLEMENTATION_INVALID`, `MECHANISM_INACTIVE`, `ACTIVE_POSITIVE`, `ACTIVE_NEUTRAL`, `ACTIVE_NEGATIVE`, `PIPELINE_INVALID`, `ITERATION_ABORTED_INFEASIBLE`) and one promotion status (`PROMOTION_PASS` or `PROMOTION_FAIL`). Explain the status from the direct Stage2 activation/contract evidence, full-epoch Stage3 run and protocol-compatible baseline. Distinguish measured comparison from causal inference; disclose that this is one locked-seed comparison and does not estimate run-to-run uncertainty. Do not run or recommend a replication/noise-estimation or root-cause iteration. Do not overgeneralize a negative/neutral outcome to the entire mechanism family. The strict target is not negotiable.

Agent A and B must work independently from this exact packet, concurrently, and write only `agent_a.md` / `agent_b.md` under this round. Each must state evidence, classification, assumptions, risks, and self-rejection conditions. Judge C starts only after both artifacts exist, checks primary files/hashes, and materializes canonical `logs/failure_attribution_iter31.md` and `logs/gate_decision_iter31.md`. The result also needs `logs/direction_decision_iter31.md` or an equivalent explicit decision artifact before Git closure. No future iteration is authorized by this packet.

## Execution DAG

PARALLEL_GROUP_1=post-run Iter31 artifact parsing; Iter29 protocol/result audit; S10 and S11 canonical evidence review; source/artifact hashing (all completed concurrently)
PARALLEL_GROUP_2=Agent A and Agent B independent result classification from this frozen packet
SERIAL_DEPENDENCIES=Judge C after both independent candidates; Git closure S13 only after canonical S12 files exist
PARALLEL_EXECUTION=YES
