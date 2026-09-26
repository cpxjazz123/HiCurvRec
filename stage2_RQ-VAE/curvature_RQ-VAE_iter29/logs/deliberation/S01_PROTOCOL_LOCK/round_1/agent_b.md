ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=logs/deliberation/S01_PROTOCOL_LOCK/round_1/source_packet.md
STAGE_ID=S01_PROTOCOL_LOCK

# Proposed iter29 Protocol Lock (candidate; not canonical)

## Full proposed manifest values

```text
PROTOCOL_ID = iter29_FCCR1_alt_closed_form_matched_iter26
DATASET_VERSION = Amazon_2023_Instruments / 2026-09 Refactor
DATASET = 24,587 items; Stage3 run records identify 57,439 evaluated users/examples (`n_eval=57439`)
STAGE1_EMBEDDING_PATH = /home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy
STAGE1_EMBEDDING_SHA256 = 6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb
STAGE1_ITEM_ID_SIDECAR = /home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json
STAGE1_ITEM_ID_SIDECAR_SHA256 = 3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30
PARENT_ITER = iter26
PARENT_COMMIT = 612a5a41dfe524205b6afa46370ad0d0ce377882
EXPERIMENT_TYPE = single_factor
CANONICAL_BASELINE_ITER = iter18 (historical headline baseline; HISTORICAL_NONCOMPARABLE for direct ranking)
CANONICAL_BASELINE_TEST_FINAL = /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json
CANONICAL_BASELINE_TEST_RECALL_AT_10 = 0.05988962203380978
DIRECT_CONTROL_ITER = iter26
DIRECT_CONTROL_TEST_FINAL = /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter26/logs/Amazon_2023_Instruments/Sep-26-2026_18-11-20/test_final.json
DIRECT_CONTROL_TEST_RECALL_AT_10 = 0.057017009349048554
STAGE2_SEED = 42
STAGE2_MAX_GLOBAL_STEPS = 100000
RQ_LAYERS = 3
CODEBOOK_SIZE_PER_LAYER = 256
STAGE2_POLICY = no Stage2 SID-quality gates; all metrics descriptive
STAGE3_CODE_COMMIT = 612a5a41dfe524205b6afa46370ad0d0ce377882
STAGE3_SEED = 42
STAGE3_EPOCHS = 150
BEAM_SIZE = 20
N_EVAL = 57439
USER_SUCCESS_CRITERION = test_recall@10 >= 0.065 (not guaranteed)
```

The mechanism equation/values are intentionally not proposed here. S14 requires one bounded alternative FCCR-1 mapping, but its choice and numerical outputs belong to later adjudicated stages, not this protocol reconstruction.

## Comparability decision

**Direct ranking / causal control: iter26 only.** Iter29 must keep the Stage1 artifacts, initialization/warm start, data, Stage2/Stage3 configuration, evaluation, and seed policy fixed to iter26. Iter26's exact Stage3 result is `0.057017009349048554` with `n_eval=57439`, at the path above. It is the recent FCCR-1 fixed-curvature parent/control identified by the approved snapshot and S14 action. The exact parent commit for iter29 is the packet-specified current post-S14 HEAD `612a5a41dfe524205b6afa46370ad0d0ce377882`; the Stage3 subtree compares cleanly (no tracked diffs) from iter26's recorded Stage3 code commit `0052f4bff117e0aa16c9a1080be1014fbd417cc6` through iter28's `bdcbbf97bd92ef6b5f0ab016cdfbae2b62c5d73f` and parent HEAD. Thus pinning Stage3 to the parent commit holds its code fixed.

**Iter18 explicit decision: `HISTORICAL_NONCOMPARABLE` for direct ranking in this lock.** Its exact output exists and its result is recorded as the canonical historical headline baseline: `test_recall@10=0.05988962203380978`, `n_eval=57439`. Primary records support that it completed 150 Stage3 epochs, used beam 20 and the iter18 SID input, and ran with the same Stage3 tracked source tree as iter26; `stage3_launch_iter18.log`, `stage3_outcome_iter18.md`, its wrapper, exact `test_final.json`, and the direct Git subtree comparison provide evidence. But no iter18 `protocol_manifest_iter18.md` exists at the expected path, and its historical records do not establish the complete seed/data/Stage2 protocol equivalence to iter26. Same `n_eval`, matching Stage3 source, and iter26's retrospective compatibility declaration are insufficient to reconstruct all missing fields under the controlling rule. It may be shown as context, not used as the direct comparator or called a current-best protocol-compatible score in this S01 lock.

This conservative classification does not erase the historical baseline identity or value. It cleanly separates the exact score reference (`CANONICAL_BASELINE_ITER=iter18`) from the direct experimental control (`DIRECT_CONTROL_ITER=iter26`). Iter25 likewise remains `HISTORICAL_NONCOMPARABLE` because its registered protocol manifest is missing. Iter28 is not FCCR-1 evidence (CAO-1/SREMA with cyclic, learnable curvature) even though its Stage3 comparison to iter18 is recorded as compatible within its own declared protocol. Iter27 was aborted before Stage2/Stage3 and supplies no result.

## Supporting primary evidence

- `logs/source_snapshot_iter29.md` identifies iter26 as the latest directly registered FCCR-1 experiment, confirms the exact Stage1 SHA-256 values above, exact iter26 result, S14 baseline-reconciliation requirement, and user threshold `>=0.065`.
- `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/protocol_manifest_iter26.md` specifies Amazon_2023_Instruments / 2026-09 Refactor, Stage2 seed 42, 100,000 steps, 3 layers × 256 codes, Stage3 seed 42, 150 epochs, beam 20, `n_eval=57439`, and its declared iter18 baseline.
- `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/mechanism_contract_iter26.json` confirms FCCR-1 fixed closed-form, non-trainable/time-invariant curvature and no schedule/regularizer/new optimizer/new auxiliary loss. Its exact registered values are `[0.6145357379232853, 0.5333020920777128, 0.3814078098431606]`; this is parent context, not an iter29 mechanism proposal.
- `stage2_RQ-VAE/curvature_RQ-VAE_iter26/logs/hypothesis_iter26.md` records the input provenance and implementation invariants. `failure_attribution_iter26.md` reports those invariance/activation checks passed and Stage3 wiring/completion; its causal interpretation is not adopted here.
- Exact iter26 JSON at the direct-control path reports `test_recall@10=0.057017009349048554`, `n_eval=57439`.
- Exact iter18 JSON reports `test_recall@10=0.05988962203380978`, `n_eval=57439`. Its launch audit and outcome record confirm beam-20 final evaluation and all 150 epochs; wrapper pins iter18 SIDs and output paths. Missing complete protocol manifest remains decisive for this lock's ranking classification.
- Direct SHA-256 verification of the present Stage1 embedding and item-ID sidecar matched the packet values. Direct `git diff --quiet` checks of `stage3_T5Train` across iter18 pre-run commit `112042ceae844e08ffa0fd4cb7dc55afbad5aaf6`, iter26 protocol's Stage3 commit, iter28 commit, and parent HEAD reported no tracked subtree differences.
- `logs/global_review_after_iter28.md` and its Judge decision require an alternative bounded fixed mapping against iter26 and explicitly require retaining other settings/seed policy, while reconciling baseline bookkeeping. They do not authorize a mechanism choice in S01.

## Fixed-versus-variable protocol and comparability limits

- The only intended conceptual change in iter29 is the separately registered alternative mapping under FCCR-1. Keep fixed: Stage1 embedding and ID sidecar; source data; parent initialization/warm-start behavior; Stage2 random seed and max steps; layer/codebook dimensions; all existing losses, optimizer, assignment/Sinkhorn, trainer, and export behavior; Stage3 source/configuration, seed, epochs, beam, and evaluation set/count. This is a single-factor experiment, not a multi-arm replication.
- Use iter26 as the single direct ranking control. Report iter18 only as the canonical historical score with the `HISTORICAL_NONCOMPARABLE` label; never compute or describe an iter29-vs-iter18 delta as a direct protocol-valid effect.
- Dataset/version string is grounded in iter26 protocol and source snapshot. The available protocol packet provides current Stage0 parquet hashes but not their paths; I do not invent paths or assert independently verified parquet identities here. Before training, the executor should match actual data paths/hashes to iter26's recorded run inputs; a discrepancy must block claiming an unchanged dataset.
- Stage3 source-tree equality across recorded commits was checked directly, but this does not prove all untracked/runtime/config inputs or random streams identical. The iter29 run must pin its code/config and record exact source commit, resolved config, input paths, seed, checkpoint, and final result.
- Results are stochastic single-run observations; a score reaching the threshold is not guaranteed by this lock. The user criterion is inclusive (`>= 0.065`); this task-level criterion controls over lower-priority strict `>0.065` phrasing in earlier protocol text.

## Assumptions

1. Packet-specified parent HEAD `612a5a41dfe524205b6afa46370ad0d0ce377882` is the authoritative starting revision; later evidence does not alter that parent selection.
2. The Stage1 byte identities verified at this point are the inputs intended for iter29; the run must recheck before execution if files can change.
3. Holding the complete iter26 run protocol fixed is feasible and does not require modifying any condition besides the not-yet-selected mapping.
4. The exact `test_final.json` values are the reported terminal evaluations; no metric normalization or rounding replaces those raw values.

## Risks

- If any Stage1/data/checkpoint/config/seed identity differs from iter26, the proposed direct comparison is compromised; record the discrepancy and do not claim protocol-compatible ranking.
- Reusing iter26's protocol does not guarantee `test_recall@10 >= 0.065`; one valid negative result remains scientifically informative.
- Historical claims of iter18 comparability can tempt downstream workers to rank against its larger metric; keep the explicit label adjacent to its path and value.
- Iter26's one-run result cannot estimate run-to-run noise or support family-wide claims. Avoid interpreting a small score delta as a discovery.
- Stage2 SID quality metrics are descriptive, not stop/adoption gates; only valid completed Stage3 evidence evaluates downstream success.

## Self-rejection criteria

Reject this candidate lock or amend before Stage2 if primary evidence shows any of the following: (1) `612a5a41dfe524205b6afa46370ad0d0ce377882` is not the actual authorized iter29 parent; (2) verified Stage1 or training data identity differs from the iter26 control; (3) seed 42, 100,000 Stage2 steps, 3×256 quantization, Stage3 seed 42/150 epochs/beam 20/`n_eval=57439`, or unchanged Stage3 code cannot be preserved; (4) Stage3 subtree diff/source pin is not clean in the actual launch tree; (5) a primary iter18 protocol record establishes all required fields and direct protocol compatibility—in that case revise its classification based on that record rather than perpetuating the conservative historical label; or (6) the final plan introduces another changed factor beyond the separately adjudicated mapping. Do not repair any failed condition by silently changing the protocol.

## Concrete autonomous next action

Judge C should adjudicate the A/B candidates against this exact packet and verify any disputed primary facts. On an accepted candidate, materialize one canonical `logs/protocol_manifest_iter29.md` declaring iter26 as the direct control, preserving iter18's exact score/path only as `HISTORICAL_NONCOMPARABLE`, and pinning the protocol values above. Then proceed autonomously to S02 for independent FCCR-1 alternative-mapping proposals. No Stage2 run or mechanism selection is authorized by this candidate alone; before launch, all required stages/preflights and the root-CLAUDE one-checkpoint/one-batch nonzero-gradient-path check must pass. If the complete iter26 control cannot be held fixed, record the exact protocol break and do not claim a direct ranking.
