# Iter31 S08 MVG — Round 4 Provenance Adjudication

## Stage identity and boundary

- Stage: `S08_MVG`, round 4; Iter31 HRA-STEP6-1, parent Iter29.
- This review was explicitly required by `round_3/judge.md` after its single authorized post-hoc reconstruction. The round-3 Judge authorized no S09, Stage2, Stage3, GPU/model work, training, checker/MVG repetition, or other experiment.
- Agent A and Agent B must independently assess the same complete evidence. They must not read each other's draft. Judge C adjudicates only after both reports exist.

## Primary evidence to inspect

1. Original sole MVG output: `logs/deliberation/S08_MVG/round_2/parent_mvg_results.md` (read in full; do not rely on summaries alone).
2. Original S08 authorization and result requirements: `logs/deliberation/S08_MVG/round_2/judge.md`.
3. Round-3 reconstruction authorization and explicit limits: `logs/deliberation/S08_MVG/round_3/judge.md`, especially `POSTHOC_ACTION`, `HISTORICAL_IDENTITY_LIMIT`, `FRESH_REVIEW_REQUIRED`, and `NO_DOWNSTREAM_AUTHORIZATION`.
4. Complete one-time post-hoc reconstruction: `logs/deliberation/S08_MVG/round_3/posthoc_batch_provenance.json`; method and caveats: `logs/deliberation/S08_MVG/round_3/posthoc_reconstruction.md`.
5. Current frozen source for logical-device and deterministic-batch-selection intent: `curvature_RQ-VAE.py` and `scripts/mvg_check.py`. Source identity SHA-256 values are recorded in the JSON; these are post-hoc current-source identities, not pre-run hashes.

## Reconstruction facts and limits

The Judge-authorized CPU/data-only script ran exactly once with `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 /tmp/s08_iter31_provenance.py` (tool-reported elapsed time 2.76 s). It inspected only the configured embedding file's memory-mapped shape header, item-ID JSON, training parquet, and current frozen source text/constants. It did not load embedding values; import/call the trainer, model, or MVG checker; load a checkpoint; query CUDA/GPU; execute a forward/backward/gradient; mutate data; or train.

It reproduced the current configured deterministic recipe: seed 42; exact parquet row order and dataset item mapping; ascending active source rows; first 640 active rows; one `np.random.choice(targets)` per source in order. Counts match the original runtime record: embedding shape `[24587, 768]`, 339519 transitions, 24474 active sources, 640 selected records (the runtime tensor shape was `[640, 32]`). The JSON contains all ordered source/future row indices and item IDs plus SHA-256 hashes of all three configured data inputs and both current source files.

**Historical identity is not established.** The original run captured no pre-run data hashes. Current post-hoc hashes identify only bytes inspected after the run; they cannot prove those were the original runtime bytes. Therefore the reconstructed IDs are conditional on the current inputs matching the bytes used during the sole MVG run. Do not call them directly captured runtime IDs or overstate provenance.

**Runtime device is not established.** Current checker source requests logical `cuda:0`; the original MVG output did not print an actual device string or physical GPU identity. The CPU/data-only reconstruction was prohibited from querying a device and cannot recover that missing runtime observation.

## Result facts to adjudicate from the full original output

The round-2 source packet records, and candidates must verify against the full raw result: fixed curvature/probe invariance; finite HRA and Euclidean outputs with shape `[640,32]`; direct HRA-vs-Euclidean output differences (report-only, no registered threshold); common-ball and denominator/clamp observations; total loss and finite/nonzero total and component gradients including `behavior_loss`; and no optimizer step or training. Limit all conclusions to the sole warm-start checkpoint and one 640-record batch. An MVG implementation pass does not imply downstream performance improvement.

## Candidate requirements

Each candidate artifact begins with the required role, independence declaration, source packet, and stage ID. Independently verify the numerical/domain/gradient/invariance claims against primary output and source. Then decide whether the post-hoc conditional ID reconstruction and source-selected logical-device evidence satisfy the explicit round-2 provenance requirements. State clearly that the missing pre-run hashes and missing runtime-device capture are not recoverable by this reconstruction. Return `PASS_S08_RESULT` only if all required evidence is actually established; otherwise return `HOLD_FOR_PROVENANCE`, identify the unmet requirement, and state the exact safe next action. Do not run tools that execute or re-execute experiment code; do not load data or select a batch; do not edit source or authorize downstream work.

## Judge C requirements

Judge C reads both complete candidate reports and the primary raw output, then writes `judge.md`. Adjudicate hard gates before preference. Explicitly decide whether S08 passes or remains held; specify the autonomous next action. No Stage2/Stage3 or further GPU/model/checker execution is authorized by this packet. If exact historical batch identity or adequate runtime-device evidence remains unestablished, keep S08 on hold; do not upgrade automatically.
