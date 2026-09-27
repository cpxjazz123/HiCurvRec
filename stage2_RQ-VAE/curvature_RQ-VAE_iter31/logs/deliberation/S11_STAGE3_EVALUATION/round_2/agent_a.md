ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S11_STAGE3_EVALUATION/round_2/source_packet.md
STAGE_ID=S11_STAGE3_EVALUATION

# Independent audit — corrected Iter31 Stage3 route

## Verdict / proposal

**REPAIR_AND_RERUN; conditionally approve exactly one corrected replacement run after the immediate prelaunch gates below pass.** This is an operational correction authorized by the user, not a seed replication, baseline rerun, or variance study. Do not use the old run for causal classification or promotion. Do not launch if a gate fails; capture the blocker without substituting a route.

The corrected active entrypoint has no active metric-triggered early-stop path under its pinned constants, the fixed 150-epoch loop is structurally unconditional, and the rank-0 test handoff is present. There is, however, a concrete stale-shutdown-file hazard that must be removed from the active shared path before launch while preserving the old file and other overwritten root logs as audit evidence.

## Evidence

### Rules and prior attempt

- Frozen packet lines 11–17 records the user authorization and explicitly supersedes the erroneous round-1 `EARLY_STOP=10` plan/protocol.
- Current root `CLAUDE.md` independently hashes to `5915dd53…58f2`; §5.1 (lines 27–33) requires full configured epochs, prohibits metric/patience/screen exits and final-test skips, and rejects an incomplete run. It also requires a source/hash prelaunch audit. This is now the controlling rule, not the S01 manifest's earlier patience-10 setting.
- The invalid attempt's metrics file independently hashes to `07d743a7…7868`. Its primary records show `num_epochs=150`, train epochs only 1–21, then `event=early_stop`, `trigger=train_loss_patience_exhausted`, `patience=10`. The subsequent `test` row is labeled epoch 150 but follows only 21 training records, so that label does not make it a 150-epoch result. Its `test_final.json` hashes to `cb254e64…5b08` and reports `n_eval=57439`, R@10 `0.021170284997997876`; this is not promotion evidence.
- The stale first attempt's root launcher log independently hashes to `292fadd2…9995`; the outer `_stage3_run.log` is present and empty. The active shared `_ddp_sync/shutdown_signal.txt` currently contains `test_done`. These fixed-path files must be preserved before the corrected launcher overwrites/reuses them, with the shutdown signal removed from its active location so new nonzero ranks cannot mistake stale state for current completion.

### Trainer behavior and DDP route

- `stage3_T5Train/train_HG-Rec.py` independently hashes to packet pin `638b8bb6…51d80`. The all-Python Stage3 scan surfaced no `early_stop`, patience, or screen-failure skip-test implementation in the active tree. The trainer constants are `NUM_EPOCHS=150`, `NO_EVAL=True`, `SKIP_TEST=False`, and `SCREEN_BASELINE_LOG=""`.
- `main()` loops `for epoch in range(1, config["num_epochs"] + 1)` (line 844 onward). No conditional break/return interrupts the loop. Best-train-loss checkpoint selection only updates checkpoint state and saves; it does not terminate training. Validation is disabled by `NO_EVAL=True`. Screen failure is not active with an empty baseline; even if a screen were populated, current code records the outcome and has no training break or test-skip branch from `screen_failed`.
- Final test is conditionally skipped only by `config["skip_test"]`; the pinned constant is false. Rank 0 builds the test dataset, reloads the selected best checkpoint when present, evaluates, writes `test_final.json`, records the test and train-end events, then writes `test_done`. Other ranks wait for the shutdown file after training. Each epoch has a DDP barrier. The four-rank launcher is hard-coded to the prescribed `torchrun`, `--standalone`, port 50201, and devices `0,1,2,3`; it injects the required NCCL settings. This route is structurally consistent with full training plus rank-0 test, provided the stale signal is cleared and runtime checks pass.
- Residual operational risk: nonzero ranks wait for shutdown-file existence with a 30-minute timeout. The final test is rank-0-only, so if evaluation exceeds that window nonzero ranks can time out before rank 0 writes completion. Prelaunch must account for this risk; postrun must verify all ranks/processes completed cleanly. Do not alter source as part of this authorized one-factor duration repair.

### Inputs, split, comparator, and output identity

- The Stage2 completion/S10 primary records establish a completed, non-aborted Stage2 handoff and the JSON schema: 24,587 dense IDs `0..24586`, four integer SID values per item. S10 treats SID quality values as descriptive, not gates.
- Independently recomputed hashes match the packet for the SID JSON (`d4b100f6…16dd7`) and Stage0 `test.parquet` (`5290abf0…bfc`). `CODE_PATH` is the exact absolute Iter31 SID JSON. `_resolve_code_file` tries that absolute path first; actual prelaunch should require it exists and resolves exactly there, with no fallback/substitution.
- The Iter29 comparator file independently hashes to `b07de15e…3662`, with locked R@10 `0.05921064085377531`, `n_eval=57439`; target remains strict `R@10 > 0.065`. The Iter31 test split's expected `n_eval=57439` and metric definitions remain unchanged.
- Trainer `LOG_PATH` and `SAVE_PATH` resolve to the fixed short result root `results/stage3_T5Train/curvature_RQ-VAE_iter31/{logs,ckpt}/`; timestamped run/checkpoint paths are derived beneath those roots. Existing results root is **not absent**: it contains the invalid attempt. A new corrected timestamp directory must be used; never clean, reuse, or overwrite the old timestamped run/checkpoint. Preserve the existing fixed launcher log, outer log, and stale shutdown signal before the launcher overwrites or depends on them.
- The metrics `variant=unknown_variant` is a disclosed metadata limitation, not a SID-selection override: source logs the explicit code path separately and its hardcoded CODE_PATH maps to Iter31's variant. Confirm this pin during final prelaunch checks; do not use the label as evidence of route identity.

## Immediate prelaunch gates (all mandatory)

1. **Preserve first, without deleting evidence.** Hash and move/copy the old root `_stage3_launcher.log`, empty `_stage3_run.log`, and stale `_ddp_sync/shutdown_signal.txt` to durable locations under the first invalid run's audit area (or otherwise immutable retained evidence). Record original/new paths and hashes. Ensure the active shutdown path is absent immediately before launch. Do not move or overwrite the invalid timestamped metrics, test, or checkpoint.
2. **Repeat source/rule gate immediately before launch.** Rehash current `CLAUDE.md`, trainer and every active Python launcher/wrapper in the Stage3 route; confirm no source/config drift. Rescan the whole active `stage3_T5Train/**/*.py` route for any metric/loss/patience/screen early-stop, break/return path, or skip-test condition; inspect shell/launcher/wrapper too. Require `NUM_EPOCHS=150`, `NO_EVAL=True`, `SKIP_TEST=False`, empty screen baseline, no metric-triggered training exit, and no screen-triggered final-test skip. Record file identities and scan outcome in the required Stage3 preflight record.
3. **Verify route and input identity.** Confirm direct no-argument trainer path and working directory; require exact resolved absolute Iter31 `item_sids.json` and current hash, valid dense 24,587-by-4 content and Stage2 completion parity. Require exact Stage0 test path/hash, dataset construction and evaluator still imply `n_eval=57439`; preserve seed 42, batch/model/optimizer/scheduler, beam 20, top-k `[5,10]`, seen-history exclusion and checkpoint semantics. Verify the Iter29 comparator hash and values unchanged.
4. **Verify output isolation.** Confirm no active same-run result directory collision; determine the next timestamped run path will be new. Preserve launcher and outer logs before fixed-path overwrite. Confirm every log/checkpoint/test/DDP sink remains inside the Iter31 short root, with the new run isolated from the prior timestamped artifacts.
5. **Verify runtime/resource/launch prerequisites.** Check the prescribed Python 3.10 and torchrun executables/imports, Python/PyTorch-CUDA runtime, all four target devices and BF16 support, no competing trainer, and port 50201 availability. Confirm launcher-visible devices and required NCCL child settings match root policy. No alternate interpreter/wrapper, CLI argument, environment override, or route is acceptable.
6. **Authorize one invocation only if all checks pass.** Execute the packet's direct, no-argument command once. Do not launch as part of this audit. If any gate is failed, ambiguous, or changed, block before launching and document the exact operational blocker.

Afterward, accept the replacement for S12 only if its own run shows 150 train events in order (epochs 1–150), no early-stop event, final test artifact with `n_eval=57439`, matching checkpoint/run path, and clean rank/supervisor completion. A test event with an epoch field of 150 alone is insufficient.

## Assumptions

- The frozen packet faithfully records the user's stated correction and the previously authorized one replacement run; this audit does not re-open that direction.
- The corrected active trainer is the direct no-argument entrypoint, and no unpinned wrapper participates. This must be reconfirmed at the immediate gate.
- Full configured 150 epochs, rather than exactly 150 under all failure modes, is the intended duration. Crashes and actual invalid execution remain operational failures, not permission for a metric-triggered exit.

## Risks / limitations

- Active stale `test_done` is a hard prelaunch blocker until preserved and removed from the active shared path; the trainer itself does not clear it.
- Fixed root launcher and outer logs would overwrite existing records if not preserved first.
- Rank-0 test plus 30-minute non-rank wait timeout creates a possible rank lifecycle mismatch on unusually long evaluation; verify completion rather than inferring from test artifact existence.
- A new Stage3 outcome, causal classification, or promotion result is unavailable before the authorized replacement completes.

## Self-rejection conditions

Reject this proposal and block the invocation if the corrected source hash or required root rule changes; any active metric-driven stop or test skip remains; constants/input/output/launcher differ from this audit; exact SID or test split identity fails; the result path cannot be isolated; preserved evidence cannot be verified; stale shutdown state remains active; runtime/GPU/process/port gate fails; or a new unexpected factor is needed to make the route run. Never “repair” by using a wrapper, changing protocol/model/data/evaluator, weakening the 150-epoch rule, or rerunning more than the single authorized replacement.
