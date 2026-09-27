# Iter31 Abort — S08 Provenance Gate

```text
STATUS=ITERATION_ABORTED_INFEASIBLE
ITERATION=31
ABORT_STAGE=S08_MVG
ABORT_ROUND=4
PARENT_ITER=29
S08_STATUS=HOLD_FOR_PROVENANCE
STAGE2_LAUNCHED=NO
STAGE3_LAUNCHED=NO
```

## Direct evidence

The one Judge-authorized S08 MVG invocation completed successfully and reported `MVG PASS`. Its raw record supports finite HRA/Euclidean outputs on one `[640,32]` batch, common-ball validity, fixed-curvature invariance, finite/nonzero total and component gradients, and no optimizer step. Those facts are limited to that sole recorded batch; they do not establish downstream performance.

The Round-2 Judge required the sole invocation's actual ordered source/future item IDs and provenance, device and batch shape, checkpoint identity/hash, and complete output. The captured raw output includes the batch shape/count and the pinned checkpoint identity/hash, but does not print the selected IDs or actual runtime device string (`logs/deliberation/S08_MVG/round_2/parent_mvg_results.md:5-27`).

The Round-3 Judge authorized exactly one bounded CPU/data-only reconstruction. It reproduced 640 ordered ID pairs and matching current aggregate counts. The preserved record (`logs/deliberation/S08_MVG/round_3/posthoc_batch_provenance.json`) explicitly states that no pre-run input hashes were captured; its post-hoc hashes identify only current bytes and cannot establish historical byte identity. The IDs remain conditional on current inputs matching those used by the sole MVG invocation. Current source selects logical `cuda:0`, but neither the raw runtime output nor the permitted reconstruction supplies an actual device observation. The reconstruction cannot recover that missing runtime fact.

Independent Agent A and Agent B round-four assessments both confirm the numerical/domain/gradient/invariance checks for the recorded batch and independently conclude that the explicit provenance gate remains unmet. Judge C's canonical decision is `ABORT_ITERATION` (`logs/deliberation/S08_MVG/round_4/judge.md`).

## Why same-iteration repair is invalid

The single-invocation MVG authorization and the round-three decision prohibit another checker/model run, repeat, device probe, or further reconstruction. A new run could not recreate the original invocation's historical input identity or device observation. Treating post-hoc hashes as pre-run identity or source-selected `cuda:0` as runtime capture would weaken the locked S08 evidence standard. Proceeding to S09 would waive a required gate. Therefore Iter31 cannot validly continue without altering its authorized verification protocol.

## Scientific interpretation and unresolved question

This is a protocol/provenance abort, not evidence that HRA is inactive, invalid, negative, neutral, or below target. No Stage2 training, SID export, Stage3 evaluation, or downstream `test_R@10` was run. The HRA Step6 performance hypothesis remains untested; there is no Iter31 performance result to compare with the Iter29 baseline.

## Next-iteration constraints

Resume only from the last appropriate valid parent (Iter29 unless fresh primary evidence establishes otherwise). Do not reuse Iter31's MVG as a valid performance result, retry it in place, or fabricate Stage2/Stage3 artifacts. Any further mechanism must be registered through fresh 2+1 deliberation. Before its sole MVG invocation, its adjudicated capture path must preserve input identity, actual ordered batch IDs/provenance, and actual runtime device as contemporaneous runtime evidence; do not rely on a post-hoc reconstruction to satisfy that gate. Stage2 remains forbidden until the new iteration passes its own S08 and S09 Judge gates and immediate gradient-path requirements.

## Closure

`logs/deliberation/S08_MVG/round_4/judge.md` is the canonical abort decision. Git closure is pending; this file records no commit, push, or remote-hash claim.
