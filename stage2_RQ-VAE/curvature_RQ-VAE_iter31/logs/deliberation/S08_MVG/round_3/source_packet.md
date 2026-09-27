# Iter31 S08 MVG — Round 3 Runtime-Result Adjudication

## Stage identity, run authorization and canonical boundary

- Stage: `S08_MVG`, round 3; Iter31 HRA-STEP6-1, parent Iter29.
- Round-1 Judge rejected source readiness because component-wise `behavior_loss` gradient coverage was missing; Round-2 Judge approved its exact checker-only repair and authorized exactly one MVG invocation (`logs/deliberation/S08_MVG/round_1/judge.md`, `round_2/judge.md`).
- The single authorized run has now executed once. Complete record is `logs/deliberation/S08_MVG/round_2/parent_mvg_results.md`; re-read the raw output, not only this summary.
- S07 Judge PASS is static-only. S08 Round-2 run authorization permitted only one MVG runtime check; it did not authorize S09, Stage2/Stage3, training or retries.

## Exact run and pre-run evidence

- Command: `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 scripts/mvg_check.py`
- Cwd: `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31`; no args.
- Tool completed successfully (exit status 0), output marker `MVG PASS`.
- Immediately beforehand the three input paths passed file/readable checks, and the pinned Iter8 checkpoint SHA-256 matched `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`.
- Runtime data line reports embedding tensor `(24587,768)`, train transitions `339519`, active sources `24474/24587`. HRA/E output shape is `[640,32]`, showing 640 observations. Source uses seed 42 and deterministically selects `active_indices[:640]` from dataset order, with per-source targets sampled from each source's `next_items` using the seeded NumPy state.
- **Provenance limitation:** captured raw stdout does not list actual source/future item IDs or the actual device string. Source explicitly sets `device=torch.device("cuda:0")` and calls `torch.cuda.set_device(device)`; do not misreport that as a printed device field. Do not fabricate the omitted ID list. Candidates must assess whether the deterministic code + recorded inputs/hash/counts/shape are sufficient, or whether Judge C must separately authorize a post-hoc, data-only reconstruction of this already selected batch's source/future IDs. Such reconstruction must not invoke the model/checker/gradient path, select another batch, train, or rerun MVG.

## Runtime facts to adjudicate from complete raw output

- Same-quantized-tensor HRA vs Euclidean outputs are finite, shape `[640,32]`; HRA norm `25.257963180541992`, Euclidean norm `24.896957397460938`, max absolute difference `0.07198049873113632`, L2 `2.018216848373413`, relative L2 `0.08106279373168945`. The registered threshold is none; this is descriptive direct effect, not a quality gate.
- Intermediate/final finite checks pass; inner/common ball-domain flags are true; common-ball max scaled norm `0.855758547782898` (<1). On this batch no source/common exp projection, source/common log clamp, or Möbius denominator floor clamp occurred; denominator minima are captured in the raw output. These observations are for one batch only.
- Fixed vector from formula remains exact; registered buffer/get_c snapshots equal the live float32 representation across train/eval and curriculum probes; fixed buffers are excluded from AdamW. `optimizer_step_performed=False`.
- Total loss `5.195184230804443`; actual `loss.backward()` reports finite nonzero encoder and codebook gradients. Component-specific nonzero gradients are present for reconstruction, quantizer and behavior_loss; behavior_loss reports encoder MLP weights.
- HRA-specific auxiliary loss remains absent. No optimizer step/save/training occurred.

## Candidate requirements

Agent A and Agent B independently analyze the complete raw output against S08 Round-2 Judge requirements and current primary source. They must adjudicate each numeric/domain/gradient/invariance field, limit conclusions to this one warm-start/640-record batch, and explicitly address omitted actual IDs/device fields. Do not run tools/checkers/MVG, compute a new batch, edit or train. Each candidate returns `PASS_S08_RESULT` or `HOLD_FOR_PROVENANCE`, evidence/line references, direct interpretation, and if needed an exact safe post-hoc reconstruction proposal.

Judge C reads both full reports and raw primary results. It writes `logs/deliberation/S08_MVG/round_3/judge.md` and may (a) pass S08 only if required provenance is sufficient, or (b) authorize only a bounded post-hoc ID reconstruction and require a fresh review/adjudication round. Judge C grants no Stage2/Stage3 permission; the next stage is separately adjudicated S09 only after S08 is canonically resolved.