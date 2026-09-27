ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I independently reviewed the complete Round 4 source packet and its named primary evidence. I did not read Agent B's artifact, run/re-run any experiment or code, load/reconstruct data, probe hardware, or use the model/checker.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S08_MVG/round_4/source_packet.md
STAGE_ID=S08_MVG
ROUND=4

# Independent evidence assessment

## Scope and source hierarchy

This review is limited to the sole Iter31 S08 MVG invocation and the one post-hoc CPU/data-only reconstruction explicitly authorized by the Round-3 Judge. Primary evidence reviewed: the complete `round_2/parent_mvg_results.md`; `round_2/judge.md`; `round_3/judge.md`; the complete `round_3/posthoc_batch_provenance.json` (including the ordered pair list and metadata); `round_3/posthoc_reconstruction.md`; current `curvature_RQ-VAE.py`; and current `scripts/mvg_check.py`. I did not inspect Agent B's draft. No evidence here authorizes another checker/model invocation, training, S09, Stage2, or Stage3.

## Observed original-run evidence

The parent raw record reports the authorized no-argument `scripts/mvg_check.py` invocation, exit status 0 and `MVG PASS`. Immediately before it, the three configured input paths were existence/readability checked, and the pinned Iter8 checkpoint SHA-256 matched the Round-2 Judge's required digest (`parent_mvg_results.md:5-12`; `round_2/judge.md:27-36`). These checks establish the recorded pre-run readability and checkpoint identity, not historical identity of the data-input bytes.

For the one recorded batch, raw stdout reports embedding shape `(24587, 768)`, 339,519 transitions, 24,474 active sources, and HRA/Euclidean output shape `[640, 32]`. It does not print actual selected IDs or a runtime device string (`parent_mvg_results.md:16-27`). The numeric, domain, invariance, and gradient evidence is affirmative for that batch:

- `same_quantized_embeddings=True`; HRA and Euclidean outputs are finite; reported norms are 25.257963180541992 and 24.896957397460938. Their max absolute, L2, and relative L2 differences are 0.07198049873113632, 2.018216848373413, and 0.08106279373168945. The recorded threshold is explicitly `none_preregistered; report_only`; the nonzero difference is an observed effect, not a preregistered pass threshold or downstream performance claim.
- Common-ball maximum scaled norm is 0.855758547782898; inner and common ball domain flags are true. All reported exp-map projection counts, log-map clamp counts, and Möbius denominator clamp counts are zero; minimum raw denominators are 0.9828716516494751 and 0.8459279537200928 (`parent_mvg_results.md:23`).
- The closed-form curvature values and live values are reported; buffer and `get_c()` snapshots match through train/eval probes at steps 0, 25,000, 50,000, and 100,000, and `after_gradient_c` matches. Fixed curvature is reported excluded from AdamW and no optimizer step performed (`parent_mvg_results.md:20-22,25-26`). These are the checker-reported probe results, not evidence of a training/update run.
- Total loss is 5.195184230804443; output lists nonzero total gradients for encoder/decoder MLP weights and all three codebook embeddings. It separately lists nonzero component gradients for reconstruction, quantizer, and `behavior_loss` (the latter on encoder MLP weights), and reports `hra_specific_auxiliary_loss=False` (`parent_mvg_results.md:24`). Checker source confirms separate component `autograd.grad` checks for reconstruction, quantizer, and behavior loss, finite/nonzero checks, followed by actual `loss.backward()` and total encoder/codebook gradient checks (`scripts/mvg_check.py:298-374`). Thus the component claims are not inferred solely from total gradients.

The current checker independently corroborates the intended calculation and checks: common-curvature aggregates and outputs are finite, shape checked, in-domain, and compared on the same quantized embeddings (`scripts/mvg_check.py:201-295`); curvature probes and tolerance are explicit (`:142-160`); fixed curvature is compared against expected values and actual selected device in current `main` is logically requested as `torch.device("cuda:0")` and passed to `torch.cuda.set_device` (`:377-399`). Source consistency supports interpretation of the recorded output; it cannot retroactively add missing stdout observations.

## Reconstructed IDs: conditional, not historical capture

The post-hoc record states that its single authorized reconstruction used CPU/data-only access, did not invoke the model/MVG, load the checkpoint, query CUDA/GPU, compute gradients, or mutate inputs (`posthoc_batch_provenance.json:1-7`; `posthoc_reconstruction.md:3`). It records seed 42, batch size 640, ascending active dataset indices, first 640, and one `np.random.choice(targets)` per source in order. It also records the expected shape/counts, dense row-ID mapping, current input hashes, and current source hashes (`posthoc_batch_provenance.json:8-33`). The full JSON contains 640 ordered source/future pairs, beginning at batch position 0 with source 0 → future 7878 and ending at position 639 with source 640 → future 19661 (`:34-4515`).

Current trainer source supports the recipe: `SEED=42` and `BATCH_SIZE=640` (`curvature_RQ-VAE.py:99-100,140`); dataset construction maps dense item IDs to row indices and appends targets in parquet row order (`:155-175`); `__getitem__` selects one target using `np.random.choice` (`:189-192`). The checker selects ascending active indices and passes the first 640 records to collation (`scripts/mvg_check.py:35-50`); main seeds NumPy and selects logical `cuda:0` (`:377-384`). Accordingly, the pair list is a reproducible reconstruction under the *current* configured inputs/source recipe, and it matches the run's aggregate shape/count record. It is not directly captured runtime IDs.

The Round-3 Judge explicitly states no pre-run data hashes were captured and limits post-hoc hashes to identifying currently inspected bytes; the reconstructed IDs are conditional on those bytes matching run-time inputs (`round_3/judge.md:35-39`; `posthoc_reconstruction.md:7`). The JSON itself states the same caveat. No reviewed evidence establishes historical input-byte identity. Therefore the reconstructed list cannot establish that the exact historical batch had those IDs, even though deterministic source logic and counts align.

## Device evidence and original hard gate

Current source expressly requests logical `cuda:0` and calls `torch.cuda.set_device(device)`. That establishes source-selected logical-device intent, not that the sole run's actual runtime device was captured. The original stdout has no device string; physical GPU model/UUID/identity is also absent. The reconstruction was explicitly prohibited from querying a device and cannot recover it (`round_3/judge.md:31,37`; `posthoc_reconstruction.md:9`; JSON `device_caveat`). I make no claim about which runtime or physical device actually executed the original run.

The original Round-2 Judge required capture of complete output including “device and batch shape, actual selected batch size/IDs and provenance, and checkpoint identity/hash” (`round_2/judge.md:34`). The original record supplies checkpoint identity/hash and batch shape/count, but lacks actual selected IDs and device capture. Post-hoc IDs plus current logical-device source selection do not meet that exact runtime-capture gate: the former remain conditional on unverified historical input identity; the latter is not a runtime observation. Round 3 authorized this bounded reconstruction and required fresh review, but did not waive the original gate; it expressly says retain HOLD if exact historical batch identity or adequate device evidence remains unestablished (`round_3/judge.md:37-41`).

## Verdict

VERDICT=HOLD_FOR_PROVENANCE

The sole run's numerical/domain/gradient/fixed-curvature-invariance evidence passes for its reported single batch, and the authorized reconstruction is internally consistent with current deterministic source and aggregate run counts. Nonetheless, the original explicit provenance requirements are not all established: historical batch identity cannot be verified from post-hoc hashes, and actual runtime device was not captured. This is a provenance hold, not a numerical MVG failure. Do not upgrade to `PASS_S08_RESULT` based on deterministic source selection, matching counts, post-hoc hashes, or source-selected logical `cuda:0`.

## Self-rejection conditions

I would reject this assessment's HOLD conclusion only if already-preserved, independently verifiable contemporaneous evidence is produced that establishes both (a) exact identity of the runtime data-input bytes sufficient to validate the reconstructed ordered IDs against the sole run, and (b) the actual runtime device information at the level required by the original Judge. Such evidence must not be inferred from the current hashes/source or obtained by repeating/probing the experiment. Conversely, I would reject any claim in this report that the numeric/domain/gradient/invariance checks passed if primary raw output did not contain the cited values or checker path; those claims are tied to the cited record and source, not independently recomputed.

## Exact safe next action

Judge C should adjudicate this report alongside Agent B's independent report and the same primary evidence. On the evidence reviewed here, retain `HOLD_FOR_PROVENANCE`; record separately that the numeric/runtime checks passed for the recorded batch and that exact historical batch/device evidence remains absent. Do not rerun MVG, repeat reconstruction, load data, probe hardware, perform model/GPU work, train, or authorize S09/Stage2/Stage3. Any possible historical evidence must be assessed as already-preserved evidence in a separately authorized adjudication; absent it, S08 remains held.