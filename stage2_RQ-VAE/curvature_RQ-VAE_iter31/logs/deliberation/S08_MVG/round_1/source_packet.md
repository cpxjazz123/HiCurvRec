# Iter31 S08 MVG — Round 1 Candidate Packet

## Stage identity and authorization boundary

- Stage: `S08_MVG`, Iter31 HRA-STEP6-1; parent Iter29.
- S07 passed in `logs/deliberation/S07_PREFLIGHT/round_3/judge.md`. This permits proceeding to S08 adjudication only. No MVG execution is authorized yet; S08 Judge C must separately approve one runtime check before Main runs it once.
- No Stage2/Stage3/GPU training is authorized here. Do not rerun or sweep checkpoints/batches; no candidate executes commands, edits, or tests.
- The two S07 preflights were static only. HRA PASS explicitly says runtime activation is not established.

## Canonical experiment contract

Read S01, S02, S04, S05, S06 canonical Judges and `logs/implementation_plan_iter31.md:87–92`; S07 round-3 Judge; both contracts; actual model/checker source. Binding constraints:

- One approved warm start, one deterministic batch, one candidate state, exact unchanged curvature vector `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`.
- S01 pinned warm-start file: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`; pinned SHA-256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`. Main verified current `sha256sum` exactly matches this value during S08 setup. Do not use any other checkpoint.
- One batch derives deterministically from Iter31 `curvature_config.py`: embeddings `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy`; item IDs `/home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json`; train transitions `/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet`; hard-coded seed 42 and per-GPU batch size 640. Verify required data files before authorized execution, then use only the script's fixed first 640 active source records; no batch search.
- Direct same-quantized-tensor effect: `z_HRA = model._step6_sum_embeddings(output)` versus unchanged Euclidean `embeddings.sum(dim=0).transpose(0,1)`. Report shapes/finiteness/values-norms/max-absolute and L2/relative-L2 differences; no preregistered effect threshold and no promotion from a proxy.
- Record actual source/common-reference exp projection, source/common-reference log-clamp incidence, Möbius denominator/clamp incidence, intermediate/final ball-domain and finite checks. Helper projection/clipping means ideal radial identities are not assumed.
- Check exact fixed curvature values, buffer/non-parameter status, `get_c()` across train/eval and curriculum probes, and exclusion from AdamW parameters without an optimizer step.
- Root `CLAUDE.md` §6 requires `total_loss.requires_grad` and `grad_fn`, actual `loss.backward()` on one checkpoint and one batch, finite nonzero relevant model/codebook gradients, component-wise `torch.autograd.grad(... retain_graph=True, allow_unused=True)` for each applicable mechanism loss, and explicit detach/inactive-path review. Fixed curvature is not expected to receive gradients. S06 says to check every relevant existing mechanism loss component where applicable; independently decide if the existing `behavior_loss`/contrastive component must also be checked in addition to reconstruction and quantizer loss. No HRA-specific auxiliary loss is registered.
- Any report is descriptive runtime gate evidence only, not a Stage2 quality gate; SID metrics are not gates.

## Implementation under review

`stage2_RQ-VAE/curvature_RQ-VAE_iter31/scripts/mvg_check.py` is the sole one-checkpoint/one-batch MVG entry; it requires CUDA and prints results without training/saving model checkpoints. `grad_check.py` delegates to its gradient check. `modules/rqvae.py` implements only the registered common-c0 Step6 aggregation; `forward` uses its output for the existing decoder/reconstruction path. No formula retuning, control curvature, alternate batch/checkpoint, model edit, or repeated run is permitted.

Main has inspected the current MVG and gradient checker source but has not executed it. Candidate A and B independently inspect whether it faithfully implements every registered/root-required observable and whether it is safe to execute once. Pay particular attention to loss-component gradient coverage and exact state/input identity. If source/tooling needs repair, request Judge authorization before edits; do not silently alter source or execute an incomplete gate.

## Candidate report requirements

Agent A and B independently review the identical frozen source packet and primary code. They must return separate recommendations (`AUTHORIZE_ONE_MVG_RUN` or `HOLD_FOR_SCOPED_REPAIR`), evidence/line references, execution risks, and exact single run boundary. Assess loss-component applicability explicitly, including `behavior_loss`; identify any error, missing proof or issue that could invalidate the run. They must not run MVG/checkers, access GPU by other scripts, edit source, or train.

Judge C reads both full reports and primary code, independently adjudicates the one-run authorization or exact repair requirement, writes `logs/deliberation/S08_MVG/round_1/judge.md`, and updates no source unless an explicit bounded repair round is subsequently authorized. If authorized, Main runs `python scripts/mvg_check.py` once from the Iter31 source directory; capture complete stdout/stderr/exit status, exact checkpoint hash, batch size/provenance, and result. A checker PASS does not imply S08 runtime validity; S08 requires the actual output to adjudicate shape/finiteness/domain/clamp/invariance/direct effect and §6 gradients. No downstream stage authorization is implied.