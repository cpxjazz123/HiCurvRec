# S06_IMPLEMENTATION — canonical source packet

STAGE_ID=S06_IMPLEMENTATION
ROUND=1
ITERATION=30

## Objective
Independently design a complete, minimal, executable implementation plan for the S02/S04/S05-approved six-run mapping comparison. Stage2/Stage3 must remain unlaunched. A/B propose plans only; Judge C selects/merges one; the orchestrator applies the approved source patch once. The plan must enforce common treatment implementation, explicit raw-input key handling, fail-closed warm start, exact no-CLI/no-env routes, and all per-run gradient gates.

## Canonical authority and evidence
- Root `CLAUDE.md` controls no-CLI/env parameters, Stage2/Stage3 output roots, each-run Stage2 gradient checks, no Stage2 quality gates, Stage2/3 launch requirements, `.gitignore` staging, and commit/push closure.
- Active `skill://curvature-rqvae-iter` §§2.5, 3, 6–10, 17–19.
- Canonical iter30 S00-S05: `logs/source_snapshot_iter30.md`, `logs/protocol_manifest_iter30.md`, `logs/hypothesis_iter30.md`, `logs/mechanism_manifest_iter30.md`, `logs/mechanism_contract_iter30.json`, `logs/one_factor_diff_iter30.md`, and each corresponding Judge record.
- S05 verdict is conditional prospective single-factor PASS only if S06 implements one common future code/reporting path for both arms; exact registered map/vector is the only scientific arm difference; same explicit inputs, seed pair, warm start, settings, and Stage3 trainer; unique route metadata only. S06 must not treat S05 as proof future source exists.
- Current implementation parent source: `stage2_RQ-VAE/curvature_RQ-VAE_iter29/` (current parent root commit `ecd01e4712a1badd38a0338255f4b2ec7b030aff`). Primary iter29 `curvature_RQ-VAE.py`, `curvature_config.py`, `modules/`, `scripts/`, and Stage3 wrapper pattern should be inspected directly. Current Stage3 trainer SHA256 is S01-locked `9861eb6b2a5b0bfac546524c50d402aba5827e5079a228e2ae5501909bc937cb`; it must not be edited.

## Immutable protocol and exact mapping identities
Three new matched seed pairs: `43`, `44`, `45`; six complete serial Stage2→Stage3 runs. Within each pair, same Stage2 and Stage3 seed; iter26 map is fresh control, iter29 map candidate. Do not pool/replace seed 42 or stop/drop a run due to target, Stage2 proxies, or paired trends. User criterion is per valid test result, inclusive `test_recall@10 >= 0.065`.

Common `[L0,L1,L2]` inputs:
```text
behavior_branching = [19.324911558712664, 1.4605688962651735, 1.0148104414712726]
raw_residual_medians = [1.0, 0.10941, 0.09331]
```
S03 passes only for historical method/value provenance (branching confidence HIGH; raw residual MEDIUM), not replay/current recalculation/historical byte identity. Use explicit plural `raw_residual_medians` only; fail closed if absent or discrepant. Never consume/fallback to ambiguous `residual_norm`, normalized layer scales `[0.001,0.932889,1.0]`, other checkpoints, or inferred values.

**Control — iter26 map** (`std` is population/ddof 0):
```text
m_min=min(m_raw)
s=log1p(B)/log1p(m_raw/m_min)
z=(s-mean(s))/(std_population(s)+1e-12)
c26=clip(0.5*exp(0.2*z),0.05,1.5)
c26=[0.6145357379232853,0.5333020920777128,0.3814078098431606]
```
**Candidate — iter29 map / S04 primary contract**:
```text
x=B/(B+2.0); y=m_raw/(m_raw+0.1); u=(x+y)/2; c29=0.05+1.45*u
c29=[1.3660953164241916,0.7347829661951981,0.6439958072706683]
```
Both fixed, precomputed, non-trainable, time-invariant. The exact 10-field canonical FCCR-1 contract at `logs/mechanism_contract_iter30.json` encodes the candidate vector only; the control vector must be checked explicitly by S07/S08 without adding a second contract.

## Existing locked Stage2/Stage3 settings
Stage2 common: input 768; hidden `[512,256,128]`; embed 32; 3 quantizer layers×256 codes; commitment 1; batch 640/GPU, 4 ranks/global 2560; 100000 steps, checkpoint every 10000; AdamW `1e-3` / `1e-4`; grad clip 1.0; configured Sinkhorn `sk_eps=0.05`/3; behavior loss weight .20 and temperature .07; curvature-reg weight zero; same model, data/sampling, M2/M3, curvature consumers, runtime, deterministic/cache/GPU settings, warm-start, SID cadence/export. Existing effective epsilon `sk_eps*(c/c_cyclic_max)` stays unchanged; its realized value can differ only as a mapped-curvature consequence. All Stage2 quality metrics are descriptive-only.

Stage3: unchanged trainer SHA above; matching seed; configured max 150 epochs preserving existing `NO_EVAL=True`/train-loss patience behavior; `SKIP_TEST=False`; final test mandatory; top K `[5,10]`; beam 20; expected `n_eval=57439`; batch 4096; infer 1024; 4 ranks/port 50201; all model, optimizer, scheduler, precision, determinism, NCCL settings unchanged. Do not repair historical `unknown_variant` metric metadata by editing trainer; use/disclose the caveat.

## Six binding run routes
For each `<label>` the exact result roots are:
```text
results/stage2_RQ-VAE/curvature_RQ-VAE_iter30/<label>/
results/stage3_T5Train/curvature_RQ-VAE_iter30/<label>/
```
The six labels are exactly:
```text
iter26_mapping_seed43, iter29_mapping_seed43,
iter26_mapping_seed44, iter29_mapping_seed44,
iter26_mapping_seed45, iter29_mapping_seed45
```
Stage2 path convention: `RQVAE_OUT_DIR=<Stage2 root>/out/rqvae/instruments/`; checkpoint/raw SIDs inside there; `SIDS_NPY=<Stage2 root>/dataset/Instruments/sids_for_hgrec.npy`; `ITEM_SIDS_JSON=<Stage2 root>/item_sids.json`; `MECHANISM_NAME=iter30_<arm>_seed<seed>`. Stage3 must receive exactly that run's absolute `item_sids.json`, same full variant label, and unique `<Stage3 root>/logs/`, `<Stage3 root>/ckpt/`, launcher and wrapper stdout logs. No overwrite or cross-profile output. All product files remain under results roots, never the source subtree.

The project scripts have no CLI arguments, user env overrides, or shell-injected run selection. Every run selection is baked into a reviewed no-argument profile/script. Stage2 and Stage3 internal torchrun flags remain hardcoded by existing launcher patterns; run serially on ports 50200 and 50201. Each profile/source/config/wrapper hash and full destination inventory must be recorded immediately before launch.

## Blocking implementation defects and required repairs
1. Existing Stage2 warm-start logic has a fail-open path: if the checkpoint is absent it warns and proceeds randomly, without locked SHA verification or guaranteed positive compatible transfer. S06 MUST make it fail closed for all six runs: verify exact absolute checkpoint and locked SHA256 before load; reject missing/unreadable/malformed model state, unexpected/unexplained missing/mismatched tensors, or zero transferred compatible tensors; skip only explicit legacy curvature entries as approved by S01; record a positive compatible tensor count and expected exact missing `_fixed_c` buffer keys. No run may start from random initialization.
2. Preserve the explicit S03 raw input key and semantics; no fallback.
3. All six actual Stage2 runs require their own immediately-prior one-checkpoint/one-batch `loss.backward()` gradient-path check (root `CLAUDE.md` §6): total loss requires grad/non-null grad_fn; finite nonzero gradients in intended model/loss paths; check detach hazards; fixed curvature remains non-trainable. A single block-level check or the S08 MVG alone does not replace per-run checks.
4. Maintain common map/consumer/reporting code for both arms and common root-compliant SID-only descriptive reporting; no HR@50, no quality gate/early stop. The source `curvature_RQ-VAE.py`, `modules/quantize.py`, `modules/rqvae.py` must visibly satisfy FCCR-1 preflight expectations: fixed buffer, no curvature Parameter, `CURVATURE_REG_WEIGHT=0`, time-invariant `get_c`.
5. The Stage3 trainer is frozen. Only no-argument wrapper/profile paths, input, seed, variant label and output destinations vary.
6. Root `.gitignore` globally ignores `.pth`/`.pt`/`.npy`/`ckpt`; the iter30 Stage2 and Stage3 deliverable roots will need narrowly scoped `iter30` exceptions so all required final results are committable. Continue excluding shared `item_emb.npy`, `_ddp_sync`, tensorboard, waits and transient `_stage3_run*.log` per existing later rules.

## Required S06 plan contents
A/B independently provide a complete file-by-file patch plan and observable checks for one common implementation. It must specify how six hardcoded no-arg profiles safely route distinct seeds/maps/paths without CLI/env selectors or arm-specific training code; how the root preflight and fixed config-path rule are satisfied; exact warm-start load validation; explicit input/contract map checks; dual-vector MVG/counterfactual signal; six per-profile gradient checks; six Stage3 wrappers; no-overwrite path inventory; `.gitignore` exceptions; and what throwaway smoke checks prove behavior without GPU. Clearly state edits are not yet applied and no Stage2/Stage3 launch is authorized.

## Candidate independence and execution boundary
A and B receive this same packet and inspect primary code independently. Write only role-specific candidate plans under `logs/deliberation/S06_IMPLEMENTATION/round_1/`. Do not modify source/config/output files or run validation. Judge C adjudicates and writes only `logs/implementation_plan_iter30.md` plus `judge.md`. After Judge approval, the orchestrator applies that canonical patch exactly once; later S07/S08/S09 are independent gates. No training/Stage3/GPU launch at S06.