ROLE=AGENT_A
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=stage2_RQ-VAE/curvature_RQ-VAE_iter29/logs/deliberation/S09_STAGE2_EXECUTION/round_1/source_packet.md
STAGE_ID=S09_STAGE2_EXECUTION

# Independent review: bounded Stage2 single-run plan

## Decision boundary

This is a launch plan only: I did not execute commands, recheck hashes, run either gate, start a process, or launch Stage2/Stage3. Agent A does not authorize training. Judge C must first select/materialize the canonical S09 plan; the subsequent pre-Stage2 deliberation gate and the listed hard prelaunch checks must pass before the orchestrator may make exactly one Stage2 invocation. Stage3 is explicitly outside this stage and must not be launched here. (S09 packet: lines 26-32; skill §§2.5, 2.7, 17-18.)

S01 locks iter26 as the immediate parent and sole direct control for a `single_factor` experiment. The sole scientific change is the per-layer FCCR-1 mapping, using the same `[L0,L1,L2]` input vectors and fixed-curvature output; no optimizer, loss, Sinkhorn parameter/rule, model, Stage1, Stage3 trainer/settings, seed, warm-start, or protocol change is authorized. The registered candidate vector is `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]`; S03's input evidence is historical method/value provenance, not checkpoint replay or historical byte-identity proof. (Canonical protocol: `logs/protocol_manifest_iter29.md:4-14,32-49,72-78`; one-factor record: `logs/one_factor_diff_iter29.md:4-10,54-80`; S06 plan: `logs/implementation_plan_iter29.md:17-35,102-114`.)

## Evidence and proposed prelaunch gates

### 1. Deliberation and previously adjudicated verification

- Preserve the final S07 decision and `MECHANISM_CONTRACT_PASS`; preserve the final S08 Judge's `MVG_PASS`. S08 establishes implementation/activation and model gradient health only, not Stage2 or Stage3 efficacy. Its run record reports the S01 checkpoint digest, fixed-vector/invariance evidence, a finite loss with `backward()` and nonzero intended model gradients, and a same-checkpoint/batch counterfactual. The S08 Judge expressly notes that Stage1/Stage0 files were not rehashed for S08 and that no batch fingerprint was emitted; do not represent either as a fresh Stage2 identity check. (S07 Judge: `logs/deliberation/S07_PREFLIGHT/round_2/judge.md:1-20`; S07 output: `logs/preflight_contract_iter29.log:1-9`; S08 Judge: `logs/deliberation/S08_MVG/round_2/judge.md:17-32`; S08 run record: `logs/mvg_check_iter29.log:1-25`.)
- After the S09 Judge decision and canonical `logs/stage2_execution_plan_iter29.md` exist, run the skill-level no-argument `deliberation_gate.py` from the iter29 source directory, and require its actual `DELIBERATION_GATE_PASS` for the pre-Stage2 phase with the exact latest S00-S09 Judge/canonical artifacts. A missing/stale decision, wrong phase, or any other output blocks launch. Do not run this gate before S09 is canonical. This candidate itself has not run it. (S09 packet: lines 26-28; skill §§17-18.)
- Root `CLAUDE.md` §6's applicable gradient-path obligation is supported by the already-adjudicated S08 one-checkpoint/one-batch backward check: total loss is differentiable and intended model parameters receive finite, nonzero gradients. The registered factor is a fixed, non-trainable curvature mapping and adds no mechanism-specific loss; do not require a gradient on the fixed curvature. Treat a contrary S08 record or an additional registered loss/mechanism as a blocker, not as justification to train first. (`CLAUDE.md:27-29`; S08 Judge and run record above.)

### 2. Exact source, configuration, and launch contract

- Use source cwd `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter29`. `curvature_config.py` declares the exact `RQVAE_OUT_DIR` as `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/out/rqvae/instruments`; the same config routes `RAW_SIDS_NPY` inside that directory, `SIDS_NPY` to `.../results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments/sids_for_hgrec.npy`, and `ITEM_SIDS_JSON` to `.../results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json`. The iter29 source tree is not the artifact destination. (`curvature_config.py:9-34`; `CLAUDE.md:3-5,53-55`.)
- As a final path gate, carry out the root-mandated `grep RQVAE_OUT_DIR <iter29>/curvature_config.py` and record its actual result: it must resolve to that exact iter29 `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/` subtree, with no wrong iteration, source-tree output, or descriptive-suffix directory. I have not run this check. (`CLAUDE.md:3-5,53-55`; source packet lines 19-20.)
- After all gates, make the single prescribed no-argument invocation from that cwd: `nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 curvature_RQ-VAE.py > logs/train_run.log 2>&1 &`. Do not append hyperparameter/CLI overrides. The entry point internally forks four `torchrun` ranks using `--standalone`, `--nproc_per_node=4`, `--master_port=50200`; `CUDA_VISIBLE_DEVICES=0,1,2,3`; worker output is captured in `logs/train_migrated.log`. (`CLAUDE.md:7-19`; `curvature_config.py:46-48`; trainer `curvature_RQ-VAE.py:904-946`.)
- Capture one attempt's launch time/PID and both log identities for process supervision. Require worker evidence of world size 4, the expected input paths and data-load success, no unexpected launcher/worker failure, and a single attempt. The launcher is already hard-coded; do not start a second process to compensate for a slow or failed first attempt. (Trainer `curvature_RQ-VAE.py:205-219,531-590,904-946`.)

### 3. Final input-identity gate

Immediately before the sole invocation, rehash and compare these current bytes to the canonical S01 values; a mismatch at any required input blocks the run:

- Stage1 embedding `stage1_GeneEmbedding/output/sentence_t5.npy`: `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`.
- Stage1 item-ID sidecar `stage1_GeneEmbedding/output/item_ids.json`: `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30`.
- Stage0 training table `results/stage0_build_parquet/train.parquet`: `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`.
- Warm-start checkpoint `results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`: `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`.

Record actual hash outputs and paths in the execution evidence; the manifest's expected values and the S08-recorded checkpoint match do not substitute for these final prelaunch checks. Input paths and hashes: `logs/protocol_manifest_iter29.md:19-24,38-39,64-68`; S08's explicit non-rehash limitation: `logs/deliberation/S08_MVG/round_2/judge.md:11-12,23-25`; source packet lines 17-18.

### 4. Fresh output ownership and routing gate

Before launch, inventory and record the state of the iter29 `RQVAE_OUT_DIR`, `sids_for_hgrec.npy`, and `item_sids.json`; do not silently accept, destroy, or overwrite an unexplained prior product. The Step0 cleaner is scoped to files matching its allowlisted patterns **inside `RQVAE_OUT_DIR` only**. `sids_for_hgrec.npy` and `item_sids.json` are outside it and survive Step0, so require them absent at their final targets or explicitly preserve/quarantine and prove their prior identity before making a fresh destination. Block if a stale external export cannot be disambiguated. Record pre-run absence/state and the Step0 cleanup count. (`modules/step_checks.py:94-168`; trainer `curvature_RQ-VAE.py:538-542`; config `curvature_config.py:30-34`; packet lines 22-25.)

All model/checkpoint/raw-SID products must remain in the repository result subtree `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/`; none may be written under `stage2_RQ-VAE/curvature_RQ-VAE_iter29/` source, except allowed code/config/scripts/logs/pycache/input copies. No S09 action may run or create Stage3 products. (`CLAUDE.md:3-5,53-59`; config `curvature_config.py:9-34`.)

## One-run monitoring, no-gate policy, and stop criteria

The run must reach `MAX_GLOBAL_STEPS=100000` with three layers/codebook size 256, seed 42, and 10,000-step checkpoint cadence; preserve the locked iter26 settings, including architecture, batch, AdamW and inherited loss/Sinkhorn settings. The source constants confirm seed 42, three layers/256 codes, 100,000 global steps, 10,000 cadence, per-GPU batch 640; the canonical one-factor record locks the remaining unchanged settings. (`curvature_RQ-VAE.py:99-111,139-145`; `logs/one_factor_diff_iter29.md:72-80,104`.)

During the one attempt, monitor `logs/train_run.log` and the rank-worker `logs/train_migrated.log` through terminal exit. Evidence must include the actual warm-start line `iter29 warm-start loaded ... from /home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth` with a nonzero transferred-tensor count; the trainer has a random-initialization warning/fallback if that checkpoint is missing, which is a failed protocol run, not an acceptable alternative. Also require correct four-rank startup, Step2 inputs, completion at global step 100,000, expected final checkpoint save, no NaN/Inf/backward/optimizer failure, no fixed-curvature contract failure, and a successful process exit. (`curvature_RQ-VAE.py:626-665,743-785,827-902`; protocol `logs/protocol_manifest_iter29.md:32-39`.)

Do **not** early-stop, veto, or decide downstream eligibility from any Stage2 SID/geometry proxy: Gini, collisions, unique SID counts, per-layer utilization, entropy, coarse/fine ratios, or HitRate@50. They are descriptive-only; the codebook warning is not a gate. The only ordinary completion target is the locked 100,000 global steps. Stop only for actual invalid execution or direct contract/protocol failure (e.g. crash/nonzero exit, NaN/Inf or backward/check failure, wrong input/warm-start, curvature drift, checkpoint corruption); once the single attempt starts, any such failure is investigated and recorded without retry/relaunch or in-place retuning. (`CLAUDE.md:11-17`; skill §10; trainer `curvature_RQ-VAE.py:771-785,788-825,827-894`.)

## Required post-run verification and Stage3 boundary

A zero exit or final checkpoint alone is not sufficient. The source raises when final 3-token SID generation fails, but catches the final 4-token export exception and only prints `[Stage3 export FAIL ...]`; therefore successful process exit cannot prove that Stage3-ready exports are fresh or valid. Require all of the following before any later stage:

1. Fresh `rqvae_best.pth` in `RQVAE_OUT_DIR`, written by this attempt and representing `global_step=100000`; fresh final raw `sids_raw.npy` in the same directory. Record post-run path, size, timestamp and SHA-256 for outputs, tied to the recorded pre-run state and this run's logs.
2. Fresh `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/dataset/Instruments/sids_for_hgrec.npy` and `results/stage2_RQ-VAE/curvature_RQ-VAE_iter29/item_sids.json`, both generated after this run started. Require `[export] SID_WIRING_PASS: 4-token extension npy + JSON 校验通过` in `train_migrated.log`; verify the NPY is `N×4`, JSON has the same item coverage/row values, and raw SIDs are the expected `N×3`. Record file hashes/timestamps and explicit consistency evidence. Any missing, partial, stale, mismatched, or unmarked export blocks Stage3, regardless of trainer exit code. (Trainer `curvature_RQ-VAE.py:446-510,827-891`; S09 packet lines 22-25.)
3. Require final training-log evidence that the fixed-curvature invariance checks passed at steps 0, 25,000, 50,000, and 100,000, against the registered candidate values within the implementation's `1e-6` invariant tolerance; a contract-failure marker blocks further execution. (`curvature_RQ-VAE.py:125-131,385-425,771-785`; S08 Judge lines 17-25.)

**Stage3 remains prohibited now**, even if Stage2 and exports pass. Do not execute `scripts/run_stage3_iter29.py` or `stage3_T5Train/train_HG-Rec.py`. First complete and adjudicate S10 Stage2/SID analysis, then S11 Stage3 wiring/evaluation planning and its Judge authorization; Stage3 itself runs once only at its proper later boundary. The wrapper confirms Stage3 would consume the external iter29 `item_sids.json` and write to `results/stage3_T5Train/curvature_RQ-VAE_iter29/`; its existence is routing evidence, not permission to launch. (Skill §§2.5, 2.7, 17-18; `scripts/run_stage3_iter29.py:6-35`; packet lines 26-32.)

## Material risks / self-rejection conditions

- Reject this plan if S00-S09 adjudication/gate sequencing is incomplete; any final input digest or resolved path is wrong; warm-start loading is absent; outputs are stale/ambiguous; or the one-factor/protocol contract has changed.
- The external `item_sids.json`/`sids_for_hgrec.npy` freshness issue is the highest operational risk: Step0 does not clean them, and final-export errors are swallowed after logging. Require the pre-state plus post-run success marker and direct file evidence; do not infer freshness from process exit.
- S08's recorded input hashes are not a final prelaunch recheck, and its batch lacks a fingerprint. Preserve that limitation; rehash the actual Stage2 inputs at the final gate and do not claim S08 demonstrated Stage2 batch identity or efficacy.
- The output could be numerically valid yet fail export/wiring; that is a Stage3 blocker, not a reason to retry the Stage2 run. A negative descriptive proxy is not a failure criterion, and an invalid runtime or registered-contract violation is not repaired by changing the registered mechanism in iter29.

No Stage2 or Stage3 launch is authorized by this artifact; the final authority is Judge C's canonical S09 decision plus all prelaunch gates above.