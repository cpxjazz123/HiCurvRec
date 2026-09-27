# Iter31 canonical Stage2 execution plan — S09

## Authorization state

- `S09_VERDICT=MERGE_AB` (`logs/deliberation/S09_STAGE2_EXECUTION/round_1/judge.md`).
- `STAGE2_AUTHORIZATION=ONE_RUN_CONDITIONAL_AFTER_IMMEDIATE_RECHECKS`.
- `LAUNCH_AUTHORIZED_NOW=NO`. The input/source identities, runtime and result destinations are packet-time evidence and have not been re-established as immediate prelaunch checks in this adjudication. Do not launch until every check below is recorded PASS and the deliberation gate reports `DELIBERATION_GATE_PASS`.
- Any failed/missing check means **do not launch**. There is no random-init fallback, automatic destination cleanup approval, retry, or user decision point; resolve through same-iteration operational repair/adjudication while keeping the registered mechanism and protocol unchanged.
- Scope is one Iter31 Stage2 training execution only. Stage3 is not authorized by S09.

## Registered run to execute once, if all checks pass

The registered run remains the Iter31 HRA-STEP6-1 / FCCR-1 experiment. Preserve current inputs, mechanism, equation, fixed-curvature contract and values, seed, topology, batch, and output routes. Current source specifies seed 42, 100,000 global steps, three codebook layers of size 256, per-rank batch 640 (global batch 2,560 at four ranks), checkpoint interval 10,000 steps, and no compile. The launcher uses the configured Python environment's torchrun, `--standalone`, four processes, port 50200, and writes its inner output to `logs/train_migrated.log`. Run to global step 100,000; do not early-stop or gate on SID/occupancy/entropy/collision metrics. Those metrics are descriptive only. Only actual execution/contract failures can invalidate the run.

Use the root-prescribed entry point exactly, with no CLI arguments, environment overrides, alternate interpreter, altered torchrun flags, or alternate code path. Working directory is `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31`:

```bash
nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 curvature_RQ-VAE.py > logs/train_run.log 2>&1 &
```

This is the only run authorized by this plan. Capture the exact command and CWD, launch/supervisor status, and both outer (`logs/train_run.log`) and inner (`logs/train_migrated.log`) logs. Do not launch from this adjudication.

## Blocking immediate prelaunch checks

Record command/output, resolved path, and PASS/FAIL for every item; the operator proceeds only when all pass.

1. **Deliberation gate.** From the Iter31 source directory, execute the required checker with the designated interpreter and no checker arguments:
   ```bash
   /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py
   ```
   Require `DELIBERATION_GATE_PASS`; abort launch on any other result. This adjudication does not assert that the gate has already been run.

2. **Fresh input identity and consumers.** Recompute resolved path, byte size, mtime and SHA-256 immediately before launch for each Stage0/Stage1 input and warm-start below; confirm current config and actual loader consumers still point to these same files. Values must match the frozen source packet, not only the historical S01 declaration:

   | Input | Expected resolved path | Expected bytes | Expected SHA-256 |
   |---|---|---:|---|
   | Stage0 train | `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet` | 11,152,937 | `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815` |
   | Stage0 valid | `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/valid.parquet` | 2,317,087 | `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9` |
   | Stage0 test | `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/test.parquet` | 2,546,378 | `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc` |
   | Stage0 items | `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/items.parquet` | 10,968,789 | `6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3` |
   | Stage1 embedding | `/fs04/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy` | 75,531,392 | `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` |
   | Stage1 item-ID sidecar | `/fs04/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json` | 161,001 | `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30` |
   | Iter8 warm-start | `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth` | 13,780,725 | `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b` |

   Reconfirm embedding shape `[24587, 768]`, finite values, dense ordered sidecar IDs `0..24586`, Stage0 item ordering agreement, and valid transition IDs in the configured train parquet. A mismatch or missing input blocks launch; no alternate input substitution is permitted.

3. **Current source/config identity and FCCR-1 continuity.** Rehash each source file immediately before launch. Each SHA-256 must equal this S09-reviewed/S08-validated source identity:

   | File | SHA-256 |
   |---|---|
   | `curvature_config.py` | `6f2f645202417118baec3ef28af879c14bda66723b42950d5b67056c2ed735bb` |
   | `curvature_RQ-VAE.py` | `b641d8b5c9897f520482d0bce5526bdcd687ae5845901f8f0be238772f49d1dd` |
   | `modules/rqvae.py` | `cf878675aa4550f96b4b13480a1f9d03928549ebd0a720675cbd6f1713e28e21` |
   | `modules/quantize.py` | `2970ad3e61de79c6a2fd17194258d7d5ed4b9cbbf836e5dde506a8635c2553f8` |
   | `modules/step_checks.py` | `03afd7f17600043c9e577e214659890571847e4de389add70eb75a3ee16caf40` |
   | `modules/sid_quality.py` | `3387d7ad2211a923b5376b5606b66411d8fbc89469baa1274f906244a5923511` |
   | `scripts/mvg_check.py` | `abf4ad518890e345c05e8b0907349067faa94cd0ef39d6dab6384bf20792318d` |
   | `logs/mvg_check_iter31.log` | `49318b59588ce9339e2173956733bf7b699fac4d5d113b3921bb77e7e8b59031` |
   | `logs/preflight_contract_iter31.log` | `e52df0e6db569c3d747ba98f093487c1a755bd991486868c397d74d2b54c7e58` |

   **Active HRA/FCCR dependency closure.** Hash these additional files immediately before launch too; equality with each verified identity is mandatory. These files participate in the active Step6/model/data/curvature path or define its registered contracts and were omitted from the initial source identity table:

   | File | SHA-256 |
   |---|---|
   | `modules/hyperbolic.py` (Step6 exp/log/Möbius helpers) | `ca7d4fdb5b7f6d4f91cc6e5083462fa9d610386ce4bda65605d3672907e22ae6` |
   | `modules/loss.py` | `c5d0106b731ee54b9e96ef40db745f7ad74725e2f76ed9c6b763e77e11108cd1` |
   | `modules/encoder.py` | `139a2cfa9fad4ecf9b795d435a640a927dd01a2d52911cea7c58a8de53c0a475` |
   | `modules/normalize.py` | `e063b3cc49e1d9571c6388251212054e269de43e2794e28238021fe9ccca67f6` |
   | `data/schemas.py` | `01054d7e928b9392391f85a5e5a2feea78de3fc2c7054c871715d0658c1e4938` |
   | `init/kmeans.py` | `7109304a3c7408d49872a079763d9c0b05820e8475f6db4d6ded0ad2e24e5e15` |
   | `scripts/compute_closed_form_curvature.py` | `5c72150f2e5484c4a1e546ba1df157e7151a4f5421f52fe45fcf10402e6d8767` |
   | `scripts/computed_behavior_branching.json` | `cceb416b158f2f9d2c3a12dc92ed1d0c08f2d02f6db51d365f3ac7a0bfafcedc` |
   | `logs/mechanism_contract_iter31.json` | `0ea83b1f55916d02fa3eed97e02d4c39f46d7229a0afa76d5fe1262d96459b32` |
   | `logs/hra_step6_contract_iter31.json` | `3e7394e91ed3f31ec4d2753e6d634c6e99590e18e5e4da98d975df0c2ae5cf17` |

   The ten added dependency hashes were independently rechecked by Judge C for this amendment; the initial source table preserves the S09 coordinator's identities. Neither set is immediate prelaunch verification. Recompute the complete union immediately before launch and require equality with every listed value. Any missing file or mismatch blocks launch and requires evidence-only same-stage repair before reconsidering execution.



4. **Warm-start key-compatibility and transfer.** Reconfirm the exact checkpoint identity in item 2 and the exact unchanged source identities in item 3. S08's `scripts/mvg_check.py:_build_model` loaded that checkpoint into the same `RqVae` state layout and explicitly required the missing-key set to be exactly `{layers.0._fixed_c, layers.1._fixed_c, layers.2._fixed_c}` with no unexpected keys. Its `codebook_kmeans_init=False` differs from training's `True` only in initialization behavior; training's `freeze_layer_scale=False` matches the model default. This S08 evidence is the key-set proof for the pinned unchanged model/checkpoint—not the trainer's transfer count. The live training loader itself calls `load_state_dict(..., strict=False)`, checks unexpected keys only, and does **not** reject or log the `missing` list. Therefore require unchanged validated code/checkpoint identity and the unchanged explicit skip set before launch; during the run require the exact Iter8 path and positive transferred tensor count with no warm-start warning. Do not claim a runtime missing-key list was emitted. If identity continuity or actual warm-start use cannot be verified, block completion and do not train from random initialization.

   S08's one-checkpoint/one-batch MVG and root `CLAUDE.md` §6 gradient checks are already satisfied by the recorded `MVG PASS`; do not repeat MVG or add another GPU check as part of S09.

5. **Destination and destructive-side-effect inventory.** Re-read current `curvature_config.py` and all current Stage2 write/cleanup callsites; verify the required `RQVAE_OUT_DIR` path check shows the short Iter31 results root. Resolve and inspect the complete destination set, including parents, existing files, and available free space:

   - `RQVAE_OUT_DIR`: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments/`; includes `RQVAE_CKPT_PATH` (`rqvae_best.pth`), `RAW_SIDS_NPY` (`sids_raw.npy`), and checkpoint/intermediate/quality artifacts. Before launch, enumerate all files matched by Step0's deletion patterns (`rqvae_step*.pt`, `rqvae_best.pt`, `rqvae_best.pth`, `rqvae_final.pt`, `sids_step*.npy`, `sids_for_hgrec*.npy`, `sids_raw.npy`, `sids_final.npy`, `item_sids.json`, `quality_step*.json`, `quality_best.json`). Step0 deletes matching files here; no whole-directory deletion is performed.
   - External Stage2 exports, not covered by Step0 cleanup: `SIDS_NPY=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/dataset/Instruments/sids_for_hgrec.npy` and `ITEM_SIDS_JSON=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json`. Final export uses `np.save` and `write_text`, so existing files can be overwritten.
   - Log targets: source-iteration `logs/train_run.log` (outer shell redirection) and `logs/train_migrated.log` (trainer opens it for write). The root rules allow logs in the source iteration, but these targets can be overwritten; inventory them as well.
   - Inspect shared-config `SAVE_DIR_ROOT`, `CONFIG_PATH`, and `BEST_CKPT_PATH`; confirm the Stage2 trainer has no write callsite to these decoder/HG-Rec config routes. Do not classify them as trainer outputs unless the current source has changed.

   The current route values must all remain under the exact Iter31 results root for data/checkpoint/SID artifacts. Confirm the source iteration contains no prohibited `.pth`, `.npy`, or `item_sids.json` products. Do not delete, replace, or silently overwrite pre-existing user/result files: the relevant deletion/overwrite targets must be absent or safely preserved outside the destinations before launch. If any destination contains an unpreserved file or any route points elsewhere, block; do not improvise cleanup or change the route.

6. **Runtime readiness.** Without changing the environment, verify the prescribed `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` and the configured torchrun executable exist and use the expected usable torch/CUDA environment. Confirm there is no conflicting Iter31 Stage2 process/launcher still running and the configured port 50200 is available. No fallback interpreter, environment install, port override, alternate worker count, or CLI parameter is permitted.

## Run-time and completion evidence

During the single authorized run, capture and verify:

- Four DDP ranks, seed 42, per-rank batch 640/global batch 2,560, and the pinned warm-start path with positive transferred tensor count; there is no warm-start-missing warning.
- No FCCR-1 contract violation or fixed-curvature drift. Use the existing registered invariance snapshots in the actual training logs; do not introduce a new mechanism check or edit.
- Training reaches and reports `global_step=100000`, with successful launcher and worker exit statuses. A crash, NaN/Inf, checkpoint/export failure, or fixed-curvature contract violation invalidates completion; do not retry under this authorization.
- Final checkpoint under `RQVAE_OUT_DIR` with step 100,000; raw three-token SID output under `RAW_SIDS_NPY`; four-token `SIDS_NPY` and `ITEM_SIDS_JSON` each cover the 24,587 dense item IDs. Confirm the final NPY has shape `[24587, 4]`, JSON keys are exactly `0..24586` with four values per item, and exporter `SID_WIRING_PASS`/round-trip checks succeeded. Verify all model/SID products are in results root and none appeared in the source iteration subtree.
- All SID quality/occupancy/collision/entropy statistics remain descriptive in the captured record. No threshold, gate, baseline rerun, or early-stop decision is permitted.

This plan authorizes no Stage3 launch or Stage3 wiring/evaluation. Stage3 remains subject to its separately adjudicated stage.

## No-change / fail path

Do not change production source, history, mechanism, contract, Stage2 outputs, or user data as part of this Judge adjudication. If any immediate prelaunch condition fails, do not issue the command. Record the exact failed identity/path/evidence and use the same-iteration operational repair path; after repair, re-establish required same-stage evidence and re-run the deliberation gate before any later launch authorization. No questions to the user are required.
