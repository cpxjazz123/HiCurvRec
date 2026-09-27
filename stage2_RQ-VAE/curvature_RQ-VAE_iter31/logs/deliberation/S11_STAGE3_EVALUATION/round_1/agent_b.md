ROLE=AGENT_B
INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.
SOURCE_PACKET=local://S11_ITER31_SOURCE_PACKET.md
STAGE_ID=S11_STAGE3_EVALUATION

# Iter31 S11 Stage3 direct-route audit — Agent B

## Decision

**Recommendation: AUTHORIZE exactly one Stage3 run, subject only to Judge C's canonical S11 authorization and fresh immediate prelaunch checks below.** The audited current direct trainer consumes the exact Iter31 Stage2 SID JSON, current source bytes match the packet's pinned hashes, the Stage0 test input matches the locked identity, evaluator/configuration match the Iter29 comparison contract, and all trainer/launcher write sinks resolve below the Iter31 short result root. The destination root and outer-redirection `logs/` parent are currently absent; create only that `logs/` parent after canonical Judge authorization. This candidate report itself grants no launch permission. Stage3 was not launched here.

## Evidence and checks

### 1. Active source identity and unchanged model/data path

Current SHA-256 values were recomputed from the active checkout and exactly match the frozen packet pins:

- `stage3_T5Train/train_HG-Rec.py`: `210fd25a7c2a4fba7c6097e937f57f77bfd9340c3d0ce39eea823a280eadfc8b`.
- `stage3_T5Train/data/dataset.py`: `8899f1348e740996c572cd95439365f4caf2fe791acc4a0e22ed7d837937030e`.
- `stage3_T5Train/data/dataloader.py`: `d490b89c82a8ce3e1d8fa7577040038ef3411d030ce327fad95a999334f98af4`.
- `stage3_T5Train/model/hg_rec.py`: `f2876a95ee25b3f25d417792867c560da3053f8e5880120eda89affd9fd5cad1`.
- `stage3_T5Train/model/utils.py`: `a471b26da34ab30372f2cb2cc7baf05a6747be8830db4f178eac3b228364623c`.

Trainer imports those exact data/model/utility modules (`train_HG-Rec.py:38-41`). It builds the unchanged `HG_Rec` T5 model (`model/hg_rec.py:10-30,43-54`) from runtime-derived SID width/vocabulary; its model behavior is not conditional on Iter31 HRA metadata. Dataset reads the named parquet as one record per row, preserving the history, target, and full seen history (`data/dataset.py:37-71`). No Stage3 model, dataset, dataloader, or evaluator source is modified by this route.

### 2. Exact SID handoff and test coverage

The active trainer hard-codes `CODE_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/item_sids.json` (`train_HG-Rec.py:121-128`); `_resolve_code_file` uses that requested absolute file (`:215-258`, as pinned in S06). Current SID JSON SHA-256 is `d4b100f68fcce4f5185f56f77fb8eef898b2b1dfa0d61d258422801239716dd7`, matching both packet and S09 completion report. The completion record documents 24,587 dense ordered keys `0..24586`, fixed width four integer rows, exact JSON/exported-NPY equality, unique exported rows, raw first-three-column parity, and extension values `[768,778]` (`logs/stage2_completion_iter31.md:20-27`). S10 canonical analysis independently carries these artifact facts and confirms SID proxies are descriptive only (`logs/sid_geometry_iter31.md:21-47`). No alternate SID path is selected by the direct trainer.

Dataset code validates dense numeric-string keys, nonempty integer SID rows and fixed width; shifts each raw component by `+1` into one shared token namespace; rejects negative values and duplicate token tuples; and derives semantic vocabulary from the observed maximum token (`data/dataset.py:80-156,182-204`). `GenRecDataset._prepare_data` raises if any target, model input history, or complete seen history is absent from the SID map (`:211-243`). The frozen packet's CPU audit of this exact SID and test split reports `n_test=57439`, zero missing targets/history/seen IDs, 16,462 distinct targets, 24,584 distinct history/seen IDs, and SID range `0..24586` (packet lines 64-67). This is consistent with the source's fail-closed behavior; four-token extension IDs remain in the dynamic common namespace, with no token/config edit.

### 3. Data split, evaluator, and strict comparison protocol

Trainer names the Stage0 directory and split files exactly: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage0_build_parquet/`, `train.parquet`, `valid.parquet`, `test.parquet` (`train_HG-Rec.py:104-128,645-651`). Current `test.parquet` SHA-256 is `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc`, matching S01 and the packet; size is packet-recorded 2,546,378 bytes. S01 locks the Instruments test split, seed 42, max 150 epochs with existing train-loss patience 10 semantics, `NO_EVAL=True`, `SKIP_TEST=False`, beam 20, top-k `[5,10]`, and `n_eval=57439` (`logs/protocol_manifest_iter31.md:57-80`). Current trainer constants match: batch 4096, infer batch 1024, max epochs 150, patience 10, `FAST=True`, `BF16=True`, `COMPILE=False`, scheduler enabled, warmup 0.10/minimum factor 0.20, LR 0.003/weight decay 0.05, four-layer 128-dimension model configuration, seed 42, beam 20, top-k `[5,10]`, and history exclusion (`train_HG-Rec.py:53-101,160-179`). Preserve existing train-loss early-stop/checkpoint behavior; the maximum is 150, not a promise to train exactly 150 epochs (`:881-900,1059-1063`).

**Additional direct CPU verification.** I instantiated the current `data.dataset.GenRecDataset` in `evaluation` mode using the prescribed Python 3.10 interpreter, exact Stage0 `test.parquet`, and exact Iter31 SID JSON; it completed without missing-item errors and reported `rows=57439`, `num_items=24587`, `n_digit=4`, `max_sid_token=779`, `vocab_size=782`, `max_token_seq_len=82` (`CPU_TEST_SID_PASS`). This was CPU-only dataset construction, not model execution, test evaluation, training, or GPU work.

The final test path constructs `test.parquet` evaluation data after training, runs one rank-0 evaluation with full seen-history exclusions and the selected best checkpoint if present (`train_HG-Rec.py:1108-1135`). Evaluator maps beams to item IDs, removes invalid/duplicate/seen items, uses the existing fallback order, computes recall/NDCG, and divides by actual `n_eval` (`:329-467`). Result JSON has `test_recall@5`, `test_recall@10`, `test_ndcg@5`, `test_ndcg@10`, `n_eval`, and checkpoint (`:1136-1155`).

The direct comparator is Iter29's primary `test_final.json`, whose current SHA-256 is `b07de15ec53e8b423f3426f091a142b00260f084bd319abf6c7cb5d605583662`; the exact result is `n_eval=57439`, R@5 `0.03953759640662268`, R@10 `0.05921064085377531`, NDCG@5 `0.026252776900288842`, NDCG@10 `0.03257647953179143` (`results/stage3_T5Train/curvature_RQ-VAE_iter29/logs/Amazon_2023_Instruments/Sep-27-2026_06-11-34/test_final.json:1-8`; S01 manifest `:19-29`). The locked adoption target remains strictly `test_R@10 > 0.065`; the Iter29 shortfall is `0.005789359146224693`. No Stage3 effect, causal conclusion, or promotion classification is known or made here.

### 4. Direct no-wrapper route and all write sinks

Current path constants implement the S06 reconciliation: `CODE_PATH` above; exact descriptive route mapping to `iter31_hra_step6_common_reference` (`train_HG-Rec.py:132-156`); `LOG_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/`; `SAVE_PATH=/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/ckpt/` (`:158-159`). The launcher uses its configured root-level `_stage3_launcher.log`, direct `os.path.abspath(__file__)`, four ranks, port 50201 and visible devices `0,1,2,3` (`:1175-1182`). It invokes the prescribed torchrun with only standalone, rank-count, and port flags, passing the trainer script directly (`:1185-1212`). There is no Iter31 wrapper in the active route; do not invoke Iter29's historical wrapper.

Audited write consumers and resolved locations:

- Outer shell redirection: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_stage3_run.log` (approved path reconciliation in S06 plan, `logs/implementation_plan_iter31.md:93-108`).
- Internal torchrun stdout/stderr: `/home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/_stage3_launcher.log` (`train_HG-Rec.py:1175-1211`).
- Per-run directory: `LOG_PATH/Amazon_2023_Instruments/<timestamp>/`; `HG_Rec.log`, `training_metrics.jsonl`, optional `screen_summary.json`, early-stop/shutdown signaling files all lie under the same `LOG_PATH` (trainer `:663-672,827-849,1076-1078,1083-1099,1158-1160`).
- Checkpoint directory: `SAVE_PATH/Amazon_2023_Instruments/<timestamp>/`; both train-loss and valid-best checkpoint writes use `best_checkpoint` beneath it (`:663-668,881-889,954-966`).
- Final `test_final.json` is in that per-run `LOG_PATH` directory (`:1110-1155`).

All resolve beneath the one required short result root `results/stage3_T5Train/curvature_RQ-VAE_iter31/`; no descriptive suffix is added to output paths. Current `stat` confirmed this Iter31 results root does not exist. The prescribed interpreter and torchrun are executable; the missing outer-redirection parent confirms the need to make exactly `.../curvature_RQ-VAE_iter31/logs` after Judge authorization. With the root absent, none of these sinks overwrites existing Iter31 results. Do not pre-create the root or any directory before canonical authorization.

**Metadata limitation:** `RQVAE_VARIANT` is correctly derived for Iter31 in the source map (`:154-156`), but is not copied into the `config` dictionary (`:528-593`); the metrics writer reads `config.get("variant", "unknown_variant")` (`:843`). As with the documented Iter29 metadata discrepancy (packet lines 71-78), metrics may therefore say `unknown_variant`. This is a provenance-label limitation, not evidence of different model/data/SID consumption: exact `code_path`, short output root, source identity and the direct file route are explicit. Preserve and disclose it; do not alter trainer code or infer a result from that label.

### 5. Current runtime and safe-run snapshot

Using the prescribed `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10`, current read-only runtime check reported PyTorch `2.13.0+cu130`, CUDA `13.0`, CUDA available, four `NVIDIA L40S` devices, and BF16 supported. `torchrun` exists and is executable. A no-training import smoke in this interpreter passed for torch, NumPy, pandas, transformers 4.45.2, tqdm, scikit-learn, and all four imported Stage3 packages/modules (output: `IMPORT_SMOKE_PASS 2.13.0+cu130 4.45.2`). Current `nvidia-smi` showed all four GPUs at 0 MiB / 0% compute; `pgrep -af train_HG-Rec.py` produced no match; port 50201 had no listener. These are transient observations, not permission, and MUST be rechecked immediately before launch.

Current launcher constants and child environment match root `CLAUDE.md` §5 and packet lines 55-56: `nproc=4`, `master_port=50201`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_TIMEOUT=3600`, `TORCH_NCCL_BLOCKING_WAIT=1` (`train_HG-Rec.py:1175-1209`). Retain existing optional NCCL defaults and source behavior; add no CLI, environment, port, or device overrides.

## Exact required immediate prelaunch checks

Only after Judge C has written canonical S11 authorization, recheck immediately before execution:

1. Recompute and require exact packet-pinned SHA-256 for all five active trainer/imported source files listed above; recheck current root `CLAUDE.md` §§5/11 and canonical S01, S06 and S10 artifacts have not changed.
2. Require exact resolved `CODE_PATH` to the Iter31 `item_sids.json`; recompute its packet-pinned hash and ensure the corresponding Stage2 completion/S10 parity, row count/schema/width/uniqueness evidence remains the exact handoff. Require exact Stage0 `test.parquet` path/hash and `n_eval=57439`; verify no path fallback or alternate SID is active.
3. Reconfirm source constants for data filenames, protocol/model/optimizer/evaluator, `NO_EVAL=True`, `SKIP_TEST=False`, seed, beam, top-k, and runtime-derived four-token vocabulary; confirm all output sinks remain exclusively in the short Iter31 result root and exact launcher constants/environment above. Do not change model, data, evaluator, or protocol.
4. Confirm the Iter31 Stage3 result root is still absent (no overwrite), then recheck prescribed Python 3.10 / PyTorch-CUDA / BF16 support, torchrun executable, all four devices, no competing `train_HG-Rec.py` process, and port 50201 free. If any check fails or identity/route differs, block before creating a directory or launching.
5. Once all checks pass, create only the missing `logs/` parent for shell redirection; do not pre-create other directories. Then issue exactly one no-argument direct launch from the Stage3 source CWD, without wrapper or overrides:

```bash
cd /home/wlia0047/ar57/wenyu/GeneRec/stage3_T5Train
mkdir -p /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs
nohup /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 train_HG-Rec.py > /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter31/logs/_stage3_run.log 2>&1 &
```

The directory creation and command above are a proposed post-authorization method only; neither was executed in this audit.

## Risks and self-rejection conditions

- Iter29's recorded `unknown_variant` and nonfatal NCCL/c10d warnings remain caveats. The variant metadata omission above should be recorded as a limitation, not repaired by changing Stage3 code in this gate; historical warnings are not evidence the current run will be warning-free.
- The result-root absence and GPU/port/process state are transient; changed state before launch requires the listed rechecks, and a pre-existing Iter31 destination blocks launch to avoid overwriting.
- Stop and reject this authorization recommendation if any pinned current source/runtime/data/SID identity differs, any test item is uncovered, evaluator/configuration is changed, any actual sink escapes the short Iter31 result root, direct no-wrapper launch cannot be preserved, the result root already exists, or immediate resource/port checks fail. Do not use a wrapper, alter Stage3 behavior, or infer results from descriptive Stage2 SID metrics.

ROLE_CONCLUSION=RECOMMEND_AUTHORIZE_ONE_RUN_AFTER_JUDGE_C_CANONICAL_APPROVAL_AND_FRESH_PRELAUNCH_CHECKS
STAGE3_LAUNCHED=NO
USER_INPUT_REQUIRED=NO