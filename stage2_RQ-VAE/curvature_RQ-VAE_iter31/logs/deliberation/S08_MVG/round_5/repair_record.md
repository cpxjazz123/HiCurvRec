# Iter31 S08 MVG — Operational Repair Record

```text
STAGE_ID=S08_MVG
ROUND=5
ROUND_TYPE=OPERATIONAL_REPAIR
LOCKED_SCIENCE_CHANGED=NO
SAME_ITERATION_REPAIR=AUTHORIZED
PARALLEL_EXECUTION=YES
PRE_RUN_CHECKS_PASS=YES
CHECKS_PASS=YES
S08_EXECUTION_STATUS=PASS
```

## Authorization and invariant

Authorized by `logs/deliberation/S08_MVG/round_5/judge.md`. This round repairs evidence capture and the shared deliberation gate only. The registered HRA-STEP6-1 equation/implementation, FCCR-1 fixed closed-form curvatures, Iter29 parent/comparator, locked data/protocol, pinned Iter8 checkpoint, seed 42, and deterministic first-640-eligible-source selection remain unchanged. Historical Iter31 S08 rounds 1–4 and the old abort record remain untouched; their incomplete provenance is invalid for the current S08 gate.

No Stage2, Stage3, SID export, training, performance comparison, or extra S08 invocation is authorized in this operational repair.

## Consolidated source repair

- `.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py`: replaced the 64-character/48-character-window negative-mention heuristic with sentence-scoped negation, contrast-boundary, and fronted-`without` handling. It continues to reject positive forbidden-action proposals and treats the recorded negated `matched-seed replication` phrase as non-positive.
- `stage2_RQ-VAE/curvature_RQ-VAE_iter31/scripts/mvg_check.py`: one unchanged first-640 eligible-source batch now emits, during the authorized run, pre-load SHA-256/path/size/mtime identities for the consumed Stage1 embedding, item-ID sidecar, Stage0 train parquet, and pinned Iter8 checkpoint; seed and exact selection rule; actual ordered selected source-target IDs; source/future tensor shapes and ID tensor shapes; runtime logical device, device name, memory, and physical UUID or explicit `UNAVAILABLE`; software versions; and the complete existing HRA/fixed-curvature/direct-effect/gradient diagnostics. Input hashing fails closed if file identity changes while hashing. The selected samples are materialized exactly once, preserving the existing RNG sequence.

## Source hashes before the authorized invocation

- `scripts/mvg_check.py`: `abf4ad518890e345c05e8b0907349067faa94cd0ef39d6dab6384bf20792318d`
- `curvature_RQ-VAE.py`: `b641d8b5c9897f520482d0bce5526bdcd687ae5845901f8f0be238772f49d1dd`
- `deliberation_gate.py`: `37e5707594ea58b523fb7031da917fef62d98d688038378a1930af309974b095`

## Deterministic pre-run checks

Independent checks were executed concurrently after the consolidated repair:

1. `python3 -m py_compile stage2_RQ-VAE/curvature_RQ-VAE_iter31/scripts/mvg_check.py .claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py` — PASS, exit 0.
2. Shared gate helper cases — PASS: the exact S00 negated phrase and direct prohibitions return false; positive proposals return true; contrastive positive actions remain detectable; `without X, use X` is not treated as a prohibition of the later action.
3. `python3 stage2_RQ-VAE/curvature_RQ-VAE_iter31/scripts/preflight_hra_step6_iter31.py` — PASS; HRA-STEP6-1/FCCR-1 source and contract scope unchanged.
4. `RQVAE_OUT_DIR` source check — PASS; points to `/home/wlia0047/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/out/rqvae/instruments`.
5. Mandated interpreter import check — PASS; `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` imports Torch `2.8.0+cu128` and reports CUDA available.

The provenance instrumentation was statically compiled before the authorized run; this single invocation captured all required contemporaneous values. Canonical output and separate stdout/stderr files are preserved below. No Stage2/Stage3 work was included in this repair.

## S08 invocation evidence

```text
COMMAND=/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 /home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31/scripts/mvg_check.py
STATUS=PASS
STDOUT=logs/mvg_check_iter31.stdout.log
STDOUT_BYTES=14445
STDERR=logs/mvg_check_iter31.stderr.log
STDERR_BYTES=0
EXIT_STATUS=0
CANONICAL_LOG=logs/mvg_check_iter31.log
CANONICAL_LOG_BYTES=14704
INVOCATION_COUNT=1
```

## Contemporaneous primary evidence

- `logs/mvg_check_iter31.log` contains the complete stdout, empty stderr, and exit status; the separate stdout/stderr files match it byte-for-byte. All 640 ordered source-target item-ID pairs are present in the stdout record, in deterministic selected-source order.
- Seed is `42` for PyTorch and NumPy. Selection is the first 640 eligible sources in ascending `TransitionDataset` row order, with one target selected by `np.random.choice` per source.
- Tensor shapes: source embeddings `[640, 768]`, future embeddings `[640, 768]`, source IDs `[640]`, future IDs `[640]`. The loaded source reports 24,587 embeddings, 339,519 train transitions, and 24,474 active sources.
- Pre-run identities captured by the invocation:
  - Stage0 train parquet `/fs04/ar57/wenyu/GeneRec/results/stage0_build_parquet/train.parquet`: SHA-256 `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`.
  - Stage1 embedding `/fs04/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy`: SHA-256 `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`.
  - Stage1 item-ID sidecar `/fs04/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json`: SHA-256 `3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30`.
  - Pinned Iter8 warm-start checkpoint `/fs04/ar57/wenyu/GeneRec/results/stage2_RQ-VAE/curvature_RQ-VAE_iter8/out/rqvae/instruments/rqvae_best.pth`: SHA-256 `189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`.
- Runtime device was observed as `cuda:0`, NVIDIA L40S, UUID `853ef1bf-bbd2-2d5f-3dfa-a787f9d0e299`, 47,665,709,056 bytes. Software: Python 3.9.25, NumPy 2.0.2, Torch 2.8.0+cu128, CUDA 12.8.
- `MVG PASS`. Registered closed-form curvature vector remained `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]` and stayed invariant across all recorded train/eval steps. Same quantized embeddings produced finite, in-domain outputs; direct HRA-vs-Euclidean difference was nonzero (`max_abs=0.0719804987`, `L2=2.0182168484`, relative L2 `0.0810627937`). Total loss was `5.1951842308`; reconstruction, quantizer, and behavior-loss gradients each reached nonzero parameters (8, 7, and 4 respectively). No optimizer step was performed.
- The repair verifier must independently inspect these primary outputs before writing the terminal `REPAIR_PASS` decision.
