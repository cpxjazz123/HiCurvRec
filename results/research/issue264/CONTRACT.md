# Issue #264 startup contract (frozen before optimized launch)

## Target

Reduce wall time from a legal Stage3 launch request to the first real training
step without changing the training semantics, recommendation model, SID, split,
hyperparameters, seed, evaluator, NCCL protocol or result identity. The measured
baseline is the live #263 Stage3 launch on this host; the #263 process is not
interrupted.

## Causal path

```
stage3_start
  -> generated harness parent
     -> [old] import train_HG-Rec (torch, transformers, data, HG_Rec)
     -> torchrun spawn
        -> each rank imports train_HG-Rec
           -> SID/parquet/Dataset/DataLoader
           -> CUDA/NCCL/DDP model
           -> first optimizer step
```

The parent does not need the trainer. Its old eager import both pays a cold
`torch`/`transformers` import on the shared filesystem and warms the page cache
for the ranks serially. The optimized parent is stdlib-only, stages the small
pure-Python Hugging Face dependency graph to a node-local content cache, then
spawns the same torchrun and rank entry. Rank-side imports, SID parsing, dataset
construction, model init, DDP and training remain the trainer's own code.

## Frozen components

- Stage3 trainer: `stage3_T5Train/train_HG-Rec.py` unchanged; `NUM_EPOCHS=150`,
  `NO_EVAL=True`, `SKIP_TEST=False`, `SEED=42`, `BEAM_SIZE=20`; 4-rank path and
  single-card fallback unchanged.
- SID, Stage2 run, split, vocab/EOS, prefix/suffix protocol, model/random seed,
  NCCL environment, port 50201 and result paths unchanged.
- The cache contains only installed pure-Python package files, never SID,
  training state, model weights, dataset samples or recommendation output.

## Cache and harness invariants

1. Cache source is `sysconfig.get_paths()["purelib"]` of the Stage3 interpreter;
   only `transformers`, `tokenizers`, `safetensors`, `huggingface_hub`,
   `sentencepiece`, `accelerate` are staged when present.
2. A manifest records every file's relative path, byte size and SHA-256. A cache
   is used only if the source tree fingerprint matches and every cached file
   verifies. A missing, corrupt, stale or unreadable cache falls back to the
   source environment; it never fabricates a package.
3. Publication is atomic (`rename`) under an exclusive `flock`; no rank or
   concurrent request sees a half-copied package. Old generations are left
   immutable.
4. The cache is node-local under `/dev/shm/genrec-stage3-deps` (overridable only
   by the existing internal environment variable) and is not a global cache
   clear or a modification of another user's environment.
5. The parent harness must not import `torch`, `transformers`, `data`, `model`
   or `train_HG-Rec`; it may import stdlib and the cache helper only.
6. The child sets the same trainer attributes (`CODE_PATH`, `RQVAE_VARIANT`,
   `LOG_PATH`, `SAVE_PATH`, `PER_USER_DUMP`) and calls the same `main`. Single-GPU
   uses `GENREC_SINGLE_GPU=1` and `GENREC_STAGE3_LAUNCHED=1`; DDP uses torchrun
   with `--standalone --nproc_per_node=4 --master_port=50201`.

## Correctness equivalence

The following are compared with the frozen SID/split: sample order and
rank sampler batches, vocabulary/EOS, prefix/suffix assignment, model
initialization hash (same seed), first batch inputs/targets, first loss, first
gradient norm and first optimizer update. Cache cold/warm/missing/corrupt/stale,
concurrent preparation, single/multi-GPU launch and failure exit propagation are
covered by tests. No recommendation `test` split is read for tuning.

## Performance measurement

Same host, interpreter, SID and frozen config. Record request timestamp,
parent start/end, cache cold/warm, torchrun start, rank start, first metric,
first real optimizer step and first-epoch throughput. Three short diagnostic
repeats are recorded (not seeds). Primary target: at least 50% reduction from the
measured old request-to-first-step; warm startup should be at most 60 s when the
external storage is controllable. Cold cache construction is reported separately
and never hidden. If the target is not met, the residual is reported and a
minimal follow-up is proposed; success is not claimed.

## Non-goals

No new full Stage3/Stage2, no GPU competition with #263, no mechanism change, no
new SID/dataset, no early-stop or protocol change, and no test-score-guided
optimization.
