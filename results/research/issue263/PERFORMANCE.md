# Issue #263 Stage2 performance record

Diagnostic only; this is the formal Stage2 stream, not a second run and not a
Stage3 result.

- Run `stage2_ae668bc04dc44d52a8bed5f9408e8e74`, immutable source snapshot.
- Recipe: DDP 4 ranks, `batch_size_per_rank=1024`, global batch 4096, 3000 epochs.
- 3000 train records observed; final epoch 3000; final SID collision diagnostic
  `0.06076381827795174`.
- All-epoch mean epoch time `0.319697 s`; first-60 mean `0.399032 s`;
  31-60 inclusive mean `0.373181 s`, median `0.358222 s`, P95 `0.442509 s`;
  epoch 50 `0.502565 s`. The 0.8 s hard threshold is met; the 31-60 mean is
  inside the requested 0.4-0.5 s band (P95 and epoch 50 are above it and are
  reported, not hidden).
- Maximum rank peak allocated memory `607,034,880` bytes (about 579 MiB);
  first-60 mean peak `375,213,346` bytes (about 358 MiB).
- Observed resident GPU memory during the run was 1.21-1.53 GiB/card; four cards
  were used. No online GNN forward or GNN optimizer state exists on this path.
- Full machine-readable record: `PERFORMANCE.json`.

No parent-throughput number was re-measured, so no speedup ratio is invented.
The candidate's absolute cost is small; recommendation quality remains the
Stage3 test.
