# #87 pilot effectiveness comparison — INVALID / NOT ADMISSIBLE

The #87 pilot reported L1 collision dropping from 0.152 to 0.062 over 128 active steps on R6-04
under a soft separation loss with lambda=0.10. Issue #88 lists six confirmed defects in the
committed pilot code, each of which is independently sufficient to invalidate that comparison:

1. `chained_replays` carries **one step-32 bundle** through steps 33..160 with the same captured
   `item_batch`, `pair_embeddings=None`, `source_ids=None`, `target_ids=None`, and unchanged
   `applied_weight=0` -- the eight "active" steps never exercised a real behavior-CE minibatch and
   the snapshot at each step reports the model that just left step 32 with one replay applied to
   the same tensors. The training-time quantity the lane was supposed to measure is therefore
   undefined.

2. `extra_loss` was added to `assemble_and_step` in commit 41cb8fa, but the pilot's chained
   replay cannot replay the BEHAVIOR-CE term either -- `pair_embeddings` is None -- so the
   auxiliary loss is composed against a `base_loss` produced by `model.compute_loss` over the
   frozen item batch only. The total loss the optimizer applied therefore differs from the host's
   production loss assembly, and the comparison is not over the host's actual training distribution.

3. `_snapshot_from_branch` calls `FS.posterior(model, population)` where `population` is the
   RAW item embedding. The quantizer expects the encoder-produced latent `z = model.encoder(embedding)`;
   feeding the raw embedding as `z` computes distances from embeddings to codewords directly,
   which is a different geometry. The reported `hard_collision_rate` is therefore `argmax` over a
   wrong-posterior and is not the host's hard L1 same-code rate. Native `model.get_indices(embedding,
   infer_use_sk=False)` would be the correct canonical hard SID.

4. `host_successor_graph()` silently returned `None` unless `trainer._train_frame` and
   `trainer._successor_indicator` were module attributes, which they were not, so every F0/F1/F2/F3
   run used a synthetic position-based pair stream against a graph the function could not find.
   `_successor_lookup` on a dense Boolean 2D tensor reads row VALUES (0/1) as successor column
   indices, returning `{0: {0, 1}, 1: {0, 1}}` for any 2x2 dense adjacency. The mask "worked" by
   accident on small toy graphs only.

5. The tau preregistration reported `median_gap = 0` because the native distance was computed
   against the wrong latent (raw embeddings after KMeans initialisation, with latent_dim meaning
   `codebook.shape[1] = 32`). With the true encoder-produced z and the host's native squared
   distance the p05/median/p95 of `d_second - d_best` is not 0, and tau=1e-3 is therefore a
   floor-driven default that the issue explicitly forbids reporting as a valid preregistration.

6. The pilot called the arms "supported" without a real CGV G1-G6 pass and without the seven-
   group spec the lane itself required; the issue's own rules do not admit that verdict.

The published numerical comparison in #87 (L1 0.152 -> 0.062, codes 130 -> 143, entropy 3.96 -> 4.21)
is preserved as evidence of what the BROKEN pilot measured, not as evidence of what the mechanism
does. The r7_f_pilot/ directory is not deleted or overwritten -- a new r7_f_harness_v2/ directory
holds the corrected attempt and its reports.
