# protocol_manifest_iter28

```
PROTOCOL_ID = CAO-1_iter28_SREMA
dataset/version = Amazon_2023_Instruments 5-core (24587 items, 57439 users)
Stage1 embedding path = /home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy
Stage1 sidecar = /home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json
parent iteration = iter18 (NOT iter27; iter27 was ABORTED)
parent commit = bdcbbf9
canonical baseline iteration = iter18
canonical baseline test_final.json = /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json
canonical baseline test_R@10 = 0.05988962203380978
Stage2 seed = 42
Stage2 max steps = 100_000
RQ layers / codebook size = 3 layers × 256 codes
Stage3 code commit = bdcbbf9
Stage3 seed = 42
Stage3 epochs = 150
beam size = 20
n_eval = 57439
```

## Compatibility statement

iter28 inherits **every** Stage2 / Stage3 protocol parameter from iter18 (not iter27):
- cyclic curvature schedule (`C_CYCLIC_MIN=0.05`, `C_CYCLIC_MAX=1.5`, `C_CYCLIC_PERIOD=100_000`).
- per-layer AdamW `β₂` frozen at init from `c_l(0)` (iter18 mechanism, unchanged).
- Stage1 sentence-T5 embeddings (identical `.npy` + `item_ids.json`).
- train/val/test parquets (identical).
- Stage3 T5 hyperparameters (identical to iter18's `decoder_instruments_hgrec_v318_amazon2023.gin`).
- Stage3 launcher env (`NCCL_IB_DISABLE=1` etc.).
- Stage3 beam=20 / n_eval=57439.

The **single** Stage2 change is: codebook `layers.X.embedding.weight` is updated by a Sinkhorn-weighted Riemannian EMA (SREMA) instead of AdamW.  No protocol-level change.  Stage3 reads the same `item_sids.json` shape (4-token TIGER-compatible), so it is fully protocol-compatible with iter18.

## What is NOT changed (one-factor diff)

- Stage1 (`sentence_t5.npy`, `item_ids.json`).
- Stage2 dataset (`train.parquet`, `valid.parquet`, `test.parquet`).
- Stage2 optimizer hyperparameters (`AdamW`, `β₁=0.9`, `β₂,l` from `c_l(0)`, `eps=1e-8`, `wd=1e-4`, `lr=1e-3`) **for all non-codebook parameters** (encoder, decoder, per-layer `c_layer_scale`).
- Stage2 assignment (`Sinkhorn 3 iters, ε=0.05`, cyclic-`c` modulated ε).
- Stage2 loss weights (`commit=1.0, behavior=0.20, curv_reg=0.005, behavior_temperature=0.07`).
- Stage2 warm-start from iter8 (`out/rqvae/instruments/rqvae_best.pth`).
- Stage3 trainer, T5 config, beam, n_eval, RQVAE_VARIANT name pattern.

## What IS changed (single mechanism)

Codebook parameters are removed from the AdamW optimizer and updated by:
1. Snapshot Sinkhorn assignment `A_ij` and query residual `z_i` during forward().
2. After `optimizer.step()` runs (non-codebook params only), compute for each codeword `j`:
   - `ξ_j = Σ_i A_ij · log_{p_j_old}^{c_l}(z_i) / (Σ_i A_ij + ε)`
3. Apply `p_j^{new} = exp_{p_j_old}^{c_l}((1 - β) · ξ_j)` and write back `w_j^{new} = log_0^{c_l}(p_j^{new})` into `embedding.weight`.

## Comparators (current best, protocol-compatible)

| Iter | R@10 | n_eval | notes |
|---|---:|---:|---|
| iter11 | 0.05976775 | 57439 | cyclic curvature, c-modulated commitment |
| iter18 | **0.05988962** | 57439 | iter18 = canonical baseline; per-layer β₂ from c_l(0) |
| iter25 | 0.05882763 | 57439 | bounded raw-residual layer-scale prior (FCCR-1) |
| iter26 | 0.057017 | 57439 | closed-form fixed curvature (FCCR-1) |
| iter27 | (aborted) | — | hyperbolic codebook trust-region — ITERATION_ABORTED_INFEASIBLE |