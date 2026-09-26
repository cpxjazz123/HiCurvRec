# protocol_manifest_iter27

```
PROTOCOL_ID = CAO-1_iter27
dataset/version = Amazon_2023_Instruments 5-core (24587 items, 57439 users)
Stage1 embedding path = /home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy
Stage1 sidecar = /home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/item_ids.json
parent iteration = iter18
parent commit = cbf38da
canonical baseline iteration = iter18
canonical baseline test_final.json = /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/Sep-26-2026_04-45-57/test_final.json
canonical baseline test_R@10 = 0.05988962203380978
Stage2 seed = 42
Stage2 max steps = 100_000
RQ layers / codebook size = 3 layers × 256 codes
Stage3 code commit = cbf38da (HEAD at iter27 launch)
Stage3 seed = 42
Stage3 epochs = 150
beam size = 20
n_eval = 57439
```

## Compatibility statement

Iter27 inherits **every** Stage2 / Stage3 protocol parameter from iter18:
cyclic curvature schedule (`C_CYCLIC_MIN=0.05, C_CYCLIC_MAX=1.5,
C_CYCLIC_PERIOD=100_000`), per-layer AdamW `β₂` (frozen at init from
`c_l(0)`), Stage1 sentence-T5 embeddings (identical `.npy` + `item_ids.json`),
train/val/test parquets (identical), Stage3 T5 hyperparameters (identical
to iter18's `decoder_instruments_hgrec_v318_amazon2023.gin`), Stage3
launcher env (`NCCL_IB_DISABLE=1` etc.), Stage3 beam=20 / n_eval=57439.

The single Stage2 change is a **post-`optimizer.step()` projection on
`layer.embedding.weight` only** — no protocol-level change.  Stage3 reads
the same `item_sids.json` shape (4-token TIGER-compatible), so it is
fully protocol-compatible with iter18.

Therefore iter27 is directly comparable to iter18, iter11, iter25, iter26.

## What is NOT changed

- Stage1 (`sentence_t5.npy`, `item_ids.json`).
- Stage2 dataset (`train.parquet`, `valid.parquet`, `test.parquet`).
- Stage2 optimizer architecture (`AdamW`, `β₁=0.9`, `eps=1e-8`,
  `wd=1e-4`, `lr=1e-3`, per-layer `β₂,l` from `c_l(0)`).
- Stage2 assignment (`Sinkhorn 3 iters, ε=0.05`, cyclic-`c` modulated ε).
- Stage2 loss weights (`commit=1.0, behavior=0.20, curv_reg=0.005,
  behavior_temperature=0.07`).
- Stage2 warm-start from iter8 (`out/rqvae/instruments/rqvae_best.pth`).
- Stage3 trainer, T5 config, beam, n_eval, RQVAE_VARIANT name pattern.

## What IS changed (single mechanism)

- A new **post-`optimizer.step()`** projection in `curvature_RQ-VAE.py`
  that, for every `Quantize` layer, after each AdamW step:
  1. snapshot `v_k_old = layer.embedding.weight.detach().clone()` before
     `optimizer.step()`;
  2. compute `p_k_old = exp_0^{c_l(t)}(v_k_old)`,
     `p_k_new = exp_0^{c_l(t)}(v_k_new)`;
  3. compute `δ_k = d_{c_l(t)}(p_k_old, p_k_new)`;
  4. compute `D_l = median_k min_{j≠k} d_{c_l(t)}(p_k_new, p_j_new)`;
  5. set `τ_l = 0.5 · max(1e-3, D_l)`;
  6. scale: `s_k = min(1, τ_l / (δ_k + 1e-9))` for `δ_k > τ_l`;
  7. write back
     `v_k_new := log_0^{c_l(t)}(exp_{p_k_old}^{c_l(t)}(s_k · log_{p_k_old}^{c_l(t)}(p_k_new)))`.

This is the only conceptual change.  No new trainable parameter, no new
loss, no new assignment rule.

## Comparators (current best, protocol-compatible)

| Iter | R@10 | n_eval | notes |
|---|---:|---:|---|
| iter11 | 0.05976775 | 57439 | cyclic curvature, c-modulated commitment |
| iter18 | **0.05988962** | 57439 | iter18 = canonical baseline; per-layer β₂ from c_l(0) |
| iter25 | 0.05882763 | 57439 | bounded raw-residual layer-scale prior (FCCR-1, contract-invalid for layer-scale prior) |
| iter26 | 0.057017 | 57439 | closed-form fixed curvature (FCCR-1) |