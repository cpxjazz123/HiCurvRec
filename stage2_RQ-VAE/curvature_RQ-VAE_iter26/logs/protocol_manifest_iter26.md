# Iter26 Protocol Manifest

```
PROTOCOL_ID = iter26_closed_form_with_raw_residual
dataset/version = Amazon_2023_Instruments / 2026-09 Refactor
Stage1 embedding path + immutable hash/id = /home/wlia0047/ar57/wenyu/GeneRec/stage1_GeneEmbedding/output/sentence_t5.npy (T5-xxl deterministic; 24587 items × 768 dims)
parent iteration + commit = parent = iter16, commit = 945f20d159d21b33026b3be3127c4d9416bbec5c (last FCCR skill bump; no functional change to Iter16 source tree)
canonical baseline iteration = iter18
canonical baseline test_final.json path = /home/wlia0047/ar57/wenyu/GeneRec/results/stage3_T5Train/curvature_RQ-VAE_iter18/logs/Amazon_2023_Instruments/<run>/test_final.json
canonical baseline test_R@10 = 0.0598896220 (n_eval=57439)
Stage2 seed = 42
Stage2 max steps = 100000
RQ layers / codebook size = 3 layers / 256 codes
Stage3 code commit = 0052f4bff117e0aa16c9a1080be1014fbd417cc6 (last push prior to Iter26 launch)
Stage3 seed = 42
Stage3 epochs = 150
beam size = 20
n_eval = 57439
```

## Comparable runs

Iter26 is directly rankable against Iter11 (canonical), Iter18 (best completed),
Iter24 (best of inverse-length family), and Iter25 (one-factor replication of the
same fixed closed-form mapping with a learnable scale anchor).

It is **not** directly comparable to Iter17 (manifold replacement — out of scope
for FCCR-1) or to Iter16 itself (uses the calibrated `[0.001, 0.932889, 1.0]`
residual vector, which the bridge hypothesis marks as a different mechanism).