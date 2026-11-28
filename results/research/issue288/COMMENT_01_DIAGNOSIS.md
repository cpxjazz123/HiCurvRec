## #288 — frozen checklist, R01 diagnosis, and the R01 launch

**Issue body sha256** `c6d1c4485bffec6df641838e444677499eebd19249782c35c39ea72cf6cde5f1`.
**INFRA_MODE = FROZEN** — the issue states it verbatim ("单Agent，INFRA_MODE=FROZEN…不修改公共infra").
**Training required: yes** — the acceptance metrics can only be produced by new checkpoint weights
(the issue demands ≥10 real 3000-epoch rounds unless all three pass earlier).

**Frozen acceptance (C1–C14 in `results/research/issue288/RUN_LOG.md`)**: one checkpoint export with
P > 0.08097285816257246, N < 0.008752316503966327, C < 0.1511367795989751 (the frozen #277 evaluator,
57,439 validation rows); ≥10 rounds R01–R10; the mechanism boundary (the curvature only selects, the
distance ranking is the only new supervision); the frozen 768/32/256×3/Sinkhorn/export protocol;
FROZEN infra; no Stage3.

**Assets reused by hash** (`results/research/issue288/INFRA.md`): `assets/manifest.json`
`cfbd9d9c…`, the triplet/curvature shards, the visible undirected graph, `sentence_t5.npy`, the frozen
TIGER checkpoint `af8efc29…` / SID `025ad8bd…`, the #287 same-anchor index `3e2cad7f…`. No asset was
rebuilt.

### The R01 diagnosis (`DIAGNOSIS.md`, `DIAGNOSIS_DATA.json`)

| export | P | N | C | in-bucket behavioural purity | unshared related pairs |
|---|---|---|---|---|---|
| TIGER | 0.092533 | 0.011457 | 0.169276 | 0.3450 | 94.58 % |
| R2 (#274) | 0.078361 | 0.008752 | 0.151137 | 0.3342 | 96.34 % |
| R3 (#274) | 0.080973 | 0.009017 | 0.154350 | 0.3331 | 96.26 % |
| #287 R1 | 0.083532 | 0.009392 | 0.150933 | 0.3534 | 95.43 % |
| #287 R7 | 0.085935 | 0.009970 | 0.153821 | 0.3542 | 94.70 % |

1. **The purity is the ceiling, and it is ~33–35 % in every export.** The mechanisms changed the
   bucket *sizes* (N) but barely *which* co-sharers are behavioural (+2 pp over five mechanisms). P is
   the target-in-bucket rate, so a bucket-size reduction removes related and unrelated items alike —
   this is the measured P/N linkage (#277 anchor-conditioned N, never the uniform-pair formula).
2. **The curvature's within-stratum information is weaker than the degree sum**: the pooled
   within-stratum AUC over 2.84 M distance-1/2 pairs is **0.7291** (curvature) vs **0.8707**
   (degree-sum control). The curvature screen is a scale proxy, as #287 measured.
3. Status: H70 (purity ceiling) SUPPORTED, H71 (curvature adds information) REFUTED, H72/H73
   (final-q shaping / same-anchor P gain) SUPPORTED from the historical exports, H74 (the R01
   combination) UNTESTED.

### R01 mechanism (one main change, pre-registered)

The **pair construction**: the #274 closed/open outer pairs → the **same-anchor** pairs (reusing
`results/research/issue287/pairs_same_anchor.npz` unchanged); the supervised representation stays the
#274 **final hard q** (all three levels), the measured best N/C form. The margin is the frozen rule's
value on the frozen TIGER final q: **0.00470653947787214** (`MARGIN_R01.json`). The gradient path: the
three codebooks, never the encoder (asserted by the tests).

Tests: 12 deterministic known-answer tests pass; 8 mutations all caught by a named assertion
(`MUTATIONS.json`, schema `genrec.issue288.r01.mutations/1`). The tree is
`stage2_RQ-VAE/TIGER_sameanchor_finalq_rq/`; the previous #287 round-7 tree is archived.

**R01 launched**: `stage2_start`, dispatch `stage2_8b87d02e2dba46338495e26cf9917cc4`, single card
(1×A100-40GB visible → the single-card branch of the frozen recipe, global batch 4096), 3000 epochs,
seed 42. The checkpoint-bound export + the frozen #277 evaluation + the round record follow
automatically. Predicted: the same-anchor P gain retained while C ≤ 0.154 and N ≤ 0.0092 (H74).
