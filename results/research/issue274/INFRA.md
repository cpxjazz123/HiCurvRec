# Issue #274 — infra record (INFRA_MODE=FROZEN)

Every line has the form `need -> command -> output summary -> sha256 -> PASS / FAIL / INFRA_FAULT`.
The public components are only called, never edited; the assets are only read, never rebuilt.

## 1. The assets (issue section 1: "只用训练图、已有全量三元组、Graphlet/Orbit/方向、曲率及索引资产")

| need | command | output | sha256 | verdict |
|---|---|---|---|---|
| the full triplet table (the centre, the two neighbours, the raw curvature, the closed/open distance, the structure class) | `python3.9 results/research/issue274/survey_triplets.py` | 28,391,948 rows over 13 shards, all valid, 13 structure classes; the class separates `bc_dist_sq == 1` (closed) from `== 4` (open) exactly | the shards: `1870d81151e3…` (000) … `c519e4fdd3e1…` (012), all MATCH `assets/manifest.json` | PASS |
| the item -> shard index | `numpy.load('assets/index.npz')` | 24,587 items, the total rows 28,391,948 = the expected total | `a5eb2935b258…` MATCH | PASS |
| the visible training graph (the direction flags and the common-neighbour count) | `numpy.load('assets/train/visible_forward.npz')` / `visible_reverse.npz` / `visible_undirected.npz` | 261,132 directed edges; the undirected projection 510,464 edges (the mean degree 20.76, the max 1,810) | `ac7155d1424a…` / `5fb26e2b1b5b…` / `097a719a9f00…` MATCH | PASS |
| the item id map | `assets/train/item_ids.npy` | 24,587 ids | `262f7ce84b27…` MATCH | PASS |
| the frozen TIGER baseline (the SID and the checkpoint) | `results/stage2_RQ-VAE/TIGER_RQ-VAE/{item_sids.json,rqvae_best.pth}` | the SID 24,587 items; the checkpoint's three codebooks `rq.vq_layers.{0,1,2}.embed.weight` | the SID `025ad8bd5809…` | PASS |
| the #265 identity (the L2 10,918) and the #265 same-caliber reconstruction | `results/stage2_RQ-VAE/TIGER_euclid_local_curve_supervision_rq/stage2_metrics/` and the #265 run's `logs/training_metrics.jsonl` | the L2 10,918, the L3 23,573, the collision 0.0412; the final reconstruction 8.180260192602873e-05 | the #265 SID `8961e4a2062e…` | PASS |
| the #269 global J values (the C7/C8 thresholds) | `results/research/issue272/round3_pass_table.json` | the full J 0.338335882630658, the 1-hop 0.798295, the 2-hop 0.278106 | the table is the frozen evidence | PASS |

No asset was rebuilt, copied, renamed or overwritten; the frozen binaries are read in place.

## 2. The public components (the zero-drift baseline)

`results/research/issue274/INFRA_SNAPSHOT_START.json`: the 61 public components
(`tools/iteration/`, `tools/research/stage2/`, `tools/research/verification/`,
`assets/manifest.json`, the Stage3 entry, `CLAUDE.md`, this skill), hashed with
`tools/iteration/integrity.py::snapshot` — the existing interface, no new tool. The end snapshot is
written before the closure and compared with `integrity.compare`.

## 3. The interpreters and the hardware

| need | command | output | verdict |
|---|---|---|---|
| the Stage2 interpreter | `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` | the analysis and the candidate tests run on it | PASS |
| the visible GPUs | `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv` | one visible card (A100, 80 GB), the port 50202 is the frozen Stage2 port | PASS (the single-card branch of the frozen recipe) |

## 4. The tree state before the new mechanism (issue section 1 / §1.0a)

`stage2_RQ-VAE/` holds `baseline/` and `_archive/` only — no old mechanism directory is left at the
top level, so the archive step of the skill is already satisfied. The candidate is copied from the
read-only `stage2_RQ-VAE/baseline/TIGER_RQ-VAE` with `ARM_NAME` as the only constant change
(CLAUDE.md §19b). `stage2_RQ-VAE/baseline/TIGER_BEHAVIOR_SAMPLE_RQ-VAE` is read as the **reference**
for the relative-distance supervision conventions (the issue requires its real code to be checked);
it is never modified and never trained.

## 5. The infrastructure faults

None so far: no public component is missing, and every frozen artefact the issue names is present
and hash-verified.
