# Issue #283 — INFRA (FROZEN)

## Assets used (referenced by hash; nothing rebuilt, nothing modified)

| item | path | sha256 |
|---|---|---|
| triplet shards (13) | `assets/shards/shard_000..012.npz` | per `assets/manifest.json`; total 28,391,948 rows, all valid |
| index | `assets/index.npz` | `a5eb2935…e656e1` |
| visible directed graph | `assets/train/visible_forward.npz` | `ac7155d1424a6a346da9b8ed6e9413b81d43223a377308af0c8ba35b6b773029` |
| visible undirected projection | `assets/train/visible_undirected.npz` | manifest `097a719a…` |
| item ids | `assets/train/item_ids.npy` | manifest |
| 253 SIDs / 129 labels / folds | `results/research/issue276/{SID_TABLE,FOLDS}.json` | `table_sha256=4c927914…` |
| #282 type audit/table | `results/research/issue282/{TYPE_AUDIT,TYPE_TABLE}.json` | committed a927098b |

Integrity spot checks: `visible_forward`, `visible_undirected`, `shard_000`, `index` hashes equal the
manifest entries; the 13 shards total 28,391,948 rows, `valid` all True; recomputing the 6-bit directed
code and `canonical_code` from `visible_forward` reproduces the stored `structure_class` on every row
(asserted in `rows283.py`).

## Environment

- Interpreter: `/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10` (Stage0/1/3 env).
- GPU: not used (analysis-only). CPU: 8 cores, multiprocessing over SIDs.
- No training entrypoint, launcher, port, MCP server or public tool was modified.

## Protected-component snapshot

`INFRA_SNAPSHOT_START.json` = `tools/iteration.integrity.snapshot(PROTECTED)` (tools/iteration,
tools/research/stage2, tools/research/verification, tools/graph_infra, tools/issue252, tools/issue253,
assets/manifest.json, Stage3 entry, CLAUDE.md, this skill). Close-out comparison must show
added=removed=changed=0.
