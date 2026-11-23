# REUSE — the 10-line version

```python
import sys; sys.path.insert(0, "<repo>")
from pathlib import Path
import json, numpy as np
from tools.graph_infra import assets, pairs, metrics, graphlet, topology, graph_distance

root = Path("<repo>/results/research/infra245/assets")
manifest = root / "manifest.json"
assets.verify_manifest(manifest)                      # fail closed on any drift

visible = assets.load_sparse(root / "graph" / "visible_forward.npz")
bank    = np.load(root / "pairs" / "bank_two_hop.npy")   # frozen pairs, do NOT resample
probe   = np.load(root / "pairs" / "probe.npz")           # frozen gradient probe + graph hops

# a model's per-pair numbers go in only here; mismatch raises
aligned = assets.join_scores(bank, model_pairs, model_scores)

# metrics with explicit denominators
metrics.spearman(aligned, model_scores)
metrics.subgraph_summary([np.array([0.9, 0.4]), np.array([0.8, 0.7])], threshold=0.5)
```

## Rules that save a rewrite

| do | don't |
|---|---|
| read the manifest and verify it | assume the assets are unchanged |
| use `bank_*` for training/eval and `bank_*_eval` only for evaluation | mix eval banks into supervision |
| pass graph hops next to a model distance | report a distortion without its unit |
| take the centre from `root_local` / `root_local_index` | assume the centre is row 0 |
| use `pairs.exclude_undirected` when adding new pairs | add pairs whose mirror image is held out |
| call `metrics.collision_rate` for the gate caliber | reuse a §63 `l3_collision` value |
| fix `tools/graph_infra/` and re-run its suites | copy the helpers into a new issue directory |

## Which asset for which question

| question | asset |
|---|---|
| structural description (allowed on the full graph) | `graph/graph_*_full.npz`, `motifs/motif_labels.npz` |
| generalisation (visible data only) | `graph/graph_visible_*`, `pairs/bank_*`, `distance/*` |
| held-out evaluation | `graph/graph_eval_forward.npz`, `pairs/bank_*_eval.npy` |
| distance sanity | `distance/bfs_visible_root_cache.npz` (unweighted hops, `+inf` = unreachable) |
| propagation baseline | `distance/ppr_visible_sources.npz` (a score, not a distance) |
| frozen gradient probe | `pairs/probe.npz` |

## Rebuild after any change

```bash
python3 tools/graph_infra/build_assets.py      # rewrites manifest + verifies 26/26
python3 tools/graph_infra/baseline_report.py   # TIGER diagnostics under the new caliber
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 tools/graph_infra/tests/test_mutation.py
```
