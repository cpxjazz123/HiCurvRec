# REUSE v2 — how a later experiment uses the frozen assets

```python
import sys; sys.path.insert(0, "<repo>")
import json, numpy as np
from tools.graph_infra import assets, config, graphlet, isolation, metrics, pairs, ppr

root = config.ASSET_ROOT
manifest = root / "manifest.json"
assets.audit_manifest(manifest)                 # schema, legal use, version
assets.verify_manifest(manifest)                # hashes; never rebuilds, never writes

visible = assets.load_sparse(root / "graph" / "visible_forward.npz")
bank    = np.load(root / "pairs" / "train_two_hop.npz")     # pairs + per-pair graph features
probe   = np.load(root / "pairs" / "probe.npz")             # frozen gradient probe
ego     = np.load(root / "ego" / "ego_full_1hop_visible.npz")
motifs  = np.load(root / "motifs" / "motif_counts.npz")
```

## Rules

| do | don't |
|---|---|
| read the v2 manifest and verify it | mix v1 (`results/research/infra245/assets`) and v2 files |
| take ego/degree/motif/distance/PPR/reference points from the **visible** graph | take them from `full_description` (that one is descriptive only) |
| block held-out pairs in **both** directions (`isolation.held_out_support`) | use weighted subtraction on sparse matrices |
| keep `pair_directed` and `pair_undirected` apart | join on positions; use `assets.join_scores_v2`, which joins by pair id |
| report `collision_prefix3` and `duplicate_prefix3` separately | report one "collision" number |
| state `direction=` for every `subgraph_summary` call | assume "lower is better" |
| read the frozen curvature from `curvature/curvature_visible.json` and keep its unit | transplant the #236 curvature to the visible graph (it is full-graph and undirected) |

## Which asset answers which question

| question | asset | legal_use |
|---|---|---|
| what does the whole train graph look like | `graph/full_description_*` | description |
| what may be propagated, supervised or calibrated on | `graph/visible_*`, `pairs/train_*`, `distance/*`, `curvature/*` | generalisation |
| what may be evaluated on, unseen | `pairs/strict_*` | evaluation |
| user-isolated but edge-seen evaluation (reported separately) | `pairs/relaxed_*` | evaluation |
| local subgraph of an item | `ego/ego_full_1hop_visible` (complete), `ego/ego_sampled_visible` (budgeted) | generalisation |
| typed local patterns | `motifs/motif_counts` (exact 3-node counts), `motifs/motif_instances_visible` (sampled instances) | generalisation |
| gradient calibration probe | `pairs/probe.npz` | generalisation |

## Rebuild and verify

```bash
python3 tools/graph_infra/build_assets_v2.py      # rebuilds v2, writes the manifest, verifies hashes
python3 tools/graph_infra/verify_assets.py --version v2
python3 tools/graph_infra/baseline_v2.py           # TIGER baseline on the v2 banks
python3 tools/graph_infra/tiger_audit.py           # SID provenance, read-only re-export
```
