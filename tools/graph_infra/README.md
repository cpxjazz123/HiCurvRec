# tools/graph_infra

Shared, model-independent graph and geometry infrastructure (issue #245).

Read first: `results/research/infra245/CONTRACT.md` (frozen conventions), then
`INFRA.md` (module map), `INVENTORY.md` (assets + known errors) and `REUSE.md` (quick start).

```bash
python3 tools/graph_infra/build_assets.py      # rebuild the 26 frozen assets, verify all hashes
python3 tools/graph_infra/verify_assets.py     # re-verify without rebuilding
python3 tools/graph_infra/baseline_report.py   # TIGER baseline under the unified caliber
for f in tools/graph_infra/tests/test_*.py; do
  /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 "$f"; done
```

No module here imports a model, a checkpoint or a Stage2/Stage3 entry point. New experiments import
this package instead of copying geometry or graph code into their own issue directory.
