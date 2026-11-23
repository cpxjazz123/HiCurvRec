# RUN_LOG — commands, exit codes, costs (issue #245)

Interpreter: `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` (repo rule §7).
All runs are CPU-only; `CUDA_VISIBLE_DEVICES=""` was set for the asset build. No GPU job was started.

| command | exit | wall | note |
|---|---|---|---|
| `build_assets.py` (final) | 0 | 57.1 s | 26 assets + manifest, 26/26 verified |
| `baseline_report.py` (final) | 0 | not timed separately | `baseline_tiger.json` written |
| `verify_assets.py` | 0 | < 1 s | 26/26 |
| all nine suites in one loop | 0 | 13.6 s | 102/102 (`test_isolation.py` alone: 2.4–3.2 s) |
| `test_mutation.py` (final) | 0 | 50.3 s | 17/17 caught |
| earlier `test_mutation.py` runs | 0 / 1 | 145.8 s, 50.9 s | 16/17, then 15/17 while a test was broken; both recorded as failures in that run's table |

Space: assets 29.8 MB on disk (largest: BFS cache 14.0 MB, 2-hop ego 5.1 MB, undirected graph 1.4 MB,
#236 curvature copy 1.3 MB). Nothing is committed as a binary: `*.npz`/`*.npy`/`*.csv` under
`results/research/` stay gitignored (§92) and are rebuilt by `build_assets.py`.

Intermediate directories created and removed during the mutation run:
`results/research/infra245/_mutation/<mutation>/` (inside the repo, never `/tmp`, §86).

## Build history worth knowing

| version | changed | why |
|---|---|---|
| 1 | graph, ego, banks, distances | first complete build (44.3 s) |
| 2 | + motifs, probe, eval banks, PPR | issue §3/§5 coverage (52.2 s) |
| 3 | banks blocked against mirror-held-out edges | `test_isolation` caught a real leak (57.2 s) |
| 4 | degree-stratified table replaces a tautological "control" | the old control recomputed the observed rate |
| 5 | ego blocks carry hop tags | contract requires root/hop marks (57.1 s) |

Each version kept the same file names; the manifest hash changes, so a stale copy cannot pass
`verify_manifest`.
