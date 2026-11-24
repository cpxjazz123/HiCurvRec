# Issue 250 infrastructure audit

PASS: read assets/manifest.json v5-gnn-geometry-249; all 67 registered SHA-256 values recomputed and matched. Exact reused paths/hashes/use appear in ACCEPTANCE.json, not a second binary asset copy.

PASS: frozen TIGER source read-only; candidate copied only from stage2_RQ-VAE/baseline/TIGER_RQ-VAE into stage2_RQ-VAE/TIGER_gnn_gate_sectional_rq. Existing unrelated active directories were not moved or deleted. Any older source trees are outside this issue's modification scope.

PASS: 4 physical NVIDIA L40S, 46068 MiB each are visible. Short numerical/gradient/entry/DDP validation is authorized; formal Stage2/Stage3 dispatch is forbidden by this issue.

Interpreter: /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9. Existing torch/numpy/scipy and shared graph utilities reused. No new dependency installation, no PPR, no all-four-node census.

Reused target: 28,391,948 rows across 13 shards, same full centre-component reference sets. Existing summary variance/mean/valid counts determine RMS target scale without recalculating graph targets. Existing embedding, train/item and TIGER SID hashes are preserved through the root manifest.

No new static asset is required by the architecture: fixed node features, standardization, graph, ego and targets already exist. Model-dependent tangent scale, lambda, GNN weights and validation outputs are dynamic candidate state/evidence, not regenerated shared graph data.

The per-step full-reference cost and peak memory must be measured before admission; no sample-only target is substituted if the full cost is large.
