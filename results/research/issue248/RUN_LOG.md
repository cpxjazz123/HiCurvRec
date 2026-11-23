# Issue 248 run log

Title: Infra v4: Full-resolution sectional curvature assets and persistent graph infrastructure (no sampling, no PPR)

Issue body sha256: 4b7e49232875ac219c00995e978e5479d6f19ed1ddbd8d23ffb6175309e70d8d

No training: the issue forbids training Teacher/GNN/RQ-VAE/Stage2/Stage3, forbids PPR and forbids retraining TIGER. Every acceptance metric is recomputed from persisted artefacts or from deterministic graph computations.

## Timeline
- Contract frozen before any code: CONTRACT.md (formula, reference set and exclusion rules, enumeration rule, storage schema, tolerances, isolation).
- Reference implementation written first (results/research/issue248/reference.py, networkx BFS + term-by-term summation, no import of the production module).
- Tests written before the production module; the first run failed on the missing module (red), then the module was written (green).
- Full enumeration built: 28,391,948 triples over 13 shards.
- First verification run failed on a bug in the verifier (the duplicate key ignored the centre); fixed and re-run.
- Second rebuild failed on a manifest-ordering bug (coverage.json was hashed before its final write); fixed and re-run.

## Frozen inputs
See INFRA.md and the manifest `inputs` block.
