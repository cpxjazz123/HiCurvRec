# REPORT — what was built, measured and verified (issue #245)

Scope respected: this pass builds and freezes infrastructure. It trains nothing, re-trains nothing,
and does not touch TIGER weights (the issue forbids starting training).

## 1. Deliverables

| deliverable | location |
|---|---|
| shared package (10 modules) | `tools/graph_infra/` |
| test suites (9 files, 102 tests) | `tools/graph_infra/tests/` |
| mutation suite (17 mutations) | `tools/graph_infra/tests/test_mutation.py` → `results/research/infra245/MUTATIONS.md` |
| frozen assets (26 entries, 29.8 MB, hash-verified) | `results/research/infra245/assets/` |
| rebuild entry | `python3 tools/graph_infra/build_assets.py` (57.1 s, CPU only) |
| baseline diagnostics | `results/research/infra245/baseline_tiger.json` |
| contract / inventory / infra / verdict / reuse | `results/research/infra245/{CONTRACT,INVENTORY,INFRA,VERDICT,REUSE}.md` |

## 2. Frozen data

| quantity | value |
|---|---|
| items / users | 24,587 / 57,439 |
| visible / eval users | 51,695 / 5,744 (`USER_SPLIT_SEED = 245001`) |
| full directed edges | 323,633 — **equal** to `behavior_graph/metadata.json` |
| undirected simple edges | 315,391 — **equal** to the canonical metadata |
| reciprocal pairs / isolated items / components | 8,242 / 31 / 32 (largest 24,556) |
| max / median out-degree | 1,295 / 6 |
| visible / eval edges | 291,954 / 33,971 |
| ego blocks | 1-hop and 2-hop, cap 32, seed 245002, coverage 1.0, 24,587 roots |
| ego hop histogram (2-hop) | 24,587 centres, 290,367 hop-1, 489,310 hop-2 |
| ego empty fraction | 0.00126 = 31 isolated items / 24,587 (consistent) |
| pair banks | 5 visible + 5 eval, 12,000 each, 0 duplicates, 0 self pairs |
| pairs dropped by mirror blocking | direct 104, shared_out 72, shared_in 73, two_hop 25, degree_matched 7 |
| BFS cache | 2,000 fixed roots, 0.485 % unreachable pairs |
| PPR | 256 fixed sources, top-32, restart 0.15, tolerance 1e-8 |

## 3. Structural description (full train graph, descriptive only)

| type | items | coverage | mean degree |
|---|---|---|---|
| chain | 24,469 | 0.995 | 26.5 |
| triangle | 18,643 | 0.758 | 32.3 |
| merge | 11,766 | 0.479 | 10.2 |
| fork | 11,606 | 0.472 | 10.3 |
| reciprocal | 4,486 | 0.182 | 76.7 |

66 items carry none of the five types. Overlaps are large and reported (`chain+triangle` 18,640,
`chain+merge` 11,717, `fork+merge` 8,037), which is why a later arm cannot present the same ego twice
as two "different graphlets".

Degree strata (rate by degree quintile): chain 0.975 → 1.000, fork 0.771 → 0.020, triangle 0.326 →
1.000, reciprocal 0.016 → 0.577. Fork and triangle move in opposite directions with degree. These are
descriptive rates; **no degree-preserving null was computed** in this pass and the report says so.

## 4. Baseline TIGER under the new caliber

| metric | value |
|---|---|
| SID collision (first three codes) | 0.1693 (matches the training log's 0.169276) |
| duplicate full-tuple items | 0.0 |
| codebook usage / entropy (L1, L2, L3) | 249 / 256 / 256 used; 4.82 / 4.54 / 5.51 bits |
| prefix@1 on visible banks | direct 0.0833, shared_in 0.0478, shared_out 0.0383, degree_matched 0.0116, two_hop 0.0112 |
| prefix@2 / @3 | direct 0.0081 / 0.0018, all other groups ≤ 0.0037 / ≤ 0.0005 |
| reconstruction error | **NA** — needs the arm's decoder; the infra layer never imports a Stage2 tree |
| relation-strength Spearman | **NA** — needs a model's geometric distance; the banks deliberately ship without one |

## 5. Verification

| check | result |
|---|---|
| full suite | 102 tests across 9 files, all passing |
| mutation suite | 17 mutations, **17 caught**, each by a designated `FAIL` line (crashes and import errors are not counted) |
| manifest verification | 26 / 26 assets re-hashed at build time |
| canonical cross-check | directed and undirected counts reproduce `behavior_graph/metadata.json` exactly, and every recorded source hash matches |
| leak checks | both directions of every held-out edge removed from the training banks; eval banks marked evaluation-only |
| index checks | centre index explicit, hop 0 exactly once per block, offsets monotone, no self loops |

Costs: assets 29.8 MB, build 57 s, suite 14 s, mutation run 50 s, all on CPU; no GPU was used.

## 6. Findings (details and evidence in `INVENTORY.md` §5)

1. the two frozen TIGER exports disagree (20,425 vs 21,873 distinct triples, 17 shared);
2. the §63 record's `l3_collision = 0.2653` cannot be reproduced from its own fields, which yield
   0.1693 — the same value as the Stage3 gate and the training log;
3. three arms exceed the 0.4 collision gate: `g237_struct_rq` 0.9655, `kappa_cost_rq` 0.8251,
   `i234_A6` 0.4119;
4. the #236 curvature is undirected and computed on the full train graph, recorded as its caliber.

## 7. Deliberately not done

* no training or Stage3 run (forbidden by the issue text);
* no dense N×N distance matrix (sparse caches only);
* no degree-preserving rewiring null (would be a separate, expensive pass; absence is stated);
* no curvature recomputation: #236 used the same graph version (identical source hashes), so reuse is
  legitimate, with the full-graph caliber recorded;
* no curvature weight, temperature, gate value or expert scale — those belong to a candidate.

## 8. Reproduce

```bash
python3 tools/graph_infra/build_assets.py                 # assets + manifest, 57 s
python3 tools/graph_infra/baseline_report.py              # TIGER diagnostics
for f in tools/graph_infra/tests/test_*.py; do           # suites
  /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 "$f"; done
python3 tools/graph_infra/tests/test_mutation.py          # 17/17
```
