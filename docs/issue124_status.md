# Issue #124 — status: the false positive is reproduced and fixed, the candidate's own path is verified, and the run finished with a negative result

## Workstream A — the false positive, the proof, and the fix

The persisted artefact `probes/successor_entropy_curvature/math_checks.json` (schema 1, `overall = PASS`) still says, verbatim:

```text
both-on assignment [0, 0, 0], behaviour-off [0, 0, 0], geometry-off [0, 1, 0], the switches change the assignment: True
```

The behaviour-off assignment is **identical** to both-on, and the verdict was an `OR` over two interventions, so the geometry-off branch alone satisfied it. The invariance that makes this a false positive is exact, not empirical:

```text
argmin_k ((1 + w_i) * d(i, k)) == argmin_k d(i, k)      for w_i > 0
```

A positive row-scalar weight scales a whole row of the score matrix and cannot move its arg-min; a codeword-independent offset cannot either. Both are measured, not asserted: `tools/research/rounds/revalidation.py::row_scalar_invariance` reports `assignments_identical: true` on the numbers it applies to.

The shared checker now reports four scopes separately (`tools/research/rounds/math_checks.py`):

| check | scope | status it can reach |
|---|---|---|
| `geometry_operator_valid` | operator | `PASS` — maps invert, curvature negative, domain guard holds, distance not Euclidean |
| `behavior_dependency` | behaviour | `PASS` only when a **codeword-dependent** behaviour term moves the assignment; a row-scalar or offset makes it `FAIL`/`INCONCLUSIVE` |
| `geometry_dependency` | geometry | `PASS` when the matched Euclidean intervention moves the top-1 indices |
| `joint_path_dependency` | candidate path | `CANDIDATE_PATH_NOT_RUN` in the probe; exercised by the candidate admission check |

Re-running the **fixed** checks on the same persisted proposal gives `overall = INCONCLUSIVE`, not `PASS`: the operator evidence stands, and no behaviour-dependence claim survives it. That is recorded in a **new, non-mutating** artefact with back-references to the original hashes:

```text
results/research/issue121/rounds/round_20261025_181918_856/probes/successor_entropy_curvature/math_validation_audit.json
```

The original `probe.json`, `math_checks.json`, `decision.json` and the `3.350` selection record are untouched (`test_the_original_artifacts_are_untouched`).

## Workstream B — the candidate's own path is verified

`results/research/issue121/trees/round_20261025_181918_856/successor_entropy_curvature/CANDIDATE_CHECK.json`:

```text
verdict                   VERIFIED_COUPLING
behaviour_moved_assignments 256 / 256
geometry_moved_assignments  256 / 256
curvature spread           0.1624 … 0.5461  (item-specific, from train-only entropy)
mean_margin_both_on        0.2954   (behaviour-off: 0.3957)
codebook gradient          150.13   encoder gradient 26743.3   finite
negative_control_rejected  true
```

The negative control — a candidate whose behaviour signal is disconnected — is rejected by the checker, so the verdict cannot be produced by a dead path.

The proposal's contradiction (curvature-conditioned quantizer vs. fixed-curvature warp followed by an unchanged Euclidean assignment) is resolved in the persisted `reconciliation` field; the frozen scoring decision and its revision chain (`decision.json`, `decision_rev2..4.json`, `audit_decision_chain.json`) are unchanged.

## Workstream C — the run finished, and it is a negative result

```text
Stage 2   3000 epochs, 4 GPUs, global batch 4096
SID       results/stage2_RQ-VAE/TIGER_successor_entropy_curvature/item_sids.json
Stage 3   NUM_EPOCHS=150, SEED=42, BEAM_SIZE=20, n_eval=57439
          test_recall@5  0.034280    test_ndcg@5  0.022753
          test_recall@10 0.050628    test_ndcg@10 0.028022
target    test_recall@10 > 0.065     passed = false
```

Comparison with the recorded comparators: `TIGER_bootstrap_code_stability` 0.063354 and `transition_busemann_gate` 0.054823 are the frozen four-card numbers; `branch_shell` 0.055851 is single-card exploratory and is not directly comparable. The candidate lands **below** `transition_busemann_gate` and **above** the HGS/branch-geodesic family (0.0457–0.0497), i.e. it is a real, fully measured negative result, not a missing measurement.

The matched `behavior_off` / `geometry_off` / `both_on` Stage 3 ablations were **defined and recorded, not run**: the repository's experiment policy (§19g) forbids training ablation arms, so no ablation arm was trained and none is claimed.

## What the #125 forensics adds to this

The per-token and beam-level forensics (see `results/research/sid_generation_forensics/report.md`) decomposes this candidate's Stage 3 result on the frozen protocol:

```text
successor_entropy   NLL T1 3.45  T2 2.33  T3 2.29  T4 0.98
                    raw beam coverage 0.0688   alive L1 0.735  L2 0.400  L3 0.133  L4 0.069
                    recall@10 0.0512   absent 0.931
```

The candidate is **not** defeated by a collapsed first level: its L1 prefix survives in 73.5% of the beams and its first token is the hardest (NLL 3.45, accuracy 0.262). Its loss is spread across all four positions and the beam prunes the gold prefix early — which is why its SID metrics looked better than its recall.

## Remaining unknowns

- Whether the behaviour-conditioned curvature is the part that hurts: the ablation arms are not trained, so the decomposition above is correlational.
- Whether the 13.4k-cell hierarchy supervision of #127 changes the picture: that experiment is running now with the sampler defect fixed.
