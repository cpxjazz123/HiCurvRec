# Evidence probe demo — round `round_20261025_181918_856`

A complete, runnable example of the pre-score evidence probe (Issue #123):
ideation → scope screen → probe → weighted reviews → pairwise debate → frozen
decision. Everything below was produced by the real pipeline on the real train
split; the artifacts are under
`results/research/issue121/rounds/round_20261025_181918_856/`.

## What the probe measured (train split only)

Source: `results/stage0_build_parquet/train.parquet`
(sha256 `…` recorded in `probes/<idea>/probe.json`), 396,958 rows, 57,439 users,
ordered `history`/`seen_history` present. 20,000 users were read per proposal
(the probe is capped), 21.7 s per proposal against a 120 s budget.

| measurement | value |
|---|---|
| directed train transition edges | 118,315 (115,582 distinct pairs) |
| mean transition entropy | 1.1063 nats |
| mass on the single most popular successor | 0.0044 |
| share of train transitions whose endpoints share the baseline SID's **first prefix** | 0.0907 |
| the same share under a **permuted-history null** (train only, seeded) | 0.0670 |

Reading: the transition signal is *not* popularity (the top successor carries
0.44% of the edge mass), and the baseline's first prefixes keep slightly more
train transitions together than a permuted null (9.07% vs 6.70%) — so there is a
measurable, if modest, amount of transition structure in the existing SIDs for a
prefix-grouping mechanism to work against.

## The other two probe sections

- **Math checks** (small CPU tensors, float64): `log_0(exp_0(v))` round trip
  PASS (max error 2.8e-17, explicitly *not* a curvature-value claim), curvature
  sign PASS (c = 0.5 → sectional curvature −0.5), non-Euclidean distance PASS
  (curved/euclidean ratio 1.4285), domain guard PASS, and a labelled
  **proxy** behaviour-off / geometry-off / both-on switch test PASS (a proxy: it
  shows the switches are meaningful, it is not the candidate's own pathway).
- **Literature** (OpenAlex, 3 mechanism queries per proposal, ≤15 papers kept,
  cached under `results/research/issue121/literature_cache/`): 15 verified records
  per proposal with real DOIs; `HVQ-VAE` and `HyperVQ` are classified
  `same_operator_or_path` — i.e. the hyperbolic-vector-quantization operator
  already exists in the literature, which is prior art the novelty score must
  answer.

## What the reviewers saw and decided

All three reviewers read the **same frozen bundle** (`bundle_sha256` per idea is
in `decision.json`), scored all seven rubric dimensions and produced one
decision.

| proposal | weighted | behaviour evidence | geometry necessity | novelty |
|---|---:|---:|---:|---:|
| `successor_entropy_curvature` | **3.350** | 3 | 4 | 3 |
| `cooccur_curved_balancer` | 2.950 | 2 | 3 | 3 |
| `coconsume_karcher_assignment` | 2.850 | 2 | 3 | 3 |

Pairwise debate (order-balanced A/B, Copeland): `successor_entropy_curvature`
+2, `cooccur_curved_balancer` 0, `coconsume_karcher_assignment` −2; no ties, no
cycles.

**Decision**: `SELECTED` = `successor_entropy_curvature`
(weighted 3.350 ≥ 3.0); backups in order: `cooccur_curved_balancer`,
`coconsume_karcher_assignment` (both `BELOW_MINIMUM_SCORE`). There was no
fallback: the winner cleared the threshold on its own merits.

```
decision.json  schema 2
  rubric        rubric-2026-10-v1  sha256 4031b1c990f8…
  decision_hash adb2546fbf2769da0d51c427110545217b1f3f93f8fddacfd5265d9eb8b88285
  evidence      bundle / probe / literature / math sha256 per proposal
```

## Reproduce it

```bash
# the probe on its own, in the interpreter that has pandas and torch
/home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10 \
  -m tools.research.rounds.probes \
  --idea '{"idea_id":"demo","title":"transition-conditioned hyperbolic assignment"}' \
  --repo . --budget 120 --max-users 20000

# the whole scoring subsystem
/usr/bin/python3 -m unittest tools.research.tests.test_issue122 tools.research.tests.test_issue123
```

## Next action

`successor_entropy_curvature` is the one mechanism that earned the slot. It now
goes through the unchanged path: implement on a fresh copy of
`stage2_RQ-VAE/TIGER_RQ-VAE`, one verifier, then Stage 2 on the frozen 4-GPU
recipe and Stage 3 on a technically valid SID. **The probe says nothing about
whether that mechanism will beat 0.063354** — it says the hypothesis is
measurable, the geometry is non-Euclidean, and the operator already exists in
the literature. That is what the evidence is for.
