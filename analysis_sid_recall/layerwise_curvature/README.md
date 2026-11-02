# Layerwise behaviour curvature (Issue #129)

What the measured geometry of the SID prefix levels can and cannot say about
Stage 3 Recall, with the behaviour-alignment and code-distribution controls, the
joint and nonlinear models, the original-graph intra/inter gap, and the verdict
on the variable-curvature arm.

## How to reproduce

```bash
# measurement (features, provenance, the original-graph gap); needs the
# accelerator interpreter only because the exact transport uses POT
python3 analysis_sid_recall/layerwise_curvature/run.py

# statistics, figures, report from the produced table (system interpreter,
# no TensorFlow): seconds, not minutes
python3 analysis_sid_recall/layerwise_curvature/rescore.py

# the contract tests
python3 -m unittest discover -s analysis_sid_recall/layerwise_curvature/tests
```

## Artefacts

| file | contents |
|---|---|
| `experiment_table.csv|json` | one row per (SID SHA-256, Stage 3 run): provenance tier, g1/g2/g3, z1/z2/z3, A1/A2/A3 with their positive and negative rates, coverage, code usage, collision, Delta1/2/3 with their nulls and counts, recall, n_eval |
| `results.json` | univariate statistics with Holm, controls, multicollinearity, joint model with leave-one-series-out and cluster bootstrap, centred quadratic fits, gap tests, tier sensitivity |
| `gap_table.json` | the same gap numbers, per arm |
| `REPORT.md` | Q1-Q4 answered separately, including the negative and undefined outcomes |
| `PREREGISTRATION.md` | the WP4 mapping and the matched arms, fixed before any training, and why no arm was launched |
| `scatter_g.svg`, `scatter_z.svg` | per-level scatter panels (g_l and z_l), coloured by parent series |
| `orc_sample_meta.json` | the fixed item-graph edge sample's coverage |

## Rules this analysis obeys

* TRAIN only: the behaviour graph is the training split's co-occurrence graph
  (symmetrised, #128), and the alignment score samples observed training edges
  against degree-matched non-edges; nothing reads validation or test labels.
* One row per (SID SHA-256, run); the join is never on the variant name.
* `z_l` is the size-preserving-null standardised curvature; `g_l` is the raw
  quotient curvature. They are different statistics and are both reported.
* Where a statistic is undefined (the L3 singleton trap, an unlabelled level) the
  table says `UNDEFINED_*` instead of a zero.

## Limits

* Provenance is content-hash based: 108 of 110 pairs match their recorded hash but
  not the Stage 3 run's own stored path, so they are content-verified rather than
  path-attested (`HASH_PROVEN` 0, `RECORD_HASH_MATCH` 108, `MISSING` 2).
  `attest.py` is the fail-closed verifier a prospective Stage 3 must call.
* The degree-preserving rewiring null is `NOT_MEASURED` for the gap (the exact
  transport needs POT and the shared filesystem stalled); the prefix-label-shuffle
  null is measured per arm.
* Delta2 and Delta3 are defined for 31 and 10 arms respectively, so they carry no
  conclusion.
