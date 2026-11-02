# The variable-curvature arms were not run (Issue #129, WP4)

No Stage 2 or Stage 3 job was submitted for this issue. The issue makes the
matched-arm experiment conditional on the diagnostics supporting an
interpretable level-dependent signal; the diagnostics do not, so launching four
matched Stage 2 runs plus their Stage 3 runs would be a claim-producing
experiment without a pre-registered reason.

## What was recorded instead

* `analysis_sid_recall/layerwise_curvature/PREREGISTRATION.md` — the mapping
  `c_l = clip(1.0 * exp(-0.5 * z_l), 0.05, 4.0)`, the radius/scale convention,
  the detached 20-epoch refresh of `z_l` on the frozen pilot SID, and the four
  arms (E: c_l = 0; S: one shared c; G: one pre-registered one-factor sweep;
  V: the mapping), all with the frozen Stage 2/Stage 3 budgets.
* `analysis_sid_recall/layerwise_curvature/REPORT.md` — Q1-Q4 answered
  separately, including the negative and undefined outcomes.
* `analysis_sid_recall/layerwise_curvature/attest.py` — the byte-exact SID
  attestation a prospective Stage 3 must call at start-up, at every checkpoint
  and before the final test; it fails closed on a hash or item-order mismatch.

## What would have to change before an arm is justified

1. A level-dependent signal that survives the sampling budget and the null
   replicate count (see `analysis_sid_recall/CONVERGENCE_FINDINGS.md`).
2. The joint model improving out-of-series prediction over the behaviour
   alignment and collision controls (see `REPORT.md`, Q2/Q3).
3. A pre-registered one-sided hypothesis and an independent set to confirm on;
   historical recall is observational evidence and cannot establish a causal
   curvature choice.

## Manifest

`manifest.json` lists the arms with `status: NOT_RUN_NOT_JUSTIFIED`, their
intended constants and the protocol they would have used. No historical artefact
under `results/stage2_RQ-VAE/` or `results/stage3_T5Train/` was read for
selection or modified.
