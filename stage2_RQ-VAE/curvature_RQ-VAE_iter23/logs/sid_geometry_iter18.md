# SID and Stage2 report — iter18

## Training and export

`logs/train_migrated.log` records successful completion at global step 100,000 (elapsed 1,346.2 s), with `rqvae_best.pth` saved at step 100,000. The final raw SID export has shape `(24587, 3)`. The four-token Stage3 export has shape `(24587, 4)` and completed with `SID_WIRING_PASS`; `item_sids.json` contains 24,587 item entries.

The final training summary reports `full_gini=0.0642`, per-layer Gini `[0.1896, 0.2173, 0.2015]`, 22,951 unique three-token SIDs, 14,812 unique `(L0,L1)` pairs, and `H(L1|L0)=5.5907` bits. The three-token collision count is 1,636 rows (about 6.65%). The final logged `hitrate@50=0.9966` is retained only as a descriptive log field; it is not a gate and was not used for a decision.

These are descriptive Stage2 observations only. Under the current Stage2 quality policy, no utilization, entropy, collision, Gini, or hitrate statistic may stop or veto a candidate. Stage3 remains mandatory.

## Mechanism checks

`logs/mvg_check_iter18.log` records `MVG PASS` on one iter8 checkpoint and one `(640,768)` batch. The initialized curvatures were `[0.050085, 0.244323, 0.273815]`, normalized layer scales `[0.001, 0.932889, 0.9999]`, and fixed AdamW `β₂` values `[0.998991, 0.990604, 0.990001]`. The required loss components retained gradient paths; quantizer embedding gradients were nonzero for all three layers. Five-step relative parameter updates were `[0.02443, 0.23058, 0.25270]`. The same-seed 200-step ON/OFF loss difference was `0.01205`, greater than `1e-6`.

The Stage2 run emitted no `Traceback`, `ERROR`, `FAIL`, `NaN`, or `Inf` matches in its training log. The iter18 `layer_norms.json` file was absent, so the existing fallback used iter1 residual scales `[0.001, 0.932889, 1.0]`; the initialized curvatures and β₂ values above reflect that fallback. This is a provenance caveat, not an unreported calibration result.

## Downstream status

The full Stage3 run completed all 150 epochs and its final beam-20 evaluation reported `test_recall@10=0.05988962203380978` (`n_eval=57439`), below the strict `>0.065` target by `0.005110377966190224`. This is a downstream NO-GO; the Stage2 metrics above remain descriptive and did not gate the run. See `stage3_outcome_iter18.md` and `gate_decision_iter18.md`.
