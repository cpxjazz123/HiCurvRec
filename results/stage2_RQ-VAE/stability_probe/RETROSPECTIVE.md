# Retrospective comparison: healthy vs collapsed arms, identical frozen probe protocol

Protocol for every arm: `probe_config` at its documented defaults (seed 42, 32 baseline steps,
128 behaviour-active steps, snapshots at 0/8/16/32/64/96/128/160), the arm's own declaration,
single process, the real loaders/loss/optimizer/clipping. Every arm reported
`status = valid` with the mechanism actually executing on 112 steps.

| arm | outcome | final Hard-L1 ER | status | risk | L1 used | L1 eff | L1 coll | 3-level coll | neg same-code | max deep/base |
|:--|:--|--:|:--|:--|--:|--:|--:|--:|--:|--:|
| `r4_08` | healthy | 16.402628 | valid | high | 88 | 18.2 | 0.9964 | 0.4769 | 0.0734 | 15.19 |
| `r6_02` | collapsed | 2.8902618750708537 | valid | high | 72 | 12.0 | 0.9971 | 0.7893 | 0.1169 | 10.21 |
| `r6_03` | collapsed | 1.8233611326609775 | valid | high | 79 | 11.1 | 0.9968 | 0.8158 | 0.1203 | 10.79 |
| `r6_04` | collapsed | 1.5416677050365089 | valid | high | 32 | 4.1 | 0.9987 | 0.9386 | 0.2547 | 6.54 |

## What the comparison shows

**The binary risk indicators are NOT discriminative.** The healthy reference arm `r4_08`, whose
historical Hard-L1 ER is 16.402628, also reaches `risk = high` under these indicators
(`l1_effective_codes_drop` and `deep_gradient_domination` fire for it as well). A high verdict
is therefore not evidence of collapse, and no automatic abort may be based on it. This is
recorded in every probe report under `risk_calibration`.

**The raw trajectory does separate the arms.** Two quantities are ordered with the final ER
across all four arms:

- three-level tuple collision at step 160: healthy 0.4769 against 0.7893 / 0.8158 / 0.9386,
  i.e. monotone in 16.40 > 2.89 > 1.82 > 1.54;
- train-only negative same-code rate at step 160: 0.0734 against 0.1169 / 0.1203 / 0.2547,
  likewise monotone.

**This is an observation, not a predictor.** One healthy arm and three failures cannot
calibrate a threshold; the issue requires a split of calibration and evaluation arms and
prospective confirmation before any predictive claim, and neither exists yet. The numbers
above are reported as evidence that the probe measures something real, not as a cutoff.

## Availability

The comparison IS rerunnable: every arm above has an executable tree, a declaration and a
stored probe report, and the protocol is fixed in `DEFAULT_PROBE_CONFIG`. The historical ER
values come from each arm's own `stage2_metrics.json` / `gate.json`.

## Consequences for the workflow

- The probe is recommended after CGV and before a 3000-epoch run, because 160 steps already
  produce the trajectory that took a full run to reveal.
- Its `risk` field is advisory. It must not cancel, approve or retune a run, and the frozen
  Hard-L1 ER remains the only Stage3 admission criterion.
- Calibrating the indicators requires predeclared arms with a calibration/evaluation split;
  until that exists, the report keeps saying `uncalibrated`.
