# gate_decision_iter27 — ITERATION_ABORTED_INFEASIBLE

## Decision: ABORT

- **STATUS**: `ITERATION_ABORTED_INFEASIBLE`
- **ABORT_STAGE**: S08_MVG (MVG round 1 — Layer D scaling sweep)
- **ABORT_REASON**: Registered `trust_radius_fraction = 0.5` produces
  `clip_frac = 0` across all tested regimes (200 MVG steps + 1000-step
  diagnostic probe).  Activation requires tightening the registered
  constant, which violates SKILL.md §2.9 "registered-spec immutability
  after hypothesis lock".

## Direct evidence

| Test | clip_frac | max δ | Source |
|---|---:|---:|---|
| MVG 5-step ON | 0.000 / 0.000 / 0.000 | 1.17e-02 | `mvg_check_iter27.log` |
| MVG 200-step ON/OFF loss delta | 0.018 (> 1e-6) | — | `mvg_check_iter27.log` |
| MVG 200-step codebook max-abs-diff | [0.146, 0.126, 0.098] | — | `mvg_check_iter27.log` |
| 1000-step probe @ trust_frac=0.5 | 0.000 / 0.000 / 0.000 | 2.64e-02 | `mvg_check_iter27.md` Layer D |
| 1000-step probe @ trust_frac=0.05 | 0.000 / 0.008 / 0.065 | 2.43e-02 | `mvg_check_iter27.md` Layer D |
| 1000-step probe @ trust_frac=0.02 | 0.072 / 0.445 / 0.914 | 2.98e-02 | `mvg_check_iter27.md` Layer D |

The mechanism is **geometrically defined** (gradient path OK,
projection runs, codebook diverges from OFF) but **operationally
inactive** (zero interventions across 1200+ tested steps under the
registered constant).

## Why this is not a clean scientific negative

The iter27 mechanism hypothesis was **never tested**, because the
mechanism never engaged.  Per SKILL.md §22, this iteration does not
count toward the 3-clean-iteration Global Review trigger.

## Promotion status

- `PROMOTION_FAIL` is **not** applicable (no Stage2/Stage3 run).
- This is **not** `MECHANISM_FAIL` (the mechanism was inactive, not
  harmful).

## Next iteration objective

A future iteration (e.g. iter28) MAY re-register the same mechanism
under CAO-1 with a tighter `trust_radius_fraction ∈ {0.05, 0.02,
0.01}`.  That iteration must:

1. Inherit iter18 as parent (not iter27).
2. Run its own S00–S08 deliberation with the new constant registered
   in `mechanism_contract.json` and `hypothesis.md` at S02/S04.
3. Verify that `clip_frac ∈ [0.05, 0.95]` under the registered constant
   before launching Stage2.

Do NOT retune inside iter27.  Do NOT silently launch Stage2 with a
different constant.

## Evidence files

- `logs/iteration_abort_iter27.md`
- `logs/mvg_check_iter27.log` (raw MVG output)
- `logs/mvg_check_iter27.md` (analysis with Layer D scaling sweep)
- `logs/preflight_contract_iter27.log` (CAO-1 preflight PASS)
- `logs/deliberation/S08_MVG/round_2/judge.md` (`ABORT_ITERATION` verdict)
- `logs/deliberation/S00..S09/round_1/judge.md` (canonical S00–S09 artifacts)