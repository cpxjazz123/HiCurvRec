# S08 MVG (round 2) Source Packet — iter27 ABORT

## Direct evidence that the registered mechanism is inactive

MVG (round 1) ran under the registered `trust_radius_fraction = 0.5`:

| Layer | clip_frac (5 ON steps) | max δ |
|---|---:|---:|
| layer 0 | 0.0000 | 1.17e-02 |
| layer 1 | 0.0000 | 1.14e-02 |
| layer 2 | 0.0000 | 1.14e-02 |

Follow-up diagnostic probe (1000 steps, iter18 ckpt regime):

| `trust_frac` | avg clip_frac (L0/L1/L2) |
|---:|---:|
| 0.5 (registered) | 0.000 / 0.000 / 0.000 |
| 0.1 | 0.000 / 0.000 / 0.000 |
| 0.05 | 0.000 / 0.008 / 0.065 |
| 0.02 | 0.072 / 0.445 / 0.914 |
| 0.01 | 0.575 / 0.983 / 1.000 |

Under the registered `trust_frac = 0.5`, the trust radius is `~10×`
larger than the per-step AdamW displacement, so the projection never
clips.  The mechanism is **geometrically defined** (gradient path OK,
projection runs, codebook measurably diverges from OFF) but
**operationally inactive** (zero interventions across the entire 200-step
MVG and the 1000-step probe).

## Why same-iteration repair is not allowed

Activating the mechanism requires changing the registered constant
`trust_radius_fraction` from `0.5` to at most `0.05` (10× tighter).  The
user's hypothesis explicitly fixed `τ_l = D_l / 2`.  Per skill §2.9
"Registered-spec immutability after hypothesis lock", this is a
violation of the one-factor rule and must trigger `ABORT_ITERATION`.

## What remains scientifically unresolved

- Whether a post-step hyperbolic codebook trust-region with a tighter
  radius (e.g. `τ_l = D_l / 20` or `τ_l = D_l / 50`) can improve Stage3
  `R@10` over iter18's `0.0599`.
- Whether the iter18 optimizer trajectory already satisfies the
  trust-region's safety constraint at every step (i.e. iter18 is
  already "trust-region safe" by construction).

## Reference

- iter18 baseline `R@10 = 0.05988962203380978` (n_eval=57439).
- `mechanism_contract_iter27.json` (registered `trust_radius_fraction = 0.5`).
- `hypothesis_iter27.md` (user-spec: `τ_l = D_l / 2`).
- `mvg_check_iter27.log` (5 ON steps; clip_frac = 0).
- `mvg_check_iter27.md` (full diagnostic + scaling sweep).

## Required fields for the abort artifact

```
STATUS=ITERATION_ABORTED_INFEASIBLE
ABORT_STAGE=S08_MVG
ABORT_STEP=MVG round 1 — Layer D scaling sweep
EVIDENCE=mvg_check_iter27.md (Layer D table); 1000-step probe result
REASON=user-fixed τ_l = D_l/2 produces clip_frac = 0 across 1200 steps; activation requires changing the registered constant
NEXT_ITERATION_CONSTRAINTS=parent remains iter18; register trust_frac in {0.05, 0.02, 0.01} as a NEW iteration (e.g. iter28) with its own S00–S08 deliberation; do not retune inside iter27
```