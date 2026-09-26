# iteration_abort_iter27 — ITERATION_ABORTED_INFEASIBLE

```
STATUS=ITERATION_ABORTED_INFEASIBLE
ABORT_STAGE=S08_MVG
ABORT_STEP=MVG round 1 — Layer D scaling sweep (5-step ON + 200-step ON/OFF + 1000-step probe)
ABORT_EVIDENCE=mvg_check_iter27.log + mvg_check_iter27.md Layer D table (clip_frac=0 across 1200+ tested steps under registered trust_frac=0.5; scaling sweep shows activation only at trust_frac ≤ 0.05).
WHY_SAME_ITERATION_REPAIR_INVALID=Activating the mechanism requires tightening the registered `trust_radius_fraction` from 0.5 to ≤ 0.05, which would mutate `mechanism_contract_iter27.json::trust_radius_fraction_default` (registered at S04), the user-locked `τ_l = D_l / 2` formula in `hypothesis_iter27.md` (registered at S02), and the constants in `one_factor_diff_iter27.md` / `implementation_plan_iter27.md`.  Per SKILL.md §2.9 "registered-spec immutability after hypothesis lock", these are immutable after S02/S04 canonical acceptance.  The skill's canonical abort example explicitly covers this exact case.
NEXT_ITERATION_CONSTRAINTS=Parent remains iter18.  A future iteration (e.g. iter28) MAY register a tighter `trust_radius_fraction ∈ {0.05, 0.02, 0.01}` under its own S00–S08 deliberation.  Do NOT retune inside iter27.  Do NOT silently launch Stage2 with a different constant.  Do NOT inherit the abortive iter27 mechanism as a parent.
```

## Direct evidence

1. **MVG round 1 — 5-step ON with registered `trust_radius_fraction = 0.5`**:
   `clip_frac = [0.000, 0.000, 0.000]` for layers 0/1/2; `max_delta =
   1.17e-02`.  Source: `mvg_check_iter27.log`.

2. **MVG round 1 — 200-step ON/OFF counterfactual**: `loss_delta =
   0.018` (> 1e-6 required); codebook max-abs-diff per layer
   `[0.146, 0.126, 0.098]` (all > 1e-7).  Mechanism **runs** and
   measurably changes the optimizer trajectory, but **never clips a
   single codeword** in 200 steps.

3. **1000-step diagnostic probe (iter18 ckpt regime)**:
   | `trust_frac` | avg clip_frac (L0/L1/L2) |
   |---:|---:|
   | 0.5 (registered) | 0.000 / 0.000 / 0.000 |
   | 0.1 | 0.000 / 0.000 / 0.000 |
   | **0.05** | **0.000 / 0.008 / 0.065** |
   | 0.02 | 0.072 / 0.445 / 0.914 |
   | 0.01 | 0.575 / 0.983 / 1.000 |

   Source: `mvg_check_iter27.md` Layer D.

4. **Geometric reason**: under iter18's per-step AdamW displacement
   (`~0.01`) versus the registered trust radius `~0.06–0.28`, the
   trust-region projection is geometrically defined but **operationally
   inactive** — the per-step displacement is `~10×` smaller than the
   trust radius.

## Why a same-iteration repair would alter the registered experiment

Activating the mechanism requires tightening the registered constant
`trust_radius_fraction` from `0.5` to ≤ `0.05` (10× tighter).  This
would change:

- `mechanism_contract_iter27.json::trust_radius_fraction_default` (registered at S04);
- the user-locked `τ_l = D_l / 2` formula in `hypothesis_iter27.md` (registered at S02);
- the constant documented in `one_factor_diff_iter27.md` and `implementation_plan_iter27.md`.

Per SKILL.md §2.9 "Registered-spec immutability after hypothesis
lock", these are immutable after S02/S04 canonical acceptance.  Same-
iteration retuning is forbidden; the canonical example in the skill is
**exactly this case**.

## What remains scientifically unresolved

- Whether a post-step hyperbolic codebook trust-region with a tighter
  radius (e.g. `τ_l = D_l / 20` or `τ_l = D_l / 50`) can improve
  Stage3 `R@10` over iter18's `0.0599`.
- Whether the iter18 optimizer trajectory is already "trust-region
  safe" by construction (i.e. iter18 satisfies every safety constraint
  the user-specified `τ_l = D_l / 2` would impose).

## Constraints for the next iteration

- **Parent remains iter18** (test_R@10=0.05988962203380978).
- A future iteration (e.g. iter28) MAY register a tighter
  `trust_radius_fraction ∈ {0.05, 0.02, 0.01}` under its **own** S00–S08
  deliberation cycle.
- The future iteration MUST re-register the mechanism contract under
  CAO-1 (active since iter27) with its own `mechanism_contract.json`
  and `hypothesis.md`.
- The future iteration MUST NOT silently inherit the abortive iter27
  mechanism as a parent; iter27's `apply_hyperbolic_trust_region`
  method on `Quantize` may be **reused as code** but the registered
  constant must change.
- The future iteration MUST NOT launch Stage2 with a different
  constant than the one registered in its own `mechanism_contract`.

## Files that are committed (audit/abort evidence only)

- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/modules/quantize.py` (CAO-1 trust-region method; code may be reused)
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/modules/rqvae.py` (snapshot + apply methods)
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/curvature_RQ-VAE.py` (post-step wiring; never executed)
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/curvature_config.py` (paths, MECHANISM_NAME)
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/configs/decoder_instruments_hgrec_iter27_hyperbolic_codebook_trust_region.gin`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/scripts/run_stage3_iter27.py`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/scripts/mvg_check.py`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/source_snapshot_iter27.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/protocol_manifest_iter27.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/hypothesis_iter27.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/mechanism_manifest_iter27.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/mechanism_contract_iter27.json`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/one_factor_diff_iter27.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/implementation_plan_iter27.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/preflight_contract_iter27.log`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/mvg_check_iter27.log`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/mvg_check_iter27.md`
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/stage2_execution_plan_iter27.md` (never executed)
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/iteration_abort_iter27.md` (this file)
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/deliberation/S00..S09/round_1/` (canonical deliberation, MERGE_AB)
- `stage2_RQ-VAE/curvature_RQ-VAE_iter27/logs/deliberation/S08_MVG/round_2/` (abort deliberation, ABORT_ITERATION)
- `.claude/skills/curvature-rqvae-iter/scripts/preflight_cao1.py` (CAO-1 preflight)

## Files that are NOT produced (deliberate)

- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/out/rqvae/instruments/rqvae_best.pth`
- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/out/rqvae/instruments/sids_raw.npy`
- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/dataset/Instruments/sids_for_hgrec.npy`
- `results/stage2_RQ-VAE/curvature_RQ-VAE_iter27/item_sids.json`
- `results/stage3_T5Train/curvature_RQ-VAE_iter27/{logs,ckpt}/...` (Stage3 was never launched)

## Skill references

- SKILL.md §2.9 (Feasibility Abort), abort condition #1 ("mechanism
  inactivity under the registered specification") and the canonical
  example matching this iteration's evidence.
- SKILL.md §2.6 (Deliberation artifact layout), `ABORT_ITERATION` fields.
- SKILL.md §22 (Iteration loop): "At any stage, a Judge C verdict of
  ABORT_ITERATION exits the loop immediately into the abort-closure
  path in §2.9."