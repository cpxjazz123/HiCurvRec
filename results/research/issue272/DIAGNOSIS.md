# Issue #272 diagnosis (before mechanism change)

## 1. Phenomenon

#271 tightened nothing: its frozen `J_max = 0.781941` came from TIGER + eps, while the recomputed #269 J
under the common ruler is `0.338336`. The candidate ended at `J = 0.351359` with `mu = 0` for all 3000
epochs, so the global term never contributed a gradient. Its local A.1 MAE fell to `1.893506`, but the
global Spearman became `-0.005483`, sign accuracy `0.392646`, effective rank `4.50`, L2 prefixes `6,096`.
#271 therefore neither exercised the constraint nor passed the local/SID criteria #272 freezes.

## 2. Measured baselines (same ruler / same SID extraction)

| arm | full J | L2 prefixes | effective rank | quant error | local MAE | local Spearman | sign |
|---|---:|---:|---:|---:|---:|---:|---:|
| TIGER | 0.739635 | 7,374 | 11.479 | 0.2509 | 7.309217 | 0.015851 | 0.432453 |
| #265 | 0.189476 | 10,918 | 3.049 | 0.00600 | 2.246 | -0.0710 | 0.3323 |
| #269 | 0.338336 | 6,246 | 4.335 | 0.00875 | 1.997 | pending | 0.3637 |
| #271 | 0.351359 | 6,096 | 4.501 | 0.00939 | 1.893506 | -0.005483 | 0.392646 |

Local scale diagnostics (400k-row fixed sample, `pred = target + error`):

| arm | var(pred)/var(target) | regression slope | Pearson | within-centre Spearman |
|---|---:|---:|---:|---:|
| TIGER | 28.844 | -0.175 | -0.0326 | -0.0166 |
| #265 | 0.312 | -0.019 | -0.0336 | 0.0875 |
| #269 | 0.430 | 0.019 | 0.0290 | 0.1066 |
| #271 | 0.335 | 0.041 | 0.0711 | 0.1254 |

## 2b. The observed frontier

No frozen arm satisfies the joint criteria. J-optimal (#265) has L2 10,918 but rank 3.05 and Spearman
-0.071; TIGER has Spearman 0.0159 and sign 0.4325 but MAE 7.309 and J 0.740; #269/#271 have the low
MAE but negative Spearman. L2 >= 10,918 and rank >= 8 never co-occur; MAE <= 1.8935 and
Spearman > 0.015851 never co-occur. The round-1 active-bound run tests whether the global channel
can break this frontier; if not, the fixed kappa_t objective is the binding cause (H272-2/H272-5).

## 3. Candidate root causes

- **R1 (constraint slack).** #271's bound was the loose TIGER value, not the #269 value; `mu` never left 0.
  Supported by the #271 training log (`global_j ~0.24-0.35`, `global_violation < 0`, `mu = 0`).
- **R2 (local objective compresses the code space).** Every arm trained with the frozen kappa_t local term
  is near rank 4.3-4.5 and L2 6,096-6,246; TIGER (no local term) has rank 11.48 but only 7,374 L2;
  #265 (Euclidean local supervision, a different objective not allowed here) reaches 10,918 L2 but rank 3.05.
  No observed arm has both L2 >= 10,918 and rank >= 8.
- **R3 (MAE gain is partly under-dispersion).** #271's prediction variance is 0.335x the target variance
  with slope 0.041; its global rank/sign are worse than TIGER's. The MAE improvement alone is not a
  valid local-quality claim.
- **R4 (reconstruction not binding).** Final training `recon`: #265 **8.18e-5**, #269 9.54e-5, #271 9.04e-5;
  the issue's `1.23e-4` reference is not the same metric as the training `recon` and must be reconciled
  in the contract (use the recomputed same-criterion #265 value).

## 4. Competing explanations

| id | explanation | separating measurement |
|---|---|---|
| H1 | A genuinely active J constraint (J_max at the #269 level) changes the geometry enough to restore rank/L2 | one 3000-epoch run with `J_max` near 0.320, report active fraction, rank/L2 |
| H2 | The frozen kappa_t local term itself caps rank and L2; the SID thresholds are infeasible without changing the objective | λ sweep is not allowed as a blind sweep, but the observed frontier (λ=0 TIGER vs λ>0 #269/#271) plus a run at the inherited λ; if rank/L2 stay <=6,246/4.5 while the local term is active, H2 gains evidence |
| H3 | The local rank/sign degradation is scale/under-dispersion, independent of the global field | variance/slope and within-centre diagnostics before/after the constraint |
| H4 | The joint criteria (J <= 0.338336, MAE <= 1.893506, Spearman > 0.015851, sign > 0.432453, L2 >= 10,918, rank >= 8) are mutually inconsistent on the observed frontier | explicit frontier table; a candidate run passing J and local but failing SID (or vice versa) |

## 5. Six admission questions (Stage2)

1. **Problem.** #271's constraint never activated and its local gain came with worse rank/sign, rank 4.50
   and L2 6,096; #272 must make the global constraint real and restore local rank/sign and SID capacity.
2. **Evidence.** The #271 log (`mu=0`, J below the loose bound), the SID inventory (L2 6,096/rank 4.50) and
   the local diagnostics (var ratio 0.335, slope 0.041) above.
3. **Why can the new mechanism act?** The dual term `mu (J - J_max)` with a tight bound is the only allowed
   global channel; with `mu > 0` its gradient reaches the encoder and codebooks (proved in #271's module
   check at a controlled violation). The local term remains the same; lambda is frozen at the #269
   calibration for round 1 so the change is the active constraint, not a new loss.
4. **Difference from #271.** Only the bound and scheduling: `J_max` from the loose 0.7819 to the #269-level
   `0.320` training bound (acceptance 0.338336), with the same mechanism and lambda. This is the issue's
   required "active global constraint" test.
5. **Metric.** Full J <= 0.338336 and per-hop tolerance; local MAE <= 1.893506, Spearman > 0.015851,
   sign > 0.432453; L2 >= 10,918, rank >= 8, prefix/behaviour >= TIGER; reconstruction <= same-criterion
   #265 x 1.05.
6. **If it fails.** It separates R1 from R2: if J passes and the local/SID criteria do not, the frozen
   kappa_t objective (not the constraint) is the binding cause, and the SID thresholds are infeasible
   without an objective change outside this issue's scope. That is a usable knowledge increment and the
   basis for the next registered mechanism.

## 6. Still unknown

- The exact recomputed #265 J and local Spearman (inventory running); the contract will freeze the
  recomputed values, not the quoted ones.
- Whether a λ/dual schedule alone can move rank from 4.5 to >= 8 without losing the local MAE.
- Which same-criterion reconstruction value corresponds to the quoted #265 `1.23e-4`.
