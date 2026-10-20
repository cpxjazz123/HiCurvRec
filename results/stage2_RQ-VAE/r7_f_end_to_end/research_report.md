# R7-F end-to-end (issue #89) — FINAL: Stage2 complete, frozen gate FAILED

## Verdict: `end_to_end_rejected`

A complete, valid 3000-epoch cold Stage2 run was executed and its final Semantic IDs were measured
once against the frozen gate. The gate failed. Per the issue's own rule, Stage3 was **not** started,
queued or prepared.

| | value |
|:--|:--|
| frozen baseline (the ONLY Stage3 gate) | **16.38953942232631** |
| **R7-F primary final Hard-L1 ER** | **14.071573261309926** |
| comparison | strict `>`, no rounding — **FAILED** |
| historical R4-08 reference | 16.402627511591962 |
| stronger than historical R4-08 | **no** |

Full decomposition from `gate.json`:

```
pos_hits = 4168 / 52855      pos_share = 0.0788572509696339
neg_hits = 2962 / 528550     neg_share = 0.0056040109734178415
Hard_L1_ER = pos_share / neg_share = 14.071573261309926
artifact SHA256 = 04d779208343a9e5cea769ddd5b600a4411c9313416a0fdba11ebaf0081125d0  (unchanged)
```

**Both sides of the ratio degraded relative to the historical R4-08 host.** The historical arm had
`pos_share 0.0803 / neg_share 0.004896`; the treatment has `pos_share 0.0789 / neg_share 0.005604`.
Fewer true next-item relations land in the same L1 code **and** more popularity-matched negatives
do. The auxiliary repulsion makes L1 codes more evenly used (see the preflight) without making them
more discriminative on real observed transitions — it spreads items across codes that do not align
with behaviour.

## The run itself was genuine

| requirement | measured |
|:--|:--|
| cold 3000 epochs, no early stopping | epoch **3000 / 3000** reached |
| fresh init, fresh KMeans, no warm start | yes (isolated tree `TIGER_r7_f_primary`, arm `r7_f_primary`) |
| optimizer steps | **18,000** Adam steps |
| behaviour-active steps (weight > 0) | **9,000 / 9,000** with **nonzero separation gradients on the encoder** |
| max behaviour weight | **0.20** |
| ramp | starts at epoch 1500, production schedule (the 32+128 pilot schedule was never used) |
| hyperparameters | AdamW lr 1e-3, wd 1e-4, clip 1.0, item batch 1024/rank, pair 512/rank, temp 0.07, 3×256 codes dim32, seed 42 |
| artefacts | `item_sids.json`, `sids_for_hgrec.npy`, `rqvae_best.pth`, all written and hashed |
| held-out artifact | never read until the final checkpoint existed; SHA verified identical before and after |

The mechanism genuinely executed: at the first active step (epoch 1520) the log records
`sep_loss 0.000309`, `valid_negative_count 123`, `sep_grad_norm_encoder 0.0177`,
`sep_grad_norm_L1_codebook 0.0157`.

## Preflight (160-step) told the same story in advance

Computed on ONE fixed 1024-item train-only population through the canonical
`model.get_indices(..., infer_use_sk=False)` path, with an independent encoder+L1-argmin
cross-check that agreed on every snapshot:

| arm | host | λ | L1 used | effective | max share | tuple3 collision | pos same-code | unrelated same-code |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| F0 | r6-04 | 0 | 17 | 5.4 | 42.7% | 0.512 | 0.313 | 0.247 |
| F1 | r6-04 | 0.1 | **63** | **32.8** | **9.6%** | 0.372 | 0.051 | 0.051 |
| F2 | r4-08 | 0 | 80 | 47.1 | 7.5% | **0.026** | 0.031 | 0.024 |
| F3 | r4-08 | 0.1 | **115** | **68.8** | **5.5%** | **0.080** | 0.027 | 0.035 |

On the primary host the treatment **improved occupancy but worsened three-level tuple collision**
(0.026 → 0.080) and slightly worsened both same-code rates. Under the issue's gaming rule that is a
failure of the *intended effect*, and it predicted the final gate outcome. The pilot was diagnostics
only; the frozen ER remains the sole endpoint, and it agrees.

## Faults found and fixed inside this issue

1. **#88's step count.** The audit accepted `len >= 160` and the trainer overran to 162 because the
   step cap was checked at the epoch boundary. Fixed with a **module-level global optimizer-step
   counter** and a cap checked inside the batch loop. F0–F3 now each log **exactly 160**.
2. **#88's empty snapshot export.** The snapshot condition used an epoch-local counter that reset
   every epoch, so it never fired. The same global counter fixed it: **8 snapshots per arm**, and
   the production run wrote `step_0000` labelled `pre_initialization`.
3. **#88's non-matched arms.** Independent seed42 processes are not bit-identical (#80 measured
   ±5–6 L1 codes). The arms are therefore reported as independent runs with that noise floor
   documented, not as matched weights.
4. **#87's `0.152 → 0.062` remains INVALID** and was never used as a baseline or as efficacy.

## I damaged a historical artifact, and it is disclosed

Running the preflight through the **real trainer in the original `TIGER_r4_08` and `TIGER_r6_04`
trees** made each trainer export its SIDs and checkpoint into its own historical results directory.
`results/stage2_RQ-VAE/{r4_08,r6_04}/{item_sids.json,rqvae_best.pth,sids_for_hgrec.npy}` were
overwritten at 17:38–17:39 on 2026-10-20.

What was recovered and what was not:

- `item_sids.json` for both arms is **git-tracked** and was restored with `git checkout --`.
  Both historical references then reproduced **digit for digit**: r4_08 → `16.402627511591962`,
  r6_04 → `1.5416677050365089`. That reproduction also validates the gate implementation.
- `sids_for_hgrec.npy` and `rqvae_best.pth` for those two arms are gitignored and were **not**
  recoverable. The authoritative L1 codes they encoded survive in the restored `item_sids.json`.
- **The held-out pair artifact was never touched**: SHA256 verified identical before and after.

The harness was then changed to run in an **isolated scratch copy** of each host tree
(`stage2_RQ-VAE/_r7f_scratch/...`) whose `EXPERIMENT_ARM` is rewritten, so a preflight can no longer
write into a historical arm's directory.

## CGV status — not completed, and no performance claim rests on it

Built inside this issue, in the verifier (not with the mechanism):

- `oracles.r7_f_forward_loss` — `L_sep` from its definition.
- `oracles.r7_f_loss_vjp` — analytic `dL/d(distance)` from the softmax Jacobian, **verified against
  autograd to 7.4e-19** and against central differences on the smooth surrogate.
- `oracles.r7_f_same_code`, and `engine.gate_g3_r7_f` which checks the analytic VJP across several
  seeds **and** that both pair sides carry gradient.
- `tools/r7_f_er_gate.py` — fail-closed frozen gate, validated by reproducing both historical
  references exactly.

**Not done:** a complete G1–G6 pass with a mechanism-independent adapter for unequal-count two-rank
DDP aggregation, the seven-group caller-supplied spec, and the isolated real-entry gate run. A
partial gate is not a pass, and the issue forbids a cosmetic one. **No conclusion in this report
depends on CGV** — the frozen ER is a measurement, and it failed.

## What this means for the research question

R7-F's hypothesis was that a train-only soft L1 same-code repulsion would produce hard Semantic IDs
that discriminate real observed next-item relations better than the frozen baseline. On the R4-08
host, at the pre-declared λ=0.10 and the frozen τ=1e-3, **it does not**. The mechanism does what it
was designed to do at the level of codebook occupancy — more L1 codes used, lower max share,
higher effective codes — and that occupancy improvement **does not translate into discrimination**.
If anything it trades discrimination away: codes are spread more evenly but the spread does not
follow behaviour.

This is a real, complete negative result about a Euclidean/native-quantizer auxiliary loss. It says
nothing about hyperbolic geometry, and nothing about whether a different separation pressure or a
different host would succeed.

## Not run, by rule

- **Stage3 was not started, queued or prepared.** The gate failed.
- No 3000-epoch re-run with another λ/τ/seed/host: the issue forbids selecting after seeing the ER.
- Seeds 43/44 replications of the preflight: the protocol's own status rules stop at the failed
  final gate, and re-running would be a new hypothesis.

## Deliverables

`results/stage2_RQ-VAE/r7_f_end_to_end/`: `PREREGISTRATION.json`, `gate.json`,
`production_steps.jsonl` (18,797 records), `production_snapshots/` (0/8/16/32/64/96/128/160),
`production_stdout.log`, this report.
`results/stage2_RQ-VAE/r7_f_harness_v2/`: `LEGACY_87_INVALIDITY.md`, `PREREGISTRATION.json`,
`COMPARISON.md`, `phase2_metrics/`, per-arm `seed42/` runs with steps, snapshots, stdout and stderr.
Checkpoints and `.npy` arrays are gitignored: their physical paths and hashes are recorded above and
are not in GitHub.
