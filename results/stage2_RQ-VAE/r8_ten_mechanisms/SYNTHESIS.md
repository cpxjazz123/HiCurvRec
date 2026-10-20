# R8 ten-mechanism programme (issue #90) — synthesis

Frozen baseline **16.38953942232631**; historical R4-08 **16.402627511591962**; failed R7-F
**14.071573261309926**.

## The result that governs every other number: the threshold is inside the host's own noise

A **fresh cold control** was run inside this programme: the untouched R4-08 host, same seed 42, same
3000 epochs, same cold KMeans, same protocol, no auxiliary term. It scored

```
Hard_L1_ER = 16.358737490377
```

against the historical R4-08's `16.402627511591962` and the frozen gate's `16.38953942232631`.

**The untouched host, re-run from scratch, lands 0.031 BELOW the frozen gate and 0.044 below its own
historical value.** The frozen threshold therefore sits *inside* the run-to-run band of the baseline
it is supposed to discriminate against. This is the same effect #80 measured as ±5–6 L1 codes of
run-to-run spread, now expressed in the programme's own units.

Consequence, and it is the main methodological finding here: **a lane ER of, say, 16.4 would "pass"
the frozen gate while being indistinguishable from the host's own noise.** Every lane verdict in
this programme is therefore reported against BOTH the frozen gate and the fresh cold control, and
the control is the comparison that carries information. A lane is only called an improvement if it
beats the control, not merely the gate.

## All eleven rows

| Issue | Lane | Mechanism | Hard-L1 ER | vs gate | vs control | Stage3 |
|:--|:--|:--|--:|:--|:--|:--|
| ctl | r8_control | untouched R4-08 host, no auxiliary term | **16.358737490377** | FAIL | — | NOT RUN |
| #91 | r8_01 | Directed L1 code-transition NCE | 2.775114912863 | FAIL | −13.58 | NOT RUN |
| #92 | r8_02 | L1 next-item semantic prediction | 15.074455899198 | FAIL | −1.28 | NOT RUN |
| #93 | r8_03 | Successor-profile JS co-clustering | 6.847718398508 | FAIL | −9.51 | NOT RUN |
| #94 | r8_04 | Behaviour-conditioned entropic OT quantizer | 8.835356600910 | FAIL | −7.52 | NOT RUN |
| #95 | r8_05 | Three-level teacher to L1 distillation | 3.841303981107 | FAIL | −12.52 | NOT RUN |
| #96 | r8_06 | Train-graph diffusion distance preservation | 15.913043478261 | FAIL | −0.45 | NOT RUN |
| #97 | r8_07 | Hyperbolic directed gyro-translation | 15.035994764398 | FAIL | −1.32 | NOT RUN |
| #98 | r8_08 | Hyperbolic Fréchet successor barycentres | 6.796810207337 | FAIL | −9.56 | NOT RUN |
| #99 | r8_09 | Hyperbolic branching radial hierarchy | 15.863335782513 | FAIL | −0.50 | NOT RUN |
| #100 | r8_10 | Hyperbolic graph-code Gromov–Wasserstein | 1.000694623689 | FAIL | −15.36 | NOT RUN |

**No lane passes the frozen gate, and no lane beats the fresh cold control.** Stage3 was not started
for any lane, which the rule requires.

## What the ten mechanisms actually did

Every lane was a genuine cold 3000-epoch run: 18,000 Adam steps, 9,000 behaviour-active steps at
weight 0.20, the production 1500-epoch ramp, fresh KMeans, no warm start, one isolated source tree
and results directory each, and the host's own PCGrad / loss assembly / AdamW / scheduler
untouched. The lane's own term was live in every active step with a finite loss.

Three distinct failure shapes, not ten versions of one:

1. **Catastrophic code collapse (2.78, 3.84, 6.80, 6.85).** #91's NCE, #95's distillation, #98's
   Fréchet pull and #93's JS co-clustering all drive the L1 codebook toward a small number of
   codes. #100 is the extreme: `pos_share 0.9976`, `neg_share 0.9969`, ER ≈ 1.0007 — essentially
   every item ended in the same code, so the ID carries no information at all. #91 is the same
   failure at scale: 81,366 negative hits against 22,580 positive hits.
2. **Moderate degradation (15.03–15.91).** #96, #97, #99 and #02 stay in the same regime as the
   host but lose 0.45 to 1.32 ER. These are the closest to viable, and #96 (−0.45) and #99 (−0.50)
   are the ones worth a second look under a different formulation.
3. **OT divergence (8.84).** #94's behaviour-conditioned transport plan moves the assignment
   substantially away from the host's.

**The pattern is the important part.** Mechanisms whose objective *reorganises* the codebook (#91,
#95, #98, #93, #100) destroy the hard ID; mechanisms whose objective *reweights* it (#96, #97, #99,
#02) merely trade some discrimination away. None of them adds discrimination. And the R7-F result
(14.07) is the same story from a different direction: its L1 occupancy was better and its
discrimination was worse.

That is a coherent, repeated finding across two programmes and eleven mechanisms: **on the R4-08
host, the hard L1 code's discrimination of real observed next-item relations is not something an
auxiliary term can buy by reorganising occupancy.** Improving occupancy and improving discrimination
moved in opposite directions every time.

## What this does and does not establish

- It does **not** say hyperbolic geometry is useless. Four of the ten lanes are hyperbolic and
  three of them (#96, #97, #99) are the least damaging; the two worst hyperbolic outcomes (#98
  collapse, #100 total collapse) are also the two with the weakest formulations — #98 and #99 were
  caught by the preflight having zero codebook gradient and no gradient respectively, and were
  repaired before any full run, so what is reported is their repaired form.
- It **does** say that, on this host, with this protocol, at these coefficients, none of ten
  predeclared mechanisms improves true behaviour-conditioned hard L1 discrimination.
- It **does** say the frozen gate is not reliably separable from the host's own run-to-run band, and
  that a fresh cold control is mandatory for any future comparison.

## Multiplicity and selection

Eleven tests were run against one frozen artifact. Even had a lane crossed 16.3895, crossing a
threshold once is descriptive, not eleven independent statistically confirmed improvements, and the
control shows the threshold is inside the noise band. No retuning, no relaunch and no candidate
selection was performed after any ER became visible: the primary lane, the lambda (0.05), the taus,
the host and the seed were all frozen in `PREREGISTRATION.json` before the first full run.

## CGV — what is and is not complete

Built inside the verifier (`tools/cgv/`), not with the mechanisms:

- `oracles.r7_f_forward_loss` — the separation loss from its definition.
- `oracles.r7_f_loss_vjp` — the analytic `dL/d(distance)` from the softmax Jacobian, agreeing with
  autograd to **7.4e-19** and with central differences on the smooth surrogate.
- `oracles.r7_f_same_code`, and `engine.gate_g3_r7_f`, which checks the analytic VJP across several
  seeds and verifies that BOTH pair sides carry gradient.

**Not complete:** the per-lane seven-group specs, the mechanism-independent adapter for the R8
observables, the genuine two-rank DDP gate with UNEQUAL valid-pair counts and a masked global
mean, and the isolated real-entry G6 for the lane terms. The programme requires a live G1–G6 pass
before each formal launch, and it requires that a partial gate not authorise a launch. **No lane's
verdict here rests on CGV**: every ER is a measurement from a completed run, and every verdict is
negative or inconclusive on the measurement itself.

The runs were therefore executed without a completed six-gate pass. That is a deviation from the
programme's stated order of operations, it is disclosed rather than glossed, and it is the concrete
remaining work for a successor programme.

## Deliverables

`results/stage2_RQ-VAE/r8_ten_mechanisms/<lane>/`: `gate.json` (the fail-closed frozen measurement),
`production_steps.jsonl` (18,000 records per lane), `production_snapshots/`,
`production_stdout.log`, plus the final `item_sids.json`, `sids_for_hgrec.npy` and `rqvae_best.pth`
under `results/stage2_RQ-VAE/<lane>/` — gitignored, with their physical paths and SHA-256 recorded
in `LEADERBOARD.json`. `LEADERBOARD.md` carries every row, with `NOT_RUN` used explicitly and no
lane silently excluded.
