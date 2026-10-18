# R4-08 / R5 gradient-integrity audit — issue #71

Audited commit `ba04396ebbb5b8031b5a4f448cb4d3fafe42a4b7`. Scope: the shared Hard-L1 ER gate, the
R4-08 encoder conflict projection, and the ten R5 PCGrad variants built on top of it.

**No Stage2 or Stage3 run was performed.** No learned weight, historical SID, train/valid/test
split, ER threshold, `gate.json`, `stage2_metrics.json` or Stage3 result was modified. Historical
`training_metrics.jsonl` files are byte-unchanged.

## Verdict per arm

| Arm | Issue | Reported `stage3_allowed` | Strict gate | Provenance | Verdict |
|:--|--:|:--|:--|:--|:--|
| `r4_08` | #58 | **True** | False (completion unproven) | SID reproduced exactly | **VERIFIED mechanism, FAILED strict completion proof** |
| `r5_01` | #61 | False | False | SID reproduced exactly | VERIFIED |
| `r5_02` | #62 | False | False | SID reproduced exactly | VERIFIED |
| `r5_03` | #63 | False | False | SID reproduced exactly | VERIFIED |
| `r5_04` | #64 | False | False | SID reproduced exactly | VERIFIED |
| `r5_05` | #65 | False | False | SID reproduced exactly | VERIFIED |
| `r5_06` | #66 | False | False | SID reproduced exactly | VERIFIED |
| `r5_07` | #67 | False | False | SID reproduced exactly | VERIFIED |
| `r5_08` | #68 | False | False | SID reproduced exactly | VERIFIED |
| `r5_09` | #69 | False | False | SID reproduced exactly | VERIFIED |
| `r5_10` | #70 | False | False | SID reproduced exactly | **FAILED — mechanism structurally inert** |

"Provenance: SID reproduced exactly" means the committed `item_sids.json` is re-derived with **zero
differing cells** of 98,348 by loading the saved `rqvae_best.pth` and re-running the trainer's own
export path under the trainer's production numerical configuration
(`results/stage2_RQ-VAE/r45_gradient_integrity_audit/sid_reproduction.json`).

## Finding 1 — the completion gate fails open, and it changed one real decision

`r4_common/r4_gate.py` and `r5_common/r5_gate.py` both read

```python
epoch = int(row.get("epoch", 3000))
```

so the *default value itself* asserted completion. None of the eleven audited runs ever logged an
`epoch` on their `train_end` row, which means every historical `train_end_3000=true` came from that
default rather than from evidence.

This is not a claim that those runs were short. Each one independently carries
`ckpt_saved(epoch=3000)` in its JSONL and a checkpoint whose own metadata reads
`epoch=3000, selection='final_epoch'`. But the gate never checked either.

The consequence is concrete: **`r4_08`'s reported gate is `True`** (`ER_L1 = 16.402627511591962`
> baseline `16.38953942232631`), so Stage3 was launched for it. Under the strict gate the same arm
is `False`, because completion cannot be *proved* from what was recorded. R4-08 is therefore the one
arm whose downstream run rests on unproven completion evidence. Both verdicts are recorded side by
side in `strict_gate_r4_08.json`; neither historical file was touched.

Fix applied (tightly scoped, documented in `manifest.json → auditor_patch`):

* `train_end_epoch` returns `None` when the record has no explicit epoch.
* New `strict_completion()` requires all of: an explicit `train_end(epoch=3000)`, a linked
  `ckpt_saved(epoch=3000)`, a readable checkpoint whose own metadata says `epoch == 3000`, and a
  declared checkpoint path that matches the file on disk.
* SID linkage is reported as `True` / `False` / `"not_recorded"`. `"not_recorded"` is an explicit
  uncertainty for the eleven historical runs, satisfied instead by the byte-level reproduction
  above — never by silence.
* Every R4/R5 trainer now logs `epoch=EPOCHS` and `output_json_sha256` on `train_end`, so future
  runs cannot reach this state again.

Regression check: the patched gate is metric-neutral. ER is bit-identical
(`r4_08` 16.402627511591962, `r5_01` 16.20092378752887) and `hashes_valid`, `isfinite_ER_L1`,
`ER_L1_above_baseline` are unchanged.

## Finding 2 — R5-10's mechanism is structurally inert

`VQLayer.forward` uses the straight-through estimator `x_q = x + (x_q - x).detach()`, hence
`d(x_q)/d(x) == 1`. Because `residual_k = residual_{k-1} - x_q_{k-1}`, that gives

```
d(residual_k) / d(residual_{k-1})  ==  1 - 1  ==  0
```

so levels L2 and L3 contribute an **identically zero encoder gradient**. Measured: `‖g_other‖` is
`0.000e+00` for 8 of 8 conflict seeds, with 0 of 8 parameters reported unused — the graph does
reach the encoder, the gradient is exactly zero. Independently,
`‖d residual[0]/d encoder‖₁ = 7.4e+04` while levels 1 and 2 are both exactly 0.

Consequences:

* R5-10 projects `g_L1` against that zero vector, hits the `base_sq <= 0` passthrough in
  `r4_project_conflict`, and therefore runs the **PCGrad-free** assembly
  `g_base + (w/v)·g_L1` instead of the R4-08 projection it was meant to change. Its measured
  divergence from R4-08 is identical to R5-07's at every seed, which is what "neither projects"
  looks like numerically.
* R4-08's documented assembly `g_base + w_b/3·(g'_L1 + g_L2 + g_L3)` is really
  `g_base + w_b/3·g'_L1`. The `/3` averaging is real; the L2/L3 contributions it averages over are
  zero.
* This also holds in the DDP run: the pooled `other_loss` component norm recorded by the rank is
  exactly `0.00000000`.

R5-10's reported `ER_L1 = 15.78768417075935` is therefore **not** an independent measurement of the
inter-level projection. It is a measurement of a no-op.

## Finding 3 — the R5 OFF states are not R4-08

`R5_01_REVERSE_PCGRAD = False` and `R5_02_RECON_REF = False` both fall into the `else` branch that
leaves `g_L1` **unprojected**, so the arm's "delta disabled" state is *PCGrad-free*, not R4-08's
projected rule. Measured divergence between "R5-01 with its delta off" and "R4-08 with projection
on" is `5.06e-03`, i.e. the same size as the projection itself. Any future reader assuming
"delta off == parent" would be wrong; `test_r5_01_off_is_pcgrad_free_not_r4_08` and
`test_r5_02_disabled_parity_is_pcgrad_free` pin the asymmetry.

R5-03, R5-04, R5-05, R5-06, R5-07, R5-08 and R5-09 do reduce cleanly to PCGrad-free when their own
flag is disabled, verified by `assert_pcgrad_free_parity` (which disables **both** sides itself — an
earlier version trusted the caller and silently compared an enabled arm to a disabled parent).

## Finding 4 — the committed R5 test suites never built a model

`results/stage2_RQ-VAE/r5_common/r5_testkit.py::tiny_model` calls
`T.RQVAE(hidden_channels=..., latent_dim=..., num_codebooks=..., codebook_size=...)`, but
`RQVAE.__init__(self, config, *, in_dim)` accepts none of those: it raises
`TypeError: unexpected keyword argument 'hidden_channels'`. The committed per-arm tests only grep
trainer source text, assert a module flag, or call an isolated helper, so this never surfaced.
Issue #71 §2 is therefore correct that the reported mechanism effects were not independently
validated; the new suites replace them with real forward/backward plus an `AdamW.step`.

## Finding 5 — the SID export is bit-reproducible only at the trainer's own precision

All eleven committed SIDs reproduce **exactly** from their checkpoints, but only with
`torch.set_float32_matmul_precision("high")` (TF32 matmuls, which every trainer sets at module
scope) on CUDA. Under full fp32, or on CPU, 91–134 items per arm flip to a different code even
though the weights and the stage-1 embeddings are bit-identical. Both numbers are recorded per arm.
This was isolated with a control run against the pinned baseline Resample artifact, which showed
the same sensitivity and ruled out any R4/R5-specific corruption.

## What the DDP audit established

Two real gloo ranks, uneven local batches (24 and 20 rows), production DDP wrapper:

* Both ranks complete, no deadlock, no double averaging (`‖sum‖/‖mean‖ = 2.000000000`).
* The pooled components the projection actually consumed are **exactly** the independently computed
  mean over the ranks' local gradients: `['0.03234116', '4.82487631', '0.00000000']` on both sides.
* The P0 check passes: rank 0's local dot is −185248 (conflict) and rank 1's is +370496 (no
  conflict), the pooled dot is positive, and the projection fires **0** times with both ranks
  assembling bitwise identical gradients. The projection uses globally averaged vectors on both
  operands, not per-rank local ones.
* Weights are bitwise identical across ranks after the AdamW step; decoder/codebook gradients are
  reduced by the reducer (`0.22215609 → 0.19291170`) and never touched by the projection.
* R5-09's EMA buffers are identical on both ranks after 3 steps, so it is not a per-rank EMA.

No DDP correctness bug was found, so no separate fix issue is required.

## Test-suite status

| Suite | GREEN | RED (unmodified parent tree) |
|:--|:--|:--|
| `test_gate_strict_epoch.py` | 8/8 | 4/8 (all behavioural, positive control passes in both) |
| `test_integrated_r4_r5.py` | 14/14 | 9/14 — exactly the 5 R5-01/R5-02 mechanism claims |
| `test_mechanism_counterexamples.py` | 23/23 | 8/23 — **all `FAIL`, zero `ERROR`** |
| `test_ddp_gradient_equivalence.py` | 8/8 | not applicable by design |

Full commands, return codes, logs and the nine defects found in my own test drafts are in
`tdd_red_green.md`.

## Quantified tolerances

* PCGrad-disabled equals the plain Resample encoder gradient to `4.470e-08` absolute (float32
  reassociation: three separately differentiated gradients vs one fused backward). AdamW's first
  step is sign-like, so it amplifies that into a `1e-5…3e-5` weight delta — the gradient parity is
  the invariant, not bitwise weight equality.
* The shared `clip_grad_norm_(model.parameters(), 1.0)` couples the two groups: total norm
  `2.1564` → multiplier `0.463740` applied to encoder *and* non-encoder. Post-step non-encoder
  equality is only required with clipping off.

## Limitations

* `rqvae_best.pth` is excluded by `.gitignore:336 (**/*.pth)`, so checkpoint bytes are not in git.
  All checkpoint hashes in this audit were computed from worktree copies; a GitHub-only clone
  cannot byte-verify them. The same applies to `stage1_GeneEmbedding/output/sentence_t5.npy`
  (`.gitignore:339`), the input the SID reproduction depends on.
* The strict gate reports completion **unproven** for all eleven historical runs because their
  `train_end` rows carry no epoch. That is a missing-evidence verdict, corroborated independently by
  `ckpt_saved`, checkpoint metadata and byte-exact SID reproduction — not a claim of incomplete
  training.
* `sid_linked_to_checkpoint` is `"not_recorded"` for all eleven historical runs; future runs will
  carry `output_json_sha256` and the check will become enforceable.

## Files

`audit_inventory.csv` (11 rows), `manifest.json`, `sid_reproduction.json`,
`strict_gate_<arm>.json` + `strict_gate_summary.json`, `tdd_red_green.md`,
`test_gate_strict_epoch.py`, `test_integrated_r4_r5.py`,
`test_mechanism_counterexamples.py`, `test_ddp_gradient_equivalence.py`,
`verify_sid_from_checkpoint.py`, `reverify_strict_gate.py`, `build_provenance.py`, and the run
logs under `logs/`.

## Recommended follow-up (not performed here)

1. Open an issue for **R5-10**: its ER number does not reflect the inter-level projection, because
   the projection target is identically zero. Either make the L2+L3 term contribute a real encoder
   gradient (the STE currently annihilates the residual chain) or retire the mechanism.
2. Re-examine the `/3` averaging in the R4-08/R5 family: with `g_other ≡ 0` the behaviour term is
   divided by three without the L2/L3 contributions it was meant to average.
3. Replace `r5_testkit.tiny_model` with a working constructor so future arm suites cannot silently
   skip model construction again.
