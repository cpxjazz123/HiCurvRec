# TDD evidence — issue #71 gradient-integrity audit

Every suite is a plain runnable script. A **RED** here always means the assertion fails for a
**behavioural** reason against an unchanged source snapshot — never because a symbol is missing.
Where a suite needs an arm module that does not exist in the parent tree, the test either filters
callee kwargs by the real signature (`arm_kwargs_for`) or treats an absent flag as "parent
behaviour" (`patched`), so the failure lands on the numeric claim.

Environment: `python3.9` at `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9`,
torch 2.8.0+cu128, CPU-only for the gradient suites, CUDA only for SID reproduction.

Fixtures: `in_dim=32`, `2*N_SRC` embedding rows with the first `N_SRC` acting as contrastive
sources and the rest as candidates (the trainer's `behavior_contrastive_components` splits on
`batch_size`), `N_SRC=24`, `behavior_weight=0.5`, AdamW `lr=1e-3, weight_decay=0.01`.

Conflict seeds — seeds whose real `<g_L1, g_base>` dot product on the R4-08 tree is negative, so
the PCGrad projection actually fires: `1, 4, 5, 12, 16, 30, 32, 36` (recomputed by
`scan_conflict_seeds()`, printed at the top of every run; the full 0..39 scan finds 21).

---

## 1. Gate strictness — `test_gate_strict_epoch.py`

| Phase | Log | Result |
|:--|:--|:--|
| RED (unmodified gate) | `logs/gate_red.log` | **4/8**, `rc=1` |
| GREEN (strict gate) | `logs/gate_green.log` | **8/8**, `rc=0` |

RED failures, all behavioural:

```
FAIL test_missing_epoch_in_train_end_must_fail_closed
     r5_gate.train_end_epoch accepted a train_end record with no 'epoch' key and reported 3000.
     This is the fail-open: the value came from the row.get('epoch', 3000) DEFAULT, not from evidence.
FAIL test_gate_fails_closed_without_epoch_even_with_high_er
     a train_end without an explicit epoch must NOT count as 3000 epochs
FAIL test_fail_closed_matrix
     gate did not fail closed for: checkpoint_metadata_wrong: stage3_allowed=True;
     checkpoint_metadata_wrong: launcher was invoked on a false gate;
     sid_checkpoint_mismatch: stage3_allowed=True; sid_checkpoint_mismatch: launcher was invoked ...
FAIL test_r4_and_r5_gates_share_the_same_defect
     r5_gate still promotes an epoch-less train_end to 3000
```

Two of those were the *absence of a safety check* rather than a wrong value: the old gate never
opened the checkpoint and never compared the exported SID against the run that produced it. The
positive control (`test_positive_control_gate_can_be_true`) passes in both phases, so the
fail-closed assertions are not vacuous.

Real defects found in my own first draft of the suite, kept here because they changed the numbers:
the fake Stage3-launcher recorder was not reset per case (spurious "launcher was invoked"), and the
positive control used a byte-string stub checkpoint, which the strict gate correctly rejects as
unreadable — the fixture had to become a genuine `torch.save` payload.

## 2. Integrated train path — `test_integrated_r4_r5.py`

| Phase | Log | Result |
|:--|:--|:--|
| GREEN (own trees) | `logs/integrated_green.log` | **14/14**, `rc=0` |
| RED (`R5_TEST_TREE=…/TIGER_r4_08`) | `logs/integrated_red.log` | **9/14**, `rc=1` |

RED fails exactly the five R5-01/R5-02 mechanism assertions, each on a numeric claim:

```
FAIL test_r5_01_projects_base_and_keeps_l1_unprojected   r5_01 must assemble g_base_projected + scale*(g_l1_original + g_other)
FAIL test_r5_01_off_is_pcgrad_free_not_r4_08             R5-01 with its delta off must reproduce the PCGrad-free assembly, off by 0.005059297662228346
FAIL test_r5_02_disabled_parity_is_pcgrad_free           R5-02 delta off must reproduce the PCGrad-free assembly, off by 0.005059297662228346
FAIL test_r5_02_uses_recon_only_reference_and_full_base  r5_02 must project g_L1 against g_recon and assemble the full g_base
FAIL test_r5_02_reference_choice_changes_the_decision    r5_02 must follow the recon reference, not the base reference
```

The nine R4-08 tests pass on the parent tree, which is correct — R4-08 *is* the parent.

Measured quantities recorded by the run:

```
conflict seeds (recomputed scan, 0..39): [1, 4, 5, 6, 7, 10, 11, 12, 13, 15, 16, 22, 23, 24, 26, 28, 29, 30, 32, 36, 37]
PCGrad-off vs plain Resample: worst encoder-grad abs diff 4.470e-08   (float32 reassociation)
dot-scan: {'no_conflict': 9, 'conflict': 11} over seeds 0..19
shared clip multiplier 0.463740 from total_norm 2.1564
r5_02 reference-choice divergence at seed 1: flips=False l2=2.135356e-01
  dot_base=-3.0019e-03 dot_recon=-1.0018e-03
```

## 3. Mechanism counterexamples — `test_mechanism_counterexamples.py`

| Phase | Log | Result |
|:--|:--|:--|
| GREEN (own trees) | `logs/counterexamples_green.log` | **23/23**, `rc=0` |
| RED (`R5_TEST_TREE=…/TIGER_r4_08`) | `logs/counterexamples_red.log` | see the log |

Per-arm divergence from the R4-08 parent encoder gradient, `clip=0`, same seed and inputs:

| Arm | Seed | `‖grad − grad_r4_08‖₂` | Counterexample |
|:--|--:|--:|:--|
| r5_03 | 1 | `7.275e-03` | `cos(g_quant, g_base) = +0.979`, and `g_base == g_recon + g_quant` |
| r5_04 | 0 | — | global dot `+2.54e-04` (no global projection) yet **2 of 4** blocks conflict |
| r5_05 | 5 | `1.126e-02` | `alpha = 0.5`; exactly half of the parent's full-strength delta |
| r5_06 | 12 | — | `‖g_L1‖ 6.208994 → restored 6.208998`; R4-08 alone would give `6.207900` |
| r5_07 | 16 | — | bisection reaches `cos = −0.200000036783` (no projection, strict `<`); `cos = −0.5` projects |
| r5_08 | 30 | — | `‖grad(2249) − grad(2250)‖₂ = 5.7048e-02` |
| r5_09 | 32 | `2.203e-02` @step2 | `‖EMA₂ − g_base₂‖₂ = 3.274e-02`; step 1 equals R4-08 exactly |
| r5_10 | 36 | — | **`‖g_other‖ == 0` for 8/8 seeds** → mechanism inert, see the report |

Projections fired per step versus the arm's own allowance:
`r5_03: 1/1, r5_04: 4/4, r5_05: 1/1, r5_06: 1/1, r5_07: 0/1, r5_08: 1/1, r5_09: 1/1, r5_10: 0/1`.

Nine defects in my own first draft of this suite, all corrected before the counts above, recorded
because each would otherwise have produced a false conclusion:

1. "plain Resample" backwarded only `base_loss`, silently dropping the behaviour term.
2. `assert_pcgrad_free_parity` relied on the caller's flag patch still being in force; the caller
   restored it first, so it compared an enabled arm against a disabled parent.
3. The R5-09 projection consumes the EMA value **after** this step's update (`m3`), not `m2`.
4. A single R5-09 step cannot differ from R4-08 at all (the first update seeds the buffer with
   `g_base`), so differentiation must be measured on step 2.
5. The R5-06 annihilation fixture had the wrong sign — projection fires only when `dot < 0`.
6. The R5-07 "disable" value `-1.0` does the opposite of disabling (`cos >= -1` always holds);
   forcing the parent rule needs a threshold above every achievable cosine.
7. The R5-07 boundary can only be hit to float32 resolution (~1e-6), not 1e-9.
8. R5-08 before epoch 2250 takes the production `loss.backward()` branch, which is the *fully
   assembled* loss, not the base term.
9. R5-04 legitimately projects once per block, so a blanket "at most 1" assertion was wrong.

## 4. Two-rank DDP — `test_ddp_gradient_equivalence.py`

Real CPU `torch.distributed` gloo, 2 processes, `fork` start method, uneven local batches
(rank 0 = 24 rows, rank 1 = 20). Comparison target is a pooled reference built in a single
process that never calls `_r4_08_global_average`, so a bug in the production pooling cannot hide
inside the reference.

| Phase | Log | Result |
|:--|:--|:--|
| GREEN | `logs/ddp_green.log` | **8/8**, `rc=0`, whole suite 37 s |
| RED | `logs/ddp_red.log` | not applicable by design — the suite compares the production DDP path against an independent pooled reference inside one snapshot; there is no parent-tree variant to regress against |

Verified results:

```
production pooled ['0.03234116', '4.82487631', '0.00000000'] == independent mean ['0.03234116', '4.82487631', '0.00000000']
||mean|| = 0.032341, ||sum|| = 0.064682, ratio = 2.000000000 (exactly 2)
local dots -185248.0 / +370496.0, projections fired = 0, ranks bitwise identical
weights bitwise identical across ranks after the step
non-encoder ||grad|| reduced by the reducer: 0.22215609 -> 0.19291170
4 blocks; pooled global dot +2.1326e-03; block dots ['+1.707e-05', '+1.008e-03', '+9.781e-04', '+1.295e-04']
EMA buffers identical across ranks (8 tensors), first |m| sums [0.127286404371, 0.013501757756, 3.599045991898]
```

The third line is the P0 result issue #71 §3 asks for: rank 0's local dot is −185248 (a
conflict) and rank 1's is +370496 (no conflict), yet the pooled dot is positive, so a correct
global projection must do nothing. It fires 0 times and both ranks assemble bitwise identical
gradients — the production path projects on globally averaged vectors, not per-rank local ones.

Four defects in my own first drafts, each of which had silently made a test vacuous:

1. With the `fork` start method a child gets a **copy** of the parent's memory, so rank results
   written into module globals never reached the parent — the suite reported "rank did not
   finish: {}" while both ranks had completed. Ranks now persist to `logs/ddp_artifacts/`.
2. Repeatedly creating and tearing down a gloo file store inside one parent process wedged after
   the first scenario and surfaced as a false "ranks hung past 180s". Each scenario now runs in a
   fresh subprocess (`--scenario <tag>`).
3. The child never wrapped the module in `DDP`, so the **reducer never ran**: decoder/codebook
   gradients stayed rank-local (norm exactly the rank-0 local value, 2x the pooled one). Worse,
   `backward()` was being called on a loss built outside `DDP.forward()`, which skips the reducer
   even when wrapped — the forward must go through the wrapper. Both are now production-faithful.
4. DDP broadcasts parameters **in place** at construction, which invalidates any graph built
   before it; and `optimizer.step()` does the same between steps. The losses are therefore rebuilt
   on the post-broadcast weights and re-forwarded every iteration, exactly as `train_rqvae.py`
   does with a fresh batch per step.

I also removed the hand-rolled full-vector reference this suite originally compared against. It
disagreed with the production path and was itself buggy (it under-counted the second rank's
gradient by 2x), and the honest fix was to drop it and assert the properties issue #71 §3 actually
names — exact pooled-component equality, cross-rank bitwise identity, reducer effect on the
non-encoder gradients — rather than loosen a tolerance until a wrong reference agreed.

## 5. SID byte-level reproduction — `verify_sid_from_checkpoint.py`

Not a TDD suite: a provenance proof. 11/11 arms `VERIFIED` — the committed `item_sids.json` is
reproduced **exactly** (0 differing cells of 98,348) by loading the saved `rqvae_best.pth` and
re-running the trainer's own export path under the production numerical configuration.

This only holds at the trainer's own precision setting. Every trainer sets
`torch.set_float32_matmul_precision("high")` (TF32 matmuls) at module scope; under full fp32, or on
CPU, 91–134 items per arm flip to a different code even though the weights and the stage-1
embeddings are bit-identical. Both numbers are recorded per arm in `sid_reproduction.json`.

A control run against the pinned baseline Resample artifact showed the same behaviour, which is
what identified the precision setting as the cause rather than any R4/R5-specific corruption.
