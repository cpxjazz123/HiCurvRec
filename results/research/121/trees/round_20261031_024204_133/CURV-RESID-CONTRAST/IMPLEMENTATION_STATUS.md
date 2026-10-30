# CURV-RESID-CONTRAST -- implementation status

Selected by the round `round_20261031_024204_133` (three scorers: two first
places, mean rank 1.33, IN_SCOPE).

## Done

* `model/curved_chain.py` -- the mechanism's mathematics: `exp_0`, `log_p`,
  `exp_p`, the Mobius addition, the specification's parallel transport, the
  conformal factor, and `CurvedChain` with the contrast objective
  `J = lambda(p)^-2 (alpha a^2 + b)` and the `argmin` over codewords, the prefix
  advancing through `exp_p`.
* `tests/test_curved_chain.py` -- 12 tests, all passing, written before the
  implementation: the exponential and logarithm are inverse, the points stay
  inside the ball, the transport is the identity at the origin and matches the
  specification's formula term by term, behaviour off reduces to the isotropic
  objective, geometry off reproduces the Euclidean chain token for token, the
  geometry changes the tokens when the codebook spans the radii, the gain changes
  the tokens, the prefix advances through the exponential map, the gradient
  reaches the codebooks and the encoder input, and the chain is deterministic.
* `prepare_behaviour.py` -- the behaviour constants, exactly the proposal's
  recipe (training split only): `S_i` capped at 32 successors, `N_i` 32 items
  drawn from the same popularity decile excluding i, its 1-hop neighbours and its
  successors, `u_i` the difference of the mean log maps, `u_hat_i` the unit
  direction and `alpha_i = clip(||u_i||_{x_i} / median, 0.5, 2.0)`.
  Built: **23,944 items with a direction, 643 isotropic fallbacks**, radial scale
  0.2484, median tangent norm 0.1666, gain range [0.5, 2.0].
* `model/model.py` -- the curved branch of `RQVAE` with the three switches
  (`curved`, `behaviour_on`, `geometry_on`), the reconstruction target being the
  same ball point the chain quantizes, and the commitment term on the curved
  prefix.

## Remaining

1. `train_rqvae.py`: carry the item ids through the data loader (the constants
   are indexed by item id), set `curved`/`behaviour_on`/`geometry_on` and
   `curved_radial_scale` in `_tokenizer_config`, and pass `direction`/`gain` to
   `forward` and to `get_indices` at export.
2. The one-epoch smoke on the real entry point.
3. `research_round_register_implementation`, the five verifiers,
   `research_round_accept`.
4. Stage 2 (four GPUs, 3000 epochs, global batch 4096, port 50202) and Stage 3
   (frozen protocol, port 50201), then the `test_recall@10` reading.

Nothing in this tree has been trained yet, and no recall value exists.

## Smoke measurement (2026-10-31 03:05-03:12)

The real entry point started (four ranks, torchrun, port 50202, `train_start`
written at wall time 80 s) and **did not finish a single epoch in seven
minutes**: `logs/training_metrics.jsonl` holds the `train_start` event and no
`train` event. The frozen baseline runs 0.16 s per epoch.

The cause is in the objective: `CurvedChain.level_objective` materialises the
`(batch, codes, dim)` difference to compare every codeword with every residual
(`1024 x 256 x 128` floats per level per step, with autograd), where the same
quantity is an algebraic function of a handful of inner products:

    a_k   = <r, u'> - <g_k, u'>,        ||r - g_k||^2 = ||r||^2 - 2<r,g_k> + ||g_k||^2
    g_k   = (1 - |p|^2) (e_k - t_k p),  t_k = <p, e_k> / (1 + sqrt(1 - |p|^2))

so every objective entry is a `(batch, codes)` scalar that needs `<r,e>`,
`<r,p>`, `<e,e>`, `<p,e>`, `<r,u>`, `<p,u>`, `<e,u>` and `|r|^2`, `|p|^2`.

Per CLAUDE.md 19c this candidate is **RESOURCE_BLOCKED**: the mechanism is
implementable and its mathematics is tested (12/12), but this implementation is
about three orders of magnitude over the frozen training budget. The
optimisation route (the algebraic reformulation above, which removes the
`(B, K, d)` tensor entirely) is the next step, and it must be followed by the
same one-epoch smoke before any formal run.

No training has been completed, and no recall value exists.

## Second smoke (after the algebraic objective)

The objective was rewritten in its algebraic form — every entry is a function of
`<r,e>`, `<p,e>`, `<e,e>`, `<r,p>`, `<r,u>`, `<p,u>`, `<e,u>`, `|r|^2`, `|p|^2` —
so the `(batch, codes, dim)` tensor is gone. A new test compares it with the
materialised form: **max |algebraic - direct| = 3.6e-15**. The suite is 13/13.

Two defects were found on the way and fixed: the conformal weight was `lambda^2`
where the specification says `lambda^-2`, and `a_k` was taken against the raw
direction instead of the transported one.

The second smoke still does not reach a first epoch: after four minutes the log
holds only `[RecBole RQ-VAE] successor attraction: 24587 items` (the PREVIOUS
round's code path), no `[curved]` line, and the four GPUs sit at 0% utilisation
with 525 MiB each. `logs/training_metrics.jsonl` is not written at all, so the
hang is between the successor-attraction build and the training loop — i.e. the
new `USE_CURVED` block that loads the behaviour constants and the codebook
initialisation have not been exercised successfully yet.

Status: the mechanism's mathematics is complete and tested, the wiring is
incomplete and the entry point does not run. **No training has been completed
and no recall value exists.**

## Stage 2 ran to completion, and the SID is degenerate

The four-GPU run finished all 3000 epochs in **477.8 s** (0.145 s per epoch, the
baseline's own speed -- the algebraic objective removed the three-orders-of-
magnitude overhead), saved the final checkpoint and exported `item_sids.json`
(24,587 items, width 4):

| level | distinct prefixes | largest group |
| --- | --- | --- |
| L1 | **2** | 24,266 |
| L2 | **2** | 24,266 |
| L3 | **2** | 24,266 |

`final_collision = 0.99992`: every raw SID is one of two codes, and the exported
SID is unique only because the disambiguation token is appended. The training
loss reached 2.3e-9 and the reconstruction 1.9e-9, i.e. the encoder collapsed.

**The defect.** The curved path reconstructs `log_0(prefix)` against the encoder
output and adds `beta * MSE(encoded, reconstructed.detach())` as the commitment.
Both terms are minimised by driving `encoded` to zero -- the ball's origin, where
every codeword is equidistant -- so the quantizer degenerates to one code per
level instead of spreading the items. Nothing in the path anchors the scale: the
baseline avoids this by reconstructing the INPUT embedding through the decoder,
which keeps the encoder output away from the origin.

**Verdict for this candidate.** `IMPLEMENTATION_NEEDS_FIX` (CLAUDE.md 19c): the
mechanism trains at the frozen budget and exports a technically valid SID, but
the SID is degenerate, so a Stage 3 run on it would measure nothing. No Stage 3
was started and **no recall value exists**.

**The fix to apply next.** Anchor the reconstruction: keep the baseline's
`decoder(prefix) -> input embedding` reconstruction (so the encoder output stays
away from the origin) and keep the curved assignment as the quantizer, with the
commitment measured between the prefix and the transported encoder tangent.
That is a change of the reconstruction target and of one loss term, testable by
the same suite plus a check that the L1 prefix count stays above, say, half the
number of codewords.

## Second Stage 2 run with the anchored reconstruction: worse, not better

The reconstruction was anchored to the input embedding through the decoder (as in
the baseline) and the commitment was left pulling only the code side. The run
finished all 3000 epochs in 603 s, and the export is **more** degenerate:

| level | distinct prefixes | largest group |
| --- | --- | --- |
| L1 | **1** | 24,587 |
| L2 | **1** | 24,587 |
| L3 | **1** | 24,587 |

`final_collision = 0.99996`. Every item ends in the same code at every level, and
the exported SID is unique only through the appended disambiguation token.

**What this rules out.** The collapse is not caused by the reconstruction target
alone: with a decoder strong enough to ignore its input, the encoder can still
send every point to one place, and the curved assignment does not resist it --
the objective has its minimum at a single codeword reached by every item. A
mechanism that resists this needs a term that makes using more codewords the
cheaper option (a usage or entropy term), and every such term is a constant that
must be derived rather than tuned (CLAUDE.md 19g), so this is mechanism work, not
a patch.

## Where the round stands

* The multi-agent chain works: the provider is unblocked, five ideation agents
  and three scorers ran, and one mechanism was selected.
* The mechanism's mathematics is implemented and tested (13/13, including the
  algebraic form against the materialised one to 3.6e-15).
* Stage 2 runs at the frozen budget and the baseline's speed (0.145 to 0.16 s per
  epoch, 3000 epochs, four GPUs), and it exports a technically valid SID.
* **The exported SID is degenerate in both variants (1 to 2 prefixes per level),
  so Stage 3 was not started and no recall value exists.**

The candidate is `IMPLEMENTATION_NEEDS_FIX`: the assignment objective has its
minimum at one codeword for every item, and no admissible (derivable, untuned)
term currently prevents that.
