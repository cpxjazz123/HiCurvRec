# Round round_20261105_144909_130 — `pmi-geomargin-l1-rq`: negative result

**The mechanism trained and exported a technically valid SID; the owner's learned
three-level-collision gate refused the Stage 3 launch, so no recall was
measured.** The owner decided to respect the gate. The diagnosis below says what
collapsed and why, and what the next round must change.

## What ran

| stage | result |
|---|---|
| ideation → probe → ranking | five proposals; `pmi-geomargin-l1-rq` selected (two first places, total rank 5) |
| formula → two Math Reviews | seven review rounds; the specification was **frozen** at `3ead324f4e71` after both reviewers passed |
| coding ×3 | three independent implementations of the frozen surface; the behavioural differential agrees in both arms (worst gap 4.8e-07) |
| implementation (main agent) | 22 tree tests written before the code (16 mechanism, 6 wiring); all nine machine checks pass, including the reference rebuild (values 1.3e-12, gradients 2.1e-12, gradcheck on both sides) and the encoder gradient (8 of 8 tensors) |
| Stage 2 | 3000 epochs on four cards (0.112 s/epoch, ~6 min); 24587 unique four-token identifiers exported |
| layered metrics | recorded **before** any Stage 3 result existed |
| Stage 3 | **not started** — the owner's gate refused the SID |

## The numbers

| metric | this arm | frozen Resample reference |
|---|---|---|
| L1 hard ER | **2.183** | 16.390 |
| L1 positive hit rate | **0.404** | 0.079 |
| L1 negative hit rate | 0.185 | 0.005 |
| level-2 distinct prefixes | **1027** | 16145 |
| level-3 distinct triples | 10870 | — |
| L3 collision (item-weighted) | **0.703** | 0.065 |
| L3 refinement gain | 4.545 bits | — |
| conditional G3 | −0.304 | — |
| level-1 codes used | **42 of 256** | 241 of 256 |
| items in the largest level-1 code | **31.0 %** | 1.0 % |

The owner's gate refuses an L3 collision above 0.35. This arm measures 0.703.

## The diagnosis

The specification replaced the baseline's **Sinkhorn-balanced** level-1
assignment with a plain **argmin** over `DIST(LIFT(r), LIFT(a_k)) - off_k`.
Sinkhorn's balancing is exactly what keeps a quantiser's codebook usage even; a
plain argmin over a distance matrix collapses onto a few codewords. The
per-codeword behaviour offset compounds it: the codewords with the largest
demand win nearly every assignment, so the collapse feeds itself.

The implementation is not at fault. It follows the frozen specification
statement by statement, and the reference contract rebuilds that specification
independently and agrees with the code. **The specification is at fault**, and it
is fixable: keep the baseline's Sinkhorn balance and apply the mechanism's score
*inside* the balanced assignment — the cost matrix becomes `D_k - off_k` — so
the behaviour shifts the assignment without removing the balancing that keeps
the codebook even.

The specification also has to state that the level-1 assignment is the
baseline's balanced assignment, not an argmin, because the frozen surface's
`k1 = argmin_k score_k` is what removed the balancing.

## What this round changed in the framework

- the **eval gate**: every defect a Math Review caught became a check that
  refuses it, the findings are decided in a ledger, and the retained tests run
  before every freeze (`tools/research/rounds/evals.py`, `eval_ledger.json`,
  `spec_tests/`);
- the **specification's surface** is the idea's expression in SymPy (with the
  eps and the clamp), the executable listing beside it is complete, and eleven
  checks now refuse a listing that is not a program;
- the **admission** runs the tree where it will train (the promoted copy under
  `stage2_RQ-VAE/`), and the DDP and gradient probes adapt to the model's
  interface instead of demanding one the frozen baseline does not have;
- the **differential** follows the surface's own callable and compares the
  implementations inside the ball, where the surface is defined.
