# Failed Mechanism Ledger

> Static reference only. It is **read** to avoid repeating a dead end and is
> **never appended to by a new run**: iterations no longer have ids, and
> SKILL.md §0.1 forbids creating new workflow records.
>
> A rejected condition leaves exactly one durable trace: the `git revert`
> commit on `main`, whose message states the mechanism and its measured
> `test_R@10`. `git log` is the ledger of record.

## Why a mechanism is rejected

Rejection means the condition was reverted after Stage3 `test_R@10` came out
at or below the parent. A revert commit is sufficient and is not duplicated
here.

## Reusable lessons from the reverted history

These are the durable findings, stated so a future mechanism does not repeat
the work:

1. **Learned or cyclic curvature can erase a closed-form prior.** Several
   early iterations used a learnable per-layer scale; the scale absorbed the
   prior instead of expressing it.
2. **Stronger Stage2 geometry proxies do not imply downstream gains.**
   Improvements in Gini, collision, or entropy repeatedly failed to move
   `test_R@10`.
3. **Optimizer-side curvature changes (curvature-conditioned AdamW beta2) were
   outside the FCCR-1 contract** and are not admissible as a fix.
4. **Stacking Sinkhorn and behavior mechanisms created attribution ambiguity.**
   One change per condition is required; the earlier multi-change iterations
   could not attribute their deltas.
5. **Some iterations tested the intended mechanism with the wrong residual
   semantics**, so a null result there is not evidence about the mechanism
   itself.
6. **Forcing the latent onto a large-radius shell failed** (collision 0.316 to
   0.853). The encoder's small natural radius is load-bearing: it keeps the
   quantizer locally Euclidean, where the Sinkhorn assignment is
   well-conditioned. Diagnosed directly: at |z| ≈ 0.024 the Poincare expmap is
   linear to 1e-4 and the distance ratio `d_poincare / d_euclidean` is a
   constant 2.0003 (std 0.0003), so curvature is currently a no-op. Activating
   the geometry must not be done by enlarging the operating point.
7. **TIGER Stage2 used to early-stop at epoch 700 of 3000** and to save
   best-collision checkpoints. Both are fixed: the baseline now runs its full
   budget and saves only the final-epoch checkpoint, so the two models are
   finally comparable. Any new number must be produced under that rule.

## Reading history

```bash
git log --oneline -20
git show <revert-commit>
```

A revert commit's message names the mechanism and the measured delta.
