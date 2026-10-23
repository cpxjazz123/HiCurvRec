# Question (B): Is the successor mean actually the successor?

## EVIDENCE

- `train_rqvae.py:691–697` constructs each transition as `(int(history[-1]), int(target))`. Thus column 0 is the source: the last item in the observed history. Column 1 is the target: the item that follows that source.
- `train_rqvae.py:1212–1213` assigns `pairs[:, 0]` to `source` and `pairs[:, 1]` to `target`.
- `train_rqvae.py:1220–1222` executes `sums.index_add_(0, target, latent[source])` and counts by `target`. Therefore, for each item in column 1, it accumulates the latent vectors of items in column 0 that transition into it: its **predecessors**.
- The helper's docstring says it computes a successor mean (`train_rqvae.py:1192–1196`), but the accumulation indices establish the opposite direction. Also, in this checked-in version, the actual factor is the normalized mean (`train_rqvae.py:1225`); it does not subtract `latent` as in the question's paraphrase.

## VERDICT

**PREDECESSOR.** Column 0 is the preceding/source item; column 1 is the following/target item. Since the target is the accumulation index and the source latent is the accumulated value, each target receives the mean of its predecessors, not the mean of its successors. The deciding lines are `train_rqvae.py:697`, `train_rqvae.py:1212–1213`, and `train_rqvae.py:1220–1225`.

## CONSEQUENCE

The implemented role factor points toward the mean latent representation of items that lead into a given item. It is an incoming-transition/predecessor signal, despite the successor wording in the helper documentation. Any interpretation or analysis treating these factors as outgoing successor preferences reverses the transition direction. This conclusion concerns the actual code in this workspace; the stated `mean - latent` expression is not present in the current helper implementation.
