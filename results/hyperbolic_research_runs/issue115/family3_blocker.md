# Family 3 withdrew too -- and this time the blocker is concrete and mine to fix

Role M's withdrawal (real session pid 861601, 25.1 s, 8 tool calls):

> "I cannot responsibly nominate a mechanism from the frozen evidence visible in this workspace. This
> is a withdrawn candidate, not a claim that every possible mechanism is refuted. I will not invent a
> trigger frequency and will not claim one was measured."

and, precisely: the required occurrence rate of the configuration the mechanism acts on is **not
available, not zero** -- "the workspace has only `opencode.json`, and lacks the assignment stream and
the frozen Stage2 artifacts needed to compute the numerator and the denominator".

## The diagnosis

The 6c requirement (measure the frequency of the configuration you act on BEFORE proposing) is
correct, and I under-provisioned the agents: I gave them the *protocol* and the *prior art*, but not
the **frozen artifacts** from which a rate can be computed. So M refused, correctly, rather than
guessing -- the same discipline that made LOTV withdraw.

## The fix (next step, deterministic)

Stage, into every research workspace, the frozen artifacts that carry the numbers:

| artifact | path | what it supplies |
|---|---|---|
| the control's exported SIDs | `results/stage2_RQ-VAE/r8_control/item_sids.json` | the frozen partition, its code usage and its per-cell structure |
| the frozen ER evaluation pairs | `results/stage3_T5Train/monotonicity/behavior_pairs_valid_s42.npz` | the positive/negative pairs the ER is computed on |
| the train parquet | `results/stage0_build_parquet/train.parquet` | the user histories (the successor structure) |
| the stage-1 embeddings | `stage1_GeneEmbedding/output/sentence_t5.npy` | the latents the assignment operates on |

With those, a candidate's occurrence rate is COMPUTABLE (numerator and denominator both come from the
frozen artifacts), and the proposal can state a measured number instead of a promise.

The campaign does not proceed by nudging an agent into proposing anyway. It proceeds by giving it the
evidence the rule requires.
