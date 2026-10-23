# Measurability review — candidate v7

## Verdict: REVISE

The declared mechanism is the unsigned 2-simplex area
\(A_k=\tfrac12\sqrt{\|u\|^2\|v_k\|^2-\langle u,v_k\rangle^2}\), tempered by distance. I find no signed-area operator or second score term in `candidate.md`; neither is reviewed here.

The proposed artifact is a useful start: with stable row identities and both winner arrays, a reviewer can independently calculate the observed full-versus-ablated flip fraction. But the schema as written does **not** let a reviewer verify all stated audit gates or establish that the emitted winners and component values really came from the declared scoring rule. The assertion in §8 that flip rate, tie share, clamp rate and verdict can all be recomputed from the file alone is therefore too strong.

## Missing fields / definitions needed for decisive falsifiers

1. **Tie share is undefined and not independently computable.** The schema supplies one full-score margin per row, but no ablated margin, tie indicator, tie count, or rule for what counts as a tie (exact equality versus tolerance, and the tolerance). A margin without its runner-up identity/score and an explicit tie convention is insufficient to reconstruct tie share, especially with floating-point scores. Emit the full and ablated tie status/counts (or all relevant top scores and a frozen comparison tolerance), and define the denominator and treatment of invalid rows.
2. **Clamp rate has no denominator.** `clamp_count` alone cannot establish the >0.1% gate. Emit the number of clamp-eligible evaluations for the same stage, define exactly what constitutes one evaluation (including points/log-map calls and repeated candidate evaluations), and report the resulting rate. The stated per-stage gate needs per-stage counts, not an ambiguous aggregate.
3. **Out-of-ball failure is not represented.** The prose says any out-of-ball point fails, but the schema has no out-of-ball count/flag or pre-projection norm evidence. Emit a count/status before projection (and the relevant population denominator), distinguishing an out-of-ball code from a clamp event.
4. **Non-finite failures are not auditable.** `error_count` is not a non-finite-score count and gives no indication which rows/operations failed. Add non-finite counts by score/operation and row, and make `ok` mechanically derivable from the emitted counts and frozen thresholds. Do not let failed rows silently disappear from the flip/tie denominators.
5. **The ablation is asserted, not demonstrated.** `winner_ablated` alone cannot verify it was computed using area set to zero while retaining the declared distance term and tie-break rule. Emit per-example ablated score inputs/score (and enough top-code data to audit the winner), along with full score components and the frozen argmin/tie-breaking semantics. As currently defined, `per_example.area` and `distance` do not specify whether they refer to the winner, every candidate, or another summary; one winner's components cannot validate an argmin over 256 codes.
6. **The claimed margin evidence is incomplete.** Only `margin_full` appears in the schema, despite §5 promising winner/runner-up margins. Define whether margins are score differences and emit them for both full and ablated assignments. For a meaningful independent audit of argmin and ties, retain at least winner and runner-up scores/indices per condition (or all 256 per-code scores/components if artifact size permits).
7. **Artifact provenance needs a checkable contract.** `candidate_sha256` is not independently useful unless the exact bytes covered by the hash and canonicalization are specified and the frozen candidate is available to the reviewer. Bind the artifact to the evaluation code/config and dataset/split manifest hashes as well; define `stage` unambiguously (level 0–2) and the exact population. `echo` is not a substitute for those bindings.

## What can be concluded from the current schema

Subject to trusting the producer and interpreting each row as one evaluated example, the listed indices and winner arrays allow recomputation of the flip rate. They do not establish that this is the declared ablation, that ties satisfy the 10% limit, or that clamping/non-finite/out-of-ball events satisfy their gates. Thus falsifier 1 is only partially measurable, while the stated audit gates as a whole are not decisive from the artifact alone.

Revise the schema and definitions above before JOIN_E. This verdict is about measurability of the audit artifact, not a claim that the mechanism has already failed its frozen ER/Recall/NDCG thresholds or partition gate.
