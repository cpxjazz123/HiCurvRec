# Invariants review — candidate v7

## Verdict: REVISE

The proposal declares one mechanism: an **unsigned 2-simplex area** in the tangent space at the current residual host, divided by `1 + d_c(p,z_k)`, with `argmin` assignment. I find no signed-area mechanism or second score term in `candidate.md`; I do not review either. The frozen protocol and strict conjunctive ER/Recall@10/NDCG@10 thresholds are not altered here. The five previously refuted families are out of scope and are not revived.

The audit is a useful start, but **the schema as written does not let a reviewer establish the mechanism's assignment effect from that artifact alone**. It permits recomputing the reported flip count/rate from the two winner arrays, but it does not supply enough information to independently verify that those winners, margins, ties, clamp rate, and failures follow the declared procedure. Since the artifact explicitly disclaims proving the winners came from the rule, this is a material gap for the requested falsifiers, not a claim that the operator itself is ambiguous.

## Missing invariants / decisive evidence

1. **Per-code inputs and scores.** `per_example.area` and `.distance` are singular arrays, with no definition of whether they refer to the winner, runner-up, or some aggregate. They cannot reconstruct an argmin over 256 codes. Emit, for every example and code, the unrounded `A_k`, `d_c(p,z_k)`, full score, ablated score, and the tie-broken ordering/key (or a lossless equivalent). Include the full and ablated winner and runner-up, and margins under a precisely specified definition. This enables an independent check of the full and area-zero winners, rather than merely trusting them.

2. **Ablation definition and exact tie semantics.** State explicitly that area-zero means setting `A_k = 0` for all codes while retaining the same distance factor and all other inputs (or specify the actual ablated score). Include the exact lexicographic comparison keys and numeric tie tolerance (if any); distinguish exact mathematical ties from floating-point ties. The prose's `(A_k, distance, k)` rule and the declared `argmin S_k` need a reproducible priority when distinct scores tie. Emit enough data to recompute tie share, and define its numerator, denominator, and whether ties are counted before or after the lexicographic break.

3. **Tie-share observability.** The schema has only a summary `tie_share`; it lacks per-row tie counts/flags or candidate scores from which to recompute it. Define whether tie share means examples with tied minima, tied area values, or something else. Emit the tied-code set/count per row (for full and ablated assignments if both are gated), together with the computed statistic and denominator.

4. **Clamp/error denominator and scope.** `clamp_count / examples.count` is not established as the specified rate: the text defines clamp rate over evaluations, while a row can involve many `log_p` evaluations (and potentially multiple points/codes). Emit `evaluation_count` and per-stage/per-operation clamp counts and rates, define what an evaluation is, and report out-of-ball counts. Likewise, an `error_count` alone cannot expose non-finite scores or identify affected rows/codes; emit their indices and failure categories. These are required to apply the stated `>0.1%`, non-finite, and out-of-ball kill criteria decisively.

5. **Dataset/stage coverage.** `examples.split` is a single `train | dev` value, but the audit is said to run per level in the evaluation loop and the frozen tests include Stage3. Specify exactly which split, examples, level, and evaluation pass each file covers; emit the complete expected index set/count and a manifest proving all required files are present, with no omissions or duplicates. Clarify whether the audit is on every evaluation example or a subset. A per-file count cannot show that the denominator is the intended population.

6. **Reproducibility and binding.** Define serialization/precision (prefer round-trip-safe values), deterministic code ordering, schema version semantics, and a manifest binding each file to the run/configuration, checkpoint/codebook state, `c`, scale `s`, epsilon, and candidate hash. Specify how `candidate_sha256` is calculated and bind the artifact to the actual run (for example, a run manifest/hash). Otherwise arrays may be internally recomputable but not demonstrably from the frozen candidate and evaluated model.

7. **Decision consistency.** `summary.ok` needs a formal Boolean definition that includes all per-level flip/tie and numerical gates, and the artifact needs to carry or link a complete stage manifest. The proposal also has downstream partition and metric gates; this assignment artifact cannot establish them. Keep those as separate required evidence rather than letting an assignment audit's `ok` imply overall approval.

## What is currently recomputable

Given intact `winner_full` and `winner_ablated` arrays, a reviewer can recompute the number and fraction of changed assignments, subject to a defined denominator and complete example coverage. The stated `changed_codes` can be recomputed as the number of distinct indices appearing among changed winners only if that is its intended definition. The listed area/distance arrays do not support checking the winner rule unless they are expanded to all candidate codes as above. Nor does `clamp_count` alone yield a valid clamp rate without the matching evaluation denominator.

## Required revision

Revise §8 (and align §5) to make the emitted JSON self-auditing: complete per-example/per-code score inputs or equivalent lossless values; exact ablation, tie, margin, and tie-share definitions; evaluation denominators and indexed numerical failures; complete coverage manifest; and a run/configuration binding. Then a reviewer should be able to recalculate the flip rate (gate `>= 5%` per level), tie share (`<= 10%` per level), and clamp rate (failure when `> 0.1%`), and verify the assignments without rerunning model inference. Until then the artifact reports outcomes but does not make the principal assignment falsifiers independently decisive.
