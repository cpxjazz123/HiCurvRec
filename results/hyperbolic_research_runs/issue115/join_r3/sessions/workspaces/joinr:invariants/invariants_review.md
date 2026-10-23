# Adversarial invariants review

## Scope and bottom line

The frozen contract remains 32 latent dimensions, three 256-way codebooks, exactly three integer SID tokens, unchanged data/splits/Stage-2 and Stage-3 recipes, and the strict conjunction ER `>16.38953942232631`, Recall@10 `>0.06406796775709883`, NDCG@10 `>0.03530651701940677`. The five previously refuted families are out of scope.

| Candidate | Verdict | Why |
|---|---|---|
| 1. Ordered-simplex signed-area assignment | **UNMEASURABLE (as currently registered)** | The current artifacts can measure partition health, implementation correctness, and task metrics, but do not establish a paired, per-example area-on/off assignment and margin record. The claimed margin/change tests require new retained outputs. Once those are emitted, it becomes falsifiable. |
| 2. Origin-chart ordered composition with tangent gyration | **UNMEASURABLE (as currently registered)** | Same artifact gap for per-example `γ||δ||²` contribution and matched assignments. CGV/H1-H7 can check declared math/gradient behavior, not whether the term changes real decisions. It becomes falsifiable if the already-proposed per-example audit is actually added. |

Neither is approved by a correct formula, CGV pass, H1-H7 pass, nonzero component variance, or frozen ER alone. Partition statistics can expose collapse/coarsening (e.g. the historical level-0 collapse to 1 distinct code and largest cell 24,587), but cannot identify *which term caused* a change. And Stage-3 may not run if ER fails, so downstream falsification is conditional on passing its scheduler gate. The current proposal’s promise to report new score components and paired margins is a request to create an artifact, not evidence that current repository artifacts already contain it.

## Shared audit: what the registered tests do and do not prove

- Ball closure, finite scores, clamp rate, Gauss–Bonnet, flat limits, transport norm preservation, and CGV/H1-H7 are **implementation/contract checks**. An identity or unused operator can pass many of them. Keep them as correctness gates, never as evidence of useful assignment.
- “Changed-code fraction ≥10%” is discriminating against a literal identity **only if** calculated from actual frozen examples, same host/codebook/checkpoint, and the named operator alone is toggled. It is not by itself evidence of a learned benefit: arbitrary perturbation or tie-breaking can change 10% of labels while worsening ER and retrieval.
- The registered `10^-4 × per-example median absolute full-score scale` is not robustly discriminating. The scale can be zero/tiny, the threshold can be below floating-point/relevant decision scale, and score units differ between candidates. A score-margin perturbation can be large on examples whose winner remains unchanged, or tiny but flip near-ties. Normalize by the *winner-vs-runner-up gap* and report actual flips; do not call a raw score delta a causal assignment effect.
- “At least 5% examples have margin change” can pass while the operator never changes a single argmin. It is therefore not a substitute for changed assignments; conversely a 10% changed-code threshold alone can be satisfied by harmful reassignment. Both must be reported, with task outcome and partition health.
- The strict frozen triple is a necessary task-level kill criterion, not a mechanism-specific falsifier. ER alone is particularly inadequate: prior runs show improved ER can coexist with downstream failure, and the unchanged-control ER itself has run-to-run spread. Report the three gates separately and require the conjunction; describe causal improvement only relative to a same-protocol control, not merely the threshold.

## Candidate 1 — Ordered-simplex signed-area assignment

### Measurability

**Not measurable from the currently guaranteed artifacts.** Existing partition reports can show per-stage distinct-code counts, largest-cell share/Gini and branches; CGV/H1-H7 can validate forward/gradient/math contracts; frozen ER and (if scheduled) Recall@10/NDCG measure task performance. None of these aggregate artifacts says, for each validation item and stage, (i) the selected code under full score versus area-zero score, (ii) the full and ablated winner/runner-up gaps, or (iii) whether the signed area term actually contributed to the decision. Proposal lines 41–42 ask to report these, but do not make them existing artifacts. Add a deterministic audit to the existing partition-statistics export; do not claim the current outputs suffice.

The flat-limit prediction is an implementation sanity check only. It does not establish finite-curvature utility. The area-zero ablation is the relevant mechanism test, but only if same points, codebook, stage host, and tie rule are used; comparing independently trained arms alone confounds the operator with optimization trajectory.

### Rewrite the invariants (non-vacuous)

1. On the exact held-out catalogue/validation rows used for partition reporting, at each of the three stages, retain full-score and area-zero per-code score arrays (or enough to reproduce them), selected indices, top-two gaps and area values. With the **same checkpoint and points**, report the paired assignment flip fraction `F_s = mean[ argmin S_full != argmin S_A=0 ]`, plus its bootstrap 95% CI. A no-op area term gives `F_s=0`; it must fail. Pre-register a meaningful minimum (retain the proposed `F_s ≥ 0.10` only as an *operator-materiality* bar, not a utility claim); require the lower CI bound to exceed zero and report every stage, not only pooled values.
2. Measure decision leverage as the fraction of rows for which the area term changes the signed top-two decision gap by at least `max(1e-3 × baseline median positive top-two gap, numerical tolerance)` **and** separately report the fraction that actually flips the winner. Drop/flag rows with zero baseline gap scale rather than normalizing by a possibly zero median absolute score. Report the distribution and sign of `A`, distance-only/full gaps, and assignment agreement. A variance/covariance or geometric identity alone does not pass.
3. Recompute partition statistics for full and area-zero assignments on the same examples: distinct codes, largest-cell share, Gini, and (where applicable) downstream branch counts. Do not assert “better partition” from usage alone; require no collapse and inspect whether the full arm preserves or improves discrimination relative to the paired control.
4. Task survival remains the frozen strict triple on the official same-recipe evaluation. A threshold pass without a paired control is not evidence the area term caused a gain. If ER fails, record that Stage-3 is unrun/not evaluable, not a downstream pass.

### Simplest decisive control

Use the candidate’s **one trained checkpoint**, fixed validation examples, hosts and codebook points; score twice: (A) full `-A + βd²`; (B) exact `A=0`, same `βd²`, projection, tie-breaking and no retraining. This isolates assignment effect without optimizer confounding and is the cheapest falsifier. Then, only if it clears the pre-registered materiality and partition checks, compare the official cold full-training arm against a cold paired `A=0` training arm (same seed/order/recipe) for causal task impact. Keep the flat `c→0` check as correctness-only. Do not use the flat limit as the control for finite-curvature utility.

## Candidate 2 — Origin-chart ordered composition with tangent gyration

### Measurability

**Not measurable from the currently guaranteed artifacts.** Partition statistics and frozen metrics can detect gross outcomes; CGV/H1-H7 can validate typed formulas, autodiff and declared operator participation. They do not supply per-example `δ_k`, `γ||δ_k||²`, the score gap with and without gyration, or paired assignments. Those are essential because a nonzero gyration vector can be irrelevant to every argmin. Add those values (or a reproducible deterministic audit) to the existing partition-statistics artifact. `PT^{rad}` is diagnostic-only and norm-preserving by construction; passing that check says nothing about assignment.

### Rewrite the invariants (non-vacuous)

1. On the same trained checkpoint/points and held-out rows, calculate full `S_k` and `γ=0` scores, per-stage chosen code, top-two gap and paired flip indicator. Report `F_s = mean[argmin S_full != argmin S_γ=0]`, with bootstrap 95% CI. Identity gyration (`δ_k=0`) must produce exactly zero flips; require the registered `F_s ≥ 0.10` in every stage plus lower CI bound above zero to call the operator decision-material. A nonzero `||δ||` distribution is not a pass.
2. For each row, report the actual differential contribution to the winner/runner-up gap, `Δ_g = γ(||δ_runner||²-||δ_winner||²)` under the full winner/runner-up pair, its normalized distribution relative to the `γ=0` positive top-two gap, and whether its sign/magnitude flips the winner. Require both a nontrivial contribution and observed flips; the existing “≥5% margin change by 10^-4 median absolute full score” is vacuous as a decision criterion and should be removed. Define zero-gap/zero-scale handling explicitly (report as ties; do not divide by zero).
3. Recompute the same partition diagnostics for full versus `γ=0` assignment on identical examples: distinct codes, largest-cell share, Gini and branch counts. This catches collapse but does not replace the paired flip test or task gate.
4. The flat-limit identity `D_oGyr→I` and score agreement with `||h+v+r||²` are correctness controls, not mechanism evidence. At frozen `c`, require the paired gyration ablation to pass the decision-materiality test and the official full arm to clear all three unchanged task thresholds. If ER fails, Stage-3 is not evidence either way.

### Simplest decisive control

At fixed trained checkpoint and exact held-out rows, compare (A) full score with its `γ||δ||²` term to (B) `γ=0`, leaving the ordered composition term, points, codebook, chart, projection and tie-breaking unchanged. Save per-example score components and assignments; this is the minimal falsifier and avoids confounding retraining. If it is material, the minimal causal task control is a paired cold Stage-2 `γ=0` arm with identical seed/order/recipe, followed by the unchanged Stage-3 recipe only when the ER gate allows it. Do not substitute the flat-limit score or a norm-preservation test.

## Decision

Both proposals have mathematically stated possible falsifiers, but **as of the artifacts guaranteed by this repository they are UNMEASURABLE**, not “falsifiable” merely because the proposal text promises diagnostics. Emit the specified paired per-example audit in the existing partition-statistics output first. Afterward, the two candidates are empirically falsifiable; the task-level decision still requires the strict frozen triple, and operator materiality alone is not success.
