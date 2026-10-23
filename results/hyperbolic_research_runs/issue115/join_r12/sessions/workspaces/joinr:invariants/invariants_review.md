# Measurability review — issue #115, JOIN R round 7

## Verdict: REVISE

This review covers exactly the mechanism declared in `candidate.md`: assignment by the **unsigned 2-simplex area**
\(A_k=\tfrac12\sqrt{\|u\|^2\|v_k\|^2-\langle u,v_k\rangle^2}\), tempered by geodesic distance, with the stated lexicographic tie rule. There is no signed-area operator or second score term in the proposal; neither is reviewed here.

The frozen protocol remains the 32-D latent, three 256-row codebooks, three integer SID tokens, and original data, splits and recipes. The conjunctive strict metric thresholds are Hard-L1 ER > 16.38953942232631, Recall@10 > 0.06406796775709883 and NDCG@10 > 0.03530651701940677. The five already-refuted families (radius-dominated product assignment, horospheric level sets, transported-residual frame, measured-vestigial role-assigned layout, and any Euclidean reformulation) are out of scope and are not reopened.

## What the declared artifact can establish

If the JSON is faithfully emitted and its values are trustworthy, a reviewer can calculate the flip rate from the two winner arrays, compare the recorded summary to the 5% per-level threshold, and derive a *reported* clamp ratio from the count and a denominator. It also carries a candidate hash, stage, row indices, per-row area/distance fields, and a summary tie share. These are useful audit hooks.

They do **not**, as specified, make all those checks decisive from the artifact alone:

1. **No executable emitter/call site is present here.** `candidate.md` describes `assignment_audit.emit` and claims it runs per level before metrics, but repository search finds no implementation or call-site—only the prose references in `candidate.md`. Thus the mandated artifact, its completeness and its placement cannot currently be inspected or verified. Do not treat the declaration as evidence that an artifact was actually produced.
2. **The ablation is underspecified.** The declared score is \(A_k/(1+d_k)\), but “area-zero score” is not defined mathematically. Setting area to zero for every code makes every score zero and leaves the lexicographic key `(A_k, d_k, k)` as the declared tie resolution; that may not be the intended ablation. State the exact ablated per-code score and selection rule. Otherwise the flip rate does not isolate the claimed area contribution.
3. **Winners cannot be independently checked.** The schema gives only `winner_full`, `winner_ablated`, one `margin_full`, and per-example `area` and `distance` arrays. It has no code-indexed component/score vectors, no runner-up identities/scores, and no ablated margin. Scalar area and distance are not identified as the winning code's values, the runner-up's values, or some aggregate. A reviewer cannot reconstruct either argmin, its margin, or whether the declared unsigned-area formula generated it. The prose’s promise of winner/runner-up margins is not reflected in the schema.
4. **Tie share is not reproducible.** Neither “tie” nor its denominator is defined (exact score equality versus tolerance; ties before or after lexicographic resolution; which arm; per-row or per-code comparisons). No tie indicators/counts or score precision/tolerance are supplied. A summary scalar alone cannot be audited.
5. **Clamp rate denominator is not justified.** `clamp_count / examples.count` is only valid if exactly one clamp opportunity is counted per example. The mechanism evaluates logs for candidates at each level and potentially multiple points per row; the schema does not state the number of clamp opportunities or separately record numerator/denominator by operation. Record both counts and define the denominator (including candidate-code evaluations) so the >0.1% threshold is recomputable. Likewise define whether `error_count` is per example, operation or score.
6. **Audit coverage and provenance are incomplete.** `split` is a single value despite possible mixed rows, `echo` is a single SID prefix rather than a per-row linkage, and there is no explicit codebook/row mapping or emitted-row manifest. `candidate_sha256` is not accompanied by a hash algorithm/canonicalization rule or a binding to the actual config, data/split manifest, codebook snapshot and implementation version. A hash string by itself does not authenticate the contents or show that all intended evaluation rows are present.
7. **“Before any metric is read” is not evidenced by the file.** The schema has a boolean assertion, but no run identifier, evaluation phase/order marker, or immutable linkage to the subsequent metric result. A boolean can be written without the claimed ordering having occurred.

## Required revisions for a decisive artifact-only audit

At minimum, freeze the ablation equation and tie semantics; emit for every row the candidate-indexed full and ablated scores (or sufficient unrounded components to recompute them), winner and runner-up indices/scores/margins, and the exact selected tie key. Define floating-point precision and equality/tolerance. Include per-row tie/clamp/error flags or counts with explicit operation-level denominators, and identify rows/splits individually. Bind the artifact to canonical proposal/config/code/data/split/codebook hashes and a run ID; include expected versus emitted row counts and a verifiable metric-phase/order linkage. The emitter and its evaluation-loop call site must be available for inspection, with tests covering degeneracy, ties, clamps and the frozen thresholds.

Keep the registered gates intact: flip rate **≥5% per level**, tie share **≤10% per level**, clamp rate **>0.1% fails**, and non-finite scores or out-of-ball points fail. The strict conjunctive ER/Recall@10/NDCG@10 thresholds above remain unchanged. Do not advance to JOIN E on the basis of a summary-only or unauthenticated artifact.

## Disposition

**REVISE.** The intended falsifiers are sensible, but with the currently declared schema a reviewer cannot recompute the score-based assignments, establish the tie share, or unambiguously recompute clamp rate from the artifact alone. Once the missing schema semantics and provenance are supplied and the emitter/call site are inspectable, the artifact can support a meaningful audit without rerunning the model; until then the flip-rate claim is not decisive evidence of the mechanism’s effect.
