# Measurability review — candidate v7

## Verdict: REVISE

The operator under review is the **unsigned 2-simplex tangent-space wedge area** in `candidate.md` §1, with score `A_k/(1+d_c(p,z_k))`. I find no signed-area operator or second score term in the candidate and do not review either. The stated frozen protocol and conjunctive ER / Recall@10 / NDCG@10 thresholds are acknowledged; this review is limited to whether `assignment_audit` can establish the claimed assignment effect from its emitted artifact without a rerun.

## What the artifact can establish in principle

Schema 3 carries the Gram operands and geodesic distances, names the full and area-zero winners, and specifies recomputation of `A_k` and `S_k`. If those operands and winners are complete and trustworthy, a reviewer can recompute all candidate scores and winners from the file, verify the reported flip fraction, and inspect clamp/tie summaries without rerunning the model. That is materially stronger than a summary-only audit.

## Why the supplied example is not decisive

The checked-in `audit_example.json` is explicitly a seeded toy batch, not a frozen-protocol evaluation artifact. It declares `run_manifest: "toy"`, `candidate_sha256` as 64 zeroes, `examples: 16`, and `codes: 8`; the candidate protocol requires 256 codes per level and three levels. Its `ok: true` and 87.5% flip rate therefore demonstrate a serialization example, not evidence about the proposed mechanism on the intended data.

There are also material gaps in the binding contract:

1. **Artifact/API mismatch:** §5 names `assignment_audit.emit(...)` and describes an artifact with components; §§8–9 instead make `emit_v3(...)` and schema 3 binding. Resolve this to one required entry point/schema and specify that every per-level evaluation uses it. As written, a run could satisfy one description while failing the other.
2. **No verifiable run/candidate identity:** the example's zero candidate digest is a placeholder. Require a non-placeholder hash of the exact candidate/config/code, plus immutable run-manifest and dataset/split/recipe identifiers (and preferably their content hashes), checkpoint/weights identity, and level/codebook identity. A free-form `run_manifest` string is not enough to bind the artifact to the frozen protocol.
3. **Coverage is self-attested:** `expected_examples` and `complete` are fields, but the file does not establish the expected evaluation population or prove there are no omitted/duplicated examples. Include stable example IDs (or a verifiable ordered-ID digest plus independently specified expected count), split, and a completeness check tied to the manifest. State whether the audit covers every evaluation example, not a sample.
4. **No precise tie-share definition:** the threshold is stated, but the denominator and what constitutes a tie are not defined (full-score ties, ablated ties, either, per-example any tie, or tied winners). Define exact tie tolerance/equality semantics, tie-count numerator and denominator, and report full and ablated tie rates separately. A generic `tie_share` cannot be independently checked from the current per-example winner lists.
5. **Recomputation does not bind winners:** the example shows operands and winners, but the stated 1e-9 check concerns stored area/score values. Emit per-example full and ablated score vectors (or sufficient operands for both arms), the exact ablation rule, and deterministically recomputable winners/margins. Specify dtype, rounding, non-finite handling, and tie-break application. Otherwise a reviewer cannot verify that `winner_full` actually minimizes the recomputed scores or that the ablated winner came from the declared area-zero score.
6. **Clamp/error accounting is underspecified:** report numerator and denominator for each operation type and stage, with an unambiguous denominator for clamp rate; distinguish clamp events from affected evaluations if multiple operations can clamp one evaluation. Include non-finite and out-of-ball counts and their checked denominators. The strict `>0.1%` boundary is good, but cannot be audited from counts without these definitions.
7. **Separate falsifiers remain outside this artifact:** the candidate correctly acknowledges the flat-control and partition-gate artifacts. To make the *overall* falsifiers decisive without rerunning, those companion artifacts must be required, identity-bound to the same run/data/codebooks, and define their comparison and gate calculations. Schema 3 alone cannot establish those falsifiers or the downstream frozen metric thresholds.

The `flip_rate` threshold itself is arithmetically checkable from complete per-example winners, provided the denominator and coverage are fixed. But the current example's `expected_examples` field is not proof of coverage, and the full/ablated winner arrays alone do not prove those winners came from the declared scoring procedure.

## Required before approval

Make schema 3 the sole binding interface; replace placeholders with immutable identities; bind complete evaluation coverage to the frozen manifest; define tie and clamp rate formulas precisely; and include enough per-example full/ablated score data to independently recompute winners, margins, flips, and thresholds. Require same-run companion artifacts for the flat control and partition gate. Until those are in place, the artifact is a useful audit format but its falsifiers are not decisive from the file alone.
