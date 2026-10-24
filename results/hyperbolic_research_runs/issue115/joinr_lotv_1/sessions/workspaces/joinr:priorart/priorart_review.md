# Prior-art review — issue #115 JOIN R round 1

## Decision

**Verdict: REVISE. Classification: ADJACENT (not established as occupied, and not cleared as novel).**

Do **not** make an unqualified novelty claim on the evidence currently in this workspace. The only recorded scan match is HiHPQ (DOI `10.1609/aaai.v38i5.28261`; arXiv `2401.07212`), and that record is explicitly **ABSTRACT_ONLY**. It cannot establish whether the complete paper uses the same operator, a direct equivalent, or something materially different.

## What is actually declared

The local `candidate.md` does **not** declare an unsigned 2-simplex area. Its one declared mechanism is LOTV: a **signed** Lorentzian 4-by-4 determinant of the item and three selected codewords, mapped to a bounded signed value and used to break ties in the existing candidate-assignment objective. It specifies a finite-precision contract and does not add a second score term or training loss.

Accordingly, this review does not treat an unsigned area as the candidate, and does not analyze a hypothetical second score term. The request's description of the operator as an unsigned 2-simplex area conflicts with the supplied complete declaration. If the unsigned-area description is intended, the artifact to review is missing; please reconcile the declaration before relying on this review. No unsigned-area mechanism is present in `candidate.md`.

## Evidence status

- **Verified from the supplied artifact:** `candidate.md` specifies the signed determinant/tie-break above; it identifies HiHPQ as an abstract-only comparison and explicitly reserves full-paper comparison. The frozen setup and strict conjunctive outcome thresholds are recorded in the candidate. The candidate also explicitly distinguishes itself from a parked unsigned tangent-space 2-area; that is a declared distinction, not independent proof of novelty.
- **Verified campaign match, as reported by the user/artifact:** HiHPQ, DOI `10.1609/aaai.v38i5.28261`, arXiv `2401.07212`, with evidence depth `ABSTRACT_ONLY`. I have no full-text evidence here to verify the method-level overlap question.
- **Recollection:** none relied upon. General recollection that hyperbolic quantization and geometry-aware ranking exist would not answer whether this particular determinant is used to choose recommender SID codes.
- **Unknown:** whether HiHPQ or other unscanned prior art uses an unsigned triangle-area term to choose a quantizer code in a recommender SID; whether it contains the LOTV determinant or a direct equivalent; and whether the proposed fixed-coordinate determinant tie-break is independently present elsewhere. The available abstract-only label cannot resolve these questions.

## Answer to the narrow question

**Is an unsigned triangle-area term used to choose a quantiser code in a recommender SID already occupied? Unknown on the evidence provided.** There is no verified match for that exact claim in the supplied scan record. That is not evidence of absence, and it is not a novelty clearance. Moreover, that unsigned-area question is not the operator declared in the local `candidate.md`.

For the actual LOTV declaration, the HiHPQ record makes the work **adjacent** at the broad level of hyperbolic/geometric quantization, but the abstract-only evidence neither verifies occupation nor supports a novelty finding. Do not convert this into “HiHPQ does not do it.”

## Required before any novelty claim

1. Obtain and inspect HiHPQ's full paper and supplementary material; compare the actual assignment rule, geometric quantity, inputs, and whether it changes code/SID selection against the declaration in `candidate.md`.
2. Search beyond the single recorded match for geometric or area-based quantization in recommender systems and discrete/SID recommenders, then record sources and exact passages supporting either overlap or distinction.
3. Resolve the unsigned-area versus signed-LOTV discrepancy in the reviewed artifact. Keep the review tied to one frozen declaration.

Until those checks are complete, the defensible wording is: **“Potentially distinct operator; novelty and non-overlap remain unverified.”** Do not claim that an unsigned area is novel, and do not claim the signed LOTV operator is novel based only on the abstract-only HiHPQ scan.
