# Iter31 S08 MVG — Operational Repair Round 5 Source Packet

```text
STAGE_ID=S08_MVG
ROUND=5
ROUND_TYPE=OPERATIONAL_REPAIR
ITERATION=31
LOCKED_SCIENCE_CHANGED=NO
SAME_ITERATION_REPAIR=AUTHORIZED
PARALLEL_EXECUTION=YES
SERIALIZATION_REASON=Only the single canonical Iter31 S08 MVG invocation is permitted after the independent checker and provenance instrumentation repairs are complete; do not duplicate it.
```

## Objective and scope

This is the current skill §2.5A fast operational-repair round, not a new scientific proposal, not a fresh A/B deliberation, and not a performance experiment. The Iter32 S00 Judge C canonical source snapshot (`stage2_RQ-VAE/curvature_RQ-VAE_iter32/logs/source_snapshot_iter32.md`) defers Iter32 and directs repair/rerun of Iter31's registered S08 before any further iteration registration. The defective earlier Iter31 S08 attempt remains invalid for gate purposes; do not rewrite any historical Iter31 record.

The repair must preserve the exact Iter31 registration: HRA-STEP6-1 Step6 equation and implementation semantics; inherited FCCR-1 fixed closed-form curvature contract and values; Iter29 parent/comparator and locked protocol; the pinned Iter8 warm-start checkpoint; seed 42 and the existing deterministic first-640-eligible-source selection policy; exact Stage1/Stage0 inputs and evaluation definition. Any detection that preserving these factors is impossible must stop the rerun and be recorded for the repair verifier; do not revise locked science within this round.

## Consolidated repair scope — all items before rerun

1. **Deliberation-gate false positive:** correct the negated-action matching defect that flags a prohibited/negated mention of `matched-seed replication` as a positive proposal. Preserve the intended rejection of genuinely positive forbidden research actions. Do not edit or rewrite historical deliberation records to make the gate pass.
2. **Contemporaneous S08 capture:** instrument the canonical Iter31 S08 verifier before its one authorized rerun so it records in the same invocation:
   - resolved paths and pre-run SHA-256 identities for the actually consumed Stage1 embedding, Stage1 item-ID sidecar, Stage0 train parquet, and pinned Iter8 checkpoint; record checkpoint path/hash explicitly;
   - seed and exact deterministic selection rule;
   - actual ordered source-target IDs selected by the verifier, not reconstructed later;
   - actual batch shape(s), row/count values and relevant tensor dimensions;
   - actual runtime logical device and physical device identity when available (record unavailable physical identity explicitly; source intent alone is not runtime observation);
   - complete existing mechanism diagnostics and checks, preserving the exact registered S08 behavior;
   - complete stdout, stderr, and process exit status for the invocation.
3. **Consolidation:** statically inspect the affected checker and S08 provenance-capture surface once, gather all independently detectable operational defects before changing files, and make one minimal repair batch. No historical artifact rewriting, post-hoc reconstruction as a substitute for contemporaneous fields, or changes to mechanism/protocol are permitted.

## Authorized and forbidden actions

Authorized after this packet's Judge authorization: prepare the minimal checker and S08 instrumentation repair; write a detailed `repair_record.md`; run the affected independent deterministic checks after repair; then perform exactly one canonical same-iteration Iter31 S08 MVG rerun with the corrected contemporaneous capture, and have a repair verifier review the record and primary outputs.

This packet authorizes **no Stage2, Stage3, training, SID export, downstream execution, performance comparison, seed variation, second S08 invocation, or GPU work other than the single authorized S08 MVG rerun**. It does not declare S08 passed. Do not register Iter32 or silently transition FCCR-1 here.

## Primary evidence

- Current skill `.claude/skills/curvature-rqvae-iter/SKILL.md`, §§2.2, 2.5A, especially pre-run MVG provenance completeness.
- Iter31 S08 initial output: `stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S08_MVG/round_2/parent_mvg_results.md`; missing actual ordered IDs and actual runtime device were documented there.
- Iter31 prior decisions/limitations: `round_2/judge.md`, `round_3/judge.md`, `round_4/judge.md`, `round_3/posthoc_reconstruction.md`, `round_3/posthoc_batch_provenance.json`, and `logs/iteration_abort_iter31.md` (read-only historical evidence).
- Iter31 locked science/protocol: `logs/hra_step6_contract_iter31.json`, `logs/mechanism_contract_iter31.json`, `logs/protocol_manifest_iter31.md`, and `logs/one_factor_diff_iter31.md`.
- Current updated checker: `.claude/skills/curvature-rqvae-iter/scripts/deliberation_gate.py`; the frozen Iter32 S00 packet records the failed S00 check on the negated forbidden-action phrase. This round authorizes repair without rewriting the records that caused the false positive.
- Canonical S00 repair-first decision: `stage2_RQ-VAE/curvature_RQ-VAE_iter32/logs/deliberation/S00_SOURCE_TRUTH/round_1/judge.md` and `logs/source_snapshot_iter32.md`.
