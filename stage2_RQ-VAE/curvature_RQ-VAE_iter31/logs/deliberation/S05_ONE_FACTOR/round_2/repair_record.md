# Iter31 S05 One-Factor — Operational Repair Record Round 2

```text
STAGE_ID=S05_ONE_FACTOR
ROUND=2
ROUND_TYPE=OPERATIONAL_REPAIR
ITERATION=31
LOCKED_SCIENCE_CHANGED=NO
SAME_ITERATION_REPAIR=AUTHORIZED
PARALLEL_EXECUTION=YES
CHECKS_PASS=YES
CANONICAL_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/one_factor_diff_iter31.md
USER_INPUT_REQUIRED=NO
```

## Gate defect and repair boundary

The gate requires the exact worker marker `INDEPENDENCE_DECLARATION=I did not read the other candidate before completing this artifact.` Agent A's round-1 file contains that literal. Agent B's round-1 file instead says: `INDEPENDENCE_DECLARATION=I did not read the other candidate draft before completing this artifact.` The added word `draft` makes it semantically explicit but not an exact match. This record does not claim Agent B wrote the gate's exact literal and does not rewrite either candidate.

## Existing adjudication and canonical evidence

- **Agent A** (`round_1/agent_a.md`): independently identifies the conditional single-factor boundary as only the registered HRA-STEP6-1 replacement of Iter29 `RqVae._step6_sum_embeddings`; preserves the inherited FCCR-1 substrate, Step4/Step5, quantizer/loss and decoder factors; and records that no implementation or training is authorized at S05.
- **Agent B** (`round_1/agent_b.md`): independently identifies the same sole model change and distinguishes path/contract updates as wiring and the HRA checker as tooling; retains the Stage3 route, provenance, and execution constraints. Its declaration remains the quoted semantic variant above.
- **Existing Judge** (`round_1/judge.md`): records `VERDICT=MERGE_AB`, `HARD_GATE_A=PASS`, `HARD_GATE_B=PASS`, both `MERGE_COMPONENTS_A/B` fields, and `PROBLEMS_B=No hard-gate failure`. It merges the candidates' consistent one-factor boundary while explicitly keeping runtime activation and Stage3 routing unresolved.
- **Canonical S05 artifact**: `logs/one_factor_diff_iter31.md` exists and contains the exact HRA equation/order, inherited-factor boundary, output wiring rules, unresolved Stage3 route, and no-execution authorization. This repair leaves it unchanged.

## Stable identities of preserved evidence

SHA-256 values were computed from the live primary files during this repair; sizes are bytes.

| Artifact | Size | SHA-256 |
|---|---:|---|
| `logs/deliberation/S05_ONE_FACTOR/round_1/agent_a.md` | 20354 | `08445d0f1736b89af17f60bc029b2d5f19c5f5e2d23b24ab5cef4daf9f11e91d` |
| `logs/deliberation/S05_ONE_FACTOR/round_1/agent_b.md` | 18755 | `fc15856fd01ea0bc9b1cdecc2ad0c93c1a5c1c3abc6aaab995f4df71a5803bc4` |
| `logs/deliberation/S05_ONE_FACTOR/round_1/judge.md` | 7226 | `9ff228a21afe02365be40b607839a8973473ff0adad4bd98231a2814f22cdcd8` |
| `logs/one_factor_diff_iter31.md` | 8719 | `7771874ffcfa6c0a0ed4f98d01b3ec37de4100002e4975515f234097c989f9fb` |

## Outcome

The round-1 candidates and Judge are preserved. This operational repair records the literal-marker mismatch and maps the existing independent-review evidence and `MERGE_AB` result to the unchanged canonical one-factor artifact. No HRA equation, single-factor boundary, scientific decision, source, checker, or authorization changed. No A/B rerun, Stage2, Stage3, GPU, or training was performed or authorized.