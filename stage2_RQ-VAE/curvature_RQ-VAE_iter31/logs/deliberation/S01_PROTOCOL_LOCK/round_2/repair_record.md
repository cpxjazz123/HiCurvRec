# Iter31 S01 Protocol Lock — Operational Repair Record Round 2

```text
STAGE_ID=S01_PROTOCOL_LOCK
ROUND=2
ROUND_TYPE=OPERATIONAL_REPAIR
ITERATION=31
LOCKED_SCIENCE_CHANGED=NO
SAME_ITERATION_REPAIR=AUTHORIZED
PARALLEL_EXECUTION=YES
CHECKS_PASS=YES
CANONICAL_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/protocol_manifest_iter31.md
USER_INPUT_REQUIRED=NO
```

## Gate defect and repair boundary

The current `deliberation_gate.py::require_judge` requires literal `MERGE_COMPONENTS_A=` and `MERGE_COMPONENTS_B=` fields when the Judge verdict is `MERGE_AB`. The frozen S01 round-1 Judge contains `VERDICT=MERGE_AB`, both hard gates, accepted evidence from A and B, and the canonical decision, but omits those two literal fields. This is a serialization omission only: do not infer that the historical Judge contained the missing fields. The original round-1 source packet, Agent A, Agent B, Judge, and canonical protocol manifest were not edited.

## Evidence mapping to the existing decision

- **Agent A contribution** (`round_1/agent_a.md`): recommends Iter29 as the sole direct comparator and cites its exact `test_final.json`, `test_recall@10=0.05921064085377531`, and `n_eval=57439`; distinguishes the reported current Iter31 parent root from Iter29 historical commits; applies the current strict `test_R@10 > 0.065` target; and records the no-training boundary.
- **Agent B contribution** (`round_1/agent_b.md`): independently reaches the same Iter29 comparator and exact result; separates Iter29's historical settings and execution facts from Iter31 effective settings; carries forward required input identity rechecks, output-root rules, and the unresolved Stage3 route constraint.
- **Existing Judge** (`round_1/judge.md`): is `MERGE_AB`, has `HARD_GATE_A=PASS` and `HARD_GATE_B=PASS`, explicitly describes both candidates' accepted evidence, and selects Iter29 as the sole comparator. Its canonical decision also preserves the strict target, root/parent identity caveats, historical-only interpretation of port/batch facts, and later Stage3 route gate. The missing literal fields do not erase this existing adjudication.
- **Canonical S01 artifact**: `logs/protocol_manifest_iter31.md` exists and records `STATUS=CANONICAL_S01_LOCK`, `PROTOCOL_ID=ITER31_HRA_SINGLE_SEED42_INSTRUMENTS_2026-09REF_VS_ITER29`, Iter29 as parent/comparator, the exact comparator metric/value/cardinality, strict target, required identity checks, short Iter31 result roots, and `NO_TRAINING_AUTHORIZATION=TRUE`.

## Stable identities of preserved evidence

SHA-256 values were computed from the live primary files during this repair; sizes are bytes.

| Artifact | Size | SHA-256 |
|---|---:|---|
| `logs/deliberation/S01_PROTOCOL_LOCK/round_1/agent_a.md` | 16545 | `97c4927bd5363d6a50299d097c27fe374561b7f604d669a49c8793200badfb8b` |
| `logs/deliberation/S01_PROTOCOL_LOCK/round_1/agent_b.md` | 19289 | `6aba73933858ccac350e6c2e2a2ebb54a89712b9fd642dbd7b92476ed5e38c42` |
| `logs/deliberation/S01_PROTOCOL_LOCK/round_1/judge.md` | 9574 | `13743a493d4f19a442774a71c0a2cc0ab375021acc5cb52ff84cc5bc5558144f` |
| `logs/protocol_manifest_iter31.md` | 10053 | `212698d114b2be0d70e45bcd0e22973a0644535257f32d276930bf5881aecc19` |

## Outcome

This record repairs only the gate's missing merge-component serialization by mapping the already accepted A/B evidence to the unchanged round-1 `MERGE_AB` decision and canonical manifest. No parent, comparator, protocol ID, metric, caveat, science, or authorization changed. No historical deliberation artifact, source, checker, or model was modified; no A/B rerun, baseline rerun, Stage2, Stage3, GPU, or training was performed or authorized.