STAGE_ID=S03_PROVENANCE
ROUND=2
PROTOCOL_ID=FCCR-1_iter30_iter26_vs_iter29_matched_seed43-45
ITERATION=30
PARENT_STAGE=S07_PREFLIGHT
PARENT_VERDICT=ACCEPT_A (static audit; subsequent actual preflight failed)
PARENT_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/deliberation/S07_PREFLIGHT/round_2/judge.md; stage2_RQ-VAE/curvature_RQ-VAE_iter30/logs/preflight_contract_iter30.log
REOPEN_REASON=The one authorized S07 preflight failed its literal manifest-term check for `behavior branching`; the canonical S03 manifest uses the key `behavior_branching` and the semantic phrase `effective branching factor`. S07 Judge directs following this direct evidence through the appropriate canonical gate, without claiming PASS or bypassing preflight.

## Objective
Independently assess the minimal canonical manifest correction, if any, that satisfies the required preflight's semantic terminology check while preserving the already-adjudicated S03 historical method/value provenance exactly. Produce an exact patch plan and risk analysis; do not edit files. This is a wording/contract-artifact compatibility repair only, not a new provenance claim or mechanism change.

## Canonical S03 decision and evidence
- Canonical S03 round 1 Judge: `logs/deliberation/S03_PROVENANCE/round_1/judge.md`, verdict `MERGE_AB`, `HARD_GATE_A=PASS`, `HARD_GATE_B=PASS`, canonical decision `PASS, limited to historical method/value provenance`.
- Canonical S03 manifest: `logs/mechanism_manifest_iter30.md`, `STATUS=PASS_LIMITED_TO_HISTORICAL_METHOD_VALUE_PROVENANCE`; it distinguishes `behavior_branching`, explicit `raw_residual_medians`, and `normalized_layer_scales`, and explicitly records raw-residual confidence as MEDIUM with replay/identity limitations.
- S03 round 1 does not establish current recalculation, historic checkpoint/SID replay, historic byte identity, or source-byte identity. Preserve all these limits. Do not relabel S03 as failed/unresolved and do not attribute raw-residual provenance to iter12.
- Registered FCCR-1 inputs remain `branching=[19.324911558712664,1.4605688962651735,1.0148104414712726]` and `raw_residual_medians=[1.0,0.10941,0.09331]`, in `[L0,L1,L2]` order. No value, semantic definition, producer attribution, formula, transformation, or confidence may change.

## Direct failure evidence
- The single authorized run is recorded at `logs/preflight_contract_iter30.log`: exit status 1, `MECHANISM_CONTRACT_FAIL: mechanism manifest missing semantic/provenance term: 'behavior branching'`. Do not rerun preflight during this deliberation.
- Required skill source `/home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py:76-80` lowercases the manifest and requires the literal substrings `raw residual`, `behavior branching`, and `normalized layer scale`.
- The current canonical manifest has literal key `behavior_branching` and semantic phrase `Effective branching factor` in its branching row; underscore is not a space, so the checker fails its exact literal test. The other required phrases occur in the raw-residual and distinction sections.
- Do not modify the skill preflight implementation or weaken its required check. Do not treat the failed run as a PASS.

## Hard constraints
- Preserve S03 round-1 `MERGE_AB` provenance conclusion and all exact provenance facts/values/limitations. No new producer, source path, checkpoint identity, raw-SID identity, recomputation, replay, confidence upgrade/downgrade, or mapping change.
- Propose only a minimal wording change to the canonical S03 manifest if direct evidence supports it; the change must clearly explain that `behavior branching` denotes the existing `behavior_branching` input/effective branching-factor semantic, without altering any numeric or historical claim.
- Do not edit the S03 artifact, preflight script, production code, other canonical artifacts, or any result path. Do not run preflight again, MVG, gradients, torchrun, trainer/model/checkpoint, Stage2/Stage3, GPU, or create outputs.
- If a safe wording-only correction cannot preserve canonical meaning, state that and recommend the valid autonomous path; no user input.

## Required independent candidate artifact
Each candidate independently reports:
1. the exact failure cause, citing preflight and manifest lines;
2. whether the canonical S03 PASS remains valid and why;
3. a minimal complete patch plan with exact file/section and exact candidate sentence, or a justified rejection of in-place repair;
4. a bounded non-execution verification plan (static review only now; no preflight rerun during deliberation);
5. risks, self-rejection criteria, and concrete autonomous next action.

## Independence and destination
Agent A and Agent B receive this identical packet and do not read or message each other. Candidate paths:
- `logs/deliberation/S03_PROVENANCE/round_2/agent_a.md`
- `logs/deliberation/S03_PROVENANCE/round_2/agent_b.md`
Judge C must adjudicate before any canonical manifest edit or additional preflight execution.
