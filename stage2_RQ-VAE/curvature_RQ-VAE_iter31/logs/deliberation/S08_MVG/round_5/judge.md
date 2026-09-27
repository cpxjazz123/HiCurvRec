# Iter31 S08 MVG — Round 5 Operational Repair Verification

```text
STAGE_ID=S08_MVG
ROUND=5
ROUND_TYPE=OPERATIONAL_REPAIR
VERDICT=REPAIR_PASS
LOCKED_SCIENCE_CHANGED=NO
CANONICAL_ARTIFACT=logs/mvg_check_iter31.log
USER_INPUT_REQUIRED=NO
CANONICAL_DECISION=REPAIR_PASS; required provenance and authorized S08 diagnostics were captured in the single same-iteration rerun, with locked science unchanged.
```

## Verification decision

The authorized operational repair passed. Primary evidence confirms one and only one same-iteration Iter31 S08 rerun, `MVG PASS`, and exit status 0, with the previously missing provenance captured contemporaneously. The defective earlier S08 verification attempt remains invalid for gate purposes. No Stage2 or Stage3 action has occurred or is authorized by this S08 repair verdict.

## Evidence verified

- `logs/mvg_check_iter31.log` records the exact authorized no-argument `scripts/mvg_check.py` command, `EXIT_STATUS=0`, `MVG PASS`, complete standard output, empty standard error, and closure markers. The separate `logs/mvg_check_iter31.stdout.log` contains the contemporaneous provenance and full diagnostics; `logs/mvg_check_iter31.stderr.log` is empty. `repair_record.md` records the separate streams and canonical log, their byte counts, and `INVOCATION_COUNT=1`.
- The contemporaneous provenance reports SHA-256 values and resolved consumed paths for all four required inputs: Stage1 embedding (`6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb`), Stage1 item-ID sidecar (`3df1f3ce6a468ae541c929a149f4dbd5e806f545b89d786b9c2d6801563b7c30`), Stage0 train parquet (`80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815`), and pinned Iter8 warm-start checkpoint (`189e0affb156b9cdbd11f38a6fdb1127804e3f3095edd23c55f84825beb56c3b`). The repaired capture code hashes pre-run and fails closed if input identity changes during hashing.
- Provenance records PyTorch and NumPy seed 42; first 640 eligible source rows in ascending order; one target selected per source by `np.random.choice`; and exactly 640 ordered source-target pairs printed in source order. Source/future embedding shapes are `[640,768]`; source/future ID shapes are `[640]`.
- Runtime device is contemporaneously observed as logical `cuda:0`, physical NVIDIA L40S, UUID `853ef1bf-bbd2-2d5f-3dfa-a787f9d0e299`. This is runtime evidence, not merely the source-declared device.
- The exact registered FCCR-1 curvature values `[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]` match the closed-form contract and remain invariant across recorded train/eval probes at steps 0, 25,000, 50,000, and 100,000. Fixed curvature is excluded from the optimizer and no optimizer step was performed.
- Direct Step6 comparison uses the same quantized embeddings; HRA and Euclidean outputs are finite, shape `[640,32]`, and differ (`max_abs_difference=0.07198049873113632`, `L2=2.018216848373413`, `relative_l2_difference=0.0810627937`). Inner and common-curvature balls are domain-valid; common-ball maximum scaled norm is `0.855758547782898`; no projection/clamp/floor incidence is reported in the recorded diagnostics.
- Total loss is finite (`5.195184230804443`) with nonzero finite total-model gradients. Reconstruction, quantizer, and behavior-loss components each have nonzero gradients (reported counts 8, 7, and 4); fixed-curvature invariance holds after gradient checks.
- The updated shared gate now scopes forbidden-action negation by clause, handles contrast boundaries and fronted `without`, and retains positive-proposal detection. `repair_record.md` documents the helper cases and all pre-run deterministic checks as passing. Historical Iter31 records were not rewritten.

## Scope and limits

This `REPAIR_PASS` certifies only the repaired Iter31 S08 operational verification. The registered HRA-STEP6-1 equation, FCCR-1 contract and curvature values, Iter29 parent/protocol, pinned checkpoint, seed, deterministic selection rule, data identity, and evaluation definition remain unchanged. The result does not establish Stage2/Stage3 performance, does not promote HRA, does not authorize Iter32, and does not constitute Stage2 execution.

## Autonomous next action

```text
AUTONOMOUS_NEXT_ACTION=Proceed to S09_STAGE2_EXECUTION deliberation and adjudication only. Do not launch Stage2 unless S09 receives its canonical Judge authorization and the separately required root CLAUDE.md §6 one-checkpoint/one-batch nonzero-gradient-path verification passes; do not launch Stage3.
```
