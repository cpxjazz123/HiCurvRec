# Iter31 S07 Preflight — Round 3 Matcher-Repair Verification

## Identity and previous adjudications

- Stage: `S07_PREFLIGHT`, round 3; Iter31 HRA-STEP6-1, parent Iter29.
- Round 1 Judge: `logs/deliberation/S07_PREFLIGHT/round_1/judge.md` authorized only an additive manifest phrase clarification and `_find_method` child capture.
- Round 2 Judge: `logs/deliberation/S07_PREFLIGHT/round_2/judge.md`, `VERDICT=REJECT_BOTH`; `logs/preflight_iter31.md` records the raw checker outcomes and diagnosis. Shared FCCR preflight passed, HRA preflight failed because `_pattern_dump` kept an `ast.Expr` wrapper for expression patterns while `_has_node` walks the underlying function and sees only inner expressions. Judge C authorized only a checker-only `_pattern_dump` repair: unwrap `ast.Expr.value` for parsed expression patterns, preserve non-expression statement AST matching and all checker policy.
- This round verifies that exact single checker repair and reruns both mandatory independent static checks. It is not S08 or execution authorization.

## Current exact repair

`scripts/preflight_hra_step6_iter31.py::_pattern_dump` now stores `parsed = ast.parse(source).body[0]`, then dumps `parsed.value` only when that node is `ast.Expr`; otherwise it dumps the original statement node. `_has_node`, its patterns/policy, both contracts, model, manifest, shared FCCR checker, Stage3 trainer, and implementation plan are unchanged in this repair.

Round-1 manifest wording and `_find_method` corrections remain present as separately authorized prior repairs; round-3 is not authorized to alter them or any other content. Inspect the exact current source/diff boundary and report any unauthorized change as a failure; candidates do not edit.

## Primary contracts/source to inspect

Read both prior S07 packets/Judges and current primary source:

- FCCR-1: `logs/mechanism_contract_iter31.json`, manifest, `curvature_RQ-VAE.py::_load_closed_form_curvatures`.
- HRA-STEP6-1: `logs/hra_step6_contract_iter31.json`, `scripts/preflight_hra_step6_iter31.py`, `modules/rqvae.py::RqVae._step6_sum_embeddings`, and current `forward` decoder consumer.
- Verify HRA remains the single Step6 scientific change; fixed FCCR remains separate; helper projection/clamp/domain are not claimed to be runtime-tested. Static PASS must not imply gradient, activation, direct effect, training, or performance evidence.
- Stage2 products remain under `results/stage2_RQ-VAE/curvature_RQ-VAE_iter31/`, absent from source Iter31 tree. Stage3 route correction remains outstanding: both prior candidates identified that trainer derives `RQVAE_VARIANT` through `_RQVAE_VARIANT_MAP.get(CODE_PATH, ...)` while the canonical S06 plan requires a direct literal. Round-2 Judge states it is a mandatory pre-S11 correction/route gate and authorizes no trainer edit here. Report it; do not edit/resolve it in S07 round 3.

## Parent-side checker commands

Candidates independently audit the source only and do not edit/run commands. Parent Main executes each exact command once for round 3, separately, and captures full output/exit status for Judge C. Cwd for both:

`/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31`

1. Unchanged shared FCCR checker:

`python /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py`

2. Iter31 HRA checker:

`python scripts/preflight_hra_step6_iter31.py`

Both must exit 0 with their own PASS markers for a possible S07 PASS. Any failure means no propagation. This is not an experiment and authorizes no S08, MVG, GPU, Stage2, or Stage3.

## Candidate requirements

Agent A and B independently inspect source packet, Round-2 Judge, exact current change, canonical S00-S06, contracts, source-boundary and paths; verify that only authorized matcher change occurred; assess semantic matcher behavior for expression vs statement AST nodes; note Stage3 map/literal deviation and pre-S11 gate; report no execution (parent-only checker outputs), no edits/training. Their review cannot substitute for actual commands.

Judge C reads both reports plus Main raw command output and primary source. It writes `logs/deliberation/S07_PREFLIGHT/round_3/judge.md` and appends a Round-3 outcome to `logs/preflight_iter31.md`. Only Judge may declare S07 PASS, only if both separate commands exit 0 and no unauthorized repair is present. Even S07 PASS authorizes only progression to separately adjudicated S08; later stages remain blocked.