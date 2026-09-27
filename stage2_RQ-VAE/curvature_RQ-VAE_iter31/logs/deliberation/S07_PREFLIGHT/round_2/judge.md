# Iter31 S07 Preflight — Judge C Adjudication

```text
STAGE_ID=S07_PREFLIGHT
ROUND=2
VERDICT=REJECT_BOTH

HARD_GATE_A=FAIL
HARD_GATE_B=FAIL

EVIDENCE_FOR_A=Agent A independently confirms that both Round-1-authorized repairs themselves are within their narrow scope: the manifest received only the additive semantic wording and _find_method now collects the matching child method while retaining its exact-one guard. It correctly identifies the separate Stage3 RQVAE_VARIANT deviation from the S06 direct-literal plan and does not claim checker execution. Parent raw results establish FCCR exit 0 and HRA exit 1; direct Judge AST reproduction confirms every one of the five guard matcher checks is false although all five guard expressions exist in _step6_sum_embeddings. Primary evidence: round_2/parent_checker_results.md; scripts/preflight_hra_step6_iter31.py:41-50,314-320; modules/rqvae.py:322-350; Agent A report.
EVIDENCE_FOR_B=Agent B independently confirms both Round-1-authorized repairs are within scope, makes the same Stage3 map-versus-literal finding, and does not claim checker execution. Parent raw results and direct Judge AST reproduction establish the same HRA false-negative cause and FCCR pass/HRA failure. Primary evidence: round_2/parent_checker_results.md; scripts/preflight_hra_step6_iter31.py:41-50,314-320; modules/rqvae.py:322-350; Agent B report.
PROBLEMS_A=The mandatory HRA checker exited 1, so the candidate cannot establish S07 PASS. The candidate’s static inspection does not replace the required checker outcome. Its Stage3 map-derived variant is inconsistent with the exact direct-literal implementation required by S06, notwithstanding the current map's matching value.
PROBLEMS_B=The mandatory HRA checker exited 1, so the candidate cannot establish S07 PASS. The candidate’s static inspection does not replace the required checker outcome. Its Stage3 map-derived variant is inconsistent with the exact direct-literal implementation required by S06, notwithstanding the current map's matching value.

WHY_NOT_A=The scoped repair review is supported, but the HRA preflight failure is a hard S07 gate; no candidate may pass on source inspection alone.
WHY_NOT_B=The scoped repair review is supported, but the HRA preflight failure is a hard S07 gate; no candidate may pass on source inspection alone.

CANONICAL_DECISION=REJECT_BOTH. The unchanged shared FCCR-1 command passed with exit status 0 and MECHANISM_CONTRACT_PASS. The distinct HRA command failed with exit status 1 and `HRA_STEP6_PREFLIGHT FAIL: Step6 shape/finiteness guards are missing`; therefore S07 fails and cannot propagate. Both candidates agree the two Round-1 repairs themselves conform to their authorized scope, and both correctly flag the Stage3 RQVAE_VARIANT discrepancy. The HRA failure is a checker false negative, independently verified as follows: `_pattern_dump` returns `ast.dump(ast.parse(source).body[0], ...)`, so each of the five expression patterns has an `ast.Expr` wrapper; the method AST contains the inner Compare/UnaryOp nodes instead. Reproducing `_has_node`'s exact AST equality produced `[False, False, False, False, False]`; comparing each pattern's `ast.Expr.value` against the method AST produced `[True, True, True, True, True]`. The source visibly contains rank/dimension checks at modules/rqvae.py:322-327 and output-shape/finiteness checks at :347-350. Thus this specific error is in matcher handling, not evidence that the model guards are absent. This static diagnosis does not establish runtime validity, activation, finite runtime output, helper-domain behavior, gradients, or performance. Authorize only one narrow edit to `scripts/preflight_hra_step6_iter31.py::_pattern_dump`: after parsing, unwrap `ast.Expr.value` before dumping expression patterns, while preserving normal dumping/matching for parsed non-expression statement nodes and leaving all checker policy, other checker code, contracts, model source and Stage3 trainer untouched. After that single edit, require a fresh independent S07 round 3 A/B source review and rerun both separate checkers; both must exit 0 before S07 can pass. The Stage3 direct-literal discrepancy is a distinct pre-S11 source-correction/route gate, not the cause of this S07 checker failure: the current generic `_RQVAE_VARIANT_MAP.get(CODE_PATH, ...)` yields the intended value via an exact path entry but violates implementation_plan_iter31.md:97-100, which requires `RQVAE_VARIANT="iter31_hra_step6_common_reference"` directly and says not to rely on the map. No Stage3 correction is authorized in this decision; it must be resolved under a later explicit authorization and verified at S11 before any Stage3 route can pass. Neither route nor runtime is verified.
CANONICAL_ARTIFACT=stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/deliberation/S07_PREFLIGHT/round_2/judge.md; stage2_RQ-VAE/curvature_RQ-VAE_iter31/logs/preflight_iter31.md
CONFIDENCE=HIGH
USER_INPUT_REQUIRED=NO
ITERATION_PURPOSE=PERFORMANCE_SEEKING_MECHANISM
SWEEP_OR_REPLICATION_ITERATION=NO
ROOT_CAUSE_ITERATION=NO
AUTONOMOUS_NEXT_ACTION=Apply only the authorized `_pattern_dump` expression-wrapper unwrapping, preserving non-expression AST matching and every checker policy; then conduct fresh independent S07 round 3 review and execute both required checkers separately. Keep S08/MVG, Stage2, Stage3, GPU, and training unauthorized unless and until their own subsequent canonical gates pass. Track the direct-literal RQVAE_VARIANT source correction as a mandatory pre-S11 route gate; no such correction is authorized here.

REPLAN_CONSTRAINTS=Do not alter `_has_node`, guard patterns/policy, contracts, model, shared FCCR checker, Stage3 trainer, or plan in this repair. Do not bypass either checker. The only authorized code delta is `_pattern_dump` unwrapping `ast.Expr.value` for expression patterns while preserving non-expression statement matching. Round 3 must review that exact edit independently and obtain exit 0 from both checkers. The S06 direct-variant requirement remains binding for S11; no route is verified. Static S07 evidence is not runtime/MVG/activation/performance evidence.
```

## Parent checker evidence

Working directory for both commands: `/home/wlia0047/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31`.

### Shared FCCR-1 preflight

Command:

```text
python /home/wlia0047/ar57/wenyu/GeneRec/.claude/skills/curvature-rqvae-iter/scripts/preflight_contract.py
```

Exit status: `0`. Raw stdout:

```text
MECHANISM_CONTRACT_PASS
iter=31
contract=FCCR-1
fixed_curvature=[1.3660953164241916, 0.7347829661951981, 0.6439958072706683]
curvature_trainable=false
curvature_time_varying=false
uses_cyclic_schedule=false
uses_curvature_regularization=false
fixed_buffer_sites=['/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31/modules/quantize.py:69']
```

### Iter31 HRA source/contract preflight

Command:

```text
python scripts/preflight_hra_step6_iter31.py
```

Exit status: `1`. Raw stderr:

```text
Traceback (most recent call last):
  File "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31/scripts/preflight_hra_step6_iter31.py", line 360, in <module>
    main()
  File "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31/scripts/preflight_hra_step6_iter31.py", line 348, in main
    _check_source()
  File "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31/scripts/preflight_hra_step6_iter31.py", line 314, in _check_source
    _require(
  File "/fs04/ar57/wenyu/GeneRec/stage2_RQ-VAE/curvature_RQ-VAE_iter31/scripts/preflight_hra_step6_iter31.py", line 30, in _require
    raise RuntimeError(f"HRA_STEP6_PREFLIGHT FAIL: {message}")
RuntimeError: HRA_STEP6_PREFLIGHT FAIL: Step6 shape/finiteness guards are missing
```

No checker was run by either candidate. The above outputs are the actual parent-side results from `round_2/parent_checker_results.md`; the independent AST reproduction in this adjudication was read-only and did not rerun either checker.
