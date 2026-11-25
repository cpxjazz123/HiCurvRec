# Issue #258 Infra faults (FROZEN — recorded, not repaired)

## FAULT-1 — recommendation validation has no identity-parameterised entry

**Need**: validation Recall@10 on the complete `valid.parquet` population for the R05 Stage3 checkpoint, one of
the preregistered predictions (issue §3: “validation R@10≥0.0583”).

**Observed**: the only implementation of this measurement is
`tools/issue256/r04/external_validation.py`. Its first import is

```
from tools.issue256.r04.identity import ROOT, ARM, RUN_ID, SOURCE, EVIDENCE
```

with `RUN_ID = 'issue256_r04_formal'`, and `inputs()` raises unless
`final['stage2_run_id'] == RUN_ID`. The generic `tools/issue256/external_validation.py` is bound to
`issue256/identity.py` (`issue256_r01_formal`) in the same way.

**Search performed** (commands and outputs recorded in `CHECK_1.md`):

```
grep -rln "valid.parquet" tools/ | grep -v pycache
# -> tools/issue256/{external_validation,r0{1,2,3,4}/external_validation}.py
#    tools/issue252/graph_audit.py, tools/genrec-mcp/stage2_quality_gate.py (not recommendation recall)
grep -rln "recommendation_validation|validation_recall|recall@10" tools/iteration tools/research tools/issue25*
# -> no generic runner
```

The frozen Stage3 `test_final.json` carries only test metrics (`NO_EVAL=True` in the frozen protocol), and
`training_metrics.jsonl` has no validation recall record.

**Impact**: C7's validation prediction cannot be measured by an existing public interface. Everything else in
C5–C9 remains measurable.

**Not done, because FROZEN forbids it**: copying the R04 runner into the candidate or the evidence directory,
monkey-patching its identity, or adding a new parameterised validation entry to `tools/`.

**Minimal relaxations for the owner**:

- **A** — authorize `BUILD` for one change only: parameterise `tools/issue256/r04/external_validation.py` (or add
  `tools/iteration/validation.py`) by `(run_id, arm, SID, Stage3 run dir)` without changing its math; then rerun
  the validation measurement and the affected checks.
- **B** — accept the round without validation Recall@10: report the SID signature, collision and final test, mark
  the recommendation prediction `NOT_AVAILABLE`, and close as COMPLETED with that limitation explicit.
- **C** — stop before the formal run and keep the preregistered R05 untouched for a later round with a validation
  interface.

**Status at registration**: documented, not blocking yet; Stage2/Stage3 and the SID-side predictions are
executable. Final classification follows the skill §6.1 after the run.
