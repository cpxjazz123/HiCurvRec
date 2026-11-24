# Issue256 Infra reuse

PASS: all 109 assets were read and SHA256-checked against the existing registry; no new or rebuilt asset. Full path:sha256/use/coverage inventory: `INFRA.json`. Manifest: `cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b`. Legacy registered evaluation asset paths remain referenced in place, not copied into assets or this issue.

Four L40S cards were idle; ports50201/50202 free; both required interpreters exist. Full semantic/train/valid/test identities and frozen TIGER source inventory are in INFRA.json. No TIGER weights or recall are fabricated and no baseline is retrained.

## Old mechanism archive

Before new design/reference/code: `git mv stage2_RQ-VAE/TIGER_prefix_curvature_gate_rq stage2_RQ-VAE/_archive/TIGER_prefix_curvature_gate_rq` (exit0). No matching live trainer/verifier or GPU process depended on it. Top level now contains only baseline/ and _archive/. All55 Python/JSON source identities were recorded in PARENT_ARCHIVE.json. Immutable `_runs/issue255_formal_r1` and the invalid first run are unchanged. Retained analysis/diagnostic path constants and the verification runner were relocated to the archive; historical reports, manifests and receipts are not rewritten. The existing result-shadow regression passes1/1 in RELOCATION_TEST.log. This is source relocation, not a new formal training generation.

## Evaluation isolation

EVALUATION.md freezes training/validation-only decisions, actual decoding and vocabulary derivation, and transparent fixed-test reuse. TEST_ACCESS.json preserves known historical exposure. No new test item/case table is read for hypothesis selection.
