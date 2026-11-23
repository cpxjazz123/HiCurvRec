# Issue 249 verdict

Infrastructure implemented. No training is required or allowed: issue goal and prohibition section explicitly exclude model training.

C1–C9 mapping and scientific limitations: REPORT.md. Final verification evidence: VERIFY_FINAL.log; final frozen-ego smoke: EGO_FINAL.log. CODE_HASHES.json binds the executed production modules and tests to SHA-256; assets/manifest.json binds every reused and newly persisted asset to SHA-256.

| Criterion | Evidence | Verdict |
|---|---|---|
| C1 immutable asset audit | root manifest and verify_reused | PASS |
| C2 complete deterministic ego batches | frozen one-hop slices; on-demand two-hop; masks, repeated-centre and chunk-offset regressions | PASS |
| C3 full target traversal | 28,391,948 rows; 13 shards; exact counts and summary agreement | PASS |
| C4 numerical contract/reference | CONTRACT.md, reference.py, test_v5_diff_sectional | PASS |
| C5 differentiation/diagnostics | finite differences, weighted and chunked tests; grad_diag | PASS |
| C6 hard-SID evaluation/isolation | FIDELITY.json; heldout_eval tests; distinct continuous and discrete reports | PASS |
| C7 TIGER identity | pure SID metrics recomputed; historical geometry unavailable and explicitly NA | PASS |
| C8 static persistence/API | 67 registered assets; assets/README.md; verify_v5.sh | PASS |
| C9 regressions/mutations/reload | VERIFY_FINAL.log, MODULE_CHECKS.json, MUTATIONS.json | PASS |

Final fixes were test-first: missing item-macro/triple-micro MAE; masks sized by edge-block count instead of edge count; chunk-global batch/centre offsets; one-hop traversal rebuilding a frozen ego. The one-hop interface now reads the existing manifest-referenced numeric arrays without object deserialization. No existing static binary is rebuilt for these fixes.

## Conclusion recheck

Read-only fidelity runner independently recomputes graph hop statistics against frozen targets: 2000 triples, maximum difference 0.0. Hard-SID sign agreement 0.7005; continuous z 0.6995. The positive-majority baseline is 0.888, and negative-target agreement is only 0.3080. These diagnostics do not establish mixed-curvature superiority, generalization, or graphlet-specific optimal geometry. Historical Stage3 SID cannot be decoded with a checkpoint from another generation; its geometry stays NA. No new recall is claimed.

## Closure checklist

- Production interfaces implemented: yes.
- Required tests executed: yes; final command and exit status recorded in the verification log.
- Required training complete: yes, vacuous because training is explicitly forbidden.
- New static assets persisted in assets/ with immutable reuse: yes.
- Remaining implementation work for this issue: no, subject to final verification success and delivery below.

Final commit SHA and successful main push are recorded in the GitHub closure comment, avoiding a self-referential commit hash in this file.
