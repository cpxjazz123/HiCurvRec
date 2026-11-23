# Issue 251 execution log

- Issue: 251 — GNN + curvature-expert cross-attention candidate, superseding issue250's single-vector gate.
- Body SHA-256: `eb05f12aee33f42cfff92568e925e00d4810fce6d78db807a31082611528af2d`.
- Acceptance: C1–C9 copied verbatim in `ACCEPTANCE.json`; live body reread during check round1.
- Contract frozen before mechanism integration: `CONTRACT.md:4c4958596a2b40a2bf177e29780d5eeda199e5d2e3495aeafafd4c265b204892`; paper/official comparison in `PAPER_COMPARISON.md`.
- **No formal training.** Issue goal says “仅实现、验证与交付可训练候选，不启动正式 Stage2/Stage3，不重训 TIGER”; C7 permits short validation only. Four-rank microstep and one-epoch admission smoke are verification, not3000-epoch dispatch. No Stage3 or R2 training-result archival applies.

## Candidate boundary

`stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq` was freshly copied from frozen `baseline/TIGER_RQ-VAE`. The original model class is retained; the public model export now points to a subclass. Semantic 768/latent32, three256-entry books, config and original losses are preserved. Standalone geometry, same-cost surrogate, full-reference curvature and provenance helpers were reused as components, not as a replacement baseline. No issue250 training was repeated. Existing issue250/user trees remain untouched.

## Verification sequence

1. Audited existing67 asset entries by hash, actual frozen assignment code and issue249 interfaces. Reused all binary assets. Added only the declarative v6 token schema under `assets/`.
2. Independent hand-computed token/attention references and geometry identities precede integration. Red logs retain missing-production, NaN-padding, invalid-target-ID and manifest-mutation failures.
3. `UNIT_FINAL.log`:60 tests PASS; `REGRESSION_249.log`:79 tests and24 module checks PASS. Nine deliberate mutations caught by specified numerical/structural/routing assertions.
4. Canonical reference extraction retained the same assertions: `REFERENCE_FINAL.log`,13 tests PASS. No test was weakened to accept a production error.
5. Controlled four-rank proof on source358db180 completed with all28,391,948 targets and exact checkpoint/SID reload. It is archived under `history_358db180/` as NOT_ACCEPTED for final model sourcecf0fd50c. Final-source runtime evidence is still being executed; no earlier measurement is substituted.

## Commands

```sh
PYTHONPATH=stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq/tests /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m unittest test_reference test_known_answers -v
/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m tools.issue251.launch_validation
```

No formal `stage2_start` or `stage3_start` is authorized by this issue.

## Source delivery and shared-worktree guard

Model snapshot: `cf0fd50c717983c13e3088384c3bc86b7375a452`.
MCP cache repair: `860c9ffcdf94dcc0a83d589dc134ff7e09c9c70e`.
Runtime/entry/full-data final measurements remain unfinished; this is not a completion claim. Selective commits exclude issue250 and unrelated user/history work.

The shared worktree contained792 changed paths/249,039 changed lines, dominated
by pre-existing research/history artifacts (including a147,524-line SID table).
`commit_guard --max-paths 750 --max-lines 300000` refused at772 paths.
The deliberate `--max-paths 850 --max-lines 300000` override then passed;
no file exceeded2MiB. Only issue251 paths and the scoped shared fixes were
selected through the commit namespace; unrelated changes were not cleaned,
staged or committed.

Three cancelled verification launches are explicitly labelled `CANCELLED_NOT_ACCEPTED`.
They ended for independent-review defects, an actual random rendezvous port33455,
and avoidable induced-graph collation cost. The controller now uses static127.0.0.1:50202.
No cancelled launch counts as passing evidence or a formal experiment.

The MCP Stage2 child was reloaded after the launcher repair; a real read-only
`stage2_status` call returned `idle` and four visible cards. Baseline logs copied
with the candidate were removed to avoid displaying historical3000-epoch rows
as candidate progress. They were never counted as this candidate's training.

## Final-source entry and MCP result retrieval

The first real frozen4096-batch entry exposed retained GNN activations exceeding
44GiB. Inner deterministic full-ego checkpointing reduced the actual32-item
training-loss saved tensors114,910,562→1,980,234, preserving exact forward/loss
and all38 parameter gradients within declared tolerances. No batch, ego or
reference target was truncated. Final-source candidate/evaluator checks75 PASS,
activation-memory regression1 PASS, launcher safety6 PASS, original infrastructure
79 tests and24 module checks PASS; see their individual execution logs.

Native MCP verification has a300-second transport deadline while its sanctioned
entry check permits3600seconds. PID2456764 continued after transport timeout.
The old machine route reran entry on every request and its cache publisher
imported deleted research code. Seven isolated cache regressions now pass on
system Python. The new route atomically persists the source/config/runtime-bound
entry result and always rechecks current behavior-graph integrity; NOT_RUN is
never cached. Reload and public-tool retrieval proof must follow the old child's
completion, not interrupt it or invent a successful receipt.

The independently audited frozen #248/#249 graphs are strict subgraphs of both
canonical directed/undirected graphs. `GRAPH_AUDIT.json` records62501 directed
edges and120318 undirected arcs missing from frozen assets, with zero extra
shared edges. Same training-input hash does not imply same graph contract.
No asset rebuild, graph whitelist or gate bypass is authorized.
