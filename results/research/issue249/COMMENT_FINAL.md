# Issue 249 final verification

C1–C9 implementation/evidence mapping: REPORT.md. Production-code hash binding: CODE_HASHES.json.

Two final regression gaps were reproduced before fixes:
- Macro error: unequal centre counts and opposing errors must yield item-macro MAE 5 and triple-micro MAE 4. The initial implementation lacked these metrics (KeyError); now independently aggregated.
- Ego batch coordinates: edge_mask had 3 entries for 180 edges. Fixed masks to count actual edges; chunked batch_index, edge_batch and root_global now include preceding chunk offsets. Regression compares full batches with chunk=1 and recovers repeated centres exactly.

- One-hop ego now slices the existing manifest-referenced frozen asset instead of traversing and rebuilding it. A regression makes graph traversal fail and checks that frozen loading still recovers the centre; 19 final ego/held-out tests pass.
Final deterministic command: `bash tools/graph_infra/verify_v5.sh`; evidence `VERIFY_FINAL.log`. Original suite: 76 tests, 24 module checks; three added regressions bring the suite to 79 tests. Mutation evidence: MUTATIONS.json, nine designated assertions.
Real-data fidelity diagnostic: 2000 deterministic sampled centres; frozen target recomputation max difference 0; hard-SID sign agreement 0.7005, continuous z 0.6995. Positive-majority baseline is 0.888. These are diagnostic numbers, not evidence for mixed-curvature superiority. Historical Stage3 table geometry remains NA; its pure SID metrics are recomputed.

No training, PPR, new graphlet census, or changes to semantic inputs or SID protocol.
