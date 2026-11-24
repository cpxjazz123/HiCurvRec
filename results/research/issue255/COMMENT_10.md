## Historical #253 generation remains verifiable after the v10 registry append

The first strict historical binding check rejected the current `assets/manifest.json`: #253 pins older registry bytes. Changing the canonical verifier or restoring the shared registry would invalidate provenance or discard v10.

`tools/issue255/historical.py` instead reads exact configuration bytes from fixed revision `6d96efe092547277173906b152f2d5b3e9172ced`; the old manifest must hash to `8c58d34521181925df2e66dbe01f64b7150fb56a1eab028f59e8b4832c381ae6`. Every old runtime input leaf, source, checkpoint, SID and export still must match its original hash. No assets were rebuilt, copied or overwritten.

Actual read-only production-path probe exited0: `paired_comparison.baseline_identity`, `validate_completed`, and `historical.load_bound_checkpoint` accepted run `stage2_787483d6ef9842d1a7c4593a61ae925c`, final checkpoint epoch3000, and the completed Stage3 population57,439. The bound `test_final.json` SHA-256 is `92cb7822e4ddaf11b8a31b3ecaeeb6e6ae6ad8300a4100a22b8125bdb1d586a4`.

Historical regression cases require exact archived config bytes and reject changed runtime leaves or source drift. A fixture initially omitted required export equality fields (3FAIL/58PASS); it was corrected to satisfy the unchanged production contract. The complete current-source suites are running. This compatibility check does not claim that #255 final evaluation or Stage3 is complete.
