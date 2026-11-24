## C9: complete corrected-source tests PASS, including real calibration regression

Command `python3 results/research/issue255/run_tests.py` exited0. The runner used the required Stage2 Python for both suites and required source inventories to be unchanged before/after:

- Candidate:104 passed,135 warnings,398.08s.
- Tools/assets/paired evaluation:61 passed,12.91s.
- No failed or skipped suite; complete receipt wall417.81s.

`TEST_RESULTS.json` SHA-256:`eca5d9bf5b60651dcc1de9763dc5f6c96177b9f2b7e3ee2b9cd32e451e398a19`.
`MUTATIONS.json` SHA-256:`4c414a0ba536c64328e79e5576d4b8b6b9b3ae17fc6bb2244d64e663456acd98`;
`LAYER_MUTATIONS.json` SHA-256:`9b96ede3451eb81913dbda5bc31a54b19b19468f391618db3a585b53c1fb8056`.

This includes the actual-data full-original-gradient λ regression, four-token geometry/routing/mutation checks, and strict historical/pairing vocabulary/provenance cases. Actual four-rank validation also passed (COMMENT_11). Dedicated profile,60-epoch inclusive benchmark and current-source machine admission remain in progress. No corrected formal training has been reserved or launched yet.
