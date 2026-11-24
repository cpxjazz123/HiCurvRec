# Issue252 infrastructure

Original requirements and input hashes: ACCEPTANCE.json. Assets were inspected before construction; existing correct binaries were reused, not regenerated. The first68 manifest leaves remain unchanged.

| Requirement | Command / evidence | Observed result |
|---|---|---|
| Raw training / split / visible graph isolation | `python3.9 -m tools.issue252.graph_audit`; `assets/v7_graph_contract/audit.json:0918afe35cddf209441fff4fe80a558a2d562e14027326d124ec2f51000acb1b` | PASS:323633 canonical /261132 visible supports;62501 fully classified; residual0 |
| Named graph registration | `verify_named_graph('infra246-visible-v2')`; REGISTERED_GRAPH_V9.json | PASS, identity `fe87a3da4a5afcb7616773950b7b57ad4df5db1adcb357e59e7122c89c544212`; hashes original leaves plus legal training-only v8/v9 ancestry and current bytes |
| Existing features / all targets | assets/manifest.json; assets/coverage.json; v8 audit | Reused24,587 six-feature rows and all28,391,948 targets, no new sampled target asset |
| Missing lossless target and full-ego runtime layout | `tools.issue252.runtime_layout`; `assets/v8_runtime_layout/metadata.json:116068365f7acb74ebb0f4aa0dcfa9249a2fbe968b0721371c8a2d7505caff60` | PASS: all target bits/order, all two-hop node populations/root positions, all directed rows. Metadata recovery did not rebuild binary arrays |
| Missing full induced sparse runtime operators | `python3.9 -m tools.issue252.ego_operators`; `assets/v9_ego_operators/metadata.json:5f8f2c0aca5203bb80b0d4779797220e5748dda3f42b41fdd07011730e7fc8db` | PASS: all24,587 complete egos at both scales;1,716,410 /513,330,917 directed edges. Integer topology only, no learned outputs |
| Training interpreter / GPU layout | Actual baseline and optimized launcher runs, four rank JSON reports | PASS: prescribed genrec_env_v2 Python3.9, Torch2.8/CUDA12.8, four L40S, global4096/port50202 |
| Incorrect controller interpreter | PROFILE_LAUNCH_FAILURE.md / CONTROLLER_INTERPRETER.json | Diagnosed: system Python has no Torch; launch refused before any GPU step. Correct interpreter succeeded; no guard bypass |
| Frozen baseline / TIGER comparison | Existing SID `025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059` | Reuse only; no new TIGER training. Unbound geometry/reconstruction/Recall remain NA |
| Runtime integrity | RED_REVIEW_CONTRACT_ASSETS.log → REVIEW_CONTRACT_ASSETS_TESTS.log | PASS:14 parseable byte mutations fail before any np.load; frozen producer and binaries unchanged |
| Stage3 interpreter / R2 | Existing project interpreter and rclone1.67.0; credentials file exists, contents never read/logged | Actual Stage3 and remote archive verification pending; existence is not an archive PASS |

New data live only in assets/v7_graph_contract, assets/v8_runtime_layout and assets/v9_ego_operators. Their manifest entries carry path/SHA256/bytes/input hashes/generator sources/use/coverage. Original assets/coverage.json is itself hash-frozen; new complete-layout coverage lives in each version's metadata/audit, with reading instructions in assets/README.md. Binary files and source archives remain local generated artifacts, not git payloads.

Formal Stage2 still requires final-source whole-suite, module/DDP, inclusive performance and actual public admission evidence. This document does not authorize training.
