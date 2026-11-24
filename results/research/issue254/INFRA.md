# Issue254 infrastructure reuse

Read-only diagnostic; no new static training assets or GPU jobs. Existing source/checkpoints/graphs are immutable.

| Frozen input | SHA256 | Registered use / verification |
|---|---|---|
| `assets/index.npz` | `a5eb2935b258024500d3558c0256238aa342bd1ade52dd0141208c1730e656e1` | training; manifest match |
| `assets/node_moments.npz` | `21a9679881d07af5f2a36577b07a0ac426fc0788a2b75c9f6d3045ef74eef6fb` | training; manifest match |
| `assets/item_summary.npz` | `1a8f46e428e66df7ca8cd5672c1bf23af57d60e23459f5d13cfaffa52a82ed12` | training; manifest match |
| `assets/train/visible_forward.npz` | `ac7155d1424a6a346da9b8ed6e9413b81d43223a377308af0c8ba35b6b773029` | training; manifest match |
| `assets/train/visible_reverse.npz` | `5fb26e2b1b5b056674b350c556df458493d56b0f450367dd4d9b7329a39896b6` | training; manifest match |
| `assets/train/visible_undirected.npz` | `097a719a9f00292477a1434c776de1ff37fb60adb7a1adc9c517a98bc7147818` | training; manifest match |
| `assets/shards/shard_000.npz` | `1870d81151e38db93de347c1b12a5b7742782d89069b4fb6241dc2cf9f5eedb2` | training; manifest match |
| `assets/v8_runtime_layout/targets.npy` | `8c0ff9f8c25b98ed31802cdb62cf921df6d3cf8bbf35ee2d3de9817d6cd0807a` | training; manifest match |
| `assets/v8_runtime_layout/target_offsets.npy` | `33f30b2979311d74cf996545bec011856ffa7acbc9cb9b995c829dbc9f87937c` | training; manifest match |
| `assets/v8_runtime_layout/two_hop_nodes.npy` | `78a3fa1f0d255b429016bc9cbba8b9b64e86ed490c398b896f9548d09a3a9dda` | training; manifest match |
| `assets/v8_runtime_layout/two_hop_offsets.npy` | `f5e233414984f26d32a4df5c5b73310b30973826612251f0fd65a4c1f4339cf7` | training; manifest match |
| `assets/v8_runtime_layout/visible_forward_bits.npy` | `17a6e7dd9ea4d264c1e68dcaa12211861b31672a02065ef9ef8155a12cb6d737` | training; manifest match |
| `results/research/issue247/assets/topology_features.npz` | `1cbbab0b7d24f06c2a8cabc5e33f1fd613c6dd7468908efdd0139b6bd5e271b3` | training; manifest match |
| `results/research/infra246/assets/motifs/motif_counts.npz` | `79e1adee8461751c5b4cdfa78a78ab0fa22ac23d8ad296c4b881d1167dfca7c9` | training; manifest match |
| `results/research/infra246/assets/motifs/motif_instances_visible.npz` | `68855edef787ae292108f3a415381be6427f1f575acaf9126c9a933214fa5349` | training; manifest match |
| `results/research/infra246/assets/motifs/motif_instances_4node_visible.npz` | `dc96eede3cc0fbe51b6117c597365ee4c3e44d4307881023c15945c6d77bc027` | training; manifest match |
| `results/stage0_build_parquet/test.parquet` | `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc` | evaluation input (not a training asset); direct frozen input SHA |
| `stage1_GeneEmbedding/output/sentence_t5.npy` | `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` | training; manifest match |

Original input split SHA checked against #253 RECONCILIATION; both generations have exact original target/SID and shard-rank checks for every row. Original dumps omit explicit user/history IDs: recovery uses source-bound split order and canonical target checks, not a fabricated historical input-tensor digest.

Stage3 trainer SHA and all immutable candidate source files match. R2 generation identities match ledger100/101, with historical download-verification receipts reused; no re-upload or credential access. Missing initial Stage3 checkpoint and directory flags were evidence-script bugs, now corrected to `test_final.best_checkpoint` and proper directory metadata.

Complete triple/graph/role coverage and transitive CPU source hashes are recorded by STRUCTURE_SUMMARY/MECHANISM_SUMMARY after actual execution. Raw diagnostic tables live only in issue254/tables and are excluded from Git/TrainingView. No active mechanism is archived or moved for this analysis.
