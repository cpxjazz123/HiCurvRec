# Issue253 infra evidence

## Mandatory archival gate

No training/validation processes were present at the safety check. Archived all 24 previous first-level mechanism directories; retained baseline/ and _archive/ unchanged. 18 tracked trees used git mv, six untracked trees used mv. No archive destination was overwritten.

```text
time=2026-11-24T06:56:27Z
git mv -- stage2_RQ-VAE/TIGER_BEHAVIOR_SAMPLE_RQ-VAE_diag stage2_RQ-VAE/_archive/TIGER_BEHAVIOR_SAMPLE_RQ-VAE_diag
git mv -- stage2_RQ-VAE/TIGER_af3_graphgeo_rq stage2_RQ-VAE/_archive/TIGER_af3_graphgeo_rq
git mv -- stage2_RQ-VAE/TIGER_baseline_recal_rq stage2_RQ-VAE/_archive/TIGER_baseline_recal_rq
git mv -- stage2_RQ-VAE/TIGER_cos_contrastive_rq stage2_RQ-VAE/_archive/TIGER_cos_contrastive_rq
git mv -- stage2_RQ-VAE/TIGER_curve_contrastive_rq stage2_RQ-VAE/_archive/TIGER_curve_contrastive_rq
git mv -- stage2_RQ-VAE/TIGER_curvfid_rq stage2_RQ-VAE/_archive/TIGER_curvfid_rq
git mv -- stage2_RQ-VAE/TIGER_curvloss_rq stage2_RQ-VAE/_archive/TIGER_curvloss_rq
git mv -- stage2_RQ-VAE/TIGER_degree_temp_rq stage2_RQ-VAE/_archive/TIGER_degree_temp_rq
git mv -- stage2_RQ-VAE/TIGER_export_matched_graphlet_rq stage2_RQ-VAE/_archive/TIGER_export_matched_graphlet_rq
git mv -- stage2_RQ-VAE/TIGER_g237_sem_rq stage2_RQ-VAE/_archive/TIGER_g237_sem_rq
git mv -- stage2_RQ-VAE/TIGER_g237_struct_rq stage2_RQ-VAE/_archive/TIGER_g237_struct_rq
git mv -- stage2_RQ-VAE/TIGER_gnn_attention_sectional_rq stage2_RQ-VAE/_archive/TIGER_gnn_attention_sectional_rq
git mv -- stage2_RQ-VAE/TIGER_gnn_gate_sectional_rq stage2_RQ-VAE/_archive/TIGER_gnn_gate_sectional_rq
mv -- stage2_RQ-VAE/TIGER_i234_A1 stage2_RQ-VAE/_archive/TIGER_i234_A1
mv -- stage2_RQ-VAE/TIGER_i234_A2 stage2_RQ-VAE/_archive/TIGER_i234_A2
mv -- stage2_RQ-VAE/TIGER_i234_A3 stage2_RQ-VAE/_archive/TIGER_i234_A3
mv -- stage2_RQ-VAE/TIGER_i234_A4 stage2_RQ-VAE/_archive/TIGER_i234_A4
mv -- stage2_RQ-VAE/TIGER_i234_A5 stage2_RQ-VAE/_archive/TIGER_i234_A5
mv -- stage2_RQ-VAE/TIGER_i234_A6 stage2_RQ-VAE/_archive/TIGER_i234_A6
git mv -- stage2_RQ-VAE/TIGER_kappa_cost_rq stage2_RQ-VAE/_archive/TIGER_kappa_cost_rq
git mv -- stage2_RQ-VAE/TIGER_kappa_sk_only_rq stage2_RQ-VAE/_archive/TIGER_kappa_sk_only_rq
git mv -- stage2_RQ-VAE/TIGER_sid_graphlet_rq stage2_RQ-VAE/_archive/TIGER_sid_graphlet_rq
git mv -- stage2_RQ-VAE/TIGER_sid_graphlet_shuffled_rq stage2_RQ-VAE/_archive/TIGER_sid_graphlet_shuffled_rq
git mv -- stage2_RQ-VAE/TIGER_topo_distill_rq stage2_RQ-VAE/_archive/TIGER_topo_distill_rq

Remaining Stage2 roots:
_archive
baseline
```

## Asset reuse

Read assets/manifest.json first. Actual SHA-256 and byte checks passed for all 91 manifest/data records (88 registered assets plus train/valid/test). No existing data were regenerated/copied and no new asset has been created. Full immutable leaf list and purposes: `asset_audit.json`.

| Path | Actual SHA-256 | Use | Result |
|---|---|---|---|
| `assets/coverage.json` | `0360e0fd2f88b52c3b83911f086b09ae2502437a6451ebba79162675f6d5327e` | training | PASS |
| `assets/distance_histogram.npz` | `0d17ed2bc5846677d8774df7b9074129909b8403297ac9b200a4aed9d131c2f9` | training | PASS |
| `assets/v5_gnn_geometry/ego_batching_index.npz` | `15535a22356eea0b21343f3c9c51413a434d3ccf39062793bb16f1cea65292ca` | training | PASS |
| `results/research/infra246/assets/ego/ego_full_1hop_visible.npz` | `2082cce4d49ea4f9363082cdc537c1730aa66c203d223fd259e8da314a74edc5` | training | PASS |
| `results/research/infra246/assets/ego/ego_sampled_visible.npz` | `1386f987ae90be1310371352aeb6ad5d58a739690a4586065b4f175c05b47b1a` | training | PASS |
| `assets/v5_gnn_geometry/evaluation_report.json` | `4ec297d4093ac0d75d116a8f75a272aaa04cce97039b834b2a66e88f8c5d2432` | description | PASS |
| `assets/v5_gnn_geometry/fidelity_report.json` | `ea31bfc5f36f2144b9af2fa1d0f8bd41db208f08c71869b45e8d92eac93f7f5c` | description | PASS |
| `results/research/infra246/assets/graph/held_out_support.npz` | `fea02c56eaee7df9fae22628c6da363ebf8a89fe3c76c762f0b4e0fc5799792e` | evaluation | PASS |
| `assets/index.npz` | `a5eb2935b258024500d3558c0256238aa342bd1ade52dd0141208c1730e656e1` | training | PASS |
| `assets/train/item_ids.npy` | `262f7ce84b271f6232e7c59893580988ae0049606d09263a5f86e3c37e14804e` | training | PASS |
| `assets/item_summary.npz` | `1a8f46e428e66df7ca8cd5672c1bf23af57d60e23459f5d13cfaffa52a82ed12` | training | PASS |
| `results/stage0_build_parquet/items.parquet` | `6408e3c0f98a65d817747477a22092c0848f7ed105dc9bf7b7327e9544a99af3` | training | PASS |
| `results/research/infra246/assets/motifs/motif_counts.npz` | `79e1adee8461751c5b4cdfa78a78ab0fa22ac23d8ad296c4b881d1167dfca7c9` | training | PASS |
| `results/research/infra246/assets/motifs/motif_instances_visible.npz` | `68855edef787ae292108f3a415381be6427f1f575acaf9126c9a933214fa5349` | training | PASS |
| `results/research/infra246/assets/motifs/motif_instances_4node_visible.npz` | `dc96eede3cc0fbe51b6117c597365ee4c3e44d4307881023c15945c6d77bc027` | training | PASS |
| `assets/v5_gnn_geometry/node_features.npz` | `580f04a9c6a8cc7142de17a865efdf287bf94e9e4a30b225504f12a75b74b4cd` | training | PASS |
| `assets/v5_gnn_geometry/node_features_report.json` | `840a52ccfce64a1c449c324dfdd8258f62375eb4be6d62000c50882fa44a10cc` | training | PASS |
| `assets/node_moments.npz` | `21a9679881d07af5f2a36577b07a0ac426fc0788a2b75c9f6d3045ef74eef6fb` | training | PASS |
| `results/research/infra246/assets/pairs/probe.npz` | `f6bde7e27622af24d2a2e462a023d60fe0e2d9d0f3ce683cd7e6264115417286` | evaluation | PASS |
| `results/research/infra246/assets/pairs/relaxed_struct_test_degree_matched.npz` | `47f34ce9f08d1c40aeef79f952ca6bb29c3c57665f4ebe59c87553e1a7a2fb2e` | evaluation | PASS |
| `results/research/infra246/assets/pairs/relaxed_struct_test_direct.npz` | `a216700f021d23ef8a007dd5afc526b031e1e179e784779847f1d50afdaeaf0a` | evaluation | PASS |
| `results/research/infra246/assets/pairs/relaxed_struct_test_shared_in.npz` | `1e1d63becb8d4542825351029cf57115edf4188612cfb1ae6b3dc5df19d55a68` | evaluation | PASS |
| `results/research/infra246/assets/pairs/relaxed_struct_test_shared_out.npz` | `3014c44743e51ace596878c75dc3ec66dba11fc0a38354e4567eef63028bd262` | evaluation | PASS |
| `results/research/infra246/assets/pairs/relaxed_struct_test_two_hop.npz` | `f3be301a575ccc7a295aac2a6e5824014fdcefbde1763ea831bde0af308baa08` | evaluation | PASS |
| `results/research/infra246/assets/pairs/relaxed_struct_val_degree_matched.npz` | `51ffe7cad12f64d545913413bbe560259bee0b09ab1946e91bdeb0c5b46a1a49` | evaluation | PASS |
| `results/research/infra246/assets/pairs/relaxed_struct_val_direct.npz` | `64ed4a57f3ef11cdfe6da94e63395bc281b4c51ca36448704f086ef480499d26` | evaluation | PASS |
| `results/research/infra246/assets/pairs/relaxed_struct_val_shared_in.npz` | `52934f9ef5e7dca28909df08828b90f1cf09e489cc48324fc8c21904bfa68fd7` | evaluation | PASS |
| `results/research/infra246/assets/pairs/relaxed_struct_val_shared_out.npz` | `1a14bb00d1440358dff2f2072ed2c191b64c76671639b260e4d194b70f98b272` | evaluation | PASS |
| `results/research/infra246/assets/pairs/relaxed_struct_val_two_hop.npz` | `4736a3ecdc1a85c7326cf9cd117be91f14a254567bf70c0e367a575c541e5e42` | evaluation | PASS |
| `results/research/issue247/assets/sectional_summary.npz` | `af41ddac19f8e7edf0a3f8323b7803dd875f8d423b339e3c7fb49b6c35cc461d` | training | PASS |
| `stage1_GeneEmbedding/output/sentence_t5.npy` | `6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` | training | PASS |
| `assets/shards/shard_000.npz` | `1870d81151e38db93de347c1b12a5b7742782d89069b4fb6241dc2cf9f5eedb2` | training | PASS |
| `assets/shards/shard_001.npz` | `6b0866880439b2fc09312fb9dbafa0e9da46ae7e1997a727eef657210248439d` | training | PASS |
| `assets/shards/shard_002.npz` | `45b08c50843a05c908b70dcc177411f0e25601a9bdfacc29caa0465f7e963b07` | training | PASS |
| `assets/shards/shard_003.npz` | `fc307613f8e62a95ea8b36cebd77e62becd855b8a2bc1b5a3a07668e7b6d9845` | training | PASS |
| `assets/shards/shard_004.npz` | `768f8989ff67c61b9177df0e342aa478b98eabfe86f7968fe4f9b436e213fde7` | training | PASS |
| `assets/shards/shard_005.npz` | `26edcbb5a478cbf1b8e5355a33781f323568713b0fc154c15103bcb371bf95cf` | training | PASS |
| `assets/shards/shard_006.npz` | `c5d51677292549c4049adeadd855be96ec0b7c01127efa01d3ed98a140e5968a` | training | PASS |
| `assets/shards/shard_007.npz` | `ff76eb5b3e7789375102403f951a3cd9d84f1d0f96026048c591491a06d4a916` | training | PASS |
| `assets/shards/shard_008.npz` | `b2859a2946ffe16bf0d6b0bea8a27dfbdfddc43e787fc5e917dfc6db5d6839ae` | training | PASS |
| `assets/shards/shard_009.npz` | `e646af02cbc1b291c7ef2cce291616701ba32b1d86771c2a78a3ec4b0ff9c1ab` | training | PASS |
| `assets/shards/shard_010.npz` | `5de0ed90a40afbacfb6d19bc980ddc03824b4b6e5f80193bef4157275a8b442f` | training | PASS |
| `assets/shards/shard_011.npz` | `8ef23c61c63663c37cd06ac097f21aeff4c56b895fd9eb1219bb1fe94e8c3d4f` | training | PASS |
| `assets/shards/shard_012.npz` | `c519e4fdd3e15ee87dc8a1600874de90f502340d6bf3fdd5245abb1dbd082444` | training | PASS |
| `results/research/infra246/assets/graph/split_users.npz` | `acfd8bdb89052646e751bc78a0050bb71d8242a88916f01214e0e4fd05d8fb68` | evaluation | PASS |
| `results/research/infra246/assets/pairs/strict_struct_test_degree_matched.npz` | `7742e34d49f18c287b33cfbc67dc92ba876a69724d730d90492ec3b87f6ae8ca` | evaluation | PASS |
| `results/research/infra246/assets/pairs/strict_struct_test_direct.npz` | `80fe725722ff618cb352761f9280d3d89e2b80cef117c9ae65f8196c4fcecbbd` | evaluation | PASS |
| `results/research/infra246/assets/pairs/strict_struct_test_shared_in.npz` | `ccb5e503da7eb255d9d73b391669657a0990923219658ac04ea1474defd625ab` | evaluation | PASS |
| `results/research/infra246/assets/pairs/strict_struct_test_shared_out.npz` | `6ea0bda9b4feedd9c7403ee9336505bfb59d168c085fe3e50aa77504b7620b1e` | evaluation | PASS |
| `results/research/infra246/assets/pairs/strict_struct_test_two_hop.npz` | `fedd4ffe3b5e541d36e8f590f59fbfb3ffeb53e1bd23b6ff24c650808fd84d4c` | evaluation | PASS |
| `results/research/infra246/assets/pairs/strict_struct_val_degree_matched.npz` | `542719e36e441fc71624413c25dc0335c2ca4eea38e3cdb349c4c678ddb4991c` | evaluation | PASS |
| `results/research/infra246/assets/pairs/strict_struct_val_direct.npz` | `bc65baff19071f1497f295261ccf8d92c7450d30d0cac41d86e6cd1e97f0d530` | evaluation | PASS |
| `results/research/infra246/assets/pairs/strict_struct_val_shared_in.npz` | `66b2d85abf63a5674b2e8fd32fd18ea29072204bdc6793d167cecbaf6e8e8a87` | evaluation | PASS |
| `results/research/infra246/assets/pairs/strict_struct_val_shared_out.npz` | `78b5fe7114b9d9fd3465c4b4656b9cad195f10c2560ad25485b3f11f72e6af96` | evaluation | PASS |
| `results/research/infra246/assets/pairs/strict_struct_val_two_hop.npz` | `d282477c65d03572ae8605d3e7a9303ea78cae9b12e47a16874bab05c640e64c` | evaluation | PASS |
| `results/research/issue247/assets/tiger.json` | `45b122b1af3b2fbb6efb969c2614181784141c0445a6e7b2345c94f4c74026cc` | description | PASS |
| `results/research/issue247/assets/topology_features.npz` | `1cbbab0b7d24f06c2a8cabc5e33f1fd613c6dd7468908efdd0139b6bd5e271b3` | training | PASS |
| `results/research/infra246/assets/pairs/train_degree_matched.npz` | `c2c85c1052e65aa61964ab0dec6ce84772862489f1cced28025a139c215d69dd` | training | PASS |
| `results/research/infra246/assets/pairs/train_direct.npz` | `22d0cf789730e0f26397fed14a7cdc3576ed0544c7bf05cd56e2e6a6c97a997c` | training | PASS |
| `results/research/infra246/assets/pairs/train_shared_in.npz` | `a6510127dca2ba3dd013f9b25709f43d98e8ecaa8897c60d9aacedb93902bc91` | training | PASS |
| `results/research/infra246/assets/pairs/train_shared_out.npz` | `9d8253cc8a045e3c4341b0d9a12ca5640d4557df2d13c9877fe18b40f34294e4` | training | PASS |
| `results/research/infra246/assets/pairs/train_two_hop.npz` | `3edc9796ae2db4db1691468dc624f77b796e47cee91775421388af27ffdc4f23` | training | PASS |
| `assets/train/visible_forward.npz` | `ac7155d1424a6a346da9b8ed6e9413b81d43223a377308af0c8ba35b6b773029` | training | PASS |
| `results/research/infra246/assets/graph/visible_forward.npz` | `ac7155d1424a6a346da9b8ed6e9413b81d43223a377308af0c8ba35b6b773029` | training | PASS |
| `assets/train/visible_reverse.npz` | `5fb26e2b1b5b056674b350c556df458493d56b0f450367dd4d9b7329a39896b6` | training | PASS |
| `results/research/infra246/assets/graph/visible_reverse.npz` | `5fb26e2b1b5b056674b350c556df458493d56b0f450367dd4d9b7329a39896b6` | training | PASS |
| `assets/train/visible_undirected.npz` | `097a719a9f00292477a1434c776de1ff37fb60adb7a1adc9c517a98bc7147818` | training | PASS |
| `assets/v6_attention_tokens/token_schema.json` | `a721a7e29e8b8a5bced06235f8033c92174b29997b3405a571259cca5e7721e8` | description | PASS |
| `assets/v7_graph_contract/audit.json` | `0918afe35cddf209441fff4fe80a558a2d562e14027326d124ec2f51000acb1b` | description | PASS |
| `assets/v7_graph_contract/contract.json` | `d61a4ca54f6e7ea29331652aceae1b0d9495e215c44fb8415ae507ed52e38687` | description | PASS |
| `assets/v8_runtime_layout/visible_forward_bits.npy` | `17a6e7dd9ea4d264c1e68dcaa12211861b31672a02065ef9ef8155a12cb6d737` | training | PASS |
| `assets/v8_runtime_layout/two_hop_nodes.npy` | `78a3fa1f0d255b429016bc9cbba8b9b64e86ed490c398b896f9548d09a3a9dda` | training | PASS |
| `assets/v8_runtime_layout/two_hop_offsets.npy` | `f5e233414984f26d32a4df5c5b73310b30973826612251f0fd65a4c1f4339cf7` | training | PASS |
| `assets/v8_runtime_layout/two_hop_root_local.npy` | `7e9dca2a869509116427dc25008677f1d2d97c6c041aa2acb70133980e856edd` | training | PASS |
| `assets/v8_runtime_layout/targets.npy` | `8c0ff9f8c25b98ed31802cdb62cf921df6d3cf8bbf35ee2d3de9817d6cd0807a` | training | PASS |
| `assets/v8_runtime_layout/target_offsets.npy` | `33f30b2979311d74cf996545bec011856ffa7acbc9cb9b995c829dbc9f87937c` | training | PASS |
| `assets/v8_runtime_layout/metadata.json` | `116068365f7acb74ebb0f4aa0dcfa9249a2fbe968b0721371c8a2d7505caff60` | description | PASS |
| `assets/v8_runtime_layout/audit.json` | `74188af783ae98f97b38dabedffc5e82bc181dc61cd1a5ff155dc0df47afeb91` | description | PASS |
| `assets/v9_ego_operators/ego1_in_col.npy` | `12359f51f960314a11387ad47c9ed99673c564526a440435934333bd5f90b694` | training | PASS |
| `assets/v9_ego_operators/ego1_in_ptr.npy` | `33c42518b8eea2f81fbcdf7b35e635cbbaf6ce4c67b78d59a2153d4e169cb8db` | training | PASS |
| `assets/v9_ego_operators/ego1_out_col.npy` | `303dd021660510d5b95e45bd593e547d1341624a9b17c3826ace081e1a5bc55f` | training | PASS |
| `assets/v9_ego_operators/ego1_out_ptr.npy` | `6bc6654825c174eb80e2ade6f082f1f56a3ddc286e1f9874b5bad66e3bfdb785` | training | PASS |
| `assets/v9_ego_operators/ego2_in_col.npy` | `cf61e25fa575695471837e05bcf8379f61e9a7da53038434582bd476d1731c17` | training | PASS |
| `assets/v9_ego_operators/ego2_in_ptr.npy` | `f5b6379cb315b2e0591a11b4f3697350521704d953fdeb89c1426e313b244f9f` | training | PASS |
| `assets/v9_ego_operators/ego2_out_col.npy` | `69c4ca48fb3ae5ecb8f0e90200e1a1d343fb5d0b9d6556cb3b2005c64568c867` | training | PASS |
| `assets/v9_ego_operators/ego2_out_ptr.npy` | `1fba4377ce2dc4330009b422c5c6566446540ab2aa21b7b840f93cd61ff5837f` | training | PASS |
| `assets/v9_ego_operators/metadata.json` | `27d5b7af803529d5b6ec735ae3bcae1d75a0f4c8cb0ff7fe4fb1c0240fa45242` | description | PASS |
| `assets/v9_ego_operators/audit.json` | `5866c35f50aa9a256483d787abc3062c68b3dc7d1c9fdd53e3712e768d7627dd` | description | PASS |
| `results/stage0_build_parquet/train.parquet` | `80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815` | input | PASS |
| `results/stage0_build_parquet/valid.parquet` | `48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9` | input | PASS |
| `results/stage0_build_parquet/test.parquet` | `5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc` | input | PASS |

## Hardware and runtime

- Four NVIDIA L40S, 46068 MiB each, 0 MiB in use / 0% at intake. Ports 50201 and 50202 had no listener.
- Stage2 interpreter: /home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9; Stage3: /home/wlia0047/ar57_scratch/wenyu/genrec_env/bin/python3.10; MCP: system python3.
- Candidate `TIGER_three_expert_attention_rq` copied only from frozen `baseline/TIGER_RQ-VAE`; all 15 source/config files byte-identical at copy. Copy-only old logs/caches removed. `source_provenance.json` binds baseline, copy, archived #252 and actually executed #252 source. The historical source difference is only execution_entry.py, the framework path adapter.
- Named graph: infra246-visible-v2; full A.1 targets 28,391,948; v9 complete induced ego CSR reused. Recommendation validation/test and structural heldout remain evaluation-only.
- LSP status: no language servers configured; symbol-aware text fallback is necessary.
- Path-reference migration for archived historical tools remains pending, not claimed PASS.
