# Issue 251 infrastructure

First action reused the existing shared `assets/` manifest. Original67 entries are frozen in `ACCEPTANCE.json`, each with immutable `path:sha256`. Hash verification passed; no binary asset was rebuilt, renamed or copied.

| Requirement | Evidence | Result |
|---|---|---|
| Existing graph/ego/features/targets |67 original manifest entries verified; `REGRESSION_249.log` covers reloading68 current entries |PASS |
| New static asset |`assets/v6_attention_tokens/token_schema.json:a721a7e29e8b8a5bced06235f8033c92174b29997b3405a571259cca5e7721e8`; declarative only, two full-ego tokens; registered in root manifest |PASS |
| Root manifest |`assets/manifest.json:23a0a6e2c28795c9b768d702e048c5bbd1fe31f749ad0ed3860709f2ef457463`; version v6-cross-attention-251,68 assets |PASS |
| Coverage |`assets/coverage.json:0360e0fd2f88b52c3b83911f086b09ae2502437a6451ebba79162675f6d5327e` |PASS |
| Full target |28,391,948 triples in13 immutable shards; whole connected-component references; same masks/centre IDs |PASS inventory; execution reported separately |
| Training interactions |`results/stage0_build_parquet/train.parquet:80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815` |PASS |
| Original semantic input |`stage1_GeneEmbedding/output/sentence_t5.npy:6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb` |PASS |
| Existing TIGER weights for controlled verification only |`results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth:af8efc29a8bcbbe5394cc5e358acf7b14c0e78ae99d8a8b9e5faf03fdddd97a8`; never retrained or used as an unbound historical geometry baseline |PASS |
| Frozen baseline |SHA-256 of sorted compact relative-source SHA map: `be740473dea6776accee5da8814c7ae9f1f9e244112aeab2444420d0c132a1ca`; read-only |PASS |
| Stage2 interpreter |`/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9`; completed unit and regression commands |PASS |
| GPUs |`nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader`:4 NVIDIA L40S,0/46068MiB before launch |PASS |
| Port |`ss -ltnp '( sport = :50202 )'`: no listener before launch |PASS |

The v5 `--verify` manifest rewrite was reproduced and repaired with a regression test. Verification now preserves exact manifest bytes and newer version identity. No dependency installation was needed. No infrastructure credentials were printed or altered.
