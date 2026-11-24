# R02 infrastructure — immutable reuse, no generated static data

- Source origin: exact15-file baseline/TIGER_RQ-VAE copy, verified against R01 LINEAGE baseline map before ARM_NAME-only change; BASELINE_COPY.json retains every source hash. No weights/results/cache copied. R01 archive prerequisite completed before this copy/design (LIVE_ARCHIVE.json).
- Assets: reuse the same109/109 audited registered inputs in root INFRA.json; fresh manifest hash cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b matched. No manifest entry, binary or coverage population replaced. Detailed per-asset immutable path:sha256/use/coverage: root INFRA.json and assets/manifest.json.
- Semantic input: stage1_GeneEmbedding/output/sentence_t5.npy:6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb.
- Train: results/stage0_build_parquet/train.parquet:80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815.
- Validation: results/stage0_build_parquet/valid.parquet:48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9. Fixed57439 recommendation rows and five immutable structure-validation banks; no structure-test bank or test-error item list selected.
- Test protocol identity only: test.parquet:5290abf0073d798401b56bd0a9ec5f710ed2c01ae9fa4cb8bd3906c1a55f0bfc. No R02 test inference or item-error access before legal final training.
- Directed train-visible CSR: assets/train/visible_forward.npz:ac7155d1424a6a346da9b8ed6e9413b81d43223a377308af0c8ba35b6b773029; all full1/2-hop/direction/curvature assets reused by registered hashes.
- Interpreters remain Stage2/tests genrec_env_v2/bin/python3.9, Stage3 genrec_env/bin/python3.10. No package/environment changes or installs.
- Four-card hardware/port admission: the only GPU execution command is the standard launcher/50202. Actual per-rank worker/PID/device/population receipts, not a guessed availability statement, will accompany each red/green/admission execution. Formal dispatch remains prohibited until current-source checks pass.

No new Infra file is required under assets/. Per-round diagnostic outputs are evidence, not replacement static assets. Historical R01 SID/checkpoint/recall/R2 hashes are retained in its immutable ROUND_RESULT.json rather than copied or renamed.
