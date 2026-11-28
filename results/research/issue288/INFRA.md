# Issue #288 — INFRA (FROZEN, checkpoint 1 = round start 2026-11-28)

## Assets reused (read-only, by hash)
- assets/manifest.json: cfbd9d9c0d3c5282f9fdd0c0b28798cbf7a86479c2626b702e499b589acc5b6b
- assets/coverage.json: 0360e0fd2f88b52c3b83911f086b09ae2502437a6451ebba79162675f6d5327e
- assets/index.npz: a5eb2935b258024500d3558c0256238aa342bd1ade52dd0141208c1730e656e1
- assets/item_summary.npz: 1a8f46e428e66df7ca8cd5672c1bf23af57d60e23459f5d13cfaffa52a82ed12
- assets/distance_histogram.npz: 0d17ed2bc5846677d8774df7b9074129909b8403297ac9b200a4aed9d131c2f9
- assets/node_moments.npz: 21a9679881d07af5f2a36577b07a0ac426fc0788a2b75c9f6d3045ef74eef6fb

- results/research/issue287/pairs_same_anchor.npz: 3e2cad7ff9c177fd76ff55b7280f81e0131ab0cde09ca04b6c8ec31c98cbff44 (the #274/#287 same-anchor index, reused)
- results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json: 025ad8bd5809ce6296047514a80a4fedd012f3145d184eb4babd51c76c140059 (frozen TIGER SID)
- results/stage2_RQ-VAE/TIGER_RQ-VAE/rqvae_best.pth: af8efc29a8bcbbe5394cc5e358acf7b14c0e78ae99d8a8b9e5faf03fdddd97a8

## Protected components (path:sha256 baseline)
- CLAUDE.md: 29f99a9486f4dd35f469123e93aff99d015c0d448a08724074cef681f0e5d82f
- .agent/skills/complete-open-issue/SKILL.md: 05ee02d9132966c8056f3cb7126a368d565cae5c56013a4dbde92eed4017c235
- stage3_T5Train/train_HG-Rec.py: 54ca6bfbcd55a42e718660dd0004370a41a5e1c2c5038a86bbfc84f32b963690
- tools/research/stage2/contract.py: e2e7fe82370f5285457d1a47b55bc1fcdb108c31d1a16361b2816049bb9f80e8
- tools/research/stage2/launcher.py: 73afff80f994062ab5f33abce093f7c8c4aa9bc3a6549434c63081be08042101
- tools/research/stage2/checks.py: 30dd5c7c93a1ee21432fa0b29841444f9d390edcc71d48a5b8ba53fed41c0ef7
- tools/research/stage2/graph_contract.py: f2a6c9cd8d59f8412856a2d222ce4a5af5dce7e3fb80311825899a87d7972481
- tools/research/stage2/sid_quality.py: ba9e95057ce98823e01da6b5d2cebc700e016301b39f868b5ac947da8d81e7b8
- tools/research/verification/admission.py: 4637400e7103a249c1c05c1a1d17d40349a2d307ccb3d07f7a52d9703ed7f647
- tools/research/verification/run_suites.py: a3f8c6008d412841c32cbc976d7ec970c5f74f9268e9625ad3bd8e495c020cc2
- tools/iteration/depgraph.py: f6d6ae56865535630fb584a31bd847249889f4eea10dad5b84d69e746c96e5d2
- tools/iteration/execution.py: 9f15a8e04e37441fe5f986ed066befc70fb9fbb02c4588593b58aa70e9db51d1

## Training data
- results/stage0_build_parquet/train.parquet: 80597e38d081029e57e7434c92ff7be2c872f82aebc2e5f072ef973d73e88815
- results/stage0_build_parquet/valid.parquet: 48a291d174654ffb014d4d023029600168c7bdffc25d2c1982c93ed422b353e9
- stage1_GeneEmbedding/output/sentence_t5.npy: 6490c71753952b2d8788bb7e4e52cff6a5b0b59819a2cfbd737bee7bc07a77fb

## Hardware
0, NVIDIA A100-SXM4-80GB, 81920 MiB

## Interpreter
3.9.25 2.8.0+cu128 True
