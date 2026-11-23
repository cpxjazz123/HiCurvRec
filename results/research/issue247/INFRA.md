# Issue 247 infrastructure evidence

| Requirement | Command | Summary | sha256 / evidence | Verdict |
|---|---|---|---|---|
| Stage0 items + train parquet | `ls results/stage0_build_parquet` | 24,587 items; train/valid/test present | recorded in manifest `inputs` | PASS |
| Semantic embeddings | `python -c "np.load(...).shape"` | (24587, 768) float32, finite | manifest `inputs` | PASS |
| Frozen v2 assets | `prepare_v3.audit_v2` | 47/47 v2 manifest hashes verified, no drift | `AUDIT_246.json` | PASS |
| User split | independent re-count | visible 45,951 / struct_val 5,744 / struct_test 5,744, pairwise intersection 0, union complete | `AUDIT_246.json.splits` | PASS |
| Directed edge recount | independent per-sequence counter | every split's CSR equals a fresh Counter of raw transitions (self-loops dropped: 0) | `AUDIT_246.json.splits` | PASS |
| Pair banks | manifest + npz | 5 train banks x 12,000; strict banks 2,726–2,940; relaxed 60–274; probe 62 | `AUDIT_246.json.banks` | PASS |
| Strict/train key overlap | set intersection | 0 | `AUDIT_246.json.strict_train_intersection` | PASS |
| v2 curvature assets | directory listing | **no curvature file exists in v2** (curvature.py implements Ollivier–Ricci, not A.1) | `AUDIT_246.json.curvature_files` | PASS (recorded absence) |
| PPR | `_ppr_cache.npz` present in v2 pairs dir | **not read, not recomputed, not referenced** by any v3 asset or command | `prepare_v3.py` skips it; manifest has no PPR entry | PASS |
| Interpreters | `/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` | CPU only, no GPU needed (no training) | all logs | PASS |
| Hardware | `nvidia-smi` | A100 40GB idle; this round trains nothing | — | PASS |
| GraphMoRE paper | arxiv 2412.11085v1 Appendix A.1 | Eq. A.1 sectional statistic | `CONTRACT.md` | PASS |
| Official implementation | github Vic-GoodLuck/GraphMoRE @1948d3c | contains `utils.cal_shortest_dis` (networkx shortest paths); **no sectional-curvature estimator is released** | `CONTRACT.md` crosswalk | PASS (limitation recorded) |
| Frozen TIGER SID | `results/stage2_RQ-VAE/TIGER_RQ-VAE/item_sids.json` | sha256 025ad8bd…, Stage3-read identity | `assets/tiger.json` | PASS |
| TIGER checkpoint / NPY | separate identities | checkpoint epoch 300 belongs to another SID generation; NPY sha 6f2682c1… | `assets/tiger.json.separate_*` | PASS (kept separate) |
| Storage policy | `.gitignore` §92 | generated npz/npy under `results/research/` stay out of git; manifest+reports are committed | commit | PASS |

Frozen assets were read only; no v2 file was modified or regenerated.
