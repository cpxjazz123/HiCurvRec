# Issue 248 infrastructure evidence

All checks below were executed on 2026-11-23 with the Stage2/infra interpreter
`/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9` (CPU only; this round trains nothing).

| Requirement | Command | Summary | Verdict |
|---|---|---|---|
| Items + train parquet | `tools.graph_infra.graph_build.item_index()` | 24,587 contiguous item ids, matching the 768-d embedding rows | PASS |
| Semantic embeddings | `np.load(stage1_GeneEmbedding/output/sentence_t5.npy, mmap_mode='r')` | shape (24587, 768), finite | PASS |
| Frozen #246 v2 assets | `build_assets_v4.audit_inputs()` | 47/47 v2 manifest hashes unchanged; v3 (#247) historical hashes unchanged | PASS |
| #247 assets kept as history | file listing | `results/research/issue247/assets/` untouched (197 MB); referenced by hash as version `v3-sectional-247` | PASS |
| Visible graph | `load_sparse(infra246/assets/graph/visible_forward.npz)` | 261,132 directed edges; reverse == forwardᵀ | PASS |
| Undirected projection | `tools.graph_infra.sectional.projection` | 510,464 undirected entries, loop-free, binary | PASS |
| Full triple population | `sum_m C(deg(m),2)` recomputed from the visible graph | **28,391,948** — the previously reported number is confirmed, not assumed | PASS |
| Components | `scipy.sparse.csgraph.connected_components` | 117 components; largest 24,471 nodes; 116 isolated items | PASS |
| All-source BFS | `scipy.sparse.csgraph.shortest_path` (chunked, unweighted) | 24587×24587 distances in ~197 s; 0.47 % of pairs unreachable (the other components) | PASS |
| Storage capacity | `df -h /fs04` | 9.0 P total, 1.4 P free (84 % used); `results/research` currently 8.6 GB | PASS |
| Persistent location | directory listing | assets live on the shared `/fs04` research volume (`results/research/issue248/assets/`), not in a temp directory or machine cache | PASS |
| Repo-approved external store | `rclone lsd r2:genrec` | bucket reachable (99 numbered directories). The numbered layout is reserved by §6b for Stage3 mechanism archives, so the v4 assets stay in the repository's research-asset tree; the manifest and the evidence files are committed | PASS (decision recorded) |
| PPR | file listing + code inspection | the v2 `_ppr_cache.npz` is skipped explicitly by the builder and referenced by no v4 asset; no v4 code path computes or reads PPR | PASS |
| Interpreters | — | stage0/1/3 `genrec_env/bin/python3.10`; stage2/infra `genrec_env_v2/bin/python3.9`; MCP `python3` | PASS |
| Hardware | `nvidia-smi` | A100 40 GB idle; this round needs no GPU | PASS |
| GraphMoRE paper | arXiv 2412.11085v1 Appendix A.1 | Eq. A.1 with the parallelogram-law interpretation | PASS |
| Official implementation | github `Vic-GoodLuck/GraphMoRE` @ `1948d3c5f19f7251cadd8d1118c06f0e294f96b3` | ships `utils.cal_shortest_dis` (networkx shortest paths) but **no sectional-curvature estimator**; the comparator is therefore an independent implementation of the paper equation | PASS (limitation recorded) |

Nothing in #246 or #247 was overwritten: the builder writes only under
`results/research/issue248/assets/`, and every reused asset is referenced by immutable hash.
