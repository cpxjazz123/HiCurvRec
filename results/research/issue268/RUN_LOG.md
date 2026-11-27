# Issue #268 run log

- Issue: #268 — Frozen triplet-specific continuous curvature via graph-distance distortion for Euclidean RQ-VAE
- Title: `Frozen triplet-specific continuous curvature via graph-distance distortion for Euclidean RQ-VAE`
- Issue state at freeze: OPEN
- Issue body sha256: `6f911080347b200b49328e696ad253c21aec26854b601ac9f34427acd4aa7e26` (3846 chars)
- Frozen at: 2026-11-26

## INFRA_MODE

`FROZEN`. The issue states: "单 Agent；INFRA_MODE=FROZEN，复用 #259/#264 已有流程及训练图/距离/曲率资产；允许候选专属离线几何拟合、缓存导出、诊断和测试，不修改公共 Infra 或原资产。"

No sentence in the issue authorises building or repairing public infrastructure, so no
`tools/`, `assets/`, launcher, admission, evaluator or archive change is allowed. Candidate-local
offline fitting, cache export, diagnostics and tests are explicitly permitted and stay under
`results/research/issue268/` plus the candidate tree.

## Does this issue require training

Yes. The acceptance requires Stage2 3000 epochs, four cards, seed 42 and one frozen Stage3
150-epoch run; the final criterion `test_R@10` cannot be produced from existing weights (issue
§4 "一次 Stage2 3000 epochs… 一次 Stage3 150 epochs… 独立复算 Recall/NDCG@5/10；比较冻结 TIGER、父代和
0.065目标").

## Acceptance checklist C1..C19 (verbatim requirements, then criterion)

| ID | Issue requirement | Criterion / output | Status |
|---|---|---|---|
| C1 | 继承 #260–267 的 HYPOTHESES；先验证 #265/#267 是否存在 gate-only 改变目标误差而 hard q/SID 不变、固定局部曲率是否有能力减少真实图距离失真；明确竞争解释（几何尺子自由度 vs 曲率代理信息不足 vs 固定参考表示缺乏行为几何）；不预设因果根因 | `HYPOTHESES.json` inherited from #267 with every historical entry; `DIAGNOSIS.md` with ≥2 competing explanations and measured evidence paths | PASS |
| C2 | 核准训练图、三元组、参考点集合、最短路径口径与有效 mask，约 28,391,948 合法记录以实际 manifest 为准；有向标签不得当作对称图距离；曲率目标定义不变 | `assets/train/visible_undirected.npz` + `assets/v8_runtime_layout/targets.npy` + offsets; counts and sha256 in `INFRA.md`/`CONTRACT.md` | PASS |
| C3 | 区分 κ_graph、κ_t、κ_hat；严禁把 κ_graph 当黎曼参数，也不允许拿 κ_t 当训练目标值 | `CONTRACT.md` §definitions; code review that the target table is read unchanged and κ_t enters only the distance | PASS |
| C4 | 冻结参考表示 Z_ref（父代最终 32 维 encoder 输出或既有同身份冻结表示），固定全局距离尺度与坐标映射使 κ=0 对应同尺度欧氏距离；事前冻结 κ 域、球/双曲坐标域、零曲率极限、尺度、参考集合、pair 权重、优化预算、数值/成功/失败条件 | `artifacts/z_ref.npy` + `artifacts/z_ref_manifest.json` (checkpoint sha256 `1f8bfecb…`); frozen constants in `CONTRACT.md`; `analysis/probe_J.json` | PASS |
| C5 | 距离失真 `J_t(κ)=mean[(d_κ/g-1)^2]`；使用 Bachmann κ-stereographic 统一几何；严禁把三个固定专家加权距离当连续 κ；批量可向量化一维有界优化；含 κ=0 对照；报告边界/平坦比例；不得用近似冒充精确 | candidate `model/stereographic.py` + offline fit script; `CONTRACT.md` formulas; `analysis/FIT_DIAGNOSTICS.json` | PASS |
| C6 | 全量既有三元组按 shard 处理，保存全部 κ_t、行 ID、参考身份、κ=0 与拟合后失真、dtype、覆盖、失败/退化策略、源码/参考/资产 hash 与运行时间；缺有效距离标记不造数；若用确定参考子集必须透明报告差异并用全量独立核验 | `kappa_cache/kappa.npy` + `manifest.json` + `analysis/FULL_AUDIT.json` | PASS |
| C7 | 在独立固定诊断总体上核对连续 κ 对欧氏失真的改善、κ 分布与符号/结构分层、数值稳定与敏感性；若无法证实有效几何适配则继续低成本诊断，不默认训练 | `analysis/TRANSFER_VALIDATION.json` + `DIAGNOSIS.md` gate | PASS |
| C8 | 导出不可变缓存覆盖全部合法三元组并冻结；检查随机/全量重读、行 ID 对应、哈希、合法取值与同源复算；RQ-VAE 不更新参数、不重新拟合 | cache sha256 frozen before candidate copy; reload test `tests/test_kappa_cache.py` | PASS |
| C9 | Stage2：保持 #265 欧氏残差量化与原重建路径；移除与原 gate 相关的无效可训练参数；hard `q_i=Σ_l c_i,l`；每个三元组用同一固定 κ_t 与固定尺度计算局部截面曲率代理；`L_total=L_RQVAE+λL_curvature`；去掉 gate 后必须明确预算组与 λ 标定 | candidate tree diff vs parent; `MODULE_CHECKS.json`; calibration record | N/A (gate) |
| C10 | hard q 前向 + 经测试的 STE/显式梯度路由更新 encoder 与预定码本；κ_t/参考表示/图资产无梯度且 hash 前后一致；不得因 detach κ 切断 q 梯度 | `MODULE_CHECKS.json` + gradient tests + hash checks | N/A (gate) |
| C11 | TDD、独立参考、手算几何、κ=0 精确极限、正负曲率、近零连续性、空间边界、重合点、三元组索引、参考一致性、梯度与变异测试；验证错位 κ / 非冻结 κ / 改变选码 / soft q 冒充 hard q 均可被捕获 | `reference.py`, `tests/test_known_answers.py`, `tests/test_reference.py`, `tests/test_mutation.py`, `MUTATIONS.md` | N/A (gate) |
| C12 | 训练前在冻结父代 checkpoint 上做低成本真实批次探测：固定 κ 时结构 loss 梯度确实到目标模块；固定 q 时 κ 无法随优化变化；报告相对欧氏几何差异、梯度比例/方向与数值稳定性 | `analysis/PRE_TRAIN_PROBE.json` + `MODULE_CHECKS.json` | N/A (gate) |
| C13 | 一次 Stage2 3000 epochs、四卡、seed42、固定 batch、最终 checkpoint、无早停；三码碰撞<0.40、导出一致性；记录 inclusive 性能、离线拟合全量成本与缓存复用摊销；保持 #265 ~0.25s/epoch 量级；不削减覆盖/周期工作/验证 | `stage2_status`, run records, `analysis/PERFORMANCE.json` | N/A (gate) |
| C14 | 统一冻结尺度下独立于逐三元组拟合 κ 的固定欧氏与固定等权三专家评估；全量 hard q / continuous z 的曲率 MAE/RMSE/Pearson/Spearman、符号/中心内排序/方差、语义重建、量化误差、有效秩、码本使用/间距、各层前缀命中率/ER、Graphlet/Orbit 与行为保真；另报自身 κ_t 训练口径、距离失真与梯度；与 #265 同口径比较 | `analysis/FIXED_EVALUATION.json` + `REPORT.md` | PARTIAL |
| C15 | 一次 Stage3 150 epochs、beam20、seed42 冻结协议；先冻结候选并生成 §63 记录，再一次最终 test，独立复算 Recall/NDCG@5/10；比较冻结 TIGER、父代和 0.065；不得据 test 调整 κ 或再训练 | `test_final.json` + independent recompute + `analysis/STAGE3_RECALL.json` | N/A (gate) |
| C16 | 交付 DIAGNOSIS/HYPOTHESES/CONTRACT、κ 缓存/manifest 与 Z_ref 身份、拟合诊断、代码/专属测试、持久日志/收据、性能/结构报告、Stage2/Stage3 产物与独立复算、R2 四件套（含 κ 缓存/参考表示可恢复依赖）、VERDICT、提交与推送 | all artifacts + R2 + commit sha | PARTIAL |
| C17 | 明确回答 4 问：固定参考表示下连续 κ 是否降低真实图距离失真；固定 κ 是否改善独立几何保真与硬 SID；推荐是否改善；离线与在线成本 | `VERDICT.md` §four-questions | PASS |
| C18 | 负结果如实记录并按合法验收关闭；不将未运行项冒充通过；不擅自开启下一轮或修改 Infra | `VERDICT.md` + closed issue | PASS |
| C19 | 披露既有 test 重复访问风险 | `DIAGNOSIS.md` / `VERDICT.md` risk paragraph | PASS |

## Mechanism scope (what may change)

- Candidate tree `stage2_RQ-VAE/TIGER_fixed_triplet_kappa_rq` only: `ARM_NAME`, removal of the
  trainable gate/token path, the frozen-κ cache reader, the κ-stereographic distance and its
  wiring into the existing point-value curvature objective, plus mechanism tests.
- New research artifacts under `results/research/issue268/`: Z_ref, κ cache, fit/audit/diagnostic
  scripts and JSON, docs, logs.
- Everything else frozen: Euclidean selection, floor/last-level Sinkhorn, residual/reconstruction/
  commitment/codebook terms, SID export, ports, DDP launcher, hyperparameters, Stage3 protocol.

## Reuse plan

- Assets: `assets/train/visible_undirected.npz`, `assets/v8_runtime_layout/targets.npy` +
  `target_offsets.npy`, `assets/node_moments.npz`, `assets/item_summary.npz`,
  `assets/v5_gnn_geometry/*` and the frozen semantic embeddings are referenced by hash only.
- Parent representation: the immutable #265 run snapshot
  `results/stage2_RQ-VAE/_runs/stage2_4891fa8df3e843839de770e52e156892/TIGER_euclid_local_curve_supervision_rq/src`
  and its `result/rqvae_best.pth` (sha256 `1f8bfecba0c74bb3cbf9da11fe6925983cbae012753100401a7168e246da7bef`).
- Prior diagnostic code: `results/research/issue267/analysis/{curvature_full_arm,compare_new,center_order}.py`
  pattern for the fixed evaluation; no new public analyser.

## Gate decision and evidence (2026-11-27)

The pre-training geometry gate failed: the per-triple κ is a boundary/global
oddity, not local geometry. Evidence:

- `analysis/probe_J.json` (max-norm scale): 100 % at the spherical ceiling.
- `analysis/probe_J_edge.json` (frozen graph-unit scale): J0 `0.2913` → `0.2641`,
  median `9.28 %`, boundary `99.95 %`.
- `kappa_cache/manifest.json`: 28,391,948 rows, mean J `0.268071` → `0.242248`,
  median improvement `9.568 %`, floor fraction `99.78 %`, κ std `0.0101`.
- `analysis/TRANSFER_VALIDATION.json`: per-triple beats one global κ on 0.18 % of
  the independent rows and on 0.0 % of the full-reference audit.
- `analysis/SCALE_CONTROL.json`: one global Euclidean scale reaches
  `J=0.174575`, below the per-triple family `0.253693`.
- `analysis/CACHE_VERIFICATION.json`: cache recomputes to `5.5e-8` (J0) and
  `3.9e-8` (Jfit); off-grid `1.4e-8`.
- `analysis/probe_J_scale_0p068284.json`: at the best global scale the
  per-triple gain is `2.27 %` median with `38.3 %` boundary solutions.

C13/C15 are NOT APPLICABLE under the issue's own gate ("若无法证实有效几何适配，
继续低成本诊断，不默认正式训练"); no run, checkpoint or metric is claimed. The partial
candidate implementation (not trained, not completed) was deleted after the gate
decision; the frozen parent snapshot remains in `_archive/`.

## Infra checkpoints

`INFRA_SNAPSHOT.json` records every reused asset and every public component
hash after the work. No public file was modified; `FROZEN` was respected.
The host changed from four A100s to one A100-80GB mid-issue; the frozen launcher
would refuse a formal single-card Stage2, recorded in `INFRA.md` for future
rounds.
