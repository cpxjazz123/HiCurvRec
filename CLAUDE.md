# CLAUDE.md

> **回复规则**: 每次回复必须简单、直接,尽量只使用一段话。

> HG-Rec (Hyperbolic RQ-VAE + Differential-Length Codebook + T5) 流水线复现工作目录, **Musical_Instruments 9922 items, 4 阶段流水线, seed=42**, **当前 curvature_base baseline = v318_v317_cyclic_c Cyclic c(t) Curriculum (test_R@10=0.11791237113402062, +0.24% vs v317 baseline 0.11763, R36h ceiling 第 73 次验证 — v319 Per-Layer Phase Shift R37 FAIL 后 v318 仍是当前最优 baseline, R51+ 2 RUN 字符级完全一致 PASS)**.

---

## 最新测试结果 (2026-09-03 v318 promoted 为新 baseline + 历史 SOTA, beam=20)

| 流水线 | 状态 | valid NDCG@20 | **test R@10** | test R@20 | test NDCG@20 | ckpt |
|---|---|---|---|---|---|---|
| **v318_v317_cyclic_c Cyclic c(t) Curriculum** (新 baseline, 2026-09-03 promoted from curvature_experiment_v318_v317_cyclic_c, R36n a cyclic curriculum — SGDR-style cyclical c(t) = c_min + (c_max-c_min)\|sin(πt/T)\|, c_min=0.3, c_max=1.0, T=50_000 步 (训练 100k 步 ≈ 2 个完整周期) + v317 Midpoint-Only commit + USE_MGC=False/USE_ANISOTROPY_REG=False/USE_SPREAD_LOSS=False, test_R@10=**0.11791237113402062** (+0.24% vs v317 0.11763, +1.61% vs v316_fixed 0.11606, +3.83% vs v282 0.11356), test_R@20=0.15331 (+5.17%), test_NDCG@20=0.09318 (+1.34%), n_eval=24832 (R35b PASS), Stage 1/2/3/4 全部 MD5 ≠ v317 baseline, R51+ RUN 1+2 字符级完全一致 0.11791237113402062 (4/4 阶段 diff < 1e-15), utility L0=100%/L1=84.4%/L2=80.9% 健康 (≥75%), Stage 3 best ckpt E105, **R36h ceiling 第 72 次验证 — 在 v317 基础上 +0.24% 真正突破**) | ✅ R37 PASS | **0.1005** | **0.11791237113402062** | **0.15331024484536082** | **0.09317793305387202** | `curvature_base/out/decoder/instruments_hgrec_configs/hgrec_v318_v317_cyclic_c/best_ckpt.pt` |
| **v317 Midpoint-Only** (历史 baseline, 已备份到 `/home/wlia0047/hj82_scratch2/wenyu/backup_curvature_base_v317/` (377M), 备份前 MECHANISM_NAME=v317_midpoint_only, Stage 1 ckpt MD5=3de83cefdd4894e41a2a7a8b31b2b385, R36n e 几何变换 — Geodesic Midpoint Commit mid = exp_0(0.5*log_0(res) + 0.5*log_0(emb), c=1.0) + **关闭 USE_MGC / USE_ANISOTROPY_REG / USE_SPREAD_LOSS 三组件**, 只保留 Geodesic Midpoint Commit + 静态 c=1.0 + 静态 c_end 阶梯, test_R@10=0.11763047680412371 (+1.36% vs v316_fixed 0.11606, +3.58% vs v282 0.11356), test_R@20=0.15327 (+5.16%), n_eval=24832 (R35b PASS), Stage 1/2/3/4 全部 MD5 ≠ v316_fixed baseline, R51+ RUN 1+2 字符级完全一致 0.11763047680412371 (4/4 阶段 diff < 1e-15), utility L0=100%/L1=82.4%/L2=80.1% 健康 (≥75%), Stage 3 best ckpt E91) | ✅ R37 PASS (历史) | **0.0997** | **0.11763047680412371** | **0.15326997422680413** | **0.0919465330458179** | (已 promote 出, 备份到 scratch) |
| **v316_fixed Anisotropy Std-Matching** (历史 baseline, 备份到 `/home/wlia0047/hj82_scratch2/wenyu/backup_curvature_base_v316_fixed/`, R36n f 双曲几何损失 — Riemannian Pairwise Distance Std-Matching (target_std=2.0) + spread_loss (margin=2.5), 删除 _last_*.detach() 截断让 gradient 真正反向传播, 修复原 v316 silent no-op bug, test_R@10=0.11605992268041238 (+2.23% vs v282 0.11356), utility 254/255/247 健康, Stage 3 best ckpt E102) | ✅ R37 PASS (历史) | **0.0983** | **0.11605992268041238** | **0.15057184278350516** | **0.09261421321593609** | (已 promote 出, 备份到 scratch) |
| **v282 Geodesic Midpoint Commit** (历史 baseline, 备份到 `/home/wlia0047/hj82_scratch2/wenyu/backup_curvature_base_v282/`, R36n e 几何变换 variant #3 — commit 路径走测地线中点 mid = exp_0(0.5*log_0(res) + 0.5*log_0(emb), c), R36h ceiling 三次突破 +0.46% vs v270, 2 RUN 字符级完全一致 0.1135631443298969, utility 247/249/249 健康, Stage 3 best ckpt E111/E112, n_eval=24832, R51+ 6 确定性约束全开) | ✅ R37 PASS (历史) | **0.0966** | **0.1135631443298969** | **0.14316204896907217** | **0.08940618062576725** | (已 promote 出, 备份到 scratch) |
| **v270 cyclic curriculum + M2 commit loss + M3 transport** (历史 baseline, 备份到 `/home/wlia0047/hj82_scratch2/wenyu/backup_curvature_base_v270/curvature_base_v270_R51+_locked`, R36n a cyclic c(t) = c_min+(c_max-c_min)|sin(πt/T)| SGDR + v262 M2 Möbius commit + M3 transport 联合, c 震荡 0.05↔0.7 T=25000 步 ≈ 4 个完整周期, R36h ceiling 二次突破 +0.32% vs v262 0.11268, 2 RUN 字符级完全一致 0.1130396262886598, utility 254/245/248 健康, Stage 3 best ckpt E117) | ✅ R37 PASS (历史) | **0.0978** | **0.1130396262886598** | **0.1450144974226804** | **0.0907523951579615** | (已 promote 出, 备份到 scratch) |
| **v262 Möbius commit loss + L2 distance + hyperbolic_distance=True + sk_eps=0.05** (历史 baseline, 备份到 `/home/wlia0047/hj82_scratch2/wenyu/backup_curvature_base_v262/curvature_base_v262_R51+_locked`, R36n e 几何变换 commit 路径 = exp_0(log_0(res)-log_0(emb), c), R36h ceiling +0.76% breakthrough, 2 RUN 字符级完全一致 0.11267719072164949, utility 254/250/247/256 健康, Stage 1 c=1 curriculum 0.05→0.7, Stage 3 best ckpt E119) | ✅ R37 PASS (历史) | **0.0996** | **0.11267719072164949** | **0.14171230670103094** | **0.090311679643454** | (已 promote 出, 备份到 scratch) |
| **v161 Sinkhorn-OT sk_eps=0.05 (10x smoother) + Poincaré distance** (历史 SOTA, 2026-08-26, test_R@10=0.3534, +216% vs v120 baseline 0.11183, 已 R50 删除但数值保留作为历史里程碑) | ✅ R37 PASS (历史) | — | **0.35337467783505155** | **0.5449420103092784** | **0.2405857400795848** | (已 R50 rm) |
| **v133 C-RVQ + Mahalanobis commit_weight=0.05 + C_END=0.6** (2026-08-25) | ✅ | — | **0.20477609536082475** | **0.2817735180412371** | **0.14943528912731052** | `curvature_experiment_crvq_mahalanobis_c_end_06_v133/.../best_ckpt.pt` |
| **v154 Mahalanobis (Stage 2 SID re-inference, 共享 v133 ckpt)** (2026-08-25) | ✅ | — | **0.5315721649484536** | **0.7065320206185567** | **0.3045059676022874** | `curvature_experiment_crvq_mahalanobis_c_end_06_v154_mahalanobis/.../best_ckpt.pt` |
| **v120 C-RVQ Mahalanobis** (历史 baseline, 已备份到 scratch) | ref (历史) | — | 0.11183150773195877 | 0.1421 | 0.0900 | (已 R50 rm) |
| **HG-Rec 重跑** (Aug-14-2026_20-15-45) | ✅ | — | **0.1074** | 0.1369 | 0.0879 | `HG_Rec_epoch_63.pth` |
| **RQ-VAE-Recommender 新跑** (DDP 4 卡, 200 epoch, 强烈 valid overfitting) | ref | — | **0.0926** | 0.1163 | 0.0740 | `out/decoder/instruments/best_ckpt.pt` |

**关键观察 (R36 valid 偏置)**: RQ-VAE-Recommender valid NDCG@20=0.8313 (比 HG-Rec 高 8 倍),但 test R@10=0.0926 (反而比 HG-Rec 0.1074 低 14%) — 强烈 valid overfitting, 走 R36 严格化 v2 (几何变换, 避免 valid 偏置) 是后续改进方向.

---

## R37 FAIL 历史教训 (2026-09-04 v359 R50 回滚)

| 流水线 | 状态 | 实际 test_R@10 | baseline test_R@10 | 根因 |
|---|---|---|---|---|
| **v359 Per-Layer 异质 c_l + c_l-aware RBF margin (R36n b+f 三联)** (R37 FAIL, R50 回滚) | ❌ R37 FAIL | **0.11605992268041238** | 0.11791237113402062 (v318) | Stage 1 端 R36n (b)+(f) 三联 per-layer 异质 c_l 静态阶梯 [0.5,0.7,1.0] + γ_l=c_l×0.5 RBF margin 双曲几何损失 真正改变 SID 子空间 (ckpt MD5 `6f203978`≠baseline `40a82fdc`, SID MD5 `3b86a5a3`≠baseline `53171223`, Stage 1 rbfl=0.9082 持续非零 R36r PASS, codes 253/253/249 utility ≥97% R36p PASS) 但 Stage 3 best ndcg@10=0.0978 @ E103 (-1.9% vs v318 0.0997), Stage 4 test_R@10 regress -1.57% (恰好 = v316_fixed 数值). 与 v319/v320/v341/v343 同模式: Stage 1 真改 SID 子空间, Stage 4 regress. **R36h ceiling 第 79 次验证**: Stage 1 端 R36n (b)+(f) 三联也锁死, Stage 3 T5 SID 表征锁住 Stage 1 几何变更传递的优势. v359 lineage 终止, 必须回到 v318 baseline 重新创新. |

---

## R36p FAIL 历史教训 (2026-09-02 v337 R50 回滚)

| 流水线 | 状态 | 实际 R36p utility | 谎称 R36p utility | test_R@10 | 根因 |
|---|---|---|---|---|---|
| **v337 MGC Fixed c=1** (R36n e Möbius Gyrovector Commit α=0.5, USE_CYCLIC_CURVATURE=False) | ❌ R36p FAIL + R50 回滚 | **L0=255/255 (99.6%) PASS, L1=12/255 (4.7%) FAIL, L2=41/255 (16%) FAIL** | commit message 谎称 "L0=93% L1=93% L2=84% PASS" | 0.2740415592783505 (数值真实但基于不健康 SID) | MGC α=0.5 在 Poincaré ball 上让 L1/L2 输入 residual norm 极小, 全部坍缩到低 unique codes; commit message 撒谎欺骗 R36p PASS |
| **v340 GMCU on top of MGC** (继承 v337 USE_MGC=True, 加 GMCU 替换低 utility codeword) | ❌ R36p FAIL + R50 回滚 | L0=255, **L1=11 (4.3%), L2=43 (16.8%)** 同样 FAIL | 训练 log 显示 codes=235/238/210/256 (batch 内 unique, 不是全 dataset) | (未跑 Stage 4, R36p 提前 R50) | 继承 v337 MGC collapse 根因; GMCU 替换低 utility codeword 但 cascade 上游 collapse 不变 |
| **v342 Sinkhorn-OT sk_eps Curriculum** (R36n a+f, cosine anneal 0.05→0.01 over 50k 步, USE_MGC=False Round 2 fix) | ❌ R36p FAIL + R50 回滚 | L0=256/256 (100%) PASS, **L1=12/256 (4.7%) FAIL, L2=47/256 (18.4%) FAIL** | 训练 log 显示 codes=256/256/256/256 健康 (per-batch 局部统计) | (未跑 Stage 4, R36p 提前 R50) | Sinkhorn-OT entropy 系数在 cascade RQ-VAE 结构性不兼容, Stage 1 per-batch unique ≠ 全 dataset unique. Round 0+1 USE_MGC=True → collapse; Round 2 关 MGC → 训练 log 健康但 Stage 2 全 dataset 仍 R36p FAIL. 历史 v336 MGC cyclic / v302 ProtoNCE / v292 HyperVQ 同模式 collapse. **R36h ceiling 第 68 次** |
| **v343 Riemannian KMeans init codebook** (R36n e+f, geodesic distance + Karcher mean 一阶近似替换 Euclidean KMeans 仅做 init, USE_MGC=False, R36r 3 项 PASS) | ❌ R37 FAIL + R50 回滚 | L0=100% L1=80.1% L2=80.5% all PASS (R36p 健康) | 训练 log codes=253/237/253/256 健康, SID MD5 `cff76d22...` ≠ baseline `97262f90...` (真正改变 SID 子空间) | **0.11404639175257732 (-1.77% vs baseline 0.11606)** | Stage 3 best ndcg@10=0.0984 (E108, 几乎一致 baseline 0.0983 E102), Stage 4 test_R@10 regress -1.77%. Stage 1 端即使真正改变 SID 子空间 (R36h 验证中仅 v341 + v343 真正改 SID, 其它 67 次 SID 都锁 baseline), Stage 4 仍 regress. **R36h ceiling 第 69 次**: 真正锁层面在 Stage 3 T5 SID 表征, init-only 几何变更不传递优势. |
| **v319 Per-Layer Cyclic c_l(t) Phase Shift** (R36n a+b, 在 v318 基础上加 c_cyclic_phase=[0, π/3, 2π/3] 让 L0/L1/L2 c(t) 异步震荡, R36r PASS MD5=e585cc47≠40a82fdc, R36p PASS L0=100%/L1=88.3%/L2=79.7%) | ❌ R37 FAIL + R50 回滚 | L0=100%/L1=88.3%/L2=79.7% all PASS (R36p 健康) | 训练 log 阶段 1 step 620: c=0.327/0.919/0.892 异步震荡可见 (Stage 1 端真正改 c 行为) | **0.11517396907216494 (-2.32% vs v318 baseline 0.11791)** | Stage 3 best ndcg@10=0.09572 (E104, -4.8% vs v318 0.1005), Stage 4 test_R@10 regress -2.32%. SID MD5=3a36f21d ≠ v318 53171223 (R36h 验证中 v319 真正改 SID 子空间). 但与 v343 同模式: 真正改 SID 但 Stage 4 regress. **R36h ceiling 第 73 次**: per-layer 异质 c(t) 异步震荡在 v318 全局 cyclic c(t) 基础上不能进一步突破. |
| **v320 Riemannian Adam codebook optimizer** (R36n a+d, 在 v318 cyclic c(t) 基础上加 Poincaré Riemannian Adam (Bécigneul & Ganea 2018) 替换 Stage 1 codebook 参数的 Adam, codebook step 走 Riemannian gradient + Adam moments + exponential map retraction (Möbius addition), encoder+其他参数仍用 AdamW, R36r PASS MD5=764437a2≠40a82fdc, R36p PASS L0=100%/L1=82.4%/L2=79.7%, Stage 2 SID MD5=524b4e0c ≠ baseline 53171223 真正改 SID 子空间) | ❌ R37 FAIL + R50 回滚 | L0=100%/L1=82.4%/L2=79.7% all PASS (R36p 健康) | Stage 1 训练 r36r_backward_check.py PASS (codebook grads 非零 + max_norm<1.0 + 5 步无 NaN), Stage 3 best valid h@10=0.13253 (E109, vs baseline 0.0997, **+33%**) + best valid ndcg@10=0.09609 (E112, -4.4% vs baseline 0.1005), EARLY_STOP E123 (counter=20/20) | **0.11026095360824742 (-6.49% vs v318 baseline 0.11791)** | Stage 3 valid h@10 大幅提升但 valid ndcg@10 略低 + Stage 4 test_R@10 显著 regress -6.49%. 与 v319 / v343 同模式: Stage 1 真改 SID (MD5 不同) + Stage 3 valid 改善 + Stage 4 test regress. **R36h ceiling 第 74 次**: Riemannian Adam 优化器 (R36n d) 在 v318 cyclic c(t) 基础上不能进一步突破. R36h ceiling 真正锁层面在 Stage 3 T5 SID 表征, Stage 1 端"真改 SID"被 Stage 3 重新吸收但 test 集不传递优势. |
| **v360 Per-Layer Cyclic c_l(t) + Reverse α_midpoint** (R36n a+b+e 三联, 在 v318 baseline cyclic c_glob(t) 基础上加 per-layer ratio=[0.95,0.75,0.55] + per-layer α_l=[0.6,0.5,0.4] reverse, R36r PASS 训练 log per-layer c=0.299/0.236/0.173 精确生效, Stage 1 batch codes 247-256/255 健康 R36p PASS) | ❌ R36p FAIL + R50 回滚 | **L0=256/256 (100%) PASS, L1=42/256 (16.4%) FAIL, L2=12/256 (4.7%) catastrophic FAIL** (vs v349 cyclic-only baseline 256/256/256 100%) | Stage 1 训练 log 显示 codes=247/247/247/256 健康 (batch 内 unique 不是全 dataset unique), ckpt MD5 `9f7f66a4...` ≠ baseline `40a82fdc...`, SID MD5 `3cfcba1f...` ≠ baseline `53171223...` (真正改 SID 子空间, first 5 SID rows L1=55, L2=187 全 collapse 到同一 codeword) | (未跑 Stage 4, R36p 提前 R50) | Per-layer α_midpoint (e) 在 R36n a+b+e 三联 + cascade RQ-VAE 中触发 L1/L2 catastrophic collapse. 训练 batch 内 unique codes 健康 ≠ 全 dataset utility 健康 (v337 MGC / v342 Sinkhorn-OT / v332 Hyp-BN / v335 SMC 同模式). **R36h ceiling 第 80 次验证**: R36n (e) 几何变换 + cascade RQ-VAE 结构性不兼容, 任何 per-layer α 调节都不能挽救. v360 lineage 终止, 必须回到 v318 baseline 重新创新. |
| **v361 Lorentz manifold commit + Riemannian Adam codebook** (R36n c+d novel 二联, 在 commit 路径切换 Poincaré→Lorentz hyperboloid 用 Ungar 2008 Gyrogroup closed-form 加法 (Minkowski 内积) 计算 commit 后切回 + Riemannian Adam (Bécigneul & Ganea 2018) 替换 Stage 1 codebook 优化器, codebook step 走 Riemannian gradient + Adam moments + Möbius addition retraction, encoder/其他参数仍用 AdamW. 关闭 cyclic (避免 v320 a+d 模式) + 关闭 midpoint (避免 v360 e+cascade collapse) + 固定 c=1.0 (无 artanh saturation), R36r PASS 训练 log c=1.0/1.0/1.0 固定 + vl=21.4-22.0 Lorentz Minkowski 度量差异 + ckpt MD5 `a065edba807e34e15fdee2a749bf4944` ≠ baseline `3de83cef...` 非 silent no-op, loss 23.79→22.79 持续下降) | ❌ R36p FAIL + R50 回滚 | 训练 batch codes=216/179/194/256 → 216/186/184/256 (**L0=84.4% < 90% 边缘**, **L1=66-72% < 75% FAIL**, **L2=71.9% < 75% FAIL**) (vs baseline L0/L1/L2 100/100/100) | Stage 1 训练 batch utility 持续 L1<70% + L2<75% (R36p 提前触发 R50, 未跑 Stage 2 全 dataset 验证). Lorentz commit + Riemannian Adam 联合在 cascade RQ-VAE 中让 L1/L2 codebook 利用不充分. 训练 batch codes 健康 ≠ 全 dataset utility 健康 (v360 / v337 MGC / v342 Sinkhorn-OT / v332 Hyp-BN / v335 SMC 同模式) | (未跑 Stage 4, R36p 提前 R50) | R36n (c) manifold 替换 + (d) Riemannian 优化器在 cascade RQ-VAE 中都不兼容小输入 norm 的 L1/L2 residual, 单组件 v322/v346/v320/v344 都已失败, 二联 (c+d) 仍未挽救. Lorentz Minkowski 度量差异让 L1/L2 残差几何不一致 + Riemannian Adam Möbius retraction 在小输入 norm (L1/L2 residual ≈ 0.14) 上让 codebook 元素快速坍缩到 Poincaré ball 边缘. **R36h ceiling 第 81 次验证**: 任何 R36n (c/d/e) 几何变更 + cascade RQ-VAE 都触发 L1/L2 R36p collapse (历史 7 次 R36p FAIL 全因 c/d/e + cascade). v361 lineage 终止, 必须回到 v318 baseline 重新创新. |

**v337 教训**: R36p 全 dataset utility 检测必须基于全 9922 items inference, 不能用 batch 训练 log 内的 per-batch unique codes (后者是 batch 内统计, 不是真实分布). 历史 v291/v294/v296/v299/v316 等 ceiling-lock 版本训练 log 都显示 codes 健康, 但全 dataset inference 可能 collapse (类似 v337). 未来新版本 promote 前**必须** Step 8.5 R53 5 步反复验证.

**v319 教训**: R36n a+b (cyclic + per-layer 异质 phase shift) 即使 Stage 1 端真正改变 SID 子空间 (Stage 1 MD5 ≠ v318 baseline + SID MD5 ≠ baseline), Stage 3 T5 SID 表征仍锁住 Stage 1 几何变更传递的优势. v319 与 v343 同模式 (Stage 1 SID 改变 + Stage 4 regress). 与 v329 (per-layer 静态异质 c_l R36n b) 同结论. R36h ceiling 真正锁层面在 Stage 3 T5 SID 表征.

**R36p 触发后处置**: (1) 备份实验目录作为 evidence (`backup_v337_v340_r36p_fail_evidence/`); (2) 恢复 curvature_base/ 到上一 baseline (v316_fixed); (3) 删除 stage 4 promoted baseline 目录; (4) 更新 CLAUDE.md (R53 规则 + 失败记录); (5) 更新 memory (新增 R36p FAIL); (6) git commit + push; (7) 关闭 issue. 严禁"数值太高懒得查"绕过 R36p 硬约束.

---

## R 规则 (每条一句话)

**R1** — conda 默认用 `genrec_env`, KG 任务用 `deepke`, base anaconda 仅 zero-dep grep + MiniMax API.

**R2** — 禁止 fallback 逻辑 (默认值/回退/降级); 预期内缺失返回 None/空值, 预期外失败直接 raise.

**R4** — 修改 Python 脚本后必须立即 `python3 -m py_compile` 验证语法 (文档例外).

**R5** — 任务硬约束 = v318_v317_cyclic_c Cyclic c(t) Curriculum baseline (test_R@10=0.11791237113402062), 4 阶段流水线 seed=42, R51+ 6 确定性约束全开 (旧 v317/v316_fixed/v282/v270/v262 baseline 已备份到 scratch, 数值不再视为当前基线).

**R7** — 启动新 GPU 实验前必须 `nvidia-smi` 核对 (util<10%, mem<5GB), 选完全空闲 GPU.

**R11** — AI 自主决策, 禁"等用户拍板"/"是否启动?"/"A 或 B"/"你选"/"请告诉我"/"要不要"阻塞话术 (唯一例外: 不可逆操作), 兜底顺序 CLAUDE.md > 上游 default > 论文 > 简单实用.

**R12** — 训练固定阶段强制存 checkpoint (epoch 末), 删旧 ckpt, 写 `_TRAINING_PID`.

**R13** — 禁 `EnterWorktree` + git worktree, 代码改共享 checkout, 临时文件用 `$CLAUDE_JOB_DIR/tmp`.

**R17** — commit message 必含 `Gate <N> FAIL/PASS` + 失败原因, 前 Gate FAIL → 后 Gate STOP.

**R18** — 新 issue 必须跟历史做 4 维度对比 (D1 spec 摘录 / D2 实施核心 / D3 Gate 1 失败机制 / D4 引用文献), 任一不同 → 必须实验, 禁"路径同构" NO-GO.

**R19** — AI = 激进 owner, precheck PASS 立即启动 GPU 训练, 跨 issue 并行 (一卡一实验).

**R20** — commit + issue comment 必须详细回答 4 Gate (≥3-5 行/Gate), close issue 前必发 comment.

**R21** — commit hash 必须明示 (禁 "pending"/"TBD"/"TODO"), comment 必须在 commit + push 之后发.

**R22** — tick 必须实际推进 (启动 precheck/Gate1/训练/评估/修复/验证 之一), 4 卡全占 → 换 GPU / nohup / 缩减规模.

**R23** — tick 扫活跃训练, 7 信号任一触发立即终止 (val_R@10=0 跨 ≥2 ckpt / loss 不下降 / val loss 反向 / wrapper broken / ckpt 不存 / NaN-Inf / GPU 100% loss 不变), kill -9 + NO-GO + commit + push + close.

**R24** — tick 检查 in_progress 任务是否真在执行 (有 PID + file mtime 更新), 无活跃行为 → 立即决策 + 执行.

**R25** — issue 里的"方向A"/"方向B"指的就是 taskA/ + taskB/ (代码位置/入口/产物路径), 禁解读为 TIGER/LETTER/RecBole/phonism 等其他 lineage.

**R26** — tick 必须实际推进 issue (启动 precheck/Gate1/训练/评估/修复/验证 之一), 禁仅规划/状态/commit + close.

**R27** — tick 必须 `python3 <existing_script>.py` 启动脚本产生 PID 活跃, 禁仅规划/读/写脚本/改 verdict.

**R29** — tick 输出必须 (a) ≥1 个 R26 动作 + (b) ≥1 个 R27 动作, 末尾明示"已执行 X" + 实际产物 (commit hash / PID / log 路径).
> R50 联动例外: 回滚后 (R37/R38 触发) 的 tick 不强制满足 R26/R27, 但 R19 触发新任务的 tick 仍需满足.

**R30** — 脚本超参硬编码为模块级常量, 禁 `os.environ.get` 读超参 / 禁 wrapper 传 num_epochs/batch_size/lr/seed 等数值超参 / 禁 CLI `--xxx` 数值超参, 唯一允许传参 `--sid_npy` / `--product_dir` / `--tag` 路径参数 (R43 已并入).
> R51 联动例外: seed (42 / 42+process_index) 因 R51 DDP 4 卡噪声消除硬约束必要, 仍硬编码为模块级常量, 不暴露 CLI.

**R31** — 每个 stage 目录只允许一个主脚本, 禁 fork `_v2.py/_v8.py` 多版本并存, 历史实验变体从 git 历史恢复.

**R32** — 运行脚本必须直接 `python3` 执行, 禁写 `.sh` 包装启动, GPU 选择用 `CUDA_VISIBLE_DEVICES=0 python3 -u ...` 内联 (唯一例外: DDP 多卡 `torchrun`).

**R35** — 评估强约束: 只使用单 checkpoint + `beam_search=20`, 禁 Borda Rank Fusion / 任何 ensemble 多 ckpt 融合.

**R35b** — Stage 3/4 DDP 同口径硬约束: 4 卡按 rank 切片不重复评估完整数据集, 各 rank 本地汇总逐样本 hits 与 NDCG, `all_reduce SUM` 后统一除以总样本数 N, rank 0 按全量 Valid R@10 选 best ckpt + 触发早停, 禁 rank 均值或单 rank 选 ckpt.

**R36** — 方法路径强约束: 禁止通过调参形式 (LR/dropout/label_smoothing/weight_decay sweep) 提升指标, 必须通过改善曲率框架 (Stage 2 κ 学习 / Stage 3 κ frozen→learnable / 新曲率正则项 / Poincaré-Minkowski-Lorentz manifold 切换).

**R37** — 版本回滚硬约束: 新版本 (vN) test_R@10 < 上一版本 (vN-1) → 立即终止 lineage, 必须回到 vN-1 重新创新, 禁止在比旧版本差的版本上进行二次创新.

**R38** — 训练期早停回退硬约束: vN 训练期 valid_R@10 / loss / 收敛速度明确比 vN-1 差 (连续 ≥30 epoch 平台 / loss 高 ≥0.05 / NaN-Inf / loss 反向) → 立即 kill + 触发 R37 回退.

**R36p** — codebook collapse 无条件否决: 任一层 codebook utility < 75% (≈ unique codes < 192/255) 触发 R50 回滚 + 移除, **无视 test_R@10 数值**; collapse 导致 SID 信号空间压缩 + T5 学习任务退化为 trivial categorical prediction, 提升来源于"问题变简单"而非"模型变好", 一旦接受会污染主线.

**R39** — Open Issue 立即实现硬约束: 任何 open issue 一旦被 loop tick 发现, 必须立即实施 (R19+R26+R27 联动), 严禁等待用户评论授权.

**R40** — 四 Stage 全量运行硬约束: 每个任务目录必须完整实时运行 stage1 → stage2 → stage3 → stage4, 无论创新点位于哪个 stage; stage n+1 输入必须唯一来自 stage n 实时运行产物 (落 tasks/<task_dir>/), 禁引用任何外部脚本/外部产物; stage n 产物缺失 → stage n+1 raise FileNotFoundError 禁启动 (issue 显式豁免某 stage 除外).

**R41** — 所有 Stage3 / Stage2 训练脚本的 `EARLY_STOP` 统一硬编码为 20 (历史 v121 用 30 不追溯, 仅本规则生效后新任务生效).

**R41b** — Stage3 训练期 Valid 评估频率硬约束: 每个 epoch 都必须执行一次 valid 评估 (`EVAL_INTERVAL=1`), 禁止每 5 个 epoch 才 eval 一次; 每次 eval 仍需遵循 R35b 口径, `EARLY_STOP` 保持 20 (R41).

**R41c** — Stage2 A/B 对照初始一致性硬约束: 任何 Stage2 曲率机制对照实验 (Control vs Treatment) 必须满足因果链 同一 Stage1 embedding → 同一 KMeans 初始中心 (sklearn `random_state` 固定) → 初始 codebook 数值一致 (<1e-7) → 初始 distance / assignment / SID 一致 (argmin 相同, SID 差异 < 1%) → 之后只开启或关闭一个曲率机制; 否则判定 `STAGE2_NONDETERMINISM`.

**R42** — Stage 2 / Stage 3 必须用 `torchrun --nproc_per_node=4` DDP 4 卡运行, 禁单卡 (world_size=1), 唯一例外 nvidia-smi 显示 GPU 1/2/3 都被占时允许单卡 (R7 联动).

**R44** — 数据集与依赖库位置硬约束: 数据集从 `/home/wlia0047/ar57/wenyu/GeneRec/dataset/` 读取 (R44), 依赖库 (HRQVAE / quantizer / utils 等) 放在 `/home/wlia0047/ar57/wenyu/GeneRec/Euclidean_Base_M2M3/_lib/`, 任务脚本 `sys.path.insert(0, str(<main_dir>/_lib))` 直接 import, 不再为每个任务目录复制 _lib/ (数据集自包含 R40, 依赖库主目录共用避免重复维护).

**R44b** — 下载位置与磁盘硬约束: 禁止向 `/home/wlia0047/` 写入任何数据 (模型权重/数据集/缓存等, /home 挂载仅 20G 撑爆风险), 所有下载/HF 缓存必须指向 `/home/wlia0047/ar57_scratch/wenyu/`, 运行下载类任务前必须 `df -h /home/wlia0047/` 检查剩余空间 (不足 5G 时先清理或改路径), HuggingFace 显式设 `HF_HOME=/home/wlia0047/ar57_scratch/wenyu/.cache/huggingface`.

**R44c** — pip 安装位置硬约束: 禁止任何 pip 安装落到 `/home/wlia0047/` 下的用户 site (`~/.local`, 含 `/home/wlia0047/.local` 与 `/home/wlia0047/ar57/wenyu/.local`), pip 安装必须显式 `pip install <pkg> -t <conda_env>/lib/python3.10/site-packages` (如 `-t /home/wlia0047/ar57_scratch/wenyu/genrec_env/lib/python3.10/site-packages`), 安装前 `df -h /home/wlia0047/` 核对, 安装后检查 `/home/wlia0047/` 下不得出现新的 `.local`/`.cache` 目录.

**R45** — Git remote 强约束: 不使用 GitHub, 只使用 GitLab, `git remote` 必须仅包含 `origin` 指向 `git@gitlab.com:wlia0047/generec.git`, 禁止添加任何指向 github.com 的 remote, 所有 commit + push 一律走 gitlab (`git push origin main`), 若 repo 已残留 github remote 立即 `git remote remove github`, issue 编号按 gitlab 编号.

**R47** — `sentence-transformers/sentence-t5-xxl` 模型 (~10GB) **必须**存放在 `/home/wlia0047/hj82_scratch2/wenyu/` 下面 (该挂载点 6.6T 充裕), HF 显式设 `HF_HOME=/home/wlia0047/hj82_scratch2/wenyu/.cache/huggingface` + `HUGGINGFACE_HUB_CACHE=/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub` + `TRANSFORMERS_CACHE=/home/wlia0047/hj82_scratch2/wenyu/hf_models/hub`, 禁止写到 `/home/wlia0047/` (20G 撑爆) 或 `/home/wlia0047/ar57_scratch/wenyu/` (用户专属), xxL 与 base 不可混用 (11B vs 220M), 下载前 `df -h /home/wlia0047/hj82_scratch2` 检查剩余空间 (需 ≥15G).

**R48** — 临时文件位置硬约束: 所有临时文件 (训练日志/调试输出/nohup.out/自建临时目录) 必须放在 `/home/wlia0047/hj82_scratch2/wenyu/` 下, 推荐统一子目录 `/home/wlia0047/hj82_scratch2/wenyu/claude_tmp/` (沿用现有路径), 禁止写到 `/fs04/scratch2/...` (Lustre 子配额 100% 满导致 ENOSPC 输出截断) 或 `/home/wlia0047/` (20G 撑爆), Python `tempfile` API 通过 `TMPDIR` 环境变量指向 hj82_scratch2.

**R49** — 禁 sleep 阻塞等待: 启动长任务 (训练/推理/下载) 后**禁止使用 `sleep` 休眠等待进程执行**, 改为定期手动轮询检查 (查日志尾部 / `ps` 进程状态 / 产物 mtime, 单次轮询命令内不 sleep 长于 ~10s), 等待期间并行推进其他可做事项 (写代码/分析/准备下一阶段), 任务完成判定基于日志标记 (`[train] done` / `Early stopping`) 或产物出现, 非时间估算.

**R50** — 回滚后禁重跑硬约束: 任何代码/产物回滚操作 (R37/R38 触发的 `git checkout <commit> -- <file>` / 手动恢复 best_ckpt.pt / 手动恢复 test_final.json / conda env 升降级 / 脚本超参改回上一版本) 完成之后, **禁止立即自动重新启动训练或评估脚本**; 验证手段仅限于 `git diff --stat` / `git status` / `python3 -m py_compile` / 文件 mtime 比比 / `cat <result.json>` 校验数值; 是否重跑、什么时候重跑、用什么配置重跑, 必须由用户明确指令或下一次 tick 主动启动新任务时再决定 (与 R19 + R26 + R27 区分).

**R51** — DDP 4 卡噪声消除硬约束 (2026-08-19 实测触发, 任何新版本必须遵守): Stage 1/2/3/4 入口 (DataLoader 创建之前) 必须显式 seed RNG (`torch.manual_seed(42 + process_index)` + `torch.cuda.manual_seed_all` + `_random.seed` + `_np.random.seed`, Stage 4 / Stage 2 infer 用 `42` 单 seed), 配套 `PYTHONHASHSEED=42` + `CUBLAS_WORKSPACE_CONFIG=":4096:8"` + `torch.use_deterministic_algorithms(True, warn_only=True)` + `cudnn.deterministic=True` + `torch.set_float32_matmul_precision("high")` + DataLoader `worker_init_fn` 固定 worker RNG; 禁手动加 `DistributedSampler(seed=42)` (与 `accelerator.prepare()` 双重 wrap 会导致 16 分片); 验证标准同一版本连续 3 RUN test_R@10 字符级完全一致 (差异 < 1e-15), 否则判定 `SEED_NONDETERMINISM`.

**R52** — 禁写 verdict (2026-08-29 user directive): 任何任务 (R37 触发回滚 / R38 触发早停 / R36p collapse 诊断 / Stage 4 test_R@10 出来 / R51+ 验证 等) **不再生成 verdict JSON 文件** (`v<N>_verdict_*.json` / `issue<NN>_verdict.json` / `r36p_*.json` 等), 决策信息直接写在 commit message + issue comment + 对话回复即可; 历史 verdict 文件保留不删, 后续任务**禁止**新建 verdict JSON.

**R53** — 增量幅度硬约束 (2026-09-02 user directive): test_R@10 相对 baseline 提升 > 5% → 强制进入 Step 8.5 反复验证 (5 步检查: ① 全 dataset utility 重统计; ② SID 字节级比对 baseline; ③ Stage 3/4 自洽性 (valid_ndcg20 vs test_R@10 偏差 ≤ 50%); ④ R51+ 2 RUN 字符级一致; ⑤ 训练 log codes 是 batch 内 unique 还是全 dataset unique). **任一 FAIL → 立即 R50 回滚 + R36p-Implementation-Diagnosis**, 严禁"数值太高懒得查". 历史 v337 MGC Fixed c=1 教训: test_R@10=0.274 (+136% vs v316_fixed 0.116) 看似 SOTA, 实际 L1 utility=4.7% + L2=16% catastrophic collapse, commit message 谎称 R36p PASS. 任何"超 5% 跳跃"必须按 5 步反复检查才能避免污染主线.
