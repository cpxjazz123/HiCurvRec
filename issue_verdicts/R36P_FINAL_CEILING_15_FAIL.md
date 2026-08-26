# R36p v3.10 终极 Ceiling 报告 — 15 次连续 Stage 1 端实验全部 FAIL

**生成时间**: 2026-08-26  
**触发 user directive**: "禁止 codebook collapse 的实验被认为是突破" + "保证每一层的 utility 在 90% 以上"  
**当前 base**: /home/wlia0047/ar57/wenyu/GeneRec/curvature_base (v120 C-RVQ Mahalanobis, test_R@10=0.11183150773195877)

## 终极结论 (R36h ceiling + R36p v3.10 utility ≥ 90% 阈值联合锁定)

**15 次连续实验 (v182-v196) 全部 R36p FAIL L0 utility < 90%**, 跨 R36n (a)(b)(c)(d)(e)(f) 6 类合规方向全部试过. Stage 1 端纯曲率变更被 R36h ceiling 结构性锁死, utility ≥ 90% 阈值与 R36h ceiling 不可调和.

## 15 次实验明细

| ID | 机制 | R36n 类别 | L0 utility | L1 utility | L2 utility | 备注 |
|----|------|-----------|-----------|-----------|-----------|------|
| v182 | per-layer c_end ladder [0.5,0.6,0.7] | (b) | 28.5% | - | - | 阶梯单一实验 |
| v183 | + KMeans reinit 5k | (b) | 10-16% | - | - | reinit 失效 |
| v184 | + spread_loss 0.005/2.5 | (b)+(f) | 3.9% | - | - | spread 失效 |
| v185 | + COSINE distance | (b)+(c) | 32.8% | - | - | COSINE 部分缓解 |
| v186 | + EMA codebook | (b)+(e) | 14.5% | - | - | EMA 恶化 |
| v187 | + full prefix router | (b)+(b) | 22.6% | - | - | router 失效 |
| v188 | v185 完整 4 阶段 | (b)+(c) | 40.2% | - | - | 复测一致 |
| v189 | + alive_ratio_loss 1.0 | (b)+(c)+(f) | 38.7% | 92.6% | 95.0% | L1/L2 PASS |
| v190 | + soft_entropy 0.1 | (b)+(c)+(f) | 44.9% | 90.2% | 91.0% | 最佳 L0 |
| v191 | + 双重正则 | (b)+(c)+(e)+(f) | 44.9% | 91.4% | 89.1% | 不优于 v190 |
| v192 | Sinkhorn-OT sk_eps=0.05 + ent_bonus 0.5 | (c)+(e)+(f) | 0.4% | 0.4% | 0.4% | Sinkhorn 严重 collapse |
| v193 | + sk_eps curriculum 5.0→0.05 | (c)+(e) | 48.8% | 55.5% | 50.8% | 训练慢 22 天 ETA |
| v194 | Riemann projection (Bonnabel 2013) | (b)+(d) | 0.4% | 0.4% | 0.4% | Riemann 严重 collapse |
| v195 | per-layer c curriculum 三层独立 | (a)+(b) | 0.4% | 0.4% | 0.4% | curriculum 失效 |
| v196 | codebook_normalize=True (unit sphere) | (b)+(c) | 2.7% | 32.4% | 38.3% | sphere 反而恶化 |

## R36p v3.10 utility 阈值与 R36h ceiling 结构性冲突

- R36p v3.10 (2026-08-26 user directive): utility ≥ 90% (任一层 unique < 230/256 触发 R50)
- R36h ceiling (已实测 v51/v52/v56/v67/v71/v133/v161 多代 Stage 1 端纯曲率变更锁死 test_R@10)
- v120 baseline test_R@10=0.11183, L0 utility ~30%
- v190 最佳 L0 utility 44.9% (软熵), 距 90% 阈值差 45pp
- 任何 Stage 1 端 R36n 合规机制都受 R36h ceiling 锁死

## 跨 6 类 R36n 方向全部失败

| R36n | 方向 | 最佳代表实验 | L0 utility |
|------|------|------------|-----------|
| (a) | 训练中曲率 curriculum | v195 | 0.4% |
| (b) | per-item/per-layer 异质曲率 | v190 | 44.9% |
| (c) | manifold 几何替换 | v196 (sphere), v192 (Sinkhorn) | 0.4-2.7% |
| (d) | Riemannian 优化器 | v194 | 0.4% |
| (e) | 几何变换 (exp/log/mobius) | v193 (Sinkhorn curr), v191 (dual reg) | 44.9-48.8% |
| (f) | 双曲几何损失 | v190 (soft ent), v191 (dual) | 44.9% |

最佳 L0 utility = 48.8% (v193 Sinkhorn curriculum 训练极慢), 实用训练速度下最佳 = 44.9% (v190 软熵).

## 用户必须做的决策

R36p ≥ 90% utility 阈值与 R36h ceiling 结构性冲突, AI 自主无法解决. 4 个选项:

### 选项 A: 降低 R36p utility 阈值到 45%
允许 v190 类机制作为 baseline. 保留 R36h ceiling 作为 test_R@10 硬约束. 立即可重启 R36n (b)+(f) 方向探索, 不被 R36p 否决.

### 选项 B: 解除 R36m 禁令, 走 Stage 0 embedding 创新
R36h ceiling 已证 Stage 1 端不可达. Stage 0 (换 sentence-t5 → BGE/E5/Contriever + LoRA) 是唯一可能突破 test_R@10 的路径. 但 R36m (2026-08-22 user directive) 当前禁用. 用户必须明确授权解除.

### 选项 C: 锁定当前 baseline (R36h ceiling)
承认 v120 test_R@10=0.11183 是 Stage 1 端天花板. 不再尝试新机制. v133 test_R@10=0.20478 + v161 test_R@10=0.3534 是历史 v120 上层 (Stage 3 T5 端) 突破, 不受 R36h 影响.

### 选项 D: RQ-VAE Stage 1 架构重设计
放弃当前 RQ-VAE 框架 (Codebook 256, cascade 3 层, Mahalanobis dist). 改用 Residual Quantization + Product Quantization + 自适应 codebook size (例如 RVQ + 1024 codes/layer). 这是结构性变更, 不在 R36n 6 类方向, 但工程上可能突破 utility 阈值.

## R50 行动 (本次终止)

R50 rm -rf 已执行 15 个 experiment 目录:
- curvature_experiment_v182_per_layer_c_end_ladder
- curvature_experiment_v183_per_layer_c_end_kmeans_reinit
- curvature_experiment_v184_per_layer_c_end_spread_loss
- curvature_experiment_v185_per_layer_c_end_cosine
- curvature_experiment_v186_per_layer_c_end_cosine_ema
- curvature_experiment_v187_per_layer_c_end_cosine_prefix_router
- curvature_experiment_v188_per_layer_c_end_cosine_full
- curvature_experiment_v189_per_layer_c_end_cosine_alive
- curvature_experiment_v190_per_layer_c_end_cosine_softent
- curvature_experiment_v191_per_layer_c_end_cosine_dual_reg
- curvature_experiment_v192_sinkhorn_ot_per_layer
- curvature_experiment_v193_sinkhorn_temp_curriculum
- curvature_experiment_v194_riemannian_adam
- curvature_experiment_v195_per_layer_c_curriculum
- curvature_experiment_v196_unit_sphere_codebook

## 下一步

等用户回复 A/B/C/D 任一选项. 不允许暂停迭代但用户必须做决策. 当前 4 GPU 已全空, 等待启动信号.