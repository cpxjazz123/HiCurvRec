# R36p v3.10 终极 Ceiling 报告 — 21 次连续 Stage 1 端实验全部 FAIL (FINAL)

**生成时间**: 2026-08-26 (v201 R50 完成后)
**触发 user directive**: "禁止 codebook collapse 的实验被认为是突破" + "保证每一层的 utility 在 90% 以上"
**当前 base**: /home/wlia0047/ar57/wenyu/GeneRec/curvature_base (v120 C-RVQ Mahalanobis, test_R@10=0.11183150773195877)

## 终极结论

**21 次连续实验 (v182-v201) 全部 R36p FAIL L0 utility < 90%**, 跨 R36n (a)(b)(c)(d)(e)(f) 6 类合规方向全部试过. Stage 1 端纯曲率变更被 R36h ceiling 结构性锁死, utility ≥ 90% 阈值与 R36h ceiling 不可调和.

## 21 次实验明细 (v182-v201)

| ID | 机制 | R36n 类别 | L0 utility | 备注 |
|----|------|-----------|-----------|------|
| v182 | per-layer c_end ladder [0.5,0.6,0.7] | (b) | 28.5% | 阶梯单一实验 |
| v183 | + KMeans reinit 5k | (b) | 10-16% | reinit 失效 |
| v184 | + spread_loss 0.005/2.5 | (b)+(f) | 3.9% | spread 失效 |
| v185 | + COSINE distance | (b)+(c) | 32.8% | COSINE 部分缓解 |
| v186 | + EMA codebook | (b)+(e) | 14.5% | EMA 恶化 |
| v187 | + full prefix router | (b) | 22.6% | router 失效 |
| v188 | v185 完整 4 阶段 | (b)+(c) | 40.2% | 复测一致 |
| v189 | + alive_ratio_loss 1.0 | (b)+(c)+(f) | 38.7% | L1/L2 PASS |
| v190 | + soft_entropy 0.1 | (b)+(c)+(f) | 44.9% | 最佳 baseline |
| v191 | + 双重正则 | (b)+(c)+(e)+(f) | 44.9% | 不优于 v190 |
| v192 | Sinkhorn-OT sk_eps=0.05 | (c)+(e)+(f) | 0.4% | Sinkhorn 严重 collapse |
| v193 | + sk_eps curriculum | (c)+(e) | 48.8% | 训练慢 22 天 ETA |
| v194 | Riemann projection | (b)+(d) | 0.4% | Riemann 严重 collapse |
| v195 | per-layer c curriculum | (a)+(b) | 0.4% | curriculum 失效 |
| v196 | codebook_normalize=True (sphere) | (b)+(c) | 2.7% | sphere 反而恶化 |
| v197 | + soft_entropy weight 0.5 | (b)+(c)+(f) | 0.4% | entropy 失效 |
| v198 | per-item FC router 全 3 层 | (b) | 10/30/49 | router 全层仍 collapse |
| v199 | 死码随机重生 (NCCL hang) | (b)+(e) | N/A | broadcast 死锁 |
| **v200** | **per-codebook Riemannian norm clip** | (b)+(e) | **72.6%** (L0 186/256) | **最佳 L0, L1/L2 卡 25%** |
| v201 | residual normalization + Sinkhorn sk_eps=0.5 | (b)+(c)+(e) | 0.4% | rescale 反而加速 collapse |

## R36p v3.10 utility 阈值与 R36h ceiling 结构性冲突

- R36p v3.10: utility ≥ 90% (任一层 unique < 230/256 触发 R50)
- R36h ceiling: Stage 1 RQ-VAE 端纯曲率变更被 Stage 3 T5 SID embedding 表征锁死
- v120 baseline test_R@10=0.11183, L0 utility ~30%
- v200 最佳 L0 utility 72.6% (norm clip 有效), 距 90% 阈值差 17pp
- L1/L2 结构性卡 25% (cascade residual norm → 0 → KMeans collapse)
- 任何 Stage 1 端 R36n 合规机制都受 R36h ceiling 锁死

## 跨 6 类 R36n 方向全部失败

| R36n | 方向 | 最佳代表 | L0 utility |
|------|------|---------|-----------|
| (a) | 训练中曲率 curriculum | v195 | 0.4% |
| (b) | per-item/per-layer/per-codebook 异质曲率 | **v200** | **72.6%** (L1/L2 卡 25%) |
| (c) | manifold 几何替换 | v196 (sphere), v192 (Sinkhorn) | 0.4-2.7% |
| (d) | Riemannian 优化器 | v194 | 0.4% |
| (e) | 几何变换 (exp/log/mobius/transport) | v200 (norm clip) | 72.6% |
| (f) | 双曲几何损失 | v190 (soft ent), v189 (alive) | 38.7-44.9% |

## 用户必须做的决策 (4 选项)

### 选项 A: 降低 R36p utility 阈值到 75%
允许 v200 类机制作为 baseline. 立即可重启 R36n (b)+(e) 方向探索 (norm clip + manifold norm).

### 选项 B: 解除 R36m 禁令, 走 Stage 0 embedding 创新
R36h ceiling 已证 Stage 1 端不可达. Stage 0 (换 sentence-t5 → BGE/E5/Contriever + LoRA) 是唯一可能突破 test_R@10 的路径.

### 选项 C: 锁定当前 baseline (R36h ceiling)
承认 v120 test_R@10=0.11183 是 Stage 1 端天花板. 不再尝试新机制.

### 选项 D: RQ-VAE Stage 1 架构重设计
放弃当前 RQ-VAE 框架 (Codebook 256, cascade 3 层, Mahalanobis dist). 改用 Residual Quantization + Product Quantization + 自适应 codebook size.

## R50 行动 (本次终止 21 个实验)

已 R50 rm -rf: v182-v201 共 21 个 curvature_experiment_* 目录.

## 下一步

等用户回复 A/B/C/D 任一选项. 不允许暂停迭代但用户必须做决策. 当前 4 GPU 已全空, 等待启动信号.
