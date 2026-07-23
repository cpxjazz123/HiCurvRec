# Task #85 — 三几何独立 RQ-VAE + SID 相似度检验

> **任务目的**: 排除"量化器不匹配几何"这一最后技术疑点；判断 Toys 数据集上, Δρ_Riemannian ≥ 0.03 是否能让双曲 κ=-1.06 子空间从 ρ=0.079 上升到 ρ≥0.11；若不达, 即"双曲本身无效"被坐实为终局性结论.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 Task #82 (v1 / v2 / v3) 系列诊断链 + Task #71 失败根因:

**Task #82 v3 已证明的差距** (Toys, 2000 子样本, ~2M 商品对):
| 子空间 | κ | ρ_L1 (欧氏 KMeans 量化) |
|--------|------|------|
| MCKG m=0 球面 | +0.845 | **-0.1341** (单子空间 ρ 实际反超 T5 -0.1207) |
| MCKG m=1 准欧氏 | -0.174 | -0.1139 |
| MCKG m=2 双曲 | -1.059 | -0.0792 |
| MCKG 加权 (三子空间) | — | -0.0934 |
| T5 Euclidean (PCA-64) | κ=0 | -0.1207 |

**但量化端**全程用了 sklearn KMeans (平方欧氏距离 + 普通 KMeans++ 初始化), 没有为 κ≠0 子空间用匹配的 κ-Stereographic 距离 / Möbius 加法 / Riemannian 优化器. 这是 Task #82 v2 verdict §3 解读第 3 条点名的 "RQ-VAE 本身天花板"。

**Task #71 已知失败根因** (E流/H流/S流 -72%~-81%):
1. **argmin 等价性失效**: 双曲欧氏等距投影回 κ 流形后, 欧氏最近邻不再是流形最近邻 (l2_normalize/Rsqrt 路径在 κ 流形上无意义)
2. **码本未用黎曼优化**: Task #71 的双曲码本用了普通 torch.optim.Adam + clamp 而非 RiemannianAdam → 码本坍缩到 9/256, 7/256
3. **投影精度损失**: Stereographic 球面投影在边界 (||x|| → 1/√κ) 处数值不稳

**本任务核心假设**:
- H1: "匹配几何的量化器"能让任一子空间 ρ_Riem > ρ_Euclid + 0.03
- H2 (否证): 即便用 Riemannian 量化, 双曲子空间仍 ρ ≤ 0.10, 则问题不在量化而在于玩具数据本身没有双曲结构 → 终局证据

---

## 2. 实验设计

**变量**: 量化器从欧氏 KMeans → κ-Stereographic Riemannian 量化 (geoopt)
**保持不变**:
- 输入: MCKG Toys 三子空间原始 embedding (`subspace_item[m]` 第 m=0/1/2 个 `(11924, 32)`)
- 输入距离公式: 仍用对应 κ 的 κ-Stereographic 测地距离算 ρ (不变)
- 样本: 2000 抽样, seed=42 (与 Task #82 v3 一致)
- K_LIST=[256,256,256], 三层独立 cascade, 互不融合
- 输出 SID tensor 格式 (11924, 3) int64

**改动点** (Phase 0 → Phase 4):

### Phase 0: 三条 pipeline 精确规格 (规格先于实现)

| 子空间 | κ | 几何 | 量化器 (用 geoopt `Stereographic` manifold 类) |
|--------|------|------|------|
| 球面 m=0 | +0.845 | 球面 Stereographic | `geoopt.Stereographic(k=+0.845).dist` + `mobius_add` + `RiemannianAdam` |
| 准欧氏 m=1 | -0.174 | 准欧氏 (≈ 退化到欧氏) | `geoopt.Stereographic(k=-0.174).dist` (退化版) |
| 双曲 m=2 | -1.059 | 双曲 Stereographic | `geoopt.Stereographic(k=-1.059).dist` + `mobius_add` + `RiemannianAdam` + 边界裁剪 \|\|x\|\|≤1/√(\|κ\|)·0.999 |

### Phase 1: 低成本码本利用率预检查 (300-500 step, 三条各跑)
- 每 50 step 记录每层 "激活码字数 / 256"
- 9 条曲线 (3 流水线 × 3 层)
- **判定**:
  - ✅ 利用率稳定 > 60% (>150/256) → 进入 Phase 2
  - ❌ < 20% 或持续下降 → 停该子空间, 回查 optimizer
  - 部分坍缩 → 只跑通过者

### Phase 2: 完整训练 (仅 Phase 1 通过者)
- 训练至收敛 (commitment + recon loss plateau)
- 每条独立 ckpt
- 日志: 重建 MSE / 每层 commitment / 每层最终码本利用率

### Phase 3: SID 相似度检验 (新增 Riemannian 对照组)

| 组别 | 距离端 | 量化端 | 用途 |
|------|--------|--------|------|
| A 对照 (有, Task #82 v3) | κ-Stereographic 测地 | 欧氏 KMeans | baseline: 量化器不匹配 |
| **B 实验 (本任务新增)** | **κ-Stereographic 测地** | **Riemannian 量化 (geoopt)** | **检验 Δρ** |

**Δρ**:
- Δρ_球面 = ρ_B(m=0) − ρ_A(m=0)=(-0.1341)
- Δρ_准欧氏 = ρ_B(m=1) − ρ_A(m=1)=(-0.1139)
- Δρ_双曲 = ρ_B(m=2) − ρ_A(m=2)=(-0.0792)

**判断**:
- Δρ_双曲 ≥ 0.03: 双曲子空间"被低估", 值得下游介入
- Δρ_双曲 ≤ 0: 量化器不是瓶颈, 双曲确实在玩具数据无结构
- 球面 Δρ ≥ +0.01: 强化"球面是首选"结论

### Phase 4 (仅 Phase 3 通过者): 下游 Recall@10
- 仅对 Δρ > 0.03 的子空间执行
- 接入 GRID Stage 3 + Stage 4 (TIGER 训练 + 推断)
- 与 baseline (T5 + 标准 RQ-VAE, R@10 = 0.0971) 对照

**启动命令** (Phase 1 预检查先跑, 模板):
```bash
source /apps/anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /home/wlia0047/ar57_scratch/wenyu/grid_toys
cd /home/wlia0047/ar57/wenyu/GeneRec

# Phase 1 三条各自 300-500 step 预检查
python3 -u task_artifacts/scripts/task85_riemannian_rqvae_pretrain.py \
  --subspace 0 --kappa +0.845 --K_list 256,256,256 \
  --steps 400 --lr 1e-3 \
  --out_dir products/task85_tri_geom_rqvae/sphere_log/
# (重复 --subspace 1 -0.174, --subspace 2 -1.059)
```

---

## 3. 决策触发 (vs Task #82 v3 baseline)

| Phase 1 码本利用率 | Phase 3 Δρ (Riemannian − 欧氏) | 决策 |
|--------------------|--------------------------------|------|
| ✅ 三空间全 > 60% | Δρ_双曲 ≥ 0.03 且任一 Δρ>0.02 | 进入 Phase 4 (下游 Recall@10) |
| ✅ 三空间全 > 60% | Δρ_双曲 ≤ 0 (基本不变) | **坐实双曲无效**: 写终局 verdict, 不进 Phase 4 |
| ⚠️ 部分坍缩 (双曲 < 20%) | — | 修复 Riemannian Adam 后重跑, 不带坍缩跑全 Phase 2 |
| ❌ 球面或全部坍缩 | — | **停止**: 代码实现 bug, 回查 geoopt 调用 |

---

## 4. 预算

| 阶段 | 估算时间 | GPU |
|------|---------|-----|
| Phase 0 规格 + 实现 (geoopt 装好 + 三脚本) | 0.5 天 | CPU |
| Phase 1 码本利用率 (3 流水线 × 500 step) | 1-2 小时 | 1 张空闲 GPU |
| Phase 1 失败排查 (如坍缩) | 0.5-1 天 | — |
| Phase 2 完整训练 (3 × 收敛) | 1-2 天 | 3 张 GPU 并行 (或 1 张串行) |
| Phase 3 SID ρ 检验 (3 流水线 × 2000 样本) | 0.5-1 小时 | CPU |
| Phase 4 下游 Recall@10 (仅 Phase 3 通过) | 1-2 天 | 1 张 GPU |
| **总计** | **4-8 天** (取决于 Phase 1/3 阻断) | |

---

## 5. 风险与缓解

**风险 1**: 双曲码本再次坍缩 (Task #71 重演)
- **缓解**: Phase 1 强制预检查, **不通过不进入 Phase 2**; 排查方向: (a) geoopt `Stereographic` manifold `k=-1.059` 边界, (b) RiemannianAdam `stabilize=True`, (c) 显式 `projx()` 把码本投影回流形

**风险 2**: argmin 等价性问题 (Task #71 主因) — 欧氏最近邻 ≠ 流形最近邻
- **缓解**: 用 `geoopt.manifold.Stereographic.dist(x, y)` 直接算测地距离, **不**通过欧氏内积或 argmin 转换

**风险 3**: 球面 κ=+0.84 边界投影数值不稳
- **缓解**: 数据预处理 `clamp_norm_(x, max=1/sqrt(κ)*0.999)`; 项目里 `task_artifacts/scripts/mckg_model/stereographic.py` 已有 `log_at_origin_kappa` 参考

**风险 4**: 准欧氏 m=1 (κ=-0.174) 接近 κ=0 退化, 数值边界
- **缓解**: κ=0 时 `Stereographic.dist` 退化为普通欧氏; κ=-0.174 → 等效 κ=0 + 小扰动, 实测两条 ρ_A vs ρ_B 应该接近 (此子空间不会因 Riemannian 量化有大提升, 是预期)

**风险 5**: ml1m v4 训练占 cuda:0, 同时开新 GPU 任务冲突
- **缓解**: 本任务 Phase 1 启动时 ml1m v4 大概率已自然停 (patience ep115 ~ 20:00), 4 张 A40 全部空闲; 若未停, 用 cuda:1/2/3 启动

---

## 6. 完成度跟踪

- [ ] Phase 0: geoopt `Stereographic` 调用验证, 三 pipeline 脚本 `task85_*` 落盘, py_compile 通过
- [ ] Phase 1: 三条 pipeline 各跑 300-500 step, 9 条码本利用率曲线产出, **≥150/256 通过**
- [ ] Phase 1.5: (条件) 若坍缩, 修复 optimizer 重跑, 直到三空间通过
- [ ] Phase 2: 完整训练 Phase 1 通过者, 各自收敛 ckpt 落盘
- [ ] Phase 3: SID ρ 检验 (Riemannian vs 欧氏), Δρ_球面/准欧氏/双曲 三数计算
- [ ] Phase 3 verdict 决策: 任一 Δρ>0.03 → 进 Phase 4, 否则直接写终局
- [ ] (条件) Phase 4: 下游 Recall@10, 仅 Phase 3 通过者
- [ ] 写 final verdict: 解释 Δρ 与 Task #71/82 的关系, paper §4.4 终稿
- [ ] 把 Task #71/69 历史 verdict 更正到 "是 KMeans 瓶颈还是几何本身无效" 取决于 Δρ
- [ ] §16 登记入 loop.md, 完成后从 §16 移除 (强制 R8)
- [ ] task_artifacts/scripts/ 添加 task85_riemannian_rqvae_pretrain.py 与诊断脚本
