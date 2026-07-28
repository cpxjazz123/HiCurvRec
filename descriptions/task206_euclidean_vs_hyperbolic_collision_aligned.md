# Task #206 — 欧式 vs 双曲 (collision 对齐) 实验 (7 阶段 κ 实验后真正决定 paper 基调)

> **任务目的**: HG-Rec 双曲几何对 Stage 4 Recall 是否有本质作用? 7 阶段 κ 实验后, 用户 2026-07-26 指出这个对照**一直没跑**, 它决定 paper 主调: 双曲 = 欧式 (HG-Rec κ 是工程优化) vs 双曲 > 欧式 (几何本质, 需 c∈[1,10] + L_hier margin).
>
> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

**7 阶段 κ 实验累加** (#84/#178/#199/#201/#203/#204/#205):
- c∈[1,10] 健康区, c=1 工程最优
- κ/降维/尺度归一都不能解锁"曲率杠杆"
- 用户 2026-07-26 14:30 发现双曲树度量 vs VQ argmin 敌对 (sort_clarity 塌缩)
- 用户提议新设计 D: 曲率用在层级 margin (不在 argmin)

**用户原话 2026-07-26 14:30**:
> "欧式 vs 双曲 (碰撞率对齐) 那个实验, 六轮以来一直没跑. **它才决定论文的基调.**"

**两种可能结论 + 对应 paper 写法**:
| 结果 | paper 写法 |
|------|----------|
| 双曲 ≈ 欧式 (对齐 collision 后 Recall 相近) | "HG-Rec κ 是工程优化, 非几何本质贡献. c=1 已足够, c 参数化是次要工程 choice." |
| 双曲 > 欧式 (对齐 collision 后 Recall 显著高) | "双曲几何对层次化量化有结构性优势. **但需要 c∈[1,10] 健康窗口 + 配合 L_hier margin (用户方向 D).**" |

---

## 2. 实验设计 (5 期流水线)

### 2.1 双臂设计

| 臂 | --loss_type | --euclidean_qloss | 含义 |
|----|-------------|-------------------|------|
| **A 双曲** (HG-Rec baseline #84) | poincare | False (默认) | 双曲双曲双曲 (recon + Sinkhorn + commit loss 全双曲) |
| **B 欧式** | mse | True | 全欧式 baseline: L2 recon + L2 argmin + L2 commit loss |

**控制相同** (跟 #84 baseline 完全对齐):
- num_emb_list=64,128,256
- e_dim=32
- beta=0.5
- lr=1e-3
- kmeans_iters=1000
- curvatures=[1,1,1]
- epochs=1000
- batch_size=256
- sk_epsilons=0.0,0.0,0.0 (HG-Rec paper, 不开 Sinkhorn)
- seed=2024
- layers=512,256,128,64

### 2.2 collision 对齐策略

**用户原意**: 在 collision_rate 对齐后比较 Recall. **R11.3 自主决策**:

方案 i (推荐): 先跑 B, 测 collision; 调整 quant_loss_weight / beta / 训练长度直到 collision 接近 A 的 8.14% (#204 c=10) 或 9.00% (#204 c=1)
方案 ii (更省): 不调超参, 接受 collision 不同, 直接比较 Recall (因为 collision 本身是结果的一部分, 双曲空间 promise 的是 collision 低 + Recall 高)
方案 iii (折中): 跑双倍 epochs (2000) 让 collision 自然收敛, 然后比较

**R11.3 决定**: 选方案 ii (最直接, 不引入调参伪影), 但同时跑方案 iii 的 2000 epoch 臂作为 sanity check.

### 2.3 启动 (5 期流水线)

| 阶段 | 时长 (估算) | 说明 |
|------|------------|------|
| Stage 1 RQ-VAE 训练 | ~15 min | GPU 0 跑 Arm B (Arm A 重用 #84 ckpt) |
| Stage 2 SID 推理 | ~5 min | Sinkhorn 解码 + 去重 |
| Stage 3 T5 训练 | ~90 min | 跟 #84 baseline 同步 |
| Stage 4 Recall 评估 | ~5 min | 跟 #84 baseline 同步 |
| 写 verdict + 对比 paper 基调 | ~30 min | 累加 7 阶段 κ 结论 |

**总计**: ~2.5 小时 (1 GPU, 顺序).

### 2.4 决策触发

| 结果 | 判定 |
|------|------|
| Arm B Recall@10 ≥ Arm A (对齐 collision 后) | 双曲只是工程 — paper 写"无本质优势" |
| Arm B Recall@10 << Arm A (≥3% 差距) | 双曲有结构性优势 — paper 写"需 c∈[1,10] + L_hier margin |
| Arm B Recall@10 ≈ Arm A (在 ±1% 噪声内) | 不确定, 需多 seed (但用户已禁止 multi-seed #R6) |

**R11.4 dry-run 提示**: 启动前先报告给用户: 选了方案 ii + iii 怎么并行, 跑哪个 GPU, 跟 #84 baseline 怎么对齐.

---

## 3. 风险与缓解

**风险 1**: Arm B 在 L2 空间里 collision 飙到 30%+ (HF-Rec 默认配置没有 L2 reference 数字)
**缓解**: 接受 collision 不同 (方案 ii), 但记录; 若 collision 影响下下游, 单独标注

**风险 2**: Sinkhorn=0 在欧式 path 下可能直接用 argmin, 跟双曲 path 不完全可比 (双曲 path 也 argmin 但距离函数不同)
**缓解**: 两个 arm 都用 argmin (Sinkhorn=0 是 HG-Rec paper 默认), 唯一变量是距离函数 L2 vs Poincare

**风险 3**: 训练时间不可控 (B 训练可能比 #84 慢, 因为 L2 距离计算和 Sinkhorn argmin 不一样)
**缓解**: GPU 0 独立跑, 不与其他任务冲突

---

## 4. 启动命令

```bash
# Arm B 启动 (后续跟 Stage 2/3/4)
bash scripts/task206_stage1_arm_B_euclidean.sh
bash scripts/task206_stage2_arm_B_sid.sh  
bash scripts/task206_stage3_arm_B_t5.sh
bash scripts/task206_stage4_arm_B_eval.sh
```

---

## 5. 完成度跟踪

- [x] 创建 task #206 (R9 max+1)
- [x] Dry-run 报告: 选方案 ii + iii, GPU 0 独立跑
- [ ] 写 Arm B 启动脚本 + sanity 测
- [ ] Stage 1 跑完, 看 collision_rate vs #84 baseline
- [ ] Stage 2 SID 推理 (Sinkhorn + 去重)
- [ ] Stage 3 T5 训练
- [ ] Stage 4 Recall@10 评估
- [ ] 写 verdict 对比 paper 基调

## 6. 关键决策点

| 时间 | 决策 | 理由 |
|------|------|------|
| 2026-07-26 15:00 | R9: 用 #206 编号 | max=205, +1=206 |
| 2026-07-26 15:00 | R7: GPU 0 分配 | 4 卡全空闲, 选 GPU 0 |
| 2026-07-26 15:00 | R11.3: collision 对齐方案 ii (不调超参, 接受差异) | 避免调参引入伪影 |
| 2026-07-26 15:00 | R11.3: 添加方案 iii (2000 epoch sanity 臂) | 双倍 epochs 看 natural collision |
| 2026-07-26 15:00 | R11.3: Stage 1 epochs=1000 (跟 #84 baseline 一致) | 用户禁止 multi-seed 抗噪声 |
| 2026-07-26 15:00 | R11.3: 复用 #84 baseline 的 Stage 2/3/4 脚本基础 | 不重写, R12 强制存 ckpt 已验证 |

**R11.3 备选方案**:
- 选 A (collision 对齐到 #204 c=1 9.00%): 需要调 quant_loss_weight 或 epochs, 可能引入和 #84 baseline 没法对比的复杂性
- 选 B (接受 collision 不同, 只看 Recall): 简单直接, 双曲 promise 是 "collision 低 + Recall 高", 两者综合看
- **本实验选 B**