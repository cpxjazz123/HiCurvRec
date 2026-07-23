# Task #88 — HG-Rec 改造: 逐层独立曲率 (c_0/c_1/c_2) 网格搜索 (Idea1 验证版)

> **任务目的**: 把 HG-Rec 的硬编码全局曲率 `c=1` (reconstruction loss) + 量化模块隐含的 c, 改为三层独立 c_0/c_1/c_2. 通过网格搜索 4-6 个候选组合, 验证 "per-layer curvature vs 单 curvature" 的下游 R@10 改进, 直接验证 [[Idea1]] 核心假设 (与 toy 数据 κ_3=-0.84 对应).

> **完成日期**: (in progress)
> **状态**: 🟡 待启动

---

## 1. 背景

承接 [[Task #84 HG-Rec 主实验]] (R@10=**0.1020**, H2 几何先验中性) + [[Task #80 Stage 1c κ 诊断]] + [[Task #70 Ollivier 真实曲率]] (Toys 数据 κ_3=-0.84 强双曲, 但 HG-Rec 复现与 vanilla MSE 持平, 即"几何先验不自动传递下游").

**关键反 evidence (前次失败教训)**:
- [[Task #82 Stage 1c Metric 弱信号]] 三重独立支撑: phonism RQ-VAE 残差几何真实接近欧氏空间 (κ=0 全层最优)
- 原因: phonism encoder 在纯欧式目标下训练好后**冻结**, 套上事后双曲距离, encoder 没机会为双曲调整
- HG-Rec 优势: encoder 与 quantization 联合训练, encoder 已经为双曲调整过

**Idea1 核心假设 (HG-Rec 改造版才能验证)**: Per-layer curvature (c_0/c_1/c_2) > single global c=1

---

## 2. 实验设计

**变量**: HG-Rec 三个量化层 (Layer 0/1/2) 的曲率参数 c_0, c_1, c_2 (独立取值 0.5~2.0)

**保持不变**:
- Stage 1: sentence-t5-base 768d 编码 (与 Task #84 完全一致, 不重跑)
- Stage 2: HG-Rec 拓扑 (encoder/decoder/quant codebook structure)
- Stage 3: T5-small 训练 (seed=42, 95 epoch 早停)
- Stage 4: 评估 (test R@5/R@10/NDCG@5/NDCG@10)
- 数据集: Musical_Instruments_5core (24,772 users / 9,922 items / 511,836 interactions)

**改造点 (dry-run 报告)**:

**A. 现有 HG-Rec hrqvae.py (line 72-89)** compute_loss 单一曲率 c=1 (6 处使用):  
```python
elif self.loss_type == 'poincare':
    out = expmap0(out, c=1)
    xs = expmap0(xs, c=1)
    out = proj_to_ball(out, c=1)
    xs = proj_to_ball(xs, c=1)
    loss_recon = torch.mean(poincare_distance(out, xs, c=1)**2)
    out = logmap0(out, c=1)
    xs = logmap0(xs, c=1)
```

**B. 量化模块 (HResidualVectorQuantization)** 当前**没有**双曲距离, 只用 MSE + SINKHORN.

**C. 改造方案** (用户已选路径 1 → 路径 1+2):
1. 给 `HRQVAE.__init__` 加 `curvature_list = [c_0, c_1, c_2]` 参数 (默认 [1.0, 1.0, 1.0] 兼容原版)
2. 在 `quantize` 内部, 对每层 residual (Layer i 的 input - codebook_i[indices_i]) 计算**双曲距离** = `poincare_distance(x_residual, 0, c=c_i)`, 加入 quant loss
3. `compute_loss` 仍用 c=1 reconstruction loss (单一曲率 decoder 输出对比 input), 改造只在 quant 层
4. 网格搜索: 4-6 个 (c_0, c_1, c_2) 候选组合, 例如:
   - [1.0, 1.0, 1.0] (原版对照组)
   - [0.5, 0.5, 0.5] (强双曲)
   - [2.0, 2.0, 2.0] (弱双曲)
   - [0.5, 1.0, 2.0] (逐层递减, 与 toy κ_3=-0.84 层级思想契合)
   - [2.0, 1.0, 0.5] (反向)
   - [1.0, 0.5, 0.5] (中→强)

**启动命令 (单 GPU 网格搜索示例)**:
```bash
# 先准备 Stage 2 codebook (与 Task #84 同一组 emb parquet, 不重跑)
# 然后每个 (c_0,c_1,c_2) 组合:
python3 /home/wlia0047/ar57/wenyu/GeneRec/HG-Rec/train_HG-Rec.py \
    --dataset Musical_Instruments \
    --curvatures 0.5,1.0,2.0 \    # 新加参数, 对应 c_0,c_1,c_2
    --loss_type poincare \
    --epochs 100 --eval_step 1 --patience 20 \
    --ckpt_dir /home/wlia0047/ar57/wenyu/GeneRec/products/task88/train
```
Stage 3 (T5-small) 和 Stage 4 训练复用 Task #84 模板,只替换 Stage 1 的 SID tensor.

---

## 3. 决策触发 (vs Task #84 baseline R@10=0.1020)

| 网格最优 R@10 | 解读 | 决策 |
|--------------|------|------|
| R@10 > 0.115 (+12.7%) | **Idea1 强证实** | 升级路径 2: c_0/c_1/c_2 设为可学习 nn.Parameter + RiemannianAdam |
| 0.105 ≤ R@10 ≤ 0.115 | **Idea1 部分证实** | 路径 2 仍可选, 但 ROI 中等, 看算力预算决定 |
| R@10 < 0.105 | Idea1 失败, 与 Task #84 持平 | 停止路径 2, 写"per-layer 在 HG-Rec 上无显著增益" verdict |
| 网格中 c 全等优于 (c_0, c_1, c_2) 异构 | 退化证据 | 升级路径 2 风险高, 直接 NO-GO 路径 2 |

---

## 4. 预算

| 阶段 | 估算时间 |
|------|---------|
| 代码改造 + py_compile + dry-run 测试 | ~2 h |
| Stage 1 (sentence-t5-base 编码, Task #84 已落盘可复用) | 0 (复用) |
| Stage 2 (HRQ-VAE 6 个 c 组合 × ~30 min/训练) | ~3 h |
| Stage 3 (T5-small 6 个组合 × ~70 min) | ~7 h |
| Stage 4 (eval 6 个组合 × ~30 s) | ~3 min |
| 写 verdict + paper section | ~1 h |
| **总计** | **~13 h GPU** |

**预算 vs Task #84 (1 h GPU)**: 7× 倍, 因为网格搜索本质上是 6 折 ablation.

---

## 5. 风险与缓解

**风险 1**: R11.4 critical — 修改 HG-Rec 上游代码 (`/fs04/ar57/wenyu/GeneRec/HG-Rec/model/hrqvae.py` 和 `train_HG-Rec.py`).
- 缓解: dry-run 先报告改动位置 + R11.3 5 字段决策点 + 仅在 `loss_type=='poincare'` 分支启用新 curvature_list, 'mse'/'l1' 路径不变
- 缓解: 修改前 git tag 一个 backup commit `pre-task88-hgrec-perlayer`, 出问题可回滚

**风险 2**: 量化模块内部 rq_loss 是否兼容 per-layer loss 项, 而不是单一 sum.
- 缓解: 先在 train_HG-Rec.py print 一次 rq_loss 的 shape, 确认 residual 形式与层数对应, 再加 c_i 项

**风险 3**: RiemannianAdam 优化器依赖 geoopt, grid_toys env 未安装.
- 缓解: 本任务路径 1 不涉及 RiemannianAdam (固定曲率), 仅路径 2 需要, 路径 2 是后续任务

**风险 4**: 网格搜索期间 GPU 占用 vs 其他任务冲突 (R7).
- 缓解: 每个组合单独 launch + nohup, GPU 0 启动前先 `nvidia-smi` 检查 0 util, 选择空闲卡

**风险 5**: 期望 R@10 0.115 vs paper HG-Rec 0.1315 (差距 -12.4%) 可能仍不满意, paper-comparison 仍要标注 absolute numbers vary.
- 缓解: paper-comparison verdict 已存在 (Task #94), 路径 1+2 完成后 verdict 引用

---

## 6. 完成度跟踪

- [ ] git backup commit `pre-task88-hgrec-perlayer`
- [ ] 改 hrqvae.py + train_HG-Rec.py 加 curvature_list 参数
- [ ] py_compile 验证 (R4)
- [ ] 网格搜索 6 个组合 (R11.3 决策 c 候选值)
- [ ] 写评测脚本 eval_task88_<组合>.py
- [ ] 6 个组合的 R@10 落盘
- [ ] 写 verdict `verdicts/task88_hgrec_perlayer_curvature_result.md`
- [ ] 升级路径 2 (可选, 看网格结果)

---

## 7. 关键决策点 (R11.3 自决, 等用户/自主记入)

1. 路径选择: 用户已回答 → 路径 1 网格搜索 → 验证后再升级路径 2 ✅
2. 网格候选值: 自主选 6 组 (上面 B 列表), 含一个对照组 [1,1,1] (R2 反对无原因弃 default)
3. Stage 1 编码复用 Task #84: R11.3 自主决策 (不重跑, 防 fallback)
4. Stage 3 (T5-small) seed 是否固定 42: 自主决策 fixed, 减少方差
5. 网格筛选策略: 全部 6 个组合都跑 (R2 禁跳过), 不只看最优
