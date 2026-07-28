# Task #192 — β 剂量扫描 (Instruments, 4 臂) + 机制因果升级

> **任务目的**: 通过 β 剂量-反应 + encoder freeze 干预两侧验证, 把 Task #191 (corr=0.861) 的相关性升级成因果.

> **完成日期**: (in progress)
> **状态**: 🟡 待启动 (等 Task #188 Phase 4 完成腾出 GPU)

---

## 1. 背景

### 1.1 既有证据 (Task #189 → #191)
- **Task #189**: 诊断 hrqvae 时间序列 (baseline + 6 epoch 系列), 建立"几何诊断"框架.
- **Task #190**: 6 epoch × 3 层码字/残差比值 → L0/‖r0‖ 单调↑ +24%, L2/‖r1‖ -6% (匹配). 假设 L0 码本超调成立.
- **Task #191**: 98 个 epoch ckpt × L0_err = ‖z − e_L0‖ → 单调↑ +207%, corr(L0_err, collision)=**0.861**. **机制: L0 K=64 码本容量受限 → 量化误差持续放大 → 残差↑ → 全层 collision 涨.**

这都是**相关性证据** — `corr(L0_err, collision) = 0.861` 不能排除隐变量 (loss 平衡漂移等).

### 1.2 用户 2026-07-25 19:55 因果设计提案

**主实验 (Task #192)**: β 剂量扫描 → 改变码本追赶速度 → 看 collision 劣化是否跟着变 (剂量-反应).

**补充实验 (Task #193)**: 冻结 encoder → 消除漂移源 → 看 collision 是否停止劣化.

两侧都指向同一机制 → 相关性升级成因果.

### 1.3 β 选择理由

RQ-VAE 损失 = `recon_loss + β · quant_loss`. β↑ 加大 quant_loss 权重, 码本更新更积极 → 码本追赶 encoder 的速度更快. β 剂量扫描本就是改变"码本 vs encoder 速度比"的直接手段.

---

## 2. 实验设计

### 2.1 Task #192 — β 扫描 (4 臂 × 1 dataset)

| 臂 | β | 码本追赶速度 | 备注 |
|----|----|--------------|------|
| A | 0.25 | 最慢 | 极端欠冲: 码本几乎不动 |
| B | **0.5** | 中等 | task188 baseline 周围 |
| C | 1.0 | 快 | task188 baseline 中有 (--save_limit=50 已经用) |
| D | 2.0 | 最快 | 极端超冲: 码本强压 encoder 输出 |

**注**: task188 --save_limit=50 训练用的是 **β=1.0**, 所以 C 臂是新跑一次作对照, 但产物应该跟 task188 兼容 (即 codebook size / curvature / loss_type / num_emb 全部相同, 只 β 不同).

### 2.2 实验变量

**唯一改动**: `beta` 参数 (从 args.beta 读取)

**保持不变** (vs task188/hrqvae_save_limit50):
- `loss_type='poincare'`, `kmeans_init=True`, `sk_epsilons=[0,0,0]`, `kmeans_iters=1000`
- `num_emb_list=[64,128,256]`, `e_dim=32`, `quant_loss_weight=1.0`
- `layers=[512,256,128,64]`
- `epochs=1000`, `batch_size=1024`, `lr=0.001`
- `--save_limit 50` (让每 5 epoch 留一个 snapshot, 跟 task188 完全一致)
- 数据集: `Instruments` (9922 items)

### 2.3 启动命令

```bash
# β=0.25 on cuda:0
python3 scripts/task188_stage1_hrqvae.py \
    --device cuda:0 \
    --beta 0.25 --epochs 1000 --save_limit 50 \
    --output_dir products/task192/hrqvae_beta0.25 \
    --data_path dataset/Instruments/item_emb.parquet \
    ...

# β=0.5 on cuda:1, β=1.0 on cuda:2, β=2.0 on cuda:3
```

4 臂在 4 GPU 同时跑, 共享 TRITON_CACHE_DIR (task192 一致性).

### 2.4 记录三条曲线 (per epoch, median + mean)

1. **`L0_err = ‖z − e_L0‖`** (中位数, 同 Task #191)
2. **`collision_rate`** (从 trainer 内 `eval()` 步取, 直接写日志)
3. **`‖z‖` 中位数** (encoder 输出范数, 看漂移本身) + **`ρ_L0` 中位数** (码字范数, 看码本有没有涨)

### 2.5 Task #193 — Encoder freeze 补充 (1 臂)

```python
# 在 hrqvae_trainer.py 的 training loop:
if epoch == 60:  # collision 最低点附近
    for p in model.encoder.parameters():
        p.requires_grad = False
    # 验证 encoder 不再更新, 但 codebook / decoder 继续
    print(f"epoch {epoch}: encoder FROZEN")
```

- β 选 0.5 (任务 #192 B 臂), `epochs=150` (60 frozen + 90 自由冻结)
- 其他配置同 2.2.
- 目的: 看冻结 encoder 后 collision 是否停止劣化.

---

## 3. 判据 (Task #192 β 扫描)

按用户 2026-07-25 19:55 提出的剂量-反应判据:

| 结果 | 结论 |
|------|------|
| β↑ → L0_err 增幅↓ → final/min collision 比值↓, **单调** | ✅ **机制通过干预确认** |
| 4 臂 final/min 比值差不多 | ❌ β 不是主因, 回到相关性, 重找 |
| 非单调 | ⚠️ 有别的因素在起作用 (例如 optimizer / lr sched) |

### 3.1 数值目标 (用户期望的"漂亮曲线")

- **baseline final/min = 1.37** (8.62% → 12.39%, 来自 task #188 同时性数据)
- β=2.0 那臂应降到 ~1.1 (码本追上, collision 几乎不劣化)
- β=0.25 那臂应升到 ~1.6 以上 (码本严重欠冲, 加速劣化)
- 跨 β 形成单调曲线 → 机制因果断

### 3.2 报两个数 (避免 β 混淆)

用户特别提示: β 不只改变追赶速度, 还改变 loss 平衡 → `min_collision` 本身会变.

- **`min_collision`** — 绝对水平 (跨 β 会变, 正常)
- **`final/min` 比值** — 跨 β **不变部分**, 这才是机制判据

### 3.3 顺带产出: ρ 和 ‖z‖ 跨 β 的变化

- β↑ 推码本追上 encoder → ρ_L0 应该跟着 ‖z‖ 涨
- 若 ρ_L0 涨 → "半径可被 β 推高" (温和选项)
- 若 ρ_L0 卡在 λ=2.0 → "必须显式约束" (需要继续 kappa 实验)

---

## 4. Task #193 判据 (freeze 补充)

| 结果 | 结论 |
|------|------|
| encoder 冻结后 L0_err 停止增长 + collision 不再劣化 | ✅ **encoder 漂移是因** |
| collision 照样涨 | ❌ 因在别处 (码本动力学 / decoder) |
| L0_err 停 + collision 仍涨 | ⚠️ 多因素, 码本/decoder 也有贡献 |

---

## 5. 预算

| 阶段 | 估算时间 |
|------|----------|
| Stage 1 4 臂并发 (1000 epoch on 4 GPU) | ~30-45 min |
| Stage 1 encoder freeze 1 臂 (150 epoch) | ~5 min |
| Task #191 风格诊断 (L0_err per epoch × 4 ckpt dirs) | ~5 min |
| Verdict + 4 臂剂量-反应图 + freeze 解读 | ~10 min |
| **总计** | ~50-65 min |

---

## 6. 风险与缓解

**风险 1**: β=0.25 极端欠冲可能让码本完全不动 (L0_err / collision 反而异常低, 因为 encoder 也学不动) → 解读时区分"码本 freeze" vs "码本欠冲"。
**缓解**: 报 `min_collision` 跟 `final_collision` 分开, 报 `L0_err` 单独曲线 (码本欠冲会让 L0_err 单调↑ 到天花板).

**风险 2**: encoder freeze 实验在 epoch 60 freeze 后, 如果之前 60 epoch 已经劣化很多, freeze 后 collision 难大幅回落 → 仍能验证"冻结止涨" (一字之差即可).
**缓解**: 选 min_collision epoch (可能略晚于 60) 作为 freeze 点, 起点对齐到 freeze 时刻 = 让 Δcollision < 0 来判.

**风险 3**: 4 臂同时在 4 GPU 跑 --save_limit=50, 每个 ckpt 22 MB × 200+ epoch × 4 臂 = ~17 GB 磁盘.
**缓解**: 跑完立即删各 β 目录 (保留 best_loss + best_collision + 每个 β 一份 final epoch snap).

---

## 7. 完成度跟踪

- [ ] Stage 1 4 臂并发跑 (β=0.25/0.5/1.0/2.0)
- [ ] Stage 1 encoder freeze 1 臂跑 (epoch 60 freeze)
- [ ] 诊断脚本 (类 task191): 4 臂 × L0_err_per_epoch × collision_per_epoch
- [ ] 4 臂 dose-response 图 + 读数
- [ ] encoder freeze 读数 (frozen 前后 L0_err / collision 趋势对比)
- [ ] Verdict (`verdicts/task192_193_mechanism_causal_verdict.md`)
- [ ] 更新 loop.md §16

---

## 8. 跟其他 task 的关系

- **Task #188 Phase 4** 也占 GPU, **必须**等 Phase 4 完成释放 GPU 再 fire β 扫描. 不能并发 (GPU 满载).
- 复用 task188 hrqvae_trainer 启动器, 改 --beta 参数即可.
- 跟 Task #191 (L0_err per epoch) 数据路径完全一致, 诊断脚本可重写 task191 做 4 dir 版本.

---

## 9. 决策触发 vs 机制假设

| 试验现象 | 假设 H1 (机制 = L0 容量受限) | 假设 H2 (机制 = 别的) |
|----------|-------------------------------|----------------------|
| 4 臂剂量-反应单降 | ✅ H1 立 | |
| 4 臂剂量-反应无差别 | | ✅ H2 立 |
| 4 臂非单调 | ⚠️ H1 局部立 | |
| encoder freeze 止涨 | ✅ H1 立 | |
| encoder freeze 不止涨 | | ✅ H2 立 |

两侧都立 → H1 因果坐实.
