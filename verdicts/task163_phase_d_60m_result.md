# Task #163 Phase D + 60M verdict — NO-GO (FreeCurv codebook 坍缩无法靠 ORC-init 解决)

> **result**: Phase D Stage 1 (真 ORC-derived θ_init) 训练后期 κ 仍饱和到 κ_max=2.0，codebook 坍缩到 ~27 unique (L0,L1,L2) tuples，Stage 2 SID col 3 退化成 cumulative occurrence counter (0-5031)。Stage 3 (T5-small 60M) 启动即崩溃 (CUDA device-side assert: token IDs 354-5384 > vocab_size=5032)。**NO-GO 结论：ORC-init 不能解决 FreeCurv 在 Instruments 上的根本性 codebook collapse，跟 #142 verdict 完全一致**。

---

## 1. 任务目的

验证 Task #163.5 (真 ORC 计算) 推出的 θ_init = [-0.0114, +0.2375, +0.2375] (对应 L0 κ=+0.766, L1 κ=+1.350, L2 κ=+1.345) 能否打破 FreeCurv 在 Instruments 上 Stage 1 后期 κ 饱和、codebook 坍缩的死循环（#142 verdict 标记为"架构根本问题"）。

---

## 2. 执行时间线

| 阶段 | 时间 | 结果 |
|------|------|------|
| Phase D Stage 1 (1000 epoch) | jul 24 训练中 | 训练到 ep 764 崩 NaN，best_loss ckpt @ ep 352 保留 (loss=1.253) |
| Phase D Stage 2 (SID 推断) | jul 25 00:47 | 产出 SID (9922, 4)，表面"all unique" |
| **Phase D Stage 2 SID 事后审计** | jul 25 01:08 | **FAIL**: col 0-2 各仅 3 unique values，col 3 有 5032 unique values 0-5031 → codebook 严重坍缩 |
| Phase D Stage 3 (T5-small 60M) | jul 25 01:06 启动 | **FAIL**: CUDA `device-side assert triggered` @ T5 encoder `cache_position = torch.arange(...)`，根因 = token IDs OOB (354-5384 > vocab_size=5032) |

---

## 3. 关键指标

### 3.1 Stage 1 κ_history (best_loss ckpt @ ep 352, 训练崩溃前)

| Layer | θ_init | κ_init | κ @ ep 352 | 解读 |
|-------|--------|--------|-----------|------|
| L0 | -0.0114 | +0.766 | +1.99 (saturated) | 接近 κ_max=2.0 |
| L1 | +0.2375 | +1.350 | +2.00 (saturated) | **完全饱和** |
| L2 | +0.2375 | +1.345 | +2.00 (saturated) | **完全饱和** |

ORC 偏置 (+0.766 ~ +1.350) 全部 saturated to κ_max = +2.0，模型把曲率推到了上限。

### 3.2 Stage 2 SID 实际内容 (致命问题)

| Column | n_unique | min | max | 解读 |
|--------|----------|-----|-----|------|
| col 0 (L0) | **3** | 12 | 31 | **只用 3/32 个 codebook** |
| col 1 (L1) | **3** | 6 | 49 | **只用 3/64 个 codebook** |
| col 2 (L2) | **3** | 101 | 110 | **只用 3/256 个 codebook** |
| col 3 (dedup) | 5032 | 0 | 5031 | 当成第 4 个独立 identifier |
| **全局 max** | - | - | **5031** | vocab_size 必须 ≥ 5032 |

**Codebook 坍缩程度**：~27 unique (L0, L1, L2) tuples, 9922 items 全靠 col 3 累积 occurrence counter 区分 → 严重信息瓶颈，T5 几乎无 SID 结构可学。

### 3.3 Stage 3 CUDA crash 根因

`data/dataset.py:71` 的 `item2code()` 把每个 code 加 offset:
```python
offsets[i] = code[i] + sum(codebook_size[0:i]) + 1
```

对于 codebook=[32,64,256,1]：
- offsets[3] = code[3] + (32+64+256) + 1 = code[3] + 353

Phase D 的 col 3 = 0~5031 → offsets[3] = 354~5384 > vocab_size=5032 → embedding lookup OOB → CUDA device-side assert。

---

## 4. 分析解读

### 4.1 跟 Task #142 (free-curv codebook collapse) verdict 完全一致

#142 结论：
> 坍缩是架构根本问题，R137 fix 解不了。Sinkhorn 无法恢复。修复方向：kmeans_init in geodesic space / Sinkhorn during train / EMA。

#163 Phase D 用 ORC-init 替代 R137 fix 同样不解决问题 → #142 结论正确：自由曲率在 Instruments 上**结构性坍缩**，无法用 init 策略修复。

### 4.2 κ-Stereographic (#164) 也无法解决

#164 短测试 (200 epoch, θ_init=[0,0,0]) 用 κ-Stereographic 距离公式重跑 → κ 同样 saturated 到 κ_max (反方向 -2.0)，codebook 利用率 <5%。**距离公式切换不是修复方案**。

### 4.3 自由曲率 RQ-VAE 三连败 (#142, #163 Phase D, #164)

三种独立修复尝试（kmeans-init geodesic + dead code reset / ORC bias init / κ-Stereographic 距离公式）全部失败 → FreeCurv on Instruments 的坍缩是**架构级别问题**，应放弃该方向。

---

## 5. 后续建议

- **NO-GO**：放弃 FreeCurv on Instruments。
- **建议方向切换**：
  - **回退到 vanilla RQ-VAE** (fixed c=1, poincare_distance 跟 #84 baseline 一致)，SID 100% 利用率，Stage 4 R@10=0.10+ 已知
  - **不再投入 GPU 到 FreeCurv 变体**
- **保留的学习**：
  - ORC-init 计算管线 (#163.5) 是有效工具，可用于其他场景（fixed-c 初始化、LR scheduler warm start）
  - κ-Stereographic 距离公式 (#164) 是干净的公式实现，codebook collapse 是 saturating 训练动力学问题不是公式 bug

---

## 6. 产物清单

| 路径 | 状态 | 用途 |
|------|------|------|
| `products/task163/phase_d_rqvae_real_orc/jul-24-2026_22-32-43/best_loss_model.pth` (4.5MB) | ✅ | Phase D Stage 1 best_loss ckpt @ ep 352 |
| `products/task163/phase_d_rqvae_real_orc/jul-24-2026_22-32-43/kappa_history.json` (9KB) | ✅ | κ 训练曲线 (验证饱和到 +2.0) |
| `HG-Rec/dataset/Instruments/Instruments_t5_rqvae_orc_init.npy` (310KB) | ⚠️ 存在但**不可用** | SID col 3 退化成 cumulative counter |
| Stage 3 launchers (Phase D + 60M) | ❌ | 已修复 cwd bug，但 SID 不可用，无意义重启 |

---

## 7. 完成度

- [x] Phase D Stage 1 (best_loss @ ep 352)
- [x] Phase D Stage 2 (SID 产出，但审计发现坍缩)
- [x] **Phase D Stage 2 SID 审计**（col 3 退化成 counter, 5032 unique）
- [x] Phase D Stage 3 启动调试（CUDA crash, 根因 = token OOB）
- [x] NO-GO 判定 + 跟 #142 verdict 交叉验证
- [x] verdict write
- [x] 提交至 P5 paper section

