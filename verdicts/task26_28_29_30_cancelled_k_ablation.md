# ❌ Cancelled — Task #26/#28/#29/#30 K ablation + #31-#35 计划

> **取消时间**: 2026-07-19 15:38
> **取消原因**: 用户反馈 "K ablation 不直接回答 Task #27 的 kNN preservation 假设, 设计逻辑有问题"
> **结果**: 4 个 Stage 2.1 已 kill, GPU 完全释放. 9 个 task list 项已 deleted. logs/ 已清空. descriptions/ 标 ❌ CANCELLED.

---

## 1. 取消范围

| 任务 | 状态 | 进展 | 处理 |
|------|------|------|------|
| Task #26 K=512 (原 #108) | ❌ Cancelled | Stage 2.1 step 7725/20000 (38.6%), 无 checkpoint | kill PID 2669803 |
| Task #28 K=128 (原 #110) | ❌ Cancelled | Stage 2.1 step 3988/20000 (19.9%), 无 checkpoint | kill PID 2709755 |
| Task #29 K=64  (原 #111) | ❌ Cancelled | Stage 2.1 step 3602/20000 (18.0%), 无 checkpoint | kill PID 2718743 |
| Task #30 K=384 (原 #112) | ❌ Cancelled | Stage 2.1 step 3078/20000 (15.4%), 无 checkpoint | kill PID 2732885 |
| Task #31 K=96  | ❌ Cancelled (未启动) | 计划 Phase D | task deleted |
| Task #32 K=192 | ❌ Cancelled (未启动) | 计划 Phase D | task deleted |
| Task #33 K=320 | ❌ Cancelled (未启动) | 计划 Phase D | task deleted |
| Task #34 K=256 seed=7 | ❌ Cancelled (未启动) | 计划 Phase D seed variance | task deleted |
| Task #35 Phase E (seed 200/999, H=2/4) | ❌ Cancelled (未启动) | 计划 Phase E | task deleted |

---

## 2. 取消的根本原因 (用户反馈)

### 2.1 原假设链
```
K 越大 
  → 码字更多 
  → 每个 item 找到更精确的码字 
  → 邻域保留越好 
  → Recall 越高
```

### 2.2 实际实验测的是什么
```
codebook_coverage (L0/L1/L2 frac_layer_coverages_step) 与 R@5 的关系
```

### 2.3 用户的核心批评
> **"你测的是'容量', 但应该测的是'邻域保留'. 两个不是一回事."**

具体:
- K=512 有 L0 cov=0.668 (未充分利用码字)
- K=64 有 L0 cov=1.000 (完全利用)
- 哪个邻域保留更好? **不知道**, 因为没测 k-NN preservation
- 实验数据不能回答 Task #27 的核心问题

---

## 3. 取消时 GPU 状态

```
GPU 0: 0 % / 0 MiB (释放)
GPU 1: 0 % / 0 MiB (释放)
GPU 2: 0 % / 0 MiB (释放)
GPU 3: 0 % / 0 MiB (释放)
```

所有 4 个 RQ-VAE worker 进程已 `pkill -9`, GPU 完全空闲.

---

## 4. 已删除/标注的资源

| 类型 | 处理 |
|------|------|
| `logs/task26_s2_train_k512/` + `.log` | ✅ 已 rm (无 checkpoint, 无意义) |
| `logs/task28_s2_train_k128/` + `.log` | ✅ 已 rm |
| `logs/task29_s2_train_k64/` + `.log` | ✅ 已 rm |
| `logs/task30_s2_train_k384/` + `.log` | ✅ 已 rm |
| `descriptions/task26/28/29/30_codebook*_ablation.md` | 🔖 标 ❌ CANCELLED, 文件保留作历史 |
| `descriptions/task27_plan_bcd_sample_expansion.md` | 🔖 标 ❌ CANCELLED |
| 9 个 task list 项 | ✅ deleted |
| `products/` (空, Stage 2.1 未完成) | 无操作 |
| `verdicts/` (空, 无 verdict 写) | 无操作 |
| `loop.md §16` | ✅ 改为 "(空 — 4 个 K ablation 已取消)" |
| `loop.md` 决策日志区 | ✅ 加取消说明 |

---

## 5. Task #27 的状态

**保留**: Task #27 (邻域排序质量 vs 下游 Recall 相关性实验, 原 #109) 本身的有效性不依赖 K ablation.
- n=6 历史分析已完成 (verdict `verdicts/task27_neighborhood_quality_result.md`)
- Spearman ρ=-0.667 (n=6, p=0.148) 非统计显著 (用户已确认)
- 脚本 `scripts/task27_knn_quality_recall.py` 仍可用于历史数据

**不再扩展**: Plan B/C/D 把 n 推到 21 的扩展方案已取消. 后续如果需要扩 n, 必须重新设计实验:
- **直接测 kNN preservation**, 而不是测 K
- 数据点应该变化"能直接影响 kNN preservation 的因素" (例如: 编码器架构, 训练步数, 嵌入维度, 度量空间类型)
- 不要变化"间接因素" (例如: K 值 — 它的影响未直接映射到 kNN preservation)

---

## 6. 后续建议

需要重新设计"邻域保留 vs Recall"的实验, 关键变量应该直接操作 kNN preservation, 而不是间接因素.

可能的实验方向 (需用户确认):
1. **冻结 RQ-VAE, 训练集不同**: 同一 RQ-VAE 编码 80% vs 100% 训练集 → 不同 SID → 测 kNN preservation 差异
2. **编码后处理**: 在固定 RQ-VAE 上, 加 / 不加 Hamming 距离后处理 → kNN preservation 提升 vs 不变 → 测下游
3. **注入噪声**: 在 SID 上注入随机 bit → kNN preservation 强制下降 → 测下游 Recall 是否同步下降 (直接因果验证)
4. **简单随机基线**: 完全随机的 4-digit SID (无 RQ-VAE) → kNN preservation ≈ 0 → 下游 Recall 接近 0 → 直接证明两者关系

**当前任务已完成, 等用户指示下一步**.

---

## 7. 时间损失统计

| 项目 | 数量 |
|------|------|
| 总 GPU 占用 | 4 张 A40 × ~1.2h = ~5 GPU·小时 |
| 浪费 Stage 2.1 进度 | K=512: 38.6% / K=128: 19.9% / K=64: 18.0% / K=384: 15.4% |
| 未产生 checkpoint | 4 个 tokenizer 全部无产物 |
| 未写 verdict | 4 个 |
| 节省资源 (vs 跑完) | 4 × (3h Stage 2.1 + 6h Stage 3 + 15min Stage 4) ≈ 36 GPU·小时 |

**结论**: 早取消省了 ~31 GPU·小时 (~7.75 张·天).
