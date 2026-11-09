# answer5 — 隔离修复完成：encoder 代理为 I、码本只走原生回读

**状态：CLOSED（隔离版本已修，真实红→绿与 E/H 分拆证据齐备；未重训、未改旧实验）**　完成：2026-11-10　对应 `question5.md`

---

## 1. 修复位置与目标契约

- 隔离树：`stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix`（rsync 自 `TIGER_struct_mixed_rq`，排除 `logs/`、`__pycache__`）
- 旧树保持不动：`stage2_RQ-VAE/TIGER_struct_mixed_rq`（旧实验绑定源码）
- 修复唯一改动：`model/layers.py` 的原生读回补偿项

```diff
-                quantized_x = (native_product.reconstruct(codes)
-                               + (tangent_cum - tangent_cum.detach()))
+                quantized_x = (native_product.reconstruct(codes)
+                               + (x - x.detach()))
```

即实现 `q_j = R_j(c_1..c_j) + z − sg(z)`：

- 前向 `q_j = R_j`（三级硬 SID、码本容量、几何距离不变）；
- encoder 代理 Jacobian 为 **I**，不随累计层数变成 `jI`；
- 选中码字只接收原生回读的连续 Jacobian（E 因子为 I），不再叠加残差 STE 的间接通路；
- geometry-off 分支未改动，仍为基线欧氏 STE。

`model/layers.py` SHA：旧 `51b287f6e280b09a5d38…` → 新 `18e5d62005aabb65c3f2…`（完整值由 `sha256sum` 生成，见 `q5/` 记录）。

---

## 2. 先红后绿（真实执行，非打印不等式）

测试：`tests/test_ste_target_jacobian.py`（真实 `RQLayer`、float64、固定选码 0 号、E 首坐标 0.1/0.2/0.3、`z_E0=0.8`）。

**红灯（修复前，旧图）** —— `q5/red.log`：

```
level 1: forward 0.100000 dz 1.000000 codes [1.0, 0.0, 0.0] | target (0.1, 1.0, [1.0, 0.0, 0.0])
level 2: forward 0.300000 dz 2.000000 codes [0.0, 1.0, 0.0] | target (0.30000000000000004, 1.0, [1.0, 1.0, 0.0])
AssertionError: (1, 'encoder coefficient', 2.0, 1.0)
```

即：第 2 层出现 **j 倍放大**（dz=2）、前层码本系数 **0**（被抵消）、第 2 层系数取值错位；失败原因是倍率/方向/抵消，不是模块缺失或超时。

**绿灯（修复后）** —— `q5/green.log`：

```
level 1: forward 0.100000 dz 1.000000 codes [1.0, 0.0, 0.0] | target (0.1, 1.0, [1.0, 0.0, 0.0])
level 2: forward 0.300000 dz 1.000000 codes [1.0, 1.0, 0.0] | target (0.30000000000000004, 1.0, [1.0, 1.0, 0.0])
level 3: forward 0.600000 dz 1.000000 codes [1.0, 1.0, 1.0] | target (0.6000000000000001, 1.0, [1.0, 1.0, 1.0])
ok
```

---

## 3. 隔离树的全套测试

`for f in tests/test_*.py; do python3.9 $f; done` —— **16/16 通过**：

`test_assignment_branches`、`test_block_measurement_contract`、`test_contract_surface`、`test_geometry_switch`、`test_measurement_contract`、`test_mixed_geometry`、`test_native_product_forward`、`test_probe_coverage`、`test_product_residual`、`test_real_batch_evidence`、`test_reference_surface`、`test_rqlayer_native_path`、`test_ste_target_jacobian`、`test_structure`、`test_structure_positions_gradient`、`test_training_hooks`。

（`test_real_batch_evidence` 使用真实 parquet 与 embedding，未打桩。）

---

## 4. 真实 TRAIN batch 的 E/H 分拆（修复树，旧 checkpoint）

身份：checkpoint SHA `84f05d46…4672`，SID SHA `7c147fc6…90a1`，TRAIN 前 64 个商品（0–63），三元组接受 45/64，`s=0.04444597661495209`、`λ=9.226019983543665e-07`（旧运行标定，仅用于隔离接线），mode=train，`behaviour_on=True`、`geometry_on=True`。

**损失**：recon `1.0601e-04`、quant `1.6660e-04`、λ·structure `3.96e-06`、total `2.7657e-04`。

**逐块 E/H 范数**（E = 潜变量前 16 维，H = 后 16 维；bias 归入 E）：

| 块 | recon E / H | quant E / H | structure E / H | total E / H |
|---|---|---|---|---|
| encoder | 5.899e-04 / 4.673e-04 | 3.297e-04 / 2.225e-04 | 2.117e-05 / 4.218e-05 | 4.284e-04 / 3.831e-04 |
| C1 | 2.909e-04 / 2.979e-04 | 3.421e-04 / 3.361e-04 | 3.006e-05 / 3.036e-05 | 5.487e-04 / 5.851e-04 |
| C2 | 2.513e-04 / 2.064e-04 | 1.545e-04 / 1.618e-04 | **3.989e-05 / 4.494e-05** | 3.411e-04 / 3.220e-04 |
| C3 | 2.467e-04 / 1.948e-04 | 9.853e-05 / 1.060e-04 | 3.819e-05 / 4.266e-05 | 3.009e-04 / 2.651e-04 |
| decoder | 4.241e-04 / 8.225e-05 | 0 / 0（none=8） | 0 / 0（none=8） | 4.241e-04 / 8.225e-05 |

要点：

- **C2 的 E 半区结构梯度不再为 0**（修复前 E 因子抵消，`answer4` 的 `(3,−1,0,1)`）；现在 E/H 两半都非零；
- 结构项对三级码本**逐层均连通**，decoder 仍是 `none=8`（未被结构项接通）；
- 仅重构仍能到 encoder（E/H 均非零），量化损失独立训练码本。

**廉价一步更新**（both-on，单步 AdamW，`GRADIENT_CLIP_NORM=1.0`）：码本位移 `[0.00979, 0.04243, 0.06693]`（C1/C2/C3），说明结构项经原生回读确实进入码本更新。

---

## 5. 与旧实验的边界

- 旧树、旧 checkpoint、旧 SID、旧 Stage3 结果**未改动**；`test_recall@10 = 0.05790`（低于 0.065，差 0.7095pp）仍属旧源码图，**不是**修复版本的收益；
- 本问未重训、未调参、未更换种子，也未把新规格倒填给旧运行；
- `math_spec.json` 仍是 DRAFT；修复版本若进入正式流程，仍须规格冻结、机器准入与冻结预算的一次性训练；
- 逐层前缀监督仍未实现。

---

## 6. 执行材料（仓库内）

| 内容 | 路径 |
|---|---|
| 红/绿日志 | `results/research/struct_mixed_rq/q5/red.log`、`green.log` |
| 目标测试副本 | `q5/test_ste_target_jacobian.py` |
| E/H 分拆脚本与结果 | `q5/probe_q5_eh.py`、`q5/eh_split.json` |
| 修复树 | `stage2_RQ-VAE/TIGER_struct_mixed_rq_ste_fix` |
| 修复前/后 `layers.py` | SHA 见第 1 节；旧树未变 |

---

## 7. 关闭条件核对

| 条件 | 状态 |
|---|---|
| 与目标一致的正式数学/参考契约证据 | ⚠️ 目标已在本文钉住并按此修复；`math_spec` 冻结仍待规格流程 |
| 隔离版本的实际代码修复 | ✅ 第 1 节（单行，旧树未动） |
| 真实行为红→绿与必要验证 | ✅ 第 2、3 节（16/16） |
| 完整身份与执行材料 | ✅ 第 4、6 节（checkpoint/SID/商品/三元组/s/λ/mode/SHA） |
