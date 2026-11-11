# #183 R48κE：表示级曲率（`exp_0^κ`）已实现，几何测试 **7 passed**

- 新树 `stage2_RQ-VAE/TIGER_r48ke_rq`（自 #182 树复制，`ARM_NAME="TIGER_r48ke_rq"`、`ARM_VARIANT="R48KE"`，路径全部标记目录推导）。
- `model/r48ke_geometry.py`：
  - `expmap0(z, κ) = tanh(√κ‖z‖)·s/(√κ‖z‖)·z`（`s = 1−1e-6` 保证 `κ‖z'‖² < 1` 严格成立；零向量取极限 1）；
  - `logmap0`：**按推导给出的精确逆** `a = atanh(√κ‖p‖/s)`、`z = p·a/(√κ‖p‖)`，与 `expmap0` 往返到 1e-9；
  - `distance`：asinh 形式（与 #179 同一公式），重合/近重合梯度有限；
  - `pair_kappa = √(κ_i κ_j)`、`behaviour_distances`（两态：`GEOMETRY_OFF` 或 `EMBED_ONLY` → `torch.cdist`；否则先映射再按对曲率测距）、κ 表按标记目录加载。
- `behavior_contrastive_loss` 已接 `r48ke_geometry.behaviour_distances`。
- 测试 `tests/test_r48ke_geometry.py` 7 项全过：映射严格球内且单调、零向量极限、**往返精确（rtol 1e-9）**、`GEOMETRY_OFF` 不改表示、重合/近重合距离有限、`EMBED_ONLY` 返回欧氏矩阵、κ 表加载与几何平均。

## 尚未完成（接线）

1. **量化器输入也必须是 `z'`**（Issue 要求"量化/残差/对比全部作用在 `z'_i`"）：需要在 `RQVAE.forward` 的 `encoder→rq` 之间按行 κ 做 `expmap0`（用与 #180 同款 `set_batch_kappa` 钩子，并让 `_encode_all` 在导出时同样注册）。
2. `GEOMETRY_OFF` 与 R48 损失**逐位一致**的三态测试、三处同一 κ 的绑定断言、一 epoch 冒烟，然后派发。

## 量化器输入也映射 + 绑定断言（本轮，**21 passed**）

- `r48ke_geometry` 增加 `set_batch_kappa/batch_kappa/map_representation`：训练器在 `model(batch)` 前后注册/清除批内逐行 κ（`GEOMETRY_OFF` 时注册 `None`）。
- `model/model.py` 的 `RQVAE.forward` 在 `encoder` 与 `rq` 之间调用 `map_representation`，于是**量化、残差、对比三处作用在同一张 `z'` 与同一 κ 表上**（Issue 验收 4）。
- 训练器：数据集改为 `(embedding, id)`、循环解包 `batch, batch_ids` 并注册 κ——与 #180 同一套 loader 契约改动。
- 新测试 `tests/test_r48ke_binding.py` 4 项：钩子注册时模型输出等于 `expmap0(encoder(x), κ)`、映射后严格在各自球内、对比用的 `pair_kappa` 与查表一致（对角等于 κ_i）、`GEOMETRY_OFF` 同时让映射与距离退回欧氏。全树 **21 passed**。

## 冒烟 **PASS** → 正式训练已启动

`SMOKE_PASS`；`applied_grad_norms` 六个全为有限正值（0.0554–0.0973）。冒烟单 epoch `collision=0.877822`（早期值仅记录）。训练已起（6 进程），epoch 576/3000、loss 2.42e-4、`behavior_weight = 0.0`（1500 前为 0）。

## 作废（owner 2026-11-11 指令）

owner 明确「不需要做 #183，直接 #184」。本臂 Stage2 已完成（`a0ca70a56`，碰撞 0.079148），但 **Stage3 在此指令下中止**（运行中的 harness 已停止，未产出 `test_final.json`），**不做 §6b 归档**、不关闭 Issue。原因是本次主线新要求：**不允许行为对比正负样本学习**，而 #183 与 #179/#182 同属源–目标对对比，故整个方向作废（#184 正文亦如此记载）。已产出的 Stage2 产物保留在 `results/stage2_RQ-VAE/TIGER_r48ke_rq/` 供审计，但不作为结论。
