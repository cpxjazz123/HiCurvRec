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
