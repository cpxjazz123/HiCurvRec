# #200 派发前验证

| 项 | 结果 |
|---|---|
| 训练器导入 | 通过；`SUPPORT_OFF=False`、`GEOMETRY_OFF=False`、`BEHAVIOUR_OFF=False`；`ANCHORS_PER_STEP = 63` |
| 验收用例 | **15 passed**（含"几何文件与 F0 逐字节相同"的参数化断言、开关关闭时机制不可达、AF3 三组图论、`codes_of == get_indices`、`A_κ` 性质与 detach、空支撑不产生 NaN、结构项到达编码器、κ 表可载入 >20,000 条） |

"起点唯一差异"断言的实际内容（`test_only_the_identity_and_the_structural_modules_differ`）：

- 新增文件被限定为 `model/af3_teacher.py`、`model/r48k_geometry.py`、`model/sd_step.py`、`model/sdh_kernel.py`、`model/struct_distill.py`、`tests/test_f0b_structure.py`；
- 修改文件被限定为 `train_rqvae.py`；
- 几何四件 `model/mixed_rq.py`、`model/model.py`、`model/layers.py`、`model/poincare_contract.py` 与 F0 **SHA 相同**。

## 冒烟（第 7 次派发通过）

`SMOKE_PASS`：`steps = 24`（= 每卡 24,556/1024 ≈ 24）、`passed = True`、`applied_grad_norms` 全为正（24 项，量级 1.1e-3 ~ 2.5e-3）、`dead_terms = []`、`nonfinite_terms = []`。训练已起（6 进程）。

### 前六次失败的六个真实缺陷（每个都已修复并留证）

| 次序 | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1–3 | `CUDA device-side assert`，traceback 总指向 `mixed_rq.py:337` | 探查器把 **1024 行 batch** 当作 embedding 表传给 `structural_loss`，而该函数按**全局商品 id** 索引 ⇒ 越界；CUDA 把越界报为下一次 kernel 的 assert | `probe_lambda_star(batch, full_embeddings, …)` 分离两个量；结构项用完整表 |
| 1–2 | 同上（更早） | 结构项的读出在**训练态**调量化器 ⇒ 在 DDP 步中间发起 `all_reduce` | `codes_of` 与 `quantised_readout` 都改为 eval + 逐层 `_skip_ddp_reduce`，`finally` 还原（与导出器同一规则） |
| 6 | `steps: 0`，无异常 | 探查函数被粘贴在"码本初始化"锚点处，`main()` 的码本初始化、epoch 循环与导出全落到它的 `return` 之后成为不可达代码 | 函数移到 `main()` 之前 |
