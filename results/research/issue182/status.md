# #182 R48κ1：三态开关与逐层距离已接线（测试运行中）

- 新树 `stage2_RQ-VAE/TIGER_r48k1_rq`，自**冻结起点 `TIGER_BEHAVIOR_R48`** 复制；继承的绝对路径已全部改为标记目录推导（`ARM_NAME="TIGER_r48k1_rq"`、`ARM_VARIANT="R48K1"`、`RESULT_DIR=REPO_ROOT/results/stage2_RQ-VAE/ARM_NAME`、torchrun 由 `sys.executable` 推导），**树内 `/home/` 计数 0**。
- `model/r48k_geometry.py` 从 #179 树复制（该模块已与冻结 `poincare_contract.distance` 在 κ=0.5/1.3/4.0 逐元素核对过），并新增第三态开关 **`LEVEL1_ONLY`（出厂 True）**。
- `behavior_contrastive_loss` 逐层选择距离：`GEOMETRY_OFF` 或 `level>=1 and LEVEL1_ONLY` → `torch.cdist`；否则 `behaviour_distances`（逐对曲率）。三态语义：
  - `GEOMETRY_OFF`：三层全欧氏 = **R48**；
  - `LEVEL1_ONLY`：第 1 层曲率、第 2/3 层欧氏 = **本臂**；
  - 两者都关：三层全曲率 = **#179**。
- 新测试 `tests/test_r48k1_levels.py` 三项：仅第 1 层弯曲（第 2 层与 `torch.cdist` 逐位相等）、全曲率时三层都与欧氏不同、`GEOMETRY_OFF` 时三层都与欧氏**逐位相等**。

## 测试：**10 passed / 0 failed**

三项三态测试全过（`LEVEL1_ONLY` 时第 1 层弯曲且第 2 层与 `torch.cdist` **逐位相等**；全曲率时三层都与欧氏不同；`GEOMETRY_OFF` 时三层都与欧氏**逐位相等**），另加 #179 移植来的 7 项 `d_κ` 性质测试（原点恒等、对称、`d(p,p)≈0`、投影球内、κ=1 与冻结参照一致、`GEOMETRY_OFF` 与 `cdist` 逐位、κ 表加载查表）。

下一步：一 epoch 真实入口冒烟（含"三层 logits 均非恒定"检查），通过后派发 Stage2。
