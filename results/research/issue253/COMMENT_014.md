## C7/C12 真实四卡同一步双分支等价 PASS

标准四卡 launcher（端口 50202、global4096、seed42）跑完一次真实 epoch，4 rank 全部 `status=pass`，每 rank 真实 batch 1024 项、完整 epoch 结束、参数哈希 4 rank 一致、导出 24587 条 SID 且同 checkpoint 重导出完全一致。证据 `results/research/issue253/validation/issue253_validation_4164325_581130150/rank_{0..3}.json`。

同一步共享（重建总体 ∪ 结构辅助总体只算一次 GNN token/gate，再按原顺序 gather 回两分支）与"两分支各自独立执行原 GNN"逐项对比：

- 重建硬 ID：完全相同（exact）
- 辅助总体硬 ID 与 hard q：完全相同（exact）
- 三项损失（重建/结构/合并）：最大绝对差 **0.0**
- 全部 38 个有梯度参数：最大绝对差 **3.4e-8**（声明容差 atol 2e-5 / rtol 2e-4）

量化总体未合并：重建 1024 项、结构辅助 256 项各自保留原 floor、Sinkhorn 与损失归一；无跨 optimizer step 缓存。逐模块梯度/裁剪/参数更新证据已落 `results/research/issue253/MODULE_CHECKS.json`。

工程修复两处并已回归：新树的 `model/estimator.py` 在模型仓库 root bootstrap 之前 import 仓库模块，导致 launcher 结果目录下 `ModuleNotFoundError: No module named 'tools'`；改为 bootstrap 之后再 import，并用独立 isolated-python 回归 `tools/issue253/test_direct_import.py` 锁定。两个继承自 #252 的 fixture 仍指向未注册 candidate 与旧 contract，已迁移到本臂身份，源码漂移检测判据保持不变。

正式训练仍未启动；接着测优化后 60 epoch inclusive 性能窗口。
