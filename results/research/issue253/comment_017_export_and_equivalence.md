## C15 三处真实实现故障的修复与证据链重建(正式训练已重新派发)

### 1. 导出清单与绑定审计的同代身份不一致

- 训练器内存导出的 SID 与 checkpoint 重新导出的 SID 在 303/24,587 项上不同(三码前缀;`same_l1=1.0`)。框架策略要求 Stage3 只读 checkpoint 自己的表,`export_check --rewrite` 因此重写 `item_sids.json` 并保留 `item_sids.trainer.json`。
- 但 `export_manifest.json` 把 `differing_items` 记为**重写前**的 303,又同时把 `same_l3` 记为 1.0;`archive_audit._check_export` 要求 `differing_items == 0`,于是归档绑定无法生成、Stage3 无法入场。两个字段自相矛盾。
- 修复:`export_check.agreement()` 统一计算"当前磁盘上的表"与 checkpoint 三码的差异;重写后 `differing_items` 为 0,重写前的差异保留在 `original_differing_items`(303)。新增 `tools/research/stage2/tests/test_export_manifest_agreement.py`(3 项):重写后的清单通过绑定审计,`differing_items=1` 或 `tokens_match=False` 仍被拒绝。
- 独立复验:用同一 checkpoint 在**新进程**中重新导出,`tokens_sha256` 与第一次完全一致(`bb08ff3a…`),`differing_items=0` → checkpoint 的导出跨进程可复现 ✓。

### 2. 绑定审计把合同文档路径写死为 #252

`archive_audit._repository_configs` 把 `mathematical_contract_sha256` 固定解释为 `results/research/issue252/CONTRACT.md`,而本臂记录的是 `results/research/issue253/CONTRACT.md` → 绑定生成失败。修复:checkpoint 现在**自报**所绑定的合同文档路径(`mathematical_contract_path`),框架按该路径校验;旧 checkpoint 只给摘要时,框架按该摘要在仓库内定位唯一的合同文档,定位不到即拒绝。新增 `tools/issue252/test_archive_binding.py::test_checkpoint_names_the_contract_document_it_was_bound_to`(命名路径、摘要定位、越界路径拒绝)。

### 3. C7 共享前向的硬输出等价判据在 float32 下不稳定

真实四卡验证曾两次在同一断言上给出不同结果:共享前向与两分支独立执行的三码在少数条目上不同。根因是 **float32 批量组成**改变每个累加和的末位:共享前向按"重建∪辅助"并集批计算,独立路径按各自总体批计算,代价矩阵因此有 ~1e-4 相对量级的差异;而 `ASSIGNMENT_FLOOR` 的容量约束分配是**全局匹配**,一个 float 级扰动会让少数条目离开自己的 argmin。

修复:判据从"逐条硬码完全相等"改为"**分配目标值必须一致**":两分支在各自代价下的容量约束分配目标之差必须落在累计数值发散界内(每层报告最大代价发散、目标值、发散界、代价尺度与不同条目数)。梯度仍按 `atol=2e-05/rtol=2e-4` 逐参数比较,重建硬 ID 仍要求完全相等。

### 证据链重建

修复触及 `tools/issue253`、`tools/issue252`、`tools/research/stage2` 与候选树源码,因此合同测试、真实四卡验证与基准全部重跑,并重新生成训练准入回执:

| 项目 | 结果 |
|---|---|
| 合同测试 | 确定性 **182**、参考 **11**、变异 **2** 全通过,`source_hashes` 未变 |
| 真实四卡验证 | 4 rank 全 `pass`;共享前向目标值差异 ~2.5e-6(相对),不同条目 0 |
| 四卡基准 | 4 rank 全 `pass`;inclusive mean **0.6396 s/epoch**、median 0.5901、p95 0.8210;0.4 目标未达、0.5 期望未达、**0.8 上限满足** |
| 训练准入回执 | `results/research/issue253/TRAINING_ADMISSION.json` sha256 `ef955f22…`,`check_receipt` 通过 |

正式训练已重新派发:`stage2_787483d6ef9842d1a7c4593a61ae925c`(4 卡 DDP、global batch 4096、seed 42、3000 epoch、端口 50202)。前一次派发 `stage2_6e4324cd326e45eba067d30295f8e4bd` 完成了 3000 epoch、collision 0.0814,但其运行凭据与修复后的回执不再逐字节一致,因此只作启动链路诊断证据,不作为结果报告。
