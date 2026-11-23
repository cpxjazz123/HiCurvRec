## 状态修正：改为 IN_PROGRESS（按新的终止状态规则）

按更新后的 skill §6.2/§6.3，之前那次「预算阻塞」关闭**不成立**：任务量大、单轮预算不足
都不属于可关闭的阻塞。该 issue 重开为 **OPEN**，断点写在 `results/research/issue243/PROGRESS.json`。

### 已完成（证据均已入库）

| 项 | 证据 |
|---|---|
| 契约 + 论文/官方实现对照 | `CONTRACT.md` |
| 独立参考实现 | `reference.py`（纯 numpy float64） |
| 已知答案测试 | `tests/test_known_answers.py` **14/14** |
| 生产 vs 参考一致性 | `tests/test_reference.py` **9/9** |
| 变异测试 | `MUTATIONS.md` **6/6 被抓到** |
| 分模块验证 | `MODULE_CHECKS.json`（5 曲率 × 128 真实商品） |
| 数学修复 | 双曲系数 2 倍、球面冒充 κ-GCN，均已修复并有反例断言 |

### 未完成（可执行，不需要 owner 决策）

1. **专家尺度标定**（0.023–2.695 相差约 100 倍）—— issue 步骤 3 的前置要求，尚未做；
2. **400 步正式 Teacher 训练 + 独立评估**（未见评估对 vs 语义/Jaccard 两个非训练对照）；
3. 重叠子图兼容性（`E_t`、最差子图、同时保真率）；
4. S_G 与修正后的 pair bank（含逐类型验证）；
5. RQ-VAE（TDD + λ 标定 + smoke）+ 一次 3000 epoch Stage2；
6. SID 评估 + 一次 150 epoch Stage3。

`next_commands` 与 `resume_order` 见 `PROGRESS.json`；下一步必须从断点继续，不重复已完成的部分。

*（`results/research/issue243/COMMENT_03.md`）*
