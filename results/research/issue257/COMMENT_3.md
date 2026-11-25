**结论：COMPLETED。** 公共执行 Infra 已真实可复用，历史故障均有回归，依赖感知验证正确命中/失效，两份真实历史结果被自动无损复算，受控真实四卡入口通过；全部证据已入库并推送（`45b2f274` 流水线、`ca42d2c2` 受控入口+VERDICT，远端 `main` 与本地同 SHA）。

| # | 验收项 | 结果 | 证据 |
|---|---|---|---|
| C1 | #251–#256 故障台账（分类/复现/修复/范围） | PASS | `results/research/issue257/FAULT_LEDGER.md`（12 行，每行指向仓库内已有证据） |
| C2 | 阶段计时（不可恢复标 NA） | PASS | `results/research/issue257/TIMING.json` |
| C3 | 公共边界（候选只管机制与测试） | PASS | `tools/iteration/*` + 复用 `tools/research/stage2/*`、`upload_stage3.py` |
| C4 | 显式父代 + 不可变快照 | PASS | `_archive/` 树 + `run_workspace`；`archive.py` 校验 binding |
| C5 | 依赖感知验证（闭包+资产+配置+运行时指纹） | PASS | `tools/iteration/depgraph.py`、`tests/test_depgraph.py` |
| C6 | 三类真实变更场景 + 最小重验集 | PASS | `VALIDATION_PLAN_DEMO.json`：仅分析→1 单元；共享算子→候选；机制→候选；无关文档→0 |
| C7 | 轮次状态机（十态、禁跳步、可重入） | PASS | `registry.py` + `tests/test_known_answers.py` |
| C8 | 按 run 身份参数化的自动分析 | PASS | `analysis.py`（无 issue/arm 字面量） |
| C9 | ≥2 个真实历史 run 只读复算 | PASS | `replay/r01/REPLAY.json`、`replay/r04/REPLAY.json`（各 13 项，含 57439 行独立归约与 5 个产物哈希） |
| C10 | 受控真实四卡入口 | PASS | `controlled_entry.json`（准入闸门 entry+behavior_graph，47.27 s）；`controlled_four_card.json`（真实四卡 `steady_profile_entry`，exit 0，四 rank `PROFILED`，67.17 s） |
| C11 | 四件套归档校验（不重复上传） | PASS | `archive.verify(R04)` → 台账 105 与 SID 前缀一致、四件齐全 |
| C12 | 文档与 skill 与实际行为一致 | PASS | `CLAUDE.md` §95–§99；`.agent/skills/complete-open-issue/SKILL.md` §8 |

复算数字（从原始产物重算，容差 1e-10）：R01 `test_R@10 = 0.05266456588728913`；R04 `test_R@10 = 0.05151552081338463`、`distinct_triples = 22995`、`l3_collision = 0.0647496644…`、`H(L3|L1L2) = 1.9957073319719747`。单次复算冷 1.6 s / 暖 1.7 s，0 GPU；流水线测试 14 passed。

失败也留证：归档树自带的 `validation_entry` 1-epoch 诊断在四卡上因该树 observer 的 `last_weights` 形状断言（期望 (N,3)，机制存 (N,2,3)）失败，详见 `FAILED_ATTEMPT.md` 与 `TIMING.json.failed_attempt`；准入路径走 `check_entry` 从不触发它，未从中得出任何结论。

**限制（如实记录）**：完整迭代 ≤2 h 预算无新完整轮次时不宣称达标，只报可验证阶段成本；本轮未新增正式 3000/150 训练、未重训 TIGER、未改历史结果。

§6.4 五问：①生产代码全实现=是（八模块，桩代码过不了 14 个测试）；②必要测试已实跑=是（14 passed）；③规定训练已完成=是（本 Issue 不要求训练，受控 1-epoch 真入口已跑）；④资产均来自 `assets/` 未重复创建=是（只按哈希读取）；⑤仍有可执行未完成项=否。

关闭：`CLOSED`（COMPLETED）。
