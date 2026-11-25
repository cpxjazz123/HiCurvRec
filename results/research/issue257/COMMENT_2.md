公共流水线已落地并通过实测（14 个测试全绿，`python3.9 -m pytest tools/issue257/tests -q` → `14 passed`）：

- **依赖感知验证**：指纹 = 测试与被测模块的传递导入闭包 + 资产 + 配置 + 测试自身 + 运行时（解释器/torch/可见设备）的 SHA-256；PASS 仅同指纹复用。实测四场景（`results/research/issue257/VALIDATION_PLAN_DEMO.json`）：
  - 全部收据存在 → `reuse: [pipeline_units, candidate_suite, stage3_protocol]`
  - 仅分析改动（`tools/issue257/analysis.py`）→ 只重验 `pipeline_units`
  - 共享算子改动（`tools/graph_infra/ego_gnn.py`）→ 重验 `candidate_suite`
  - 机制改动（`model/mechanism.py`）→ 重验 `candidate_suite`
  - 无关文档改动（`RUN_LOG.md`）→ 不重验任何单元
  - 闭包突变测试：关闭导入闭包后共享算子场景即漏检（证明测的是闭包，不是路径）
- **状态机**：`registry.py` 十态、禁止跳步、running 相位可幂等重入；`tests/test_known_answers.py` 覆盖非法跳转与重复注册。
- **只读复算（≥2 个真实历史 run）**：R01 与 R04 各 13 项检查全 PASS，含 57439 行预测的独立归约（`independent_reduction:test_recall@10` 等）与 5 个产物哈希：
  - `results/research/issue257/replay/r01/REPLAY.json`、`REPLAY.md`
  - `results/research/issue257/replay/r04/REPLAY.json`、`REPLAY.md`
- **计时**：单次只读复算冷 1.6 s / 暖 1.7 s（CPU，0 GPU）；流水线测试 0.4 s；归档校验 0.1 s（`results/research/issue257/TIMING.json`）。
- **归档校验**：`archive.verify(R04)` → PASS（台账 `105` 绑定 SID 前缀一致 + 四件齐全），未重复上传。
- **文档**：`CLAUDE.md` §95–§99 写接口、凭据失效与产物位置；`.agent/skills/complete-open-issue/SKILL.md` §8 写实际调用接口。

下一节点：受控短四卡入口 + 关闭判定。
