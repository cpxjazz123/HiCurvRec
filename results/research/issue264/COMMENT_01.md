## #264 根因诊断与冻结范围

- 正文 sha256：`6506a531ac7e623f1be3222c9d2dc9bdc575871b5e6ba3a6039f11d7b11a47bc`
- `INFRA_MODE=BUILD`，授权原句：单 Agent 明确允许修复公共 Stage3 启动、数据读取/预处理缓存、日志/状态、性能测试及配套文档；复用 #259 接口。按 issue 要求，不中断/修改 #263 Stage3，不启动新正式训练。
- 验收 C1–C10 与 CONTRACT：`results/research/issue264/RUN_LOG.md`、`CONTRACT.md`。
- 旧启动实测：#263 `launched_at` → torchrun **275.78 s**，→首个 epoch train 记录 **332.61 s**；旧 harness 在 parent 先导入 `train_HG-Rec`（torch/transformers）。历史 #258 记录首次 transformers 导入约 6 分钟。
- 新轻量 parent + 节点本地缓存：parent 三次热启动 **2.72/2.71/2.76 s**；rank import source 首次 **103.725 s** → local cache **5.386 s**；冷源缓存构建 **96.958 s**，热复用 **2.519 s**。目前只完成 parent/rank import 诊断，尚未宣称首个真实 step 已验收；待 #263 释放 GPU 后做短时隔离诊断。
- 证据：`results/research/issue264/BASELINE_TIMELINE.json`、`PERFORMANCE.json`、`diagnostics/`。当前 #263 运行未被修改。
