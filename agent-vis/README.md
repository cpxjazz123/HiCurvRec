# agent-vis（DeepSeek Harness 插件）

把 agent-vis 作为 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（dsh）的一个
Cordis 插件运行：页面的 agent 由 `ctx.agents` 创建，工具走 `ctx.tools` 的 scoped 注册，
系统提示词走 `ctx.systemPrompt`，模型路由由 profile 里的 `llm-pi-ai` 提供。
agent-vis 自己只保留 HTTP/SSE、沙箱工具实现和 dsh 事件 → UI message stream 的翻译。

## 组成

| 文件 | 职责 |
|---|---|
| `plugin.js` | dsh 插件入口：`name` / `inject` / `apply`，起 HTTP 服务并登记 harness 上下文 |
| `dsh-toolset.js` | 只注册工具与提示词的插件（不起端口），供 Harness Web 会话使用 |
| `server.js` | 静态资源、`/api/*`、UI message stream（SSE） |
| `live-agent.js` | 权限 → 工具表、`ctx.agents.create` 会话适配、dsh 事件翻译、沙箱工具实现 |
| `config.js` | 仓库边界、权限目录、人设与提示词规则、模型路由常量 |
| `dsh/profile/` | profile 模板：`@deepseek-ai/dsh-base` + 模型路由 + 关闭宿主工具 + 挂载插件 |
| `dsh/install.mjs` | 生成 `$DSH_HOME/profiles/agent-vis/` |
| `dsh/start.sh` | 找 Node ≥ 24、导出模型 key、启动 `dsh --profile agent-vis` |

## 安装与运行

```sh
cd agent-vis
npm install                 # 安装 @deepseek-ai/dsh 与 typebox
./dsh/start.sh              # 默认 DSH_HOME=agent-vis/.data/dsh-home，端口取 config.js
```

首次启动会生成 `.data/dsh-home/profiles/agent-vis/`。模型 key 从
`~/.config/opencode/commandcode.key` 读进 `COMMANDCODE_API_KEY`（只进进程环境，
不落盘、不打印）；也可以用环境变量显式给出。

`start.sh` 会优先使用 Node ≥ 24（dsh 0.2.2 要求 ≥ 24.9）；`DSH_NODE=/path/to/node`
可显式指定。

## 模型路由

`config.js` 的 `PROVIDER_NAME` / `BASE_URL` / `MODEL_ID` / `CONTEXT_WINDOW` 是唯一来源：
`dsh/install.mjs` 把它们写进 profile patch 的 `llm-pi-ai` 路由（OpenAI 兼容通道，
DeepSeek V4.1 Flash），agent 创建时用同一组值选 provider/model。

宿主自带的 `bash` / `edit` / `write` 等工具在 profile 里关闭；agent-vis 的 agent 只看到
按本轮权限裁出来的沙箱工具（`read` / `ls` / `grep` / `find` / `write` / `edit` /
`deleteFile` / Stage2/3 工具……），全部限制在 `AGENT_WORKDIR` 之内。

## 在 Harness Web 里使用工具集

`dsh-toolset.js` 是同一个仓库工具集的"无页面"版本：只注册工具与提示词，不起 8123。
在 Web profile 的 `cordis.patch.yml` 里插入：

```yaml
- insert:
    - id: agent-vis-toolset
      name: file:///abs/path/agent-vis/dsh-toolset.js
      config:
        permissions: [read, network, web]   # 默认只读；write/delete/execute/train 需显式列出
```

它会跳过 `read` / `write` / `edit` / `ls` / `grep` / `find`（宿主已有同名工具），
只注册 `checkIdea`、`checkJson`、`treeDiff`、`specCheck`、`readDiff`、`stage2*`、
`stage3*`、`runTests`、`copyBaselineTree`、`deleteFile`、`listIssues`、`readIssue`、
`gitLog`、`webSearch` 中权限允许的部分，并把权限说明/语言约束/idea 规则挂进系统提示词。

## 运行主机

`config.js` 的 `AGENT_HOST = 'm3d102'` 是硬约束；面板「全局设置」可以覆盖（同一目录下的
设置会持久化）。主机或工作目录不符时 `/api/run` 直接 503，不会跑在别处。

## 测试

```sh
npm test
```

覆盖：插件接线（`test/plugin.js`、`test/toolset-plugin.js`，假 harness）、dsh 事件翻译与多步轮次
（`test/dsh-agent.js`）、静态资源/协议/权限/沙箱（`test/smoke.js`）、工具与 idea 规则
（其余文件）。`test/browser.js` 是真实浏览器 + 真实模型的端到端检查，需要先启动本插件。
