/**
 * agent-vis 的 DeepSeek Harness（dsh）插件入口。
 *
 * dsh 的每个能力都是 Cordis 插件：这里把 agent-vis 的 HTTP 服务与 agent
 * 运行器挂到 harness 上，模型、工具注册表、会话日志、系统提示词装配全部
 * 用 ctx 上的服务，而不是自己再起一套。
 *
 * 挂载方式（见 dsh/ 目录）：
 *   - id: agent-vis
 *     name: file:///abs/path/agent-vis/plugin.js
 *     config: { host: 127.0.0.1, port: 8123 }
 *
 * 插件只注册自己需要的东西；宿主 profile 负责模型路由（llm-pi-ai）、
 * 会话持久化与 sandbox 策略。
 */
import { createServer } from './server.js';
import { setHarnessContext } from './live-agent.js';
import { DEFAULT_PUBLIC_PATH, HOST, PORT } from './config.js';

/** Cordis 插件名（harness 日志、插件管理里显示的名字）。 */
export const name = 'agent-vis';

/**
 * agent-vis 的 agent 由 harness 的 agents 服务创建，工具与提示词走
 * tools / systemPrompt；缺任何一个服务就不该加载。
 */
export const inject = ['agents', 'tools', 'systemPrompt'];

/** 等 server 真正监听（或明确失败）再算插件加载完成。 */
function listen(server, port, host) {
  return new Promise((resolve, reject) => {
    const onError = (err) => {
      server.off('listening', onListening);
      reject(err);
    };
    const onListening = () => {
      server.off('error', onError);
      resolve(server.address());
    };
    server.once('error', onError);
    server.once('listening', onListening);
    server.listen(port, host);
  });
}

/**
 * 挂载 agent-vis。
 *
 * @param ctx - harness 的 Cordis 上下文。
 * @param config - profile patch 里的插件配置。
 *   host / port：HTTP 监听地址；publicPath：路径前缀门槛；
 *   dataDir：agent-vis 自己的页面配置与聊天记录目录（与 dsh 的会话日志分开）。
 */
export async function apply(ctx, config = {}) {
  const host = typeof config.host === 'string' && config.host !== '' ? config.host : HOST;
  const port = Number.isInteger(config.port) ? config.port : PORT;
  const publicPath = typeof config.publicPath === 'string' ? config.publicPath : DEFAULT_PUBLIC_PATH;

  setHarnessContext(ctx);
  const server = createServer({
    publicPath,
    dataDir: typeof config.dataDir === 'string' && config.dataDir !== '' ? config.dataDir : undefined,
  });
  const address = await listen(server, port, host);
  const actualPort = address && typeof address === 'object' ? address.port : port;
  ctx.logger?.info?.('agent-vis listening on http://%s:%d', host, actualPort);

  // 插件卸载：先摘掉上下文（后续请求会明确指出 harness 不可用），再关端口。
  ctx.effect(() => () => {
    setHarnessContext(null);
    server.close();
  });

  mounted = { server, host, port: actualPort };
}

/** 最近一次 apply 的挂载信息（测试与诊断用；Cordis 插件的返回值必须是 effect）。 */
export let mounted = null;
