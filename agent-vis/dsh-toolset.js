/**
 * agent-vis 工具集的 harness 插件（不含页面）。
 *
 * 与 plugin.js 的区别：plugin.js 是"agent-vis 自己的页面 + agent 运行器"，
 * 会起 8123；这个插件只把 agent-vis 的仓库工具与提示词规则注册进 harness，
 * 供 DeepSeek Harness Web 的会话直接使用，不监听任何端口。
 *
 * 挂载（web profile 的 cordis.patch.yml）：
 *   - insert:
 *       - id: agent-vis-toolset
 *         name: file:///abs/path/agent-vis/dsh-toolset.js
 *         config:
 *           permissions: [read, network]
 *
 * 权限只决定"注册哪些工具"：默认只给只读工具，写/训练工具必须显式列出。
 * read / write / edit / ls / grep / find 由 harness 自己的文件工具提供，
 * 这里不注册同名工具，避免覆盖宿主能力。
 */
import { DSH_TOOLS, toolsForPermissions } from './live-agent.js';
import { IDEA_RULE, LANGUAGE_RULE, buildPermissionPrompt, normalizePermissions } from './config.js';

/** Cordis 插件名。 */
export const name = 'agent-vis-toolset';

/** 只需要工具注册表与系统提示词装配。 */
export const inject = ['tools', 'systemPrompt'];

/** 宿主已提供的文件工具：同名注册会覆盖宿主能力，必须跳过。 */
const HOST_FILE_TOOLS = new Set(['read', 'write', 'edit', 'ls', 'grep', 'find']);

/** 默认权限：只读。写/训练工具必须显式配置。 */
const DEFAULT_PERMISSIONS = ['read'];

/** 最近一次 apply 的挂载信息（测试与诊断用）。 */
export let mounted = null;

/**
 * 挂载工具集。
 *
 * @param ctx - harness 上下文。
 * @param config - `permissions`：agent-vis 权限 id 列表（read/write/network/execute/train）。
 */
export function apply(ctx, config = {}) {
  const requested = Array.isArray(config.permissions) && config.permissions.length > 0
    ? config.permissions
    : DEFAULT_PERMISSIONS;
  const permissions = normalizePermissions(requested);

  const tools = toolsForPermissions(permissions).filter((tool) => !HOST_FILE_TOOLS.has(tool.name));
  if (tools.length === 0) {
    throw new Error(`agent-vis-toolset: 权限 ${permissions.join(',')} 没有可注册的工具`);
  }
  for (const tool of tools) ctx.tools.register(tool);

  const section = [
    '【agent-vis 仓库工具】以下工具直接作用于本仓库，并受仓库边界约束；',
    '涉及 README/issue/公式/候选树/训练状态时优先用它们，而不是自己拼命令。',
    buildPermissionPrompt(permissions),
    LANGUAGE_RULE,
    IDEA_RULE,
  ].join('\n');
  ctx.systemPrompt.section({
    name: 'agent-vis:toolset',
    order: 1001,
    text: section,
  });

  ctx.logger?.info?.(
    'agent-vis toolset: %d tools for permissions %s',
    tools.length,
    permissions.join(','),
  );
  mounted = { tools: tools.map((tool) => tool.name), permissions };
}

export { DSH_TOOLS };
