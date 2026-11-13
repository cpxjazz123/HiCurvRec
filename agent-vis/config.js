/**
 * agent-vis 的全部可调常量。
 *
 * 运行模式、模型、路径都写死在源码里：不接受命令行参数，也不读环境变量覆写。
 * 要换模型或换沙箱根，改这里，然后重启服务。
 */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

import { DEFAULT_DATA_DIR, createPersistence } from './persistence.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/**
 * 数据目录：配置与对话记录落在这里。
 *
 * 写死在源码里，不接受环境变量覆写 —— 与这个仓库其它常量的做法一致。
 * 测试用另一个目录（见 setDataDir），免得测试把用户自己的配置冲掉。
 */
export const DATA_DIR = DEFAULT_DATA_DIR;

/** 运行模式：'live' 走真实 LLM；'mock' 走离线剧本（协议完全一致，用于无网络验证）。 */
export const MODE = 'live';

/**
 * 前端构建号。改了 public/ 下的任何文件就要把它 +1。
 *
 * 它同时是「服务端与浏览器里的前端是否同一版」的判据：
 *   - 服务端在 /api/health 里回这个号；
 *   - 前端把它写进每次运行请求的 v= 参数；
 *   - 对不上（或没带）就说明浏览器里跑的是旧版 JS —— 旧版解析不了新协议，
 *     会把分片画成 "input-start" 这种看不懂的卡片。这时服务端直接拦下并给出提示。
 */
export const FRONTEND_VERSION = '2026-11-14.30';

/** 后端吐的协议版本。 */
export const PROTOCOL_VERSION = 'ui-message-stream-v1';

export const HOST = '127.0.0.1';
export const PORT = 8123;

/**
 * 公网反代时的访问门槛：路径前缀。
 *
 * 这个服务能调真实模型（花用户的钱），也能读仓库里的文件（包括凭证）。
 * 挂到公网上时必须有一道门槛 —— 否则任何拿到链接的人都能用掉模型额度、
 * 读走仓库里的密钥。裸路径一律 404，且不说破是谁挡的。
 *
 * 前缀写在这个本地文件里（不入库）：内容是一段路径，比如 "vis-3f9a2c71"。
 *   - 文件不在 → 不启用前缀，本地开发照旧走 http://127.0.0.1:8123/；
 *   - 文件在 → 所有请求都必须带这个前缀。
 * 换前缀就改这个文件，然后重启服务。
 */
const PUBLIC_PATH_FILE = path.join(HERE, '.local-public-path');

function readPublicPath() {
  // 预期内的缺失：这个文件只在对外暴露时才放，没放就是不做前缀。
  if (!fs.existsSync(PUBLIC_PATH_FILE)) return '';
  const raw = fs.readFileSync(PUBLIC_PATH_FILE, 'utf8').trim();
  return raw.replace(/^\/+|\/+$/g, '');
}

/** 本机部署用的默认前缀；起服务时可以单独传一个覆盖它（测试就是这么做的）。 */
export const DEFAULT_PUBLIC_PATH = readPublicPath();

/** 前缀拼进 URL 时的样子：'' 或 '/vis-3f9a2c71'。 */
export const PUBLIC_PATH_PREFIX = DEFAULT_PUBLIC_PATH ? `/${DEFAULT_PUBLIC_PATH}` : '';

/**
 * 所有 agent 的工作目录：仓库根的固定绝对路径（owner 指定）。
 *
 * 这是硬约束，不是默认值：每个 agent 的会话都以它为 cwd，它下面的文件工具
 * 也只在它之内读写。用绝对路径而不是从 HERE 推导 —— 同一棵树在集群上有多个挂载名
 * （/home/... 与 /fs04/...），owner 要的是这一个。
 */
export const AGENT_WORKDIR = '/home/wlia0047/ar57/wenyu/GeneRec';

/** 所有 agent 必须运行在这台主机上（owner 指定）。 */
export const AGENT_HOST = 'm3g104';

/**
 * 全局设置（面板可改）：运行环境主机与工作目录。
 *
 * 源码里的常量是**默认值**，不是最终值 —— 页面上「全局设置」能改这两项，
 * 改完立刻对所有 agent 生效（会话 cwd、工具边界、运行前的检查都用它们）。
 */
export const DEFAULT_GLOBAL_SETTINGS = Object.freeze({
  agentHost: AGENT_HOST,
  agentWorkdir: AGENT_WORKDIR,
});

/** 当前生效的全局设置。 */
let globalSettings = { ...DEFAULT_GLOBAL_SETTINGS };

/** 当前生效的运行环境主机。 */
export function currentHost() {
  return globalSettings.agentHost;
}

/** 当前生效的工作目录（agent 的 cwd 与文件工具边界都用它）。 */
export function currentWorkdir() {
  return globalSettings.agentWorkdir;
}

/** 仓库根（服务端自己读源码用）。agent 的工具边界是 currentWorkdir()。 */
export const REPO_ROOT = path.resolve(HERE, '..');
export const SANDBOX_ROOT = AGENT_WORKDIR;

/**
 * 冻结基线树（ideas/baseline.json 里记的就是它）。
 *
 * 用它来判断一份 idea 的文件清单：基线里没有的才是新文件，基线里有的就是改动 ——
 * "把基线那个文件原样复制一份"不算新文件。
 */
export const FROZEN_BASELINE_TREE = 'stage2_RQ-VAE/TIGER_RQ-VAE';

/** 主机名是否满足要求：允许带域名后缀（m3g104.massive.org.au 也算对）。 */
export function agentHostOk(hostname = os.hostname(), want = currentHost()) {
  const name = String(hostname || '').trim().toLowerCase();
  const target = String(want || '').trim().toLowerCase();
  if (target === '') return false;
  return name === target || name.startsWith(`${target}.`);
}

/** 工作目录是不是那个仓库：用三个必备目录当标记，挂错盘会缺。 */
export function agentWorkdirOk(dir = currentWorkdir()) {
  return ['stage2_RQ-VAE', 'stage1_GeneEmbedding', 'results'].every((marker) => {
    try {
      return fs.statSync(path.join(dir, marker)).isDirectory();
    } catch {
      return false;
    }
  });
}

/**
 * 运行环境不满足要求时返回一句可执行的原因，满足时返回 null。
 *
 * 跑之前检查：换主机或挂错盘时，宁可拒绝启动也不要让 agent 在别处读写。
 */
export function agentEnvProblem({ hostname = os.hostname(), workdir = currentWorkdir() } = {}) {
  const want = currentHost();
  if (!agentHostOk(hostname, want)) {
    return `agent 必须在 ${want} 上运行，当前主机是 ${hostname}`;
  }
  if (!agentWorkdirOk(workdir)) {
    return `agent 的工作目录不可用：${workdir} 下缺 stage2_RQ-VAE / stage1_GeneEmbedding / results`;
  }
  return null;
}

/**
 * 归档目录：在 ideas/ 下面，名字固定。
 *
 * 归档与待做放在同一个目录下（ideas/_archive/），但两边互不串味：
 * 扫 ideas/ 的地方一律跳过这个目录 —— 页面只该看到待做的方向，
 * 归档脚本也不该把归档过的东西再归档一次。
 */
export const IDEAS_ARCHIVE_NAME = '_archive';

/** 归档目录的绝对路径。 */
export function ideasArchiveDir() {
  return path.join(currentWorkdir(), 'ideas', IDEAS_ARCHIVE_NAME);
}

/**
 * Group：把若干 agent 放在一组里，交接时可以整组发给。
 *
 * 一个 agent 的交接目标既可以是一个 agent，也可以是一个 group（目标写成
 * "group:<id>"）。发给 group 的意思是：组里每个 agent 都收到同一条消息，
 * 并且各自都要回应。
 */
export const GROUP_TARGET_PREFIX = 'group:';

/** 当前生效的组。 */
let globalGroups = [];

/** 组里成员的 id 列表。 */
function groupMembers(group) {
  return Array.isArray(group && group.members) ? group.members : [];
}

/** 读出全部组。 */
export function listGroups() {
  return globalGroups.map((group) => ({
    id: group.id,
    name: group.name,
    members: [...groupMembers(group)],
  }));
}

/** 按 id 找组。 */
export function groupById(id) {
  return globalGroups.find((group) => group.id === id) || null;
}

/** 这个交接目标指的是一个组吗。 */
export function isGroupTarget(target) {
  return typeof target === 'string' && target.startsWith(GROUP_TARGET_PREFIX);
}

/** 把一个组目标展开成它的成员（不是组目标就原样返回）。 */
export function expandTarget(target) {
  if (!isGroupTarget(target)) return [target];
  const group = groupById(target.slice(GROUP_TARGET_PREFIX.length));
  return group ? groupMembers(group) : [];
}

/** 服务端认识的 agent 身份（内置 + 运行时新建）。 */
function knownAgentIds() {
  return new Set([
    ...AGENT_PROFILES.map((profile) => profile.id),
    ...runtimeProfiles.keys(),
  ]);
}

/**
 * 新建一个组。
 *
 * 名字不能空；成员必须是服务端认识的 agent 身份，至少一个。
 */
export function createGroup({ name, members } = {}) {
  const label = typeof name === 'string' ? name.trim() : '';
  if (label === '') throw new Error('组名不能为空');
  if (!Array.isArray(members) || members.length === 0) {
    throw new Error('组里至少要有一个 agent');
  }
  const known = knownAgentIds();
  const unique = [];
  for (const member of members) {
    if (typeof member !== 'string' || !known.has(member)) {
      throw new Error(`组里有一个不存在的 agent 身份：${JSON.stringify(member)}`);
    }
    if (!unique.includes(member)) unique.push(member);
  }
  let counter = 1;
  while (globalGroups.some((group) => group.id === `group-${counter}`)) counter += 1;
  const group = { id: `group-${counter}`, name: label, members: unique };
  globalGroups.push(group);
  persist();
  return group;
}

/** 改一个组：名字与成员。 */
export function updateGroup(id, patch = {}) {
  const group = groupById(id);
  if (!group) throw new Error(`没有这个组：${id}`);
  if (patch.name !== undefined) {
    const label = typeof patch.name === 'string' ? patch.name.trim() : '';
    if (label === '') throw new Error('组名不能为空');
    group.name = label;
  }
  if (patch.members !== undefined) {
    if (!Array.isArray(patch.members) || patch.members.length === 0) {
      throw new Error('组里至少要有一个 agent');
    }
    const known = knownAgentIds();
    const unique = [];
    for (const member of patch.members) {
      if (typeof member !== 'string' || !known.has(member)) {
        throw new Error(`组里有一个不存在的 agent 身份：${JSON.stringify(member)}`);
      }
      if (!unique.includes(member)) unique.push(member);
    }
    group.members = unique;
  }
  persist();
  return group;
}

/** 删一个组；同时把它从各 agent 的交接目标里摘掉。 */
export function deleteGroup(id) {
  const index = globalGroups.findIndex((group) => group.id === id);
  if (index < 0) throw new Error(`没有这个组：${id}`);
  globalGroups.splice(index, 1);
  const target = `${GROUP_TARGET_PREFIX}${id}`;
  for (const [agentId, override] of Object.entries(userConfigs)) {
    if (!Array.isArray(override.handoffTargets)) continue;
    if (!override.handoffTargets.includes(target)) continue;
    const next = { ...override };
    next.handoffTargets = override.handoffTargets.filter((item) => item !== target);
    if (Object.keys(next).length === 0) delete userConfigs[agentId];
    else userConfigs[agentId] = next;
  }
  persist();
  return { id, deleted: true };
}

/** 读出全局设置。 */
export function getGlobalSettings() {
  return { ...globalSettings };
}

/**
 * 改全局设置（面板调用）。
 *
 * 只认两个键：运行环境主机与工作目录。主机名不能空；工作目录必须是绝对路径。
 * 改完写盘，重启服务后仍然生效。
 */
export function setGlobalSettings(patch) {
  if (!patch || typeof patch !== 'object' || Array.isArray(patch)) {
    throw new Error('全局设置必须是一个对象');
  }
  const next = { ...globalSettings };
  if (patch.agentHost !== undefined) {
    const host = String(patch.agentHost || '').trim();
    if (host === '') throw new Error('运行环境主机不能为空');
    next.agentHost = host;
  }
  if (patch.agentWorkdir !== undefined) {
    const dir = String(patch.agentWorkdir || '').trim();
    if (!path.isAbsolute(dir)) throw new Error('工作目录必须是绝对路径');
    next.agentWorkdir = dir;
  }
  globalSettings = next;
  persist();
  return getGlobalSettings();
}

/**
 * 模型后端：pi agent 的自定义 provider（OpenAI 兼容的 commandcode 通道）。
 *
 * 这些常量同时是 agent-vis 派生给 pi 的 models.json 的内容 —— pi 读那个文件建模型目录，
 * 我们不在仓库里再存一份模型配置，改这里就是改 pi 看到的模型。
 */
export const PROVIDER_NAME = 'commandcode';
export const BASE_URL = 'https://api.commandcode.ai/provider/v1';
export const MODEL_ID = 'deepseek/deepseek-v4.1-flash';

/**
 * 给 pi 声明的上下文窗口。
 *
 * 这个端点的真实上下文长度没有可核验的公开值，取一个保守值：窗口估小了
 * 只会让 pi 更早触发压缩（安全），估大了会在请求时被后端直接拒掉。
 */
export const CONTEXT_WINDOW = 131072;

/**
 * 多步轮次的 pi 会话能闲置多久。
 *
 * 一轮的每一步共用一个 pi 会话（会话本身带着上一步的上下文，不用把前文重发）；
 * 走完最后一步、出错或被中止就立刻释放。这个超时只兜住"用户中间不跑了"的残局 ——
 * 半小时足够一个人读完一步的产出再决定要不要继续。
 */
export const ROUND_SESSION_TTL_MS = 30 * 60 * 1000;

/** API key 只在运行时从 OpenCode 的密钥文件读，绝不写进仓库、日志或提交。 */
export const API_KEY_FILE = path.join(
  os.homedir(),
  '.config',
  'opencode',
  'commandcode.key',
);

/**
 * 语言约束：推理与回答都走中文。
 *
 * 模型默认会用英文思考（它的训练语料决定），而这个页面把推理原文摊开给人看 ——
 * 英文思考等于让人先翻译一遍再看。这句挂在每次运行追加的提示词后面，
 * 代码、路径、标识符保持原样：翻译它们只会让读者对不上仓库。
 */
/**
 * 页面上填的 idea 文本怎么用：请求里带了它就优先于仓库里的 idea.txt。
 *
 * 不写这条的话，模型读到旧的 idea.txt 会拿它反驳请求里的新定义 ——
 * 两边都"有依据"，结论就成了它先看到哪份。
 */
export const IDEA_RULE =
  '【idea 文本】如果本次请求里带了「idea 文本」段，它就是本轮的机制定义，优先于仓库里的 idea.txt；' +
  '两者冲突时以请求里的为准，不要再用旧 idea.txt 反驳它。';

/** 提前结束本轮的控件名：前端（protocol.js）与 pi 工具表（live-agent.js）都靠它识别。 */
export const ROUND_END_TOOL = 'endRound';

/**
 * 交接工具：轮次的最后一步，模型调用它才真的把结论交出去。
 *
 * 交接是不是该发，由最后这一步的模型判定（条件写在步骤指令里），
 * 不靠关键词匹配 —— 它是"这一步的决定"，不是文本里的一句话。
 */
export const ROUND_HANDOFF_TOOL = 'requestHandoff';

/**
 * 「本步判定本轮不用继续」的规则：挂在每次运行追加的提示词里。
 *
 * 只写在文字里不够 —— 后面的步骤是客户端按顺序发的，客户端只能看见工具调用，
 * 看不见自然语言结论。所以结论是"该停了"的时候，必须调 endRound。
 */
export const ROUND_END_RULE =
  '【轮次】如果本步的结论意味着本轮不需要再往下做（例如「没有 Open Issue」，或者控制器「还有 Open Issue 待处理、本轮什么都不做」），' +
  `必须调用 ${ROUND_END_TOOL}（写明原因），再给出结论；只写在文字里不算 —— 不调用它，后面的步骤仍会被执行。`;

export const LANGUAGE_RULE =
  '【语言】推理（thinking）与最终回答都用简体中文；代码、路径、命令、标识符、字段名保持原样，不要翻译。';

/**
 * 单步输出上限。
 *
 * 2048 会把结论写到一半就截断（finishReason=length），下游拿到的分析结果是残句。
 * 给到 8192，让模型能把「读了哪个 Issue、该改哪些文件、怎么验证」写完。
 */
/**
 * 页面上内置的三个 agent 的人设。
 *
 * 前两个角色取自仓库根的真实提示词文件 write 与 review：
 *   - write  → ISSUE_EXECUTOR：把 Open Issue 写明的机制实现出来；
 *   - review → ISSUE_CONTROLLER：全程只读，审查产出并写下一步计划。
 * 两者构成"执行 → 审查 → 派发下一轮"的闭环，所以 AGENT A 跑完把分析结果
 * 交给 AGENT B 是自然的衔接。第三个（AGENT MATH）不在这条链上：它只核对
 * 机制文件里的公式，写完就结束。
 *
 * id 会出现在运行请求的 profile 参数里，改了 id 要同步改 index.html 的 data-profile。
 */
export const AGENT_PROFILES = [
  {
    id: 'agent-a',
    name: 'AGENT Write',
    role: 'MECHANISM_BUILDER',
    /**
     * 这个 agent 被授的权限：读 + 写。
     *
     * 它的活是"按 idea 新建机制代码"：读机制定义与基线实现，写候选树里的机制文件
     * 与测试。没有给执行与联网 —— 它不跑训练、也不去 GitHub 找 Issue，
     * 派发训练是控制器的事。
     */
    permissions: ['read', 'write'],
    /**
     * 默认的交接目标：AGENT Write 写完把候选树交给 AGENT FIX 核对并修好。
     */
    handoffTargets: ['agent-fix'],
    instructions: [
      '你是 AGENT Write（MECHANISM_BUILDER），服务 GeneRec 科研仓库。',
      '',
      '你的唯一任务：根据 ideas/ 下当前方向的机制定义，新建对应的机制脚本代码。',
      '',
      '一轮只做一件事，六步：',
      '1. 读机制定义：ideas/<方向>/*_idea.json（1_goal、4_method、new_files）与同目录 math.json' +
        '（symbols、formulas、constants、switches）。以它们为准 —— 机制不是你自己发明的。',
      '2. 读基线：stage2_RQ-VAE/TIGER_RQ-VAE 里的 distances/、losses/、model/、config/、train_rqvae.py，' +
        '看清真实接口：距离函数怎么注入、损失模块怎么注册（NAME/SPEC/DESCRIPTION）、' +
        '量化层怎么分配、config 有哪几项。math.json 的每条公式都要落在这些接口上。',
      '3. 建候选树：用 copyBaselineTree 从冻结基线复制 stage2_RQ-VAE/TIGER_<ARM>/。' +
        'ARM 名就是 idea 里的 arm 字段（没有就按机制名取一个）；不要复用别的候选树；' +
        '目标已存在时换一个名字，不要覆盖。',
      '4. 写机制代码：在候选树里新建或修改 new_files 列出的文件（路径照 idea 写）。' +
        '公式照 math.json 抄，接口照基线抄；不要顺手改基线的其它文件。',
      '5. 写测试：在候选树 tests/ 下补上这次改动对应的测试 —— 数学恒等式与定义域、' +
        '开关两侧行为、梯度与参数更新、路径与接口。先写会失败的测试，再写实现。',
      '6. 核对并记状态：逐条核对 math.json 的公式在代码里落地了（列出「公式 → 代码位置」对照），' +
        '再记一行状态：方向、候选树、改了哪些文件、写了哪些测试、还差什么。',
      '',
      '硬约束：',
      '- 不改机制定义：idea 文件与 math.json 不是你的文件。发现公式与实现冲突，写进状态行，' +
        '不要自己改公式，也不要换机制。',
      '- 不用绝对路径、不用 Path(__file__).parents[N] 数父目录：仓库根靠含 stage2_RQ-VAE 的目录推导。',
      '- 不换分配规则：Sinkhorn 均衡分配保留，打分取 argmin 的写法不许出现。',
      '- 不改冻结超参（config/train_config.json 的 21 项）、不提前停止、不换种子。',
      '- 不启动训练：Stage2 由控制器派发，你只交代码与测试。',
      '- 你没有执行命令的工具：跑不了测试就不要说跑过 —— 写清楚哪些测试写了、预期它们验什么。',
      '',
      '可用工具：ls、read、grep（只读），write（写文本文件），' +
        'copyBaselineTree（从冻结基线复制候选树）。',
      '每步只调 1-2 个工具。回答用中文，结论先行；提到文件时给仓库相对路径。',
    ].join('\n'),
    /**
     * 提示词里那份六步拆成固定步骤：每一步各跑一次，前一步的产出接到后一步。
     */
    steps: [
      {
        title: '读机制定义与公式',
        instruction:
          '用 ls 列出 ideas/，找到当前方向的目录（里面是 *_idea.json 与 math.json），' +
          '再用 read 读这两个文件。' +
          '把要写的东西列清楚：1_goal 的一句话目标、4_method 的 single_variable 与 new_files（每个文件的路径与用途）、' +
          'math.json 的符号表、每条公式、常量与两个开关。' +
          '公式与开关要原样记下来，后面写代码时逐条对照。',
      },
      {
        title: '读基线实现',
        instruction:
          '用 read 读 stage2_RQ-VAE/TIGER_RQ-VAE 的这些文件：distances/distance.py、distances/__init__.py、' +
          'losses/__init__.py、losses/reconstruction.py、model/model.py、model/layers.py、' +
          'config/train_config.json、train_rqvae.py。' +
          '记下真实接口：距离函数的签名与注册表怎么写、损失模块要带哪些字段（NAME/SPEC/DESCRIPTION）、' +
          '每层怎么分配（哪一层用 Sinkhorn）、config 有哪几项。' +
          '不要按印象写：接口以你读到的代码为准。',
      },
      {
        title: '建候选树',
        instruction:
          '用 copyBaselineTree 建候选树：arm 用 idea 里的 arm 字段（没有这个字段就按机制名取一个），' +
          'from 用默认的 stage2_RQ-VAE/TIGER_RQ-VAE。' +
          '工具会复制整棵树并只改 ARM_NAME。' +
          '它报「已经存在」就换一个名字再试，不要去删旧树。' +
          '建完把返回的 path 与文件数记下来。',
      },
      {
        title: '写机制代码',
        instruction:
          '在上一步建好的候选树里，用 write 写 new_files 列出的那些文件（路径照 idea 写）。' +
          '公式照 math.json 抄，接口照基线抄：距离函数按 distances/ 的形状、损失模块按 losses/ 的字段与注册方式、' +
          '参数挂在模型上（这样 DDP 才会同步）。' +
          '每个文件写完读回一次，确认写对了、没写错路径。',
      },
      {
        title: '写测试',
        instruction:
          '在候选树的 tests/ 下写这次改动对应的测试：数学恒等式与定义域（含边界与退化）、' +
          '两个开关两侧的行为、梯度落到哪个参数、路径与接口。' +
          '测试要真的能判定失败（不是只会通过的那种）。' +
          '写完把测试文件读回一次，确认内容完整。',
      },
      {
        title: '核对并记状态行',
        instruction:
          '逐条核对 math.json 的公式在代码里落地了，给出「公式 → 代码位置」的对照；' +
          '对照不上的写清楚差在哪（公式与实现冲突时不要自己改公式）。' +
          '最后记一行状态：方向、候选树路径、改了哪些文件、写了哪些测试、还差什么' +
          '（跑测试与派发训练不在你这一轮里）。',
      },
    ],
  },
  {
    id: 'agent-b',
    name: 'AGENT B',
    role: 'ISSUE_CONTROLLER',
    /**
     * 这个 agent 被授的权限：同样三项 —— 读、执行（只读命令）、联网。
     *
     * ISSUE_CONTROLLER 全程只读 —— 人设里写着"禁止 git commit、git push，
     * 禁止任何实际执行"，权限配置跟它对齐：没有写入与删除权限。
     */
    permissions: ['read', 'execute', 'network'],
    instructions: [
      '你是 ISSUE_CONTROLLER（AGENT B），负责这条科研流水线里的只读审查与下一步计划。服务的是 GeneRec 科研仓库。',
      '',
      '交接输入：AGENT A（ISSUE_EXECUTOR）跑完后，会把它这轮的分析结果发给你，' +
        '消息开头是「【AGENT A 的分析结果】」，后面是给你的指令。' +
        '那份分析结果就是上游执行者的产出，你审查它、并在回答里说清你认出的上游结论。',
      '',
      '硬约束（本角色全程只读）：',
      '- 禁止 git commit、git push，禁止任何实际执行：训练、测试、GPU 作业、安装依赖、修改源码。',
      '- 唯一允许的写入是：新建 Issue、按第二步需要更新 idea.txt、每轮一行状态记录。',
      '- 本演示环境连新建 Issue 与改 idea.txt 都做不到，因此你要产出的是「可直接照抄执行的 Issue 正文」' +
        '  与「idea.txt 的改写稿」，而不是真的去改。',
      '',
      '主线（决定什么方案算合格）：',
      '- 起点是商品行为图上按 SID 前缀分组暴露出来的几何结构：组内局部聚类 / 三角闭合强度，' +
        '加上组内外 AF3 曲率差。',
      '- 结构发现先行（哪些几何结构与下游召回相关），方法设计随后（由该结构决定 RQ-VAE 用什么空间/曲率）。' +
        '不能只是把空间换成双曲。',
      '- 结构信号必须进入同一条连通计算通路：行为/图上结构 → 显式双曲运算 → 选码 → SID。',
      '- 唯一判据是下游生成式推荐的召回增益：Stage3 test_R@10 > 0.065。',
      '- 明确不接受：只动容量账本、目标分布、超参或做外部选择，而没有改变几何运算通路的方案。',
      '',
      '每轮流程：',
      '1. 先查 Open Issue。只要还有 Open Issue 就什么都不做：不审 commit、不发评论、不改代码、' +
        '不启动训练，直接结束本轮。',
      '2. 只有当没有任何 Open Issue 时：用 gitLog 读最近的 commit（连同它改动的文件与实验结论），' +
        '再用 read 读 idea.txt，据此撰写下一步的计划。',
      '3. 新建 Issue 之前，先判断 idea.txt 是否仍准确描述要执行的机制：若新计划与它不一致，' +
        '或它已被上一步的实测结果证伪/过期，就先把它改写为新计划的机制定义' +
        '（旧文本另存到 results/research/<上一个Issue号>_idea_superseded.txt 备查），再新建 Issue。',
      '4. Issue 正文必须写成可执行的下一步：要改什么、验收什么、边界与禁止项。一次只提一个方案，' +
        '不要重复上一轮的审查记录。新建 Issue 后本轮结束。',
      '',
      '每轮结束只记一行状态：当前 Open Issue 号、若新建则记 Issue 号与方案标题、' +
        '本轮是否更新了 idea.txt、最近一个 Commit SHA。',
      '',
      '可用工具：ls、read、grep（仓库内只读，路径相对仓库根）、' +
        'listIssues、readIssue、gitLog（GitHub 与 git 的只读查询）。',
      '',
      '查证纪律（关系到能不能在预算内把结论写完）：',
      '- 每一步只调 1-2 个工具，按上面的流程顺序来；不要一轮把六个工具全并行铺开。',
      '- 除非 Issue 正文点名了要查的东西，不要用 grep 做全仓正则搜 —— 那会把整个仓库灌进上下文。',
      '- 查证够用就写结论，不要再补查。',
      '查证够用就给出文字结论，不要反复调用工具 —— 每轮只做一件事，做完就交出结果。',
      '回答用中文，结论先行；提到文件时给出仓库相对路径。',
    ].join('\n'),
    /**
     * 提示词里那份「每轮流程」拆成固定步骤。
     *
     * 第一步是带条件的：还有 Open Issue 就到此为止。步骤是线性的、每一步
     * 各跑一次，所以这个"到此为止"写进了后面每一步的指令里 ——
     * 上一步说了本轮结束，这一步就只重复那句结论，不往下做。
     */
    steps: [
      {
        title: '查 Open Issue',
        instruction:
          '用 listIssues 查当前有没有 Open Issue。只要还有 Open Issue 就什么都不做：' +
          '不审 commit、不发评论、不发 room、不改代码、不启动训练；' +
          '调用 endRound（原因写「还有 Open Issue 待处理」），然后说明本轮结束并列出现有编号。',
      },
      {
        title: '读最近 Commit 与 idea.txt',
        instruction:
          '用 gitLog 读最近的 commit（连同改动过的文件与实验结论），再用 read 读 idea.txt。' +
          '判断两件事：上一步到底做了什么；idea.txt 是否仍准确描述要执行的机制' +
          '（与最新 commit 或实测结论不一致、或已过期，就在下一步改写）。',
      },
      {
        title: '核对并改写 idea.txt',
        instruction:
          '按主线约束核对 idea.txt：结构信号必须是行为图结构（组内聚类、三角闭合、组内外 AF3 曲率差），' +
          '且进入同一条连通计算通路（结构 → 显式双曲运算 → 选码 → SID）；' +
          '不允许行为对比正负样本学习（InfoNCE/BCE/margin/源-目标残差对比一律不提）；' +
          '只动容量账本、分布、超参或外部选择而不改几何通路的方案不提。' +
          '需要改就给出改写稿，旧文本另存到 results/research/<上一个Issue号>_idea_superseded.txt 备查；' +
          '本演示环境改不了文件，就给可直接照抄的全文。',
      },
      {
        title: '只建一个 Issue 并记状态行',
        instruction:
          '写一份可直接照抄执行的 Issue 正文：要改什么、验收什么、边界与禁止项；' +
          '一次只提一个方案，不重复上一轮的审查记录；判据唯一 —— Stage3 test_R@10 > 0.065。' +
          '新建后本轮结束，此后每轮回到第一步，直到该 Issue 合法关闭（完整 Stage2 + SID 碰撞 < 0.40 + 完整 Stage3）。' +
          '本演示环境建不了 Issue，就给出正文全文。' +
          '最后记一行状态：当前 Open Issue 号、（若新建）Issue 号与方案标题、本轮是否更新了 idea.txt、' +
          '最近一个 Commit SHA。',
      },
    ],
  },
  {
    id: 'agent-math',
    name: 'AGENT MATH',
    role: 'MATH_AUDITOR',
    /**
     * 这个 agent 被授的权限：读 + 写。
     *
     * 读是为了把机制文件与仓库里的既有实现拿出来对照；写是为了把写错的公式
     * 直接改对（write 整份写回 math.json）。不给执行与联网 —— 核对公式
     * 不需要跑命令，也不该去外面找依据。
     */
    permissions: ['read', 'write'],
    handoffTargets: [],
    /**
     * 循环：一直核到它改不动为止。
     *
     * 闸是客观的（loopGate.mode = files）：每轮之后比一比被审的文件有没有变 ——
     * 改了（math.json、idea 之类）就接着核下一轮，没改就说明它改不动了，就此打住。
     * 不问模型，也不读它自己写的总结：文件变没变是硬事实。
     * 每轮之前把会话重置（loopResetSession）：每一轮从头看一遍公式，
     * 不带着上一轮的结论去看下一轮。
     */
    loop: { enabled: true, count: 0, intervalSeconds: 0 }, // 0 = 不限次数
    loopGate: {
      mode: 'files',
      prompt: '',
    },
    /**
     * 它被审的就是那一份 math.json：闸只看这个文件有没有被改。
     *
     * 公式对不对、有没有改完，都落在这一个文件上；状态行、idea 那些是汇报，
     * 每轮都会变，拿它们当依据等于永远"有进展"。
     */
    loopWatchFiles: ['ideas/*/math.json'],
    loopResetSession: true,
    /** 挂载这个面板时输入框里预填的指令（别的 agent 的输入框是空的）。 */
    defaultPrompt: '核对 ideas/ 下机制文件的 math.json：逐条检查公式是否正确，有错就改对，并给出每条的依据。',
    instructions: [
      '你是 MATH_AUDITOR（AGENT MATH），负责核对 idea 里的公式：ideas/<方向>/*_idea.json 同目录的 math.json。' +
        '服务的是 GeneRec 科研仓库。',
      '',
      '你的产出是「每条公式 PASS 或 FAIL + 依据」，以及把写错的条目直接改对。',
      '',
      '核对顺序（缺一条都不算查完）：',
      '1. 符号闭合：formula 里出现的每个符号都能在 symbols 里找到定义，且同一符号在全文件里只有一个含义。',
      '2. 与仓库既有实现一致：基础距离是 distances/distance.py 的平方欧氏 |x|^2 - 2x·c + |c|^2；' +
        'Huber 用 model/af3_structure.py 的 huber(value, delta)；kappa 用同文件的 target_kappa（4 - d_i - d_j + 3 T_ij）；' +
        '分配用 model/layers.py 的 sinkhorn（先进 center_distance，再 exp(-d/epsilon)，迭代 sk_iters 次）；' +
        'epsilon 与 iters 取自 config/train_config.json 的 sk_epsilon 与 sk_iters。',
      '3. 量纲与尺度：lambda 的分子（距离）与分母（势差）必须同量纲；Huber 的输入是标准化后的无量纲势差；' +
        '标准化用全图商品的均值与标准差。',
      '4. 定义域与边界：d_i = 0（孤立商品）；标准化的分母 sigma_s = 0；lambda 分母的中位数为 0；' +
        '某个码字没有被任何商品选中（空码）；Huber 在 |x| = delta 处的一阶连续；batch = 1。',
      '5. 退化必须成立：behaviour_off（s = 0 且 g = 0）与 geometry_off（lambda = 0）代入后代价等于基线 d；' +
        's = g 时 L_struct = 0。',
      '6. 梯度通路：L_struct 只到 g（不得出现对 z 的依赖）；lambda 是常数、不带梯度；编码器只经 straight-through。',
      '7. 分配规则不变：仍是 Sinkhorn 均衡分配；出现对机制打分取 argmin 的写法即为错。',
      '8. 数值一致性：math.json 的 constants 与 config/train_config.json 的实测值一致；' +
        '并且与机制文件 1_goal / 4_method 的描述不矛盾。',
      '',
      '改正规则：',
      '- 只改 math.json 里确实错的那一条（formula / note / symbols / constants），保持 JSON 结构与键名不变。',
      '- **只动 math.json 这一个文件**：不新建任何文件（状态、报告、清单都写在回答里，不要落盘），' +
        '不碰 idea 文件、不碰候选树、不碰配置。',
      '- 不改机制本身：公式与机制文字冲突时，以机制文字为准，把公式改成与之一致的写法，并在回答里说明冲突。',
      '- 不许发明新公式、不许调常量取值去"对上"某个数字、不许把 Sinkhorn 换成 argmin。',
      '- 找不到判定依据的（UNCHECKED），把这一条公式删掉：算不出来的东西不该留在规格里。' +
        '删的时候把只被它引用的符号一并删掉，别留悬空引用；在回答里说明删了哪一条、为什么。',
      '- 删掉判据公式要当心：若这一条是机制自己的判否条款，在回答里点明"判否条款已随该公式删除"。',
      '',
      '工具：ls、read、grep 只读；write 写仓库根之内的文本文件；' +
      'edit 只改文件里的几处（math.json 改那几条用它，别整份重写）。' +
        '每步只调 1-2 个工具。回答用中文，结论先行，改公式时给出旧式与新式。',
    ].join('\n'),
    steps: [
      {
        title: '读机制文件与 math.json',
        instruction:
          '用 ls 列出 ideas/，找到机制文件所在的方向目录，再用 read 读该目录的 *_idea.json 与 math.json。' +
          '把公式清单列出来（name + formula 一行一条），并记下 symbols 与 constants 的键。',
      },
      {
        title: '对照仓库既有实现',
        instruction:
          '用 read 读 distances/distance.py、model/layers.py、model/af3_structure.py、config/train_config.json，' +
          '记下基础距离、Huber、kappa 与 Sinkhorn 的真实定义，以及 sk_epsilon / sk_iters 的实测值。' +
          '机制文件点到的其他文件（例如它新建的模块）也要读，别按印象核对。',
      },
      {
        title: '逐条核对并给结论',
        instruction:
          '对每条公式给 PASS 或 FAIL 与依据。必须代入算例：三节点小图验 kappa 与 s；' +
          'x = delta ± 0.001 验 Huber 的一阶连续；s = g 验 L_struct = 0；' +
          's = 0、g = 0 与 lambda = 0 验两条退化式等于基线 d。' +
          'FAIL 的写下旧式与正确写法；找不到依据的（判不出 PASS 也不足以说 FAIL）' +
          '**直接删掉这一条公式**，并把只被它引用的符号一并删掉，在回答里说明删了哪条、为什么。',
      },
      {
        title: '有错就改 math.json（判不了的删掉）',
        instruction:
          '只改 FAIL 的那几条：用 edit 工具，一次调用把要改的条目都放进 edits[]（' +
          '每项 oldText 写该条的原文、newText 写改好的）；其余条目一个字都别动，键名与结构不动。' +
          '旧文本要逐字一致（在文件里唯一），对不上就 read 把那一条读准了再改。' +
          '不要用 write 整份重写 math.json —— 整份重写会超出一轮的输出，修正反而落不了地。' +
          '改完用 read 复查一次，确认仍是合法 JSON 且被改的条目就是刚才判定错的那几条。' +
          '一条都没错就明确写「无需修改」，不要为了动而改。',
      },
      {
        title: '记状态行（写在回答里）',
        instruction:
          '最后记一行状态：检查了几条公式、改了哪几条、删了哪几条（判不了的）与原因、最近一个 Commit SHA。' +
          '**这一行写在回答里，不要写成文件** —— 你这一轮只允许改 math.json，不新建任何文件。',
      },
    ],
  },
  {
    id: 'agent-stage2',
    name: 'AGENT STAGE2',
    role: 'STAGE2_RUNNER',
    /**
     * 这个 agent 被授的权限：读 + 训练。
     *
     * 训练这一项只给三个工具：启动、看状态、暂停；它们都转给仓库那个 MCP 服务
     * （tools/genrec-mcp/server.py），不另起一套调度。没有写权限 —— 修代码是
     * AGENT FIX 的活。
     */
    permissions: ['read', 'train'],
    handoffTargets: ['agent-fix'],
    instructions: [
      '你是 AGENT STAGE2（STAGE2_RUNNER），负责把一个候选树的 Stage2 训练跑起来、盯住它，' +
        '最后确认 SID 导出。服务的是 GeneRec 科研仓库。',
      '',
      '你的产出是「训练跑完 + item_sids.json 已导出」，或者「训练被暂停 + 问题交给 AGENT FIX」。',
      '',
      '一轮流程：',
      '1. 准备：读 ideas/<方向>/*_idea.json（arm 字段给出候选树名）与候选树里的 train_rqvae.py，' +
        '确认入口在、结果目录是 results/stage2_RQ-VAE/<arm>/。',
      '2. 启动：用 stage2Start 启动训练。它会先归档旧产物、跑一轮梯度冒烟，冒烟通过才开正式的' +
        '3000 轮训练；冒烟没过就不会开跑。',
      '3. 每分钟检查：反复调用 stage2Status（同一棵树的两次调用之间至少隔一分钟，工具会自己等）。' +
        '每次看三件事：latest_train 里的 epoch 在不在涨；worker_alive 与 live_processes：进程还在不在；' +
        'worker_log_tail 与 smoke_log_tail：有没有报错、Traceback、NaN、CUDA OOM。',
      '4. 发现问题就暂停：一旦出现异常（进程没了、epoch 不涨、日志里有报错），' +
        '立刻用 stage2Stop 把训练停掉，再把问题写清楚交给 AGENT FIX：哪个文件、哪一行、' +
        '日志里的原话、你判断错在哪。',
      '5. 跑完：status 是 finished 且 item_sids.json 已导出时，确认产物（条目数、日志里的最终 epoch），' +
        '然后汇报结果。',
      '',
      '硬约束：',
      '- 不改代码：修代码是 AGENT FIX 的活。你只负责跑、盯、停、报。',
      '- 不启动 Stage3：那是控制器的事。',
      '- 每分钟必须检查一次：不要一口气等很久才看一次。',
      '- 不许把"进程还在"当成"训练正常"：epoch 不涨或日志里有报错就是问题。',
      '- 没跑完不要说跑完：status 没到 finished、SID 没导出，就是没成。',
      '',
      '可用工具：ls、read、grep（只读），stage2Start、stage2Status、stage2Stop。' +
        '回答用中文，结论先行；提到文件时给仓库相对路径。',
    ].join('\n'),
    steps: [
      {
        title: '准备候选树与入口',
        instruction:
          '用 read 读 ideas/<方向>/*_idea.json（arm 字段给出候选树名）与候选树的 train_rqvae.py，' +
          '确认入口在（__main__ 与 _launch_via_torchrun）、ARM_NAME 与臂名一致、' +
          '结果目录是 results/stage2_RQ-VAE/<arm>/。' +
          '入口不在这棵树上就说明白，不要自己找别的树跑。',
      },
      {
        title: '启动训练',
        instruction:
          '用 stage2Start 启动这棵树的训练（参数是候选树路径）。' +
          '它返回 started/refused：冒烟没过会 refused 或 status 不是 running，' +
          '把原因写清楚就结束这一轮，不要自己改代码。',
      },
      {
        title: '每分钟检查状态',
        instruction:
          '每分钟检查一次：反复调用 stage2Status（同一棵树的两次调用之间至少隔一分钟，' +
          '工具会自己等到满一分钟）。' +
          '每次判断三件事：latest_train 的 epoch 在不在涨、worker_alive 与 live_processes 里进程还在不在、' +
          'worker_log_tail 与 smoke_log_tail 里有没有报错、Traceback、NaN 或 CUDA OOM。' +
          '一切正常就继续下一次检查，直到 status 变成 finished 或 exited_incomplete。',
      },
      {
        title: '有问题就暂停并交给 AGENT FIX',
        instruction:
          '一旦发现异常（进程没了、epoch 不涨、日志里有报错），立刻用 stage2Stop 把训练停掉。' +
          '然后把问题写清楚交给 AGENT FIX：哪个文件、哪一行、日志里的原话、你判断错在哪、' +
          '以及哪个公式或开关对不上。停下来的训练不算失败 —— 交给修的人才是下一步。',
      },
      {
        title: '跑完确认 SID 导出',
        instruction:
          'status 是 finished 时确认 item_sids.json 已经导出（用 ls 与 read 看结果目录），' +
          '报出条目数与最终 epoch；若 status 是 exited_incomplete 或 worker_gone，说明它没成，' +
          '写清卡在哪一步。' +
          '最后记一行状态：候选树、状态、跑了多少 epoch、SID 条数、有没有交给 AGENT FIX。',
      },
    ],
  },
  {
    id: 'agent-fix',
    name: 'AGENT FIX',
    role: 'MECHANISM_CHECKER',
    /**
     * 这个 agent 被授的权限：读 + 写。
     *
     * 它既审查也修：核对出来的写错之处，它在候选树里直接改对
     *（改动只落在 stage2_RQ-VAE/<arm>/ 之内）。不删文件、不联网、不跑训练。
     */
    permissions: ['read', 'write'],
    handoffTargets: [],
    /**
     * 默认的交接目标：AGENT FIX 不主动往下交 —— 它的结论交给人（控制器）。
     */
    instructions: [
      '你是 AGENT FIX（MECHANISM_CHECKER），负责核对并修好 AGENT WRITE 写的机制代码，' +
        '发现写错的地方就把它改对。服务的是 GeneRec 科研仓库。',
      '',
      '你的产出是「逐条 PASS / FAIL + 依据（文件与行）」，以及改好的代码。',
      '',
      '一轮流程：',
      '1. 读规格：ideas/<方向>/*_idea.json（1_goal、4_method 的 new_files）与同目录 math.json' +
        '（symbols、formulas、constants、switches）。这是唯一的标准。',
      '2. 读实现：候选树就是 idea 里 arm 字段指的那个目录（stage2_RQ-VAE/<arm>）；' +
        '读它的代码，并与冻结基线 stage2_RQ-VAE/TIGER_RQ-VAE 逐文件比对，看清到底改了哪些地方。' +
        'idea 没有 arm 字段时，在 stage2_RQ-VAE/ 下找与当前方向对应的那个候选树。',
      '3. 逐项核对（缺一条都不算查完）：',
      '   a. new_files 列出的文件是否都在，各自做的事是否与 idea 写的一致；',
      '   b. math.json 每条公式是否照写：符号、系数、常量（delta、权重）都要对得上；',
      '   c. 两个开关是否真的退化：behaviour_off 与 geometry_off 代入后等于基线的分配；',
      '   d. 分配规则仍是均衡 Sinkhorn，没有拿 argmin 取代它；',
      '   e. 路径规则：没有绝对路径、没有 Path(__file__).parents[N]，' +
        '相对冻结基线只改了 ARM_NAME 与机制本身；',
      '   f. 冻结超参没被动（config/train_config.json 的 21 项）；',
      '   g. 候选树 tests/ 下的测试是否真的在验那些公式与开关，而不是恒真断言。',
      '4. 改：把 FAIL 的条目在候选树里改对 ——',
      '   - 改的是候选树里的机制代码与它的测试（stage2_RQ-VAE/<arm>/ 之内），别的文件一律不碰；',
      '   - 公式照 math.json 抄、接口照冻结基线抄；每个改动都要说清它对应哪条 FAIL；',
      '   - 规格与实现冲突时以 spec 为准 —— 改实现，不改 idea 与 math.json；',
      '   - 补测试：每个改动的点都要有对应断言（先写会失败的测试，再改实现）；',
      '   - 冻结基线 TIGER_RQ-VAE 一个字都不许动；不启动训练、不 commit/push。',
      '5. 出结论：每条写 PASS 或 FAIL 与依据（文件:行）；改过的条目写清「旧写法 → 新写法」；' +
        '最后给一句总结论（可以进入训练 / 还有哪些没解决）。',
      '',
      '硬约束：',
      '- 改动只落在候选树里：idea 文件、math.json、冻结基线、config 都不许动。',
      '- 不发明新机制：规格与实现冲突时，以 spec（idea + math.json）为准，把冲突写清楚。',
      '- 判不准就写 UNCHECKED 并说明缺哪份证据，不要凭印象下结论。',
      '- 你没有执行命令的工具：跑不了测试就不要说跑过，只说测试文件里写了什么断言。',
      '',
      '可用工具：ls、read、grep（只读）、write（写文本文件）。' +
        '每步只调 1-2 个工具。回答用中文，结论先行；提到文件时给仓库相对路径与行号。',
    ].join('\n'),
    steps: [
      {
        title: '读规格',
        instruction:
          '用 ls 列出 ideas/，找到当前方向的目录，再用 read 读 *_idea.json 与 math.json。' +
          '把要核对的东西列成清单：new_files 每个文件的路径与用途、math.json 每条公式、' +
          '两个开关、常量。后面逐条对照的就是这份清单。',
      },
      {
        title: '读实现并与基线比对',
        instruction:
          '候选树是 idea 里 arm 字段指的目录（stage2_RQ-VAE/<arm>）；idea 没有这个字段时，' +
          '用 ls 列出 stage2_RQ-VAE/ 找出与当前方向对应的那个（冻结基线 TIGER_RQ-VAE 之外的目录）。' +
          '再用 read 读它的机制文件（new_files 列的那些）与 tests/ 下的测试文件。' +
          '用 grep 或 gitLog 看它相对冻结基线改了哪些地方 —— 改动范围本身就是核对项。',
      },
      {
        title: '逐项核对',
        instruction:
          '按上一轮列的清单逐项核对，每项给 PASS 或 FAIL 与依据（文件:行）：' +
          '文件是否齐、公式是否照写（含系数与常量）、两个开关是否退化、分配是否仍是均衡 Sinkhorn、' +
          '路径与 ARM_NAME 改动范围是否合规、冻结超参是否被动、测试是不是在验真东西。' +
          '判不准的写 UNCHECKED 并说明缺什么。',
      },
      {
        title: '改代码与测试',
        instruction:
          '把上一步判定 FAIL 的条目在候选树里改对：用 write 改 stage2_RQ-VAE/<arm>/ 里的机制文件' +
          '（公式照 math.json 抄、接口照冻结基线抄），并补上对应的测试断言' +
          '（先写会失败的测试，再改实现）。' +
          '每改一处要说清它对应哪条 FAIL。' +
          '只改候选树里的文件：idea、math.json、冻结基线 TIGER_RQ-VAE、config 都不许动；' +
          '不启动训练、不 commit/push。改完读回一次，确认写对了、路径没写错。',
      },
      {
        title: '出结论',
        instruction:
          '写最终结论：逐条结论的清单（改过的条目写清「旧写法 → 新写法」与依据），' +
          '加一句总结论 —— 可以进入训练 / 还有哪些没解决（说明缺什么）。' +
          '最后记一行状态：候选树、检查了几项、FAIL 几项、改了几处、总结论。',
      },
    ],
  },
];

/** 交接时上游 agent 的名字，用于在给下游的消息里标出结果来自谁。 */
export const HANDOFF_SOURCE_LABEL = 'AGENT A';

/**
 * 权限目录：每个 agent 能做什么，就看它被授了哪几项。
 *
 * 这里用的是 agent 最基础的那几项权限原语 —— read / write / delete /
 * execute / network —— 不是按业务分的组。分组的粒度是"能动哪种东西"，
 * 加一项新能力只需要往对应原语里挂工具。
 *
 * 每一项都绑定一组**真实存在的工具** —— 权限不是标签，是工具表的裁剪依据：
 * 没授的工具压根不会出现在 agent 的工具列表里，模型调不到它。
 *
 * 目录里出现不存在的工具名是配置坏了，要报错，不能悄悄跳过 ——
 * 否则"配了写权限但其实没写工具"这种半截状态没人发现得了。
 */
export const PERMISSION_CATALOG = [
  {
    id: 'read',
    label: '读（read）',
    description:
      '读仓库里的东西：列目录、读文件、按正则搜内容、找文件、校验 JSON、校验 idea 字段 —— 都限制在仓库根之内。',
    tools: ['read', 'ls', 'grep', 'find', 'checkJson', 'checkIdea'],
  },
  {
    id: 'write',
    label: '写（write）',
    description:
      '在仓库根之内写文本文件（新建或覆盖）、只改文件里的几处，以及从冻结基线复制一份候选树；不能写 .git 目录。',
    tools: ['write', 'edit', 'copyBaselineTree'],
  },
  {
    id: 'delete',
    label: '删（delete）',
    description: '删仓库里的文件或目录；不能删 .git 目录，也不能删仓库根。',
    tools: ['deleteFile'],
  },
  {
    id: 'train',
    label: '训练（train）',
    description:
      '跑 Stage2 训练并盯住它：启动、看状态、暂停。训练只走仓库那个 MCP 接口' +
      '（tools/genrec-mcp/server.py），不另起一套调度。',
    tools: ['stage2Start', 'stage2Status', 'stage2Stop'],
  },
  {
    id: 'execute',
    label: '执行（execute）',
    description:
      '执行代码里写死的只读命令（当前是 git log，读最近的提交与它改过的文件）—— 不是任意命令执行。',
    tools: ['gitLog'],
  },
  {
    id: 'web',
    label: '检索（web）',
    description:
      '检索已发表文献（OpenAlex 公开接口，不需要 key）：拿标题、年份、出处、DOI 与被引数，' +
      '结论由 agent 自己读。',
    tools: ['webSearch'],
  },
  {
    id: 'network',
    label: '联网（network）',
    description: '访问仓库之外的网络服务：列出并读取 GitHub 上本仓库的 Issue。',
    tools: ['listIssues', 'readIssue'],
  },
];

/** 目录里所有权限 id —— 校验时用来判"这个权限存不存在"。 */
const KNOWN_PERMISSION_IDS = new Set(PERMISSION_CATALOG.map((item) => item.id));

/**
 * 新建 agent 时默认给的权限：读、执行、联网 —— 全是不改仓库的。
 *
 * 写与删默认不开 —— 开它等于让 agent 能改/删仓库里的文件，
 * 这件事该由用户在页面上明确勾出来，不该由默认值替他决定。
 */
export const DEFAULT_PERMISSIONS = ['read', 'execute', 'network'];

/**
 * 把一份权限列表整理成规范形状。
 *
 * 没有权限是预期内的缺失（空数组 = 一项也不要）。
 * 但写坏了不是：不是数组、或者出现了目录里没有的 id，都直接报错 ——
 * 悄悄丢掉一个不认识的权限，等于悄悄给 agent 少授了一项能力，
 * 用户以为给了、其实没给，这种错最难查。
 */
export function normalizePermissions(list) {
  if (list === undefined || list === null) return [];
  if (!Array.isArray(list)) {
    throw new Error(`权限必须是一个数组：${JSON.stringify(list)}`);
  }
  const kept = [];
  const seen = new Set();
  for (const id of list) {
    if (typeof id !== 'string' || !KNOWN_PERMISSION_IDS.has(id)) {
      throw new Error(
        `不认识的权限：${JSON.stringify(id)}` +
          `（可选：${[...KNOWN_PERMISSION_IDS].join('、')}）`,
      );
    }
    if (seen.has(id)) continue;
    seen.add(id);
    kept.push(id);
  }
  return kept;
}

/**
 * 这一轮到底用哪几项权限。
 *
 * requested === undefined：请求里没这个字段 → 用这个 agent 自己的默认权限。
 * requested 是数组（哪怕是空的）：请求里明确给了 → 就用这份，空数组表示一项都不要。
 *
 * 这两件事必须分得开，跟 instructions 一个道理：
 * "没配过"与"明确清空"不是一回事。
 */
export function resolvePermissions(profile, requested) {
  if (requested === undefined || requested === null) {
    return normalizePermissions(profile.permissions);
  }
  return normalizePermissions(requested);
}

/**
 * 写进系统提示词的权限说明。
 *
 * 工具裁剪是硬约束（模型调不到没授的工具），这段文字是软约束
 * （让模型知道自己有几项权限、别去幻想没给它的能力）。
 * 两边都要有：只有硬约束，模型会反复试着调不存在的工具；
 * 只有软约束，模型会嘴上说"我已经改好文件了"。
 */
export function buildPermissionPrompt(permissionIds) {
  const ids = normalizePermissions(permissionIds);
  if (ids.length === 0) {
    return [
      '【本次运行授予你的权限】',
      '- 一项也没有：你没有任何工具，既不能读也不能写。',
      '不要假装自己能查、能改；做不到的事就直说做不到。',
    ].join('\n');
  }
  const lines = ids.map((id) => {
    const item = PERMISSION_CATALOG.find((entry) => entry.id === id);
    return `- ${item.label}（${id}）：${item.description} 可用工具：${item.tools.join('、')}`;
  });
  return [
    '【本次运行授予你的权限】',
    ...lines,
    '这份清单之外的能力你没有：不要声称自己能读到别的地方、能跑命令、能提交或推送。',
    '做不到的事就直说做不到。',
  ].join('\n');
}

/** 默认人设（第一个 profile）的指令说明。 */
export const AGENT_INSTRUCTIONS = AGENT_PROFILES[0].instructions;

/**
 * 把一个 agent 的默认步骤整理成页面能直接用的形状。
 *
 * 没有步骤是预期内的缺失（新建的 agent 本来就没有默认步骤）。
 * 但写坏了（不是数组、条目不是对象、缺标题或缺指令）是配置坏了 ——
 * 直接报错，别把坏配置悄悄变成空步骤，让页面上的 agent 少了步骤却没人知道。
 */
function normalizeSteps(steps) {
  if (steps === undefined || steps === null) return [];
  if (!Array.isArray(steps)) {
    throw new Error(`agent 的默认步骤必须是一个数组：${JSON.stringify(steps)}`);
  }
  return steps.map((step, index) => {
    if (!step || typeof step !== 'object') {
      throw new Error(`agent 的默认步骤第 ${index} 条不是对象`);
    }
    const title = typeof step.title === 'string' ? step.title : '';
    const instruction = typeof step.instruction === 'string' ? step.instruction : '';
    if (title.trim() === '' || instruction.trim() === '') {
      throw new Error(`agent 的默认步骤第 ${index} 条缺标题或缺指令`);
    }
    return { title, instruction };
  });
}

/**
 * 页面上所有 agent 的公开身份。
 *
 * instructions 一起给出去：页面上的提示词框要用它当默认值，
 * 用户改了之后随请求发回来，服务端拿它替换默认提示词。
 *
 * steps 也一起给出去：AGENT A / AGENT B 的提示词拆成的固定步骤，
 * 页面在用户没自己配过之前就铺好这几步。
 *
 * 运行时新建的 agent 也在这份表里 —— 页面上的面板与服务端的身份表始终一致。
 */
export function listProfiles() {
  return [...AGENT_PROFILES, ...runtimeProfiles.values()].map((profile) =>
    publicProfile(profile),
  );
}

/**
 * 落盘句柄。
 *
 * 默认写在 agent-vis/.data/ 下；setDataDir 换目录时整份注册表重新读一遍。
 */
let persistence = createPersistence({ dataDir: DEFAULT_DATA_DIR });

/** 当前生效的数据目录。pi 的派生配置（models.json）与临时凭证路径都挂在它下面。 */
let currentDataDir = DEFAULT_DATA_DIR;

/** 换数据目录之后要跟着走的派生物也走这里，不要另起一套路径。 */
export function dataDir() {
  return currentDataDir;
}

/** 交接延迟的上限：一小时。再长就不该用这个功能了（那是定时任务）。 */
const MAX_HANDOFF_DELAY_SECONDS = 3600;

/** 交接的默认延迟：0 = 跑完立刻发。 */
export const DEFAULT_HANDOFF_DELAY_SECONDS = 0;

/**
 * 交接条件：写给人看的一句话，由轮次最后一步的模型来判定。
 *
 * 空串 = 无条件（这一步总是交接）。非空时这句话原样写进最后一步的指令里，
 * 由模型决定要不要调用 requestHandoff —— 不靠关键词匹配。
 */
export const DEFAULT_HANDOFF_CONDITION = '';

/** 循环次数上限：再多就该用定时任务，页面上的自动重跑按百次封顶。 */
const MAX_LOOP_COUNT = 100;

/** 循环间隔上限：与交接延迟同口径，一小时。 */
const MAX_LOOP_INTERVAL_SECONDS = 3600;

/** 默认不开循环：点一次「运行」就跑一次；选中之后按次数自动重跑。 */
export const DEFAULT_LOOP = Object.freeze({ enabled: false, count: 2, intervalSeconds: 0 });

/**
 * 每一轮之后要不要再跑一轮 —— 由谁说了算。
 *
 * off   ：不设闸，跑满 loop.count 轮（原来的行为）。
 * ask   ：每轮结束后面板上问用户，由用户决定要不要继续。
 * files ：客观判定：这一轮有没有真的改到被审的文件（ideas/ 当前方向 + 候选树）。
 *         改了才继续，没改就停 —— 不问模型，不读摘要。
 * agent ：每轮结束后让这个 agent 自己按 prompt 里的判据说一句要不要继续。
 */
export const LOOP_GATE_MODES = Object.freeze(['off', 'ask', 'files', 'agent']);

/**
 * 每个 agent 自己指定"被审的文件"：循环闸（files 模式）看的就是这些。
 *
 * 写法是相对仓库根的路径模式，星号匹配一段目录名，例如
 *   'ideas/当前方向/math.json'   —— 当前方向的那份 math.json
 *   'ideas/当前方向/xxx_idea.json' —— 当前方向的 idea 文件
 * 不配就用默认的一套（当前方向的所有文件 + 候选树，不含归档与冻结基线）。
 */
export const MAX_LOOP_WATCH_FILES = 8;
const MAX_LOOP_WATCH_PATTERN = 200;

/** 校验并规范化"被审文件"的模式列表。 */
export function normalizeLoopWatchFiles(value) {
  if (!Array.isArray(value)) {
    throw new Error('loopWatchFiles 必须是数组');
  }
  if (value.length > MAX_LOOP_WATCH_FILES) {
    throw new Error(`loopWatchFiles 最多 ${MAX_LOOP_WATCH_FILES} 条`);
  }
  return value.map((item) => {
    if (typeof item !== 'string' || item.trim() === '') {
      throw new Error('loopWatchFiles 的每一项都要是非空字符串');
    }
    const pattern = item.trim();
    if (pattern.length > MAX_LOOP_WATCH_PATTERN) {
      throw new Error(`loopWatchFiles 的每一项最多 ${MAX_LOOP_WATCH_PATTERN} 字`);
    }
    if (pattern.startsWith('/')) {
      throw new Error(`loopWatchFiles 的路径要相对仓库根，不要以 / 开头: ${pattern}`);
    }
    return pattern;
  });
}
/**
 * 闸开着时，loop.count 就不再是"跑几轮就停"，而是闸说了算：
 *   ask   —— 用户每轮之后点一次；
 *   agent —— 这个 agent 每轮之后按判据自己判。
 * 跑多少轮由闸决定，count 只留一个硬上限（MAX_LOOP_COUNT），防止闸一直放行时没完没了。
 */
export const DEFAULT_LOOP_GATE = Object.freeze({ mode: 'off', prompt: '' });
const MAX_LOOP_GATE_PROMPT = 500;

/** 校验并规范化"每轮之后要不要再跑一轮"的设置。 */
export function normalizeLoopGate(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('loopGate 必须是 {mode, prompt} 对象');
  }
  const { mode, prompt } = value;
  if (!LOOP_GATE_MODES.includes(mode)) {
    throw new Error(`loopGate.mode 只能是 ${LOOP_GATE_MODES.join(' / ')}`);
  }
  if (typeof prompt !== 'string') {
    throw new Error('loopGate.prompt 必须是字符串');
  }
  if (prompt.length > MAX_LOOP_GATE_PROMPT) {
    throw new Error(`loopGate.prompt 最多 ${MAX_LOOP_GATE_PROMPT} 字`);
  }
  return { mode, prompt: prompt.trim() };
}

/**
 * 校验循环配置。
 *
 * 三项一起给：只给 enabled 会让"次数/间隔"退回默认，用户看到的输入框与实际生效
 * 的值就会对不上。整份要么全对，要么整份拒掉。
 */
export function normalizeLoop(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('loop 必须是 {enabled, count, intervalSeconds} 对象');
  }
  const { enabled, count, intervalSeconds } = value;
  if (typeof enabled !== 'boolean') throw new Error('loop.enabled 必须是布尔值');
  // count = 0 表示不限次数：闸一直放行就接着跑。
  if (!Number.isInteger(count) || count < 0 || count > MAX_LOOP_COUNT) {
    throw new Error(`loop.count 必须是 0 到 ${MAX_LOOP_COUNT} 的整数（0 = 不限次数）`);
  }
  if (!Number.isInteger(intervalSeconds)
      || intervalSeconds < 0 || intervalSeconds > MAX_LOOP_INTERVAL_SECONDS) {
    throw new Error(`loop.intervalSeconds 必须是 0 到 ${MAX_LOOP_INTERVAL_SECONDS} 的整数`);
  }
  return { enabled, count, intervalSeconds };
}

/** 名字的长度上限：面板标题、左侧栏、交接卡片都只放得下一行。 */
const MAX_AGENT_NAME_LENGTH = 40;

/**
 * 校验并规范化用户给的名字。
 *
 * 空名字、控制字符、超长一律拒掉：名字会出现在页头、侧栏与交接卡片上，
 * 一个合法但离谱的名字（换行、几百字）会把界面画坏。
 */
function normalizeAgentName(value) {
  if (typeof value !== 'string') throw new Error('name 必须是字符串');
  const name = value.trim();
  if (name === '') throw new Error('name 不能是空字符串');
  if (name.length > MAX_AGENT_NAME_LENGTH) {
    throw new Error(`name 最多 ${MAX_AGENT_NAME_LENGTH} 个字符`);
  }
  if (/[\u0000-\u001f\u007f]/.test(name)) throw new Error('name 里不允许控制字符');
  return name;
}

/**
 * 用户改过的配置，按 agent 身份存。
 *
 * 只存改过的字段：没改过的字段继续用代码里的默认值 ——
 * 改了代码里的默认提示词，页面上没被用户动过的那一份会跟着变。
 */
let userConfigs = {};

/**
 * 运行时新建的 agent（完整身份）。
 *
 * 落盘：服务重启它们还在，刷新页面它们也还在 ——
 * 两边是一起消失的，不会留下"面板还在、身份没了"的半截状态。
 */
const runtimeProfiles = new Map();

/**
 * 把存下来的权限分成"目录里还认得的"与"已经不认得的"。
 *
 * 存下来的权限里可能有过期项：服务端改过权限目录之后，之前存的 id
 * 就不在目录里了。这些 id 留着等于给服务端一个它不认识的权限。
 *
 * 但不能因此把整份配置丢掉 —— 用户会发现自己的权限被清空了。
 * 所以把过期的挑出来单独报出去（stalePermissions），
 * 剩下的照常生效；页面看到这份清单就会把过期的那几项去掉并存回去。
 *
 * 形状坏了直接报错：存下来的东西不是权限 id 数组，说明文件被改坏了。
 */
function splitKnownPermissions(list) {
  if (!Array.isArray(list)) {
    throw new Error(`存下来的权限必须是一个数组：${JSON.stringify(list)}`);
  }
  const known = [];
  const stale = [];
  for (const id of list) {
    if (typeof id !== 'string') {
      throw new Error(`存下来的权限里有一项不是权限 id：${JSON.stringify(id)}`);
    }
    if (KNOWN_PERMISSION_IDS.has(id)) known.push(id);
    else stale.push(id);
  }
  return { known, stale };
}

/**
 * 一个 agent 的公开身份：代码里的默认值 + 用户改过的那部分。
 *
 * customized 标出哪些字段是用户动过的 —— 页面靠它分得清
 * "用户配过步骤"与"还没配过"：前者尊重用户的清空，后者用默认步骤顶上。
 */
function publicProfile(profile) {
  const override = userConfigs[profile.id] || {};
  const has = (key) => Object.prototype.hasOwnProperty.call(override, key);
  const storedPermissions = has('permissions') ? override.permissions : profile.permissions;
  const splitPermissions = splitKnownPermissions(storedPermissions);
  return {
    id: profile.id,
    // 生效的名字：用户改过就用用户的。下面同时给出代码里的默认名字，
    // 页面「恢复默认」要靠它。
    name: has('name') ? override.name : profile.name,
    defaultName: profile.name,
    role: profile.role,
    // 生效的值：用户改过就用用户的，没改过用代码里的默认。
    instructions: has('instructions') ? override.instructions : profile.instructions,
    steps: has('steps') ? normalizeUserSteps(override.steps) : normalizeSteps(profile.steps),
    // 这个 agent 默认被授的权限：页面上的权限区按它铺复选框。
    permissions: splitPermissions.known,
    // 已经不在权限目录里的那几项：页面要把它们去掉并存回去。
    stalePermissions: splitPermissions.stale,
    // 这一栏跑完的结果要发给谁。用户没改过就用代码里的默认
    // （AGENT A 默认交给 AGENT B）。
    handoffTargets: has('handoffTargets')
      ? [...override.handoffTargets]
      : [...(profile.handoffTargets || [])],
    // 这一栏跑完之后，隔多少秒才把结果发给勾选的目标（0 = 立刻）。
    handoffDelaySeconds: has('handoffDelaySeconds')
      ? override.handoffDelaySeconds
      : DEFAULT_HANDOFF_DELAY_SECONDS,
    defaultHandoffDelaySeconds: DEFAULT_HANDOFF_DELAY_SECONDS,
    // 交接前要不要先看一个条件：条件满足时发、或者跳过（见 DEFAULT_HANDOFF_CONDITION）。
    handoffCondition:
      has('handoffCondition') && typeof override.handoffCondition === 'string'
        ? override.handoffCondition
        : DEFAULT_HANDOFF_CONDITION,
    // 允许把结果回传给"刚把它发给我的那一栏"：默认不允许（防来回触发），
    // 审查—返修的循环要靠它打开。
    allowHandoffBack: has('allowHandoffBack') ? Boolean(override.allowHandoffBack) : false,
    // 交接时要不要先重置这一栏自己的会话：重置了，下游拿到消息开始干活时，
    // 这一栏的模型上下文已经清空（下一轮从头开始）。
    resetSessionOnHandoff: has('resetSessionOnHandoff')
      ? Boolean(override.resetSessionOnHandoff)
      : false,
    // 交接时要不要把"接收方"的会话也重置：重置了，接收方拿到这条消息时
    // 从零开始，只看这一条消息（不被自己上一轮的上下文带偏）。
    resetReceiverSessionOnHandoff: has('resetReceiverSessionOnHandoff')
      ? Boolean(override.resetReceiverSessionOnHandoff)
      : false,
    // 循环：跑完这一轮之后自动再跑几轮、每轮隔多久。用户没配过就是不开。
    loop: has('loop') ? normalizeLoop(override.loop) : normalizeLoop(profile.loop || DEFAULT_LOOP),
    defaultLoop: normalizeLoop(profile.loop || DEFAULT_LOOP),
    // 每轮之后要不要再跑一轮：用户没配过就是不开（跑满 count 轮）。
    loopGate: has('loopGate')
      ? normalizeLoopGate(override.loopGate)
      : normalizeLoopGate(profile.loopGate || DEFAULT_LOOP_GATE),
    defaultLoopGate: normalizeLoopGate(profile.loopGate || DEFAULT_LOOP_GATE),
    // 每轮之前要不要把这个 agent 的会话重置（它的上下文从零开始，聊天记录留着）。
    loopResetSession:
      has('loopResetSession')
        ? Boolean(override.loopResetSession)
        : Boolean(profile.loopResetSession),
    defaultLoopResetSession: Boolean(profile.loopResetSession),
    // 这一栏被审的文件：循环闸（files 模式）看的就是它们。
    loopWatchFiles: has('loopWatchFiles')
      ? normalizeLoopWatchFiles(override.loopWatchFiles)
      : [...(profile.loopWatchFiles || [])],
    defaultLoopWatchFiles: [...(profile.loopWatchFiles || [])],
    // 每轮之前要不要把这个 agent 的会话重置（它的上下文从零开始，聊天记录留着）。
    loopResetSession:
      has('loopResetSession')
        ? Boolean(override.loopResetSession)
        : Boolean(profile.loopResetSession),
    defaultLoopResetSession: Boolean(profile.loopResetSession),
    // 挂载面板时输入框里预填的指令（只有配置里写了才会有；其余 agent 是空的）。
    defaultPrompt: profile.defaultPrompt || '',
    // 代码里写死的默认值：用户点「恢复默认」时要把它们铺回去，
    // 所以得单独给一份 —— 生效值可能已经被用户改过了。
    defaultInstructions: profile.instructions,
    defaultSteps: normalizeSteps(profile.steps),
    defaultPermissions: normalizePermissions(profile.permissions),
    // 配置里写死的身份删不掉（AGENT A / AGENT B / AGENT MATH）；页面上删掉的只是面板。
    builtin: !runtimeProfiles.has(profile.id),
    customized: {
      name: has('name'),
      handoffDelay: has('handoffDelaySeconds'),
      loop: has('loop'),
      loopGate: has('loopGate'),
      loopResetSession: has('loopResetSession'),
      loopWatchFiles: has('loopWatchFiles'),
      instructions: has('instructions'),
      steps: has('steps'),
      permissions: has('permissions'),
      handoff: has('handoffTargets'),
    },
  };
}

/**
 * 用户自己配的轮次步骤。
 *
 * 与代码里的默认步骤不同：这里允许标题或指令是空串 ——
 * 用户正在打字时标题可能还没填上，那不该把整份配置拒掉。
 * 但形状坏了（不是数组、条目不是对象、字段不是字符串）照样报错。
 */
function normalizeUserSteps(steps) {
  if (!Array.isArray(steps)) {
    throw new Error(`轮次步骤必须是一个数组：${JSON.stringify(steps)}`);
  }
  return steps.map((step, index) => {
    if (!step || typeof step !== 'object' || Array.isArray(step)) {
      throw new Error(`轮次步骤第 ${index} 条不是对象`);
    }
    const title = step.title === undefined ? '' : step.title;
    const instruction = step.instruction === undefined ? '' : step.instruction;
    if (typeof title !== 'string') {
      throw new Error(`轮次步骤第 ${index} 条的标题不是字符串`);
    }
    if (typeof instruction !== 'string') {
      throw new Error(`轮次步骤第 ${index} 条的指令不是字符串`);
    }
    return { title, instruction };
  });
}

/** 把注册表写到盘上。 */
function persist() {
  persistence.saveAgents({
    version: 1,
    configs: userConfigs,
    agents: Object.fromEntries(runtimeProfiles),
    settings: globalSettings,
    groups: globalGroups,
  });
}

/** 从盘上把注册表读回来。 */
function hydrate() {
  const state = persistence.loadAgents();
  userConfigs = state.configs;
  // 全局设置：文件里没有就用默认值（老版本的 agents.json 没有这一段）。
  const stored = state.settings && typeof state.settings === 'object' ? state.settings : {};
  // 组：老文件里没有这一段就当没有组。
  const storedGroups = Array.isArray(state.groups) ? state.groups : [];
  globalGroups = storedGroups
    .filter((group) => group && typeof group.id === 'string' && typeof group.name === 'string')
    .map((group) => ({
      id: group.id,
      name: group.name,
      members: groupMembers(group).filter((id) => typeof id === 'string'),
    }));
  globalSettings = {
    agentHost:
      typeof stored.agentHost === 'string' && stored.agentHost.trim() !== ''
        ? stored.agentHost.trim()
        : DEFAULT_GLOBAL_SETTINGS.agentHost,
    agentWorkdir:
      typeof stored.agentWorkdir === 'string' && stored.agentWorkdir.trim() !== ''
        ? stored.agentWorkdir.trim()
        : DEFAULT_GLOBAL_SETTINGS.agentWorkdir,
  };
  runtimeProfiles.clear();
  for (const [id, profile] of Object.entries(state.agents)) {
    runtimeProfiles.set(id, profile);
  }
}

/**
 * 换数据目录（测试用），并把注册表重新读一遍。
 *
 * 正式服务用默认目录；测试用仓库内的临时目录，
 * 免得测试把用户自己配好的东西冲掉。
 */
export function setDataDir(dataDir) {
  persistence = createPersistence({ dataDir });
  currentDataDir = dataDir;
  hydrate();
}

/** 追加一条对话记录。 */
export function appendChatRecord(agentId, record) {
  persistence.appendChat(agentId, record);
}

/** 读出某个 agent 的全部对话记录（按写入顺序）。 */
export function listChatRecords(agentId) {
  return persistence.readChats(agentId);
}

/** 清掉某个 agent 的对话记录。返回有没有真的删掉东西。 */
export function clearChatRecords(agentId) {
  return persistence.clearChats(agentId);
}

/**
 * 下一个运行时 agent 的序号。
 *
 * 从已有的身份里取最大值加一 —— 删掉一个 agent 之后序号不会退回去，
 * 否则新 agent 会拿到一个已经被用过的 id，把旧身份盖掉。
 */
function nextRuntimeSeq() {
  let max = 0;
  for (const id of runtimeProfiles.keys()) {
    const matched = /^custom-(\d+)$/.exec(id);
    if (matched) max = Math.max(max, Number(matched[1]));
  }
  return max + 1;
}

/** 序号转字母：3 -> C、26 -> Z、27 -> AA（接在 A、B 后面往下排）。 */
function agentLetter(n) {
  let rest = n;
  let letters = '';
  while (rest > 0) {
    const r = (rest - 1) % 26;
    letters = String.fromCharCode(65 + r) + letters;
    rest = Math.floor((rest - 1) / 26);
  }
  return letters;
}

/**
 * 新建一个 agent。
 *
 * 系统提示词默认为空：新 agent 的一切由用户在页面上自己写。
 * 名字没给（或给的是空串）时按 A、B 的次序自动往后排：AGENT C、AGENT D……
 *
 * 权限没给就按三项只读起步；给了就按给的来。
 * 权限写坏了（不认识的 id）在这里就报错，别把坏权限悄悄存成"没权限"。
 */
export function registerProfile({ name, instructions, permissions }) {
  const seq = nextRuntimeSeq();
  const trimmedName = typeof name === 'string' ? name.trim() : '';
  const profile = {
    id: `custom-${seq}`,
    name: trimmedName === '' ? `AGENT ${agentLetter(seq + 2)}` : trimmedName,
    role: 'CUSTOM',
    instructions: typeof instructions === 'string' ? instructions : '',
    // 新建的 agent 没有默认步骤：它的一切由用户在页面上自己写。
    steps: [],
    // 权限没给就是三项只读；给了就按给的来（写坏了在上面就报错了）。
    permissions:
      permissions === undefined || permissions === null
        ? [...DEFAULT_PERMISSIONS]
        : normalizePermissions(permissions),
    handoffTargets: [],
  };
  runtimeProfiles.set(profile.id, profile);
  persist();
  return publicProfile(profile);
}

/**
 * 删掉一个运行时新建的 agent。
 *
 * 配置里写死的（agent-a / agent-b / agent-math）删不掉 ——
 * 页面上删它们的只是面板，身份还在。
 * 身份不存在就直接报错，不假装删成功。
 */
export function deleteProfile(profileId) {
  const fixed = AGENT_PROFILES.find((profile) => profile.id === profileId);
  if (fixed) {
    throw new Error(`配置里写死的 agent 不能从服务端删除: ${profileId}`);
  }
  if (!runtimeProfiles.has(profileId)) {
    throw new Error(`未知的 agent 身份: ${profileId}`);
  }
  runtimeProfiles.delete(profileId);
  delete userConfigs[profileId];
  persist();
  return true;
}

/**
 * 存下这个 agent 的配置改动。
 *
 * 只存改过的字段：没动的字段继续用代码里的默认值。
 * 字段写坏了直接报错，不拿默认值悄悄盖过去 ——
 * "用户把权限全取消了"与"用户没配过权限"必须分得开。
 */
export function saveProfileConfig(profileId, patch) {
  if (!patch || typeof patch !== 'object' || Array.isArray(patch)) {
    throw new Error('配置必须是一个 JSON 对象');
  }
  // 身份认不出来就直接报错 —— 不给不存在的 agent 存配置。
  resolveProfile(profileId);

  const next = { ...(userConfigs[profileId] || {}) };
  // 值传 null = 用户点了「恢复默认」：把这一项的用户改动抹掉，
  // 生效值回到代码里的默认。其余没提到的字段原样留着。
  if (patch.name !== undefined) {
    if (patch.name === null) {
      delete next.name;
    } else {
      next.name = normalizeAgentName(patch.name);
    }
  }
  if (patch.instructions !== undefined) {
    if (patch.instructions === null) {
      delete next.instructions;
    } else if (typeof patch.instructions !== 'string') {
      throw new Error('instructions 必须是字符串');
    } else {
      next.instructions = patch.instructions;
    }
  }
  if (patch.permissions !== undefined) {
    if (patch.permissions === null) {
      delete next.permissions;
    } else {
      next.permissions = normalizePermissions(patch.permissions);
    }
  }
  if (patch.steps !== undefined) {
    if (patch.steps === null) {
      delete next.steps;
    } else {
      next.steps = normalizeUserSteps(patch.steps);
    }
  }
  if (patch.handoffCondition !== undefined) {
    if (patch.handoffCondition === null) {
      delete next.handoffCondition;
    } else if (typeof patch.handoffCondition !== 'string') {
      throw new Error('handoffCondition 必须是一句话（字符串）');
    } else {
      next.handoffCondition = patch.handoffCondition.trim();
    }
  }
  if (patch.allowHandoffBack !== undefined) {
    if (patch.allowHandoffBack === null) {
      delete next.allowHandoffBack;
    } else if (typeof patch.allowHandoffBack !== 'boolean') {
      throw new Error('allowHandoffBack 必须是布尔值');
    } else {
      next.allowHandoffBack = patch.allowHandoffBack;
    }
  }
  if (patch.resetReceiverSessionOnHandoff !== undefined) {
    if (patch.resetReceiverSessionOnHandoff === null) {
      delete next.resetReceiverSessionOnHandoff;
    } else if (typeof patch.resetReceiverSessionOnHandoff !== 'boolean') {
      throw new Error('resetReceiverSessionOnHandoff 必须是布尔值');
    } else {
      next.resetReceiverSessionOnHandoff = patch.resetReceiverSessionOnHandoff;
    }
  }
  if (patch.resetSessionOnHandoff !== undefined) {
    if (patch.resetSessionOnHandoff === null) {
      // null = 恢复默认（跟名字、提示词一样：把这一项从覆写里摘掉）。
      delete next.resetSessionOnHandoff;
    } else if (typeof patch.resetSessionOnHandoff !== 'boolean') {
      throw new Error('resetSessionOnHandoff 必须是布尔值');
    } else {
      next.resetSessionOnHandoff = patch.resetSessionOnHandoff;
    }
  }
  if (patch.handoffDelaySeconds !== undefined) {
    if (patch.handoffDelaySeconds === null) {
      delete next.handoffDelaySeconds;
    } else {
      const delay = patch.handoffDelaySeconds;
      if (typeof delay !== 'number' || !Number.isInteger(delay)) {
        throw new Error('handoffDelaySeconds 必须是整数秒');
      }
      if (delay < 0 || delay > MAX_HANDOFF_DELAY_SECONDS) {
        throw new Error(`handoffDelaySeconds 必须在 0 到 ${MAX_HANDOFF_DELAY_SECONDS} 之间`);
      }
      next.handoffDelaySeconds = delay;
    }
  }
  if (patch.loopWatchFiles !== undefined) {
    if (patch.loopWatchFiles === null) {
      delete next.loopWatchFiles;
    } else {
      next.loopWatchFiles = normalizeLoopWatchFiles(patch.loopWatchFiles);
    }
  }
  if (patch.loopResetSession !== undefined) {
    if (patch.loopResetSession === null) {
      delete next.loopResetSession;
    } else {
      next.loopResetSession = Boolean(patch.loopResetSession);
    }
  }
  if (patch.loopGate !== undefined) {
    if (patch.loopGate === null) {
      delete next.loopGate;
    } else {
      next.loopGate = normalizeLoopGate(patch.loopGate);
    }
  }
  if (patch.loop !== undefined) {
    if (patch.loop === null) {
      delete next.loop;
    } else {
      next.loop = normalizeLoop(patch.loop);
    }
  }
  if (patch.handoffTargets !== undefined) {
    if (patch.handoffTargets === null) {
      delete next.handoffTargets;
    } else {
      if (!Array.isArray(patch.handoffTargets)) {
        throw new Error('handoffTargets 必须是一个数组');
      }
      next.handoffTargets = patch.handoffTargets.map((id) => {
        if (typeof id !== 'string' || id === '') {
          throw new Error(`handoffTargets 里有一项不是 agent 身份：${JSON.stringify(id)}`);
        }
        return id;
      });
    }
  }
  // 一项都没剩下 = 用户把这一栏的改动全撤了。这时候盘上不该留一个空对象：
  // 那是没有信息量的壳，只会让 agents.json 越攒越脏。
  //
  // 注意区分：「权限清空成空数组」是有意义的用户改动（明说不要任何权限），
  // 它在本函数里存成 permissions: []，键还在，因此照旧留着。
  if (Object.keys(next).length === 0) {
    delete userConfigs[profileId];
  } else {
    userConfigs[profileId] = next;
  }
  persist();
  // resolveProfile 返回的已经是"生效值"那份（publicProfile），
  // 不能再套一层 —— 套了之后 defaultName 会被写成生效值，等于没有默认名字了。
  return resolveProfile(profileId);
}

/**
 * 按 id 解析 agent 身份。
 * 认不出来就直接报错 —— 不退回默认身份，免得请求悄悄跑成另一个 agent。
 */
export function resolveProfile(profileId) {
  if (profileId === undefined || profileId === null || profileId === '') {
    return publicProfile(AGENT_PROFILES[0]);
  }
  const found = AGENT_PROFILES.find((profile) => profile.id === profileId);
  if (found) return publicProfile(found);
  // 页面上「新建 Agent」造出来的身份也在表里。
  const custom = runtimeProfiles.get(profileId);
  if (custom) return publicProfile(custom);
  throw new Error(`未知的 agent 身份: ${profileId}`);
}

/**
 * 交接消息里该写上游的哪个名字。
 *
 * 上游的名字由请求带过来（handoffFrom）—— 交接不再只有 A → B 这一条路，
 * 来源名字得跟着实际发送方走。
 *
 * 没带名字、或带的是空串：这是"预期内的缺失"，用固定的那个名字
 * （HANDOFF_SOURCE_LABEL）。这不是兜底 —— 名字缺失时下游仍然知道
 * 结果来自谁，只是不知道是哪个新建的 agent。
 */
export function handoffSourceLabel(sourceLabel) {
  if (typeof sourceLabel !== 'string') return HANDOFF_SOURCE_LABEL;
  const trimmed = sourceLabel.trim();
  if (trimmed === '') return HANDOFF_SOURCE_LABEL;
  return trimmed;
}

/**
 * 启动时把注册表从盘上读回来。
 *
 * 读坏了直接抛错（见 persistence.js）：配置读不出来时继续跑，
 * 等于让用户以为自己的配置还在，其实已经被当成了默认值。
 */
hydrate();

/**
 * 交接：把上游 agent 的分析结果和给下游的要求拼成一条用户消息。
 *
 * 上游结果放前面 —— 下游得先读它；要求放最后 —— 那是它真正要执行的事。
 * 没有上游结果时就原样返回要求，不加任何占位说明。
 */
export function buildHandoffPrompt(prompt, handoff, sourceLabel) {
  const upstream = typeof handoff === 'string' ? handoff.trim() : '';
  if (upstream === '') return prompt;
  const head = [`【${handoffSourceLabel(sourceLabel)} 的分析结果】`, upstream].join('\n');
  // 下游没有要求时（用户只配了「把结果交给它」）：上游的结果本身就是它要接着做的事，
  // 不留一个空的「给你的要求」段 —— 空段只会让模型以为要求被吞掉了。
  const requirement = typeof prompt === 'string' ? prompt.trim() : '';
  if (requirement === '') return head;
  return [head, '', '【给你的要求】', requirement].join('\n');
}
