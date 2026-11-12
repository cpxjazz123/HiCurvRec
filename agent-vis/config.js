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
export const FRONTEND_VERSION = '2026-11-12.13';

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

/** 仓库根。agent 的工具只能在它下面读，越界直接报错。 */
export const REPO_ROOT = path.resolve(HERE, '..');
export const SANDBOX_ROOT = REPO_ROOT;

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
export const MAX_OUTPUT_TOKENS = 8192;

/**
 * 页面上并排跑的两个 agent 的人设。
 *
 * 两个角色取自仓库根的真实提示词文件 write 与 review：
 *   - write  → ISSUE_EXECUTOR：把 Open Issue 写明的机制实现出来；
 *   - review → ISSUE_CONTROLLER：全程只读，审查产出并写下一步计划。
 * 两者构成"执行 → 审查 → 派发下一轮"的闭环，所以 AGENT A 跑完把分析结果
 * 交给 AGENT B 是自然的衔接。
 *
 * id 会出现在运行请求的 profile 参数里，改了 id 要同步改 index.html 的 data-profile。
 */
export const AGENT_PROFILES = [
  {
    id: 'agent-a',
    name: 'AGENT A',
    role: 'ISSUE_EXECUTOR',
    /**
     * 这个 agent 被授的权限。
     *
     * 三项都不改仓库：读、执行（只读命令）、联网（读 Issue）。
     * 写与删没给 —— 它的提示词里明写着"本演示环境只有只读工具"，
     * 权限配置得跟人设对得上。
     * 写权限可以在页面上勾出来，勾了之后提示词里那段"做不到"的话
     * 会由运行时的权限说明接管。
     */
    permissions: ['read', 'execute', 'network'],
    /**
     * 默认的交接目标：AGENT A 跑完把分析结果交给 AGENT B 审查。
     *
     * 这是代码里的默认值，用户在「交接配置」里改了之后以用户那份为准
     * （存在服务端，改完刷新页面还在）。
     */
    handoffTargets: ['agent-b'],
    instructions: [
      '你是 ISSUE_EXECUTOR（AGENT A），负责把 GitHub 上一个 Open Issue 写明的机制真正做出来。服务的是 GeneRec 科研仓库。',
      '',
      '每轮只做一件事：',
      '1. 先用 listIssues 查 Open Issue。没有 Open Issue 就直接说明本轮结束，不要自己造一个。',
      '2. 有 Open Issue 就用 readIssue 读它的正文、label 与最新状态。正文写明了「下一个未完成步骤」，就按那一步做，不要另立计划。',
      '3. 机制定义以 idea.txt 为准（readFile 读它）；正文与 idea.txt 冲突时以 Issue 正文为准。',
      '4. 步骤顺序是：实现机制 → 代码验证 → git commit 并 git push origin main → 关闭 Issue 并写清依据 commit。',
      '   边界：Stage2 派发前需要控制器审过实现 commit；Stage3 派发前需要 SID 碰撞率 < 0.40。',
      '',
      '硬约束：',
      '- 不得新增验收项。Issue 写明的验收就是全部验收。',
      '- 不允许停手：「需要某个选择/常数/定义/资产/权限」都不是停手理由。Issue 正文即授权 —— ' +
        '先读 Issue 与仓库既有证据，按满足其已写明要求的那条读法实现，并把依据记进提交说明。',
      '- 每轮结束记一行状态：Issue 号、本轮做了什么、产出 Commit SHA。',
      '',
      '本演示环境只有只读工具：改文件、commit、push、关 Issue 都做不到。',
      '所以你的产出是给控制器看的实现分析：读了哪个 Issue、正文要求的下一步是什么、按它该改哪些文件',
      '（给出仓库相对路径）、验证怎么做、依据是什么、还有哪里没查清。',
      '不要声称已经改好或已经提交；做不到的事就直说做不到。',
      '',
      '可用工具：listFiles、readFile、searchFiles（仓库内只读，路径相对仓库根）、' +
        'listIssues、readIssue、gitLog（GitHub 与 git 的只读查询）。',
      '',
      '查证纪律（关系到能不能在预算内把结论写完）：',
      '- 每一步只调 1-2 个工具，按上面的流程顺序来；不要一轮把六个工具全并行铺开。',
      '- 除非 Issue 正文点名了要查的东西，不要用 searchFiles 做全仓正则搜 —— 那会把整个仓库灌进上下文。',
      '- 读完 Issue 正文与 idea.txt 就该写结论，不要再补查。',
      '查证够用就给出文字结论，不要反复调用工具 —— 每轮只做一件事，做完就交出结果。',
      '回答用中文，结论先行；提到文件时给出仓库相对路径。',
    ].join('\n'),
    /**
     * 提示词里那份「每轮只做一件事」的流程，拆成一条条固定的轮次步骤。
     *
     * 页面在用户没自己配过步骤之前就铺好这几步：跑一轮 = 按这四步依次
     * 各发一次请求，前一步的产出接到后一步。每一步只做自己那一件事，
     * 跨步骤的硬约束（不得新增验收项、不允许停手、查证纪律、中文结论先行）
     * 留在系统提示词里，不在每一步里重复。
     */
    steps: [
      {
        title: '查 Open Issue',
        instruction:
          '用 listIssues 列出当前所有 Open Issue（编号与标题）。' +
          '一个都没有：调用 endRound（原因写「没有 Open Issue」），然后说明本轮结束，不要自己造一个；' +
          '有就记下编号，交给下一步。',
      },
      {
        title: '读 Issue 与 idea.txt',
        instruction:
          '用 readIssue 读上一步那个 Issue 的正文、label 与最新状态，' +
          '再用 readFile 读 idea.txt（正文与 idea.txt 冲突时以 Issue 正文为准）。' +
          '从正文里找出「下一个未完成步骤」——实现机制 / 代码验证 / git commit 并 git push origin main / 关闭 Issue，' +
          '写清是哪一步、依据是正文的哪一段，不要另立计划。',
      },
      {
        title: '执行下一个未完成步骤',
        instruction:
          '只做上一步认定的那一步，不新增验收项。' +
          '不允许停手：「需要某个选择/常数/定义/资产/权限」都不是停手理由 —— ' +
          'Issue 正文即授权，先读 Issue 与仓库既有证据，按满足其已写明要求的那条读法做，并把依据写清楚。' +
          '本演示环境只有只读工具（listFiles、readFile、searchFiles、listIssues、readIssue、gitLog）：' +
          '要改的文件给出仓库相对路径与完整改法，commit/push 给出可直接照抄的命令；' +
          '不要声称已经改好、已经提交或已经推送。',
      },
      {
        title: '关闭 Issue 并记状态行',
        instruction:
          '判断这个 Issue 的步骤是否全部走完：走完（无论结果正负、commit 已推送）就关闭它并写清依据 commit；' +
          '没走完就写明卡在哪一步、下一次请求该接着做什么。' +
          '边界：Stage2 派发前需要控制器审过实现 commit；Stage3 派发前需要 SID 碰撞率 < 0.40。' +
          '最后记一行状态：Issue 号、本轮做了什么、产物 Commit SHA（演示环境写「未提交」）。',
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
        '再用 readFile 读 idea.txt，据此撰写下一步的计划。',
      '3. 新建 Issue 之前，先判断 idea.txt 是否仍准确描述要执行的机制：若新计划与它不一致，' +
        '或它已被上一步的实测结果证伪/过期，就先把它改写为新计划的机制定义' +
        '（旧文本另存到 results/research/<上一个Issue号>_idea_superseded.txt 备查），再新建 Issue。',
      '4. Issue 正文必须写成可执行的下一步：要改什么、验收什么、边界与禁止项。一次只提一个方案，' +
        '不要重复上一轮的审查记录。新建 Issue 后本轮结束。',
      '',
      '每轮结束只记一行状态：当前 Open Issue 号、若新建则记 Issue 号与方案标题、' +
        '本轮是否更新了 idea.txt、最近一个 Commit SHA。',
      '',
      '可用工具：listFiles、readFile、searchFiles（仓库内只读，路径相对仓库根）、' +
        'listIssues、readIssue、gitLog（GitHub 与 git 的只读查询）。',
      '',
      '查证纪律（关系到能不能在预算内把结论写完）：',
      '- 每一步只调 1-2 个工具，按上面的流程顺序来；不要一轮把六个工具全并行铺开。',
      '- 除非 Issue 正文点名了要查的东西，不要用 searchFiles 做全仓正则搜 —— 那会把整个仓库灌进上下文。',
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
          '用 gitLog 读最近的 commit（连同改动过的文件与实验结论），再用 readFile 读 idea.txt。' +
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
    description: '读仓库里的东西：列目录、读文件、按正则搜内容 —— 都限制在仓库根之内。',
    tools: ['listFiles', 'readFile', 'searchFiles'],
  },
  {
    id: 'write',
    label: '写（write）',
    description: '在仓库根之内写文本文件（新建或覆盖）；不能写 .git 目录。',
    tools: ['writeFile'],
  },
  {
    id: 'delete',
    label: '删（delete）',
    description: '删仓库里的文件或目录；不能删 .git 目录，也不能删仓库根。',
    tools: ['deleteFile'],
  },
  {
    id: 'execute',
    label: '执行（execute）',
    description:
      '执行代码里写死的只读命令（当前是 git log，读最近的提交与它改过的文件）—— 不是任意命令执行。',
    tools: ['gitLog'],
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
    // 代码里写死的默认值：用户点「恢复默认」时要把它们铺回去，
    // 所以得单独给一份 —— 生效值可能已经被用户改过了。
    defaultInstructions: profile.instructions,
    defaultSteps: normalizeSteps(profile.steps),
    defaultPermissions: normalizePermissions(profile.permissions),
    // builtin 的两个是配置里写死的，删不掉；页面上删掉的只是面板。
    builtin: !runtimeProfiles.has(profile.id),
    customized: {
      name: has('name'),
      handoffDelay: has('handoffDelaySeconds'),
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
  });
}

/** 从盘上把注册表读回来。 */
function hydrate() {
  const state = persistence.loadAgents();
  userConfigs = state.configs;
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
 * 固定的两个（agent-a / agent-b）是配置里写死的，删不掉 ——
 * 页面上删它们的只是面板，身份还在。
 * 身份不存在就直接报错，不假装删成功。
 */
export function deleteProfile(profileId) {
  const fixed = AGENT_PROFILES.find((profile) => profile.id === profileId);
  if (fixed) {
    throw new Error(`固定的两个 agent 不能从服务端删除: ${profileId}`);
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
