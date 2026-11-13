#!/usr/bin/env node
/**
 * agent-vis 的 HTTP 服务。
 *
 * 只做两件事：
 *   1. 提供前端静态文件；
 *   2. 把 agent 的一次运行按标准 AI SDK v7 UI message stream 协议吐成 SSE。
 *
 * 真实运行（live）与离线剧本（mock）产出的是同一种协议分片，
 * 前端只有一套解析器；模式由 config.js 的 MODE 常量决定。
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

import * as config from './config.js';
import {
  listProfiles,
  registerProfile,
  deleteProfile,
  resolveProfile,
  resolvePermissions,
  normalizePermissions,
  saveProfileConfig,
  appendChatRecord,
  listChatRecords,
  clearChatRecords,
  setDataDir,
} from './config.js';
import { readBaseline, readIdea, readIdeaDoc, readIdeaFile, readMainLine, readMath, readVerification, writeIdea } from './idea-store.js';
import { runMockChunks } from './mock-agent.js';
import { resetAgentSessions, runLiveChunks } from './live-agent.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(HERE, 'public');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
};

/** SSE 保活间隔：长时间没有分片时告诉浏览器连接还活着。 */
const KEEP_ALIVE_MS = 15000;

/**
 * UI message stream 的响应头。
 *
 * 这是前端解析器认的协议头（AI SDK v7 的同名常量就是这五个键值），
 * 写在这里而不是从 ai 包里 import —— 后端只产出这个协议，不需要整个 SDK。
 */
const UI_MESSAGE_STREAM_HEADERS = {
  'content-type': 'text/event-stream',
  'cache-control': 'no-cache',
  connection: 'keep-alive',
  'x-vercel-ai-ui-message-stream': 'v1',
  'x-accel-buffering': 'no',
};

/**
 * 静态资源里用来占位「本次响应的资产版本号」的那个串。
 *
 * 每次请求首页都会把它替换成一个全新的随机值，于是首页里引用的
 * app.js / style.css 的 URL 每次都不一样 —— 浏览器的 HTTP 缓存键包含查询串，
 * URL 一变就必须重新下载。这样即便浏览器缓存过旧版 JS，也拿不到旧版。
 */
const ASSET_TOKEN = '__ASSET_V__';

/**
 * 首页里占位「服务端当前前端版本号」的串。
 *
 * 页面守卫要靠它判断浏览器里跑的 JS 是不是旧版：旧版 JS 也会设置
 * 启动标记（只是值是旧版本号），只看标记在不在会误判成"已启动"。
 */
const VERSION_TOKEN = '__FRONTEND_VERSION__';

/**
 * 首页里占位「本次部署的 API 根前缀」的串。
 *
 * 页面里的接口调用全都要带上这个前缀 —— 挂在路径前缀下时，
 * 请求打到裸路径 /api/... 会被门槛挡下，前端必须知道该往哪发。
 */
const API_BASE_TOKEN = '__API_BASE_TOKEN__';

/** 需要替换资产版本号的文本类型。 */
const TEXT_EXT = new Set(['.html', '.js', '.css']);

function logRequest(req, url) {
  const ua = (req.headers['user-agent'] || '').slice(0, 90);
  const version = url.searchParams.get('v') || '-';
  process.stdout.write(
    `[${new Date().toISOString()}] ${req.method} ${url.pathname} v=${version} ua=${ua}\n`,
  );
}

function serveStatic(req, res, url, apiBase = '') {
  let pathname = url.pathname;
  if (pathname === '/') pathname = '/index.html';

  const filePath = path.join(PUBLIC_DIR, path.normalize(pathname));
  if (!filePath.startsWith(PUBLIC_DIR + path.sep)) {
    res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('forbidden');
    return;
  }

  fs.readFile(filePath, (err, buf) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('not found');
      return;
    }
    const ext = path.extname(filePath);
    let body = buf;
    if (TEXT_EXT.has(ext)) {
      const nonce = crypto.randomBytes(8).toString('hex');
      body = Buffer.from(
        buf
          .toString('utf8')
          .split(ASSET_TOKEN)
          .join(nonce)
          .split(VERSION_TOKEN)
          .join(config.FRONTEND_VERSION)
          .split(API_BASE_TOKEN)
          .join(apiBase),
      );
    }
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    res.end(body);
  });
}

/**
 * 浏览器里跑的是旧版前端时给它的提示。
 *
 * 旧版前端不认识新协议，会把 tool-input-start 之类的分片画成 "input-start"
 * 这种看不懂的卡片 —— 与其让它画错，不如明确告诉用户怎么修。
 */
function staleFrontendMessage(clientVersion) {
  const shown = clientVersion ? `v=${clientVersion}` : '没有声明版本（是旧版前端）';
  return (
    `浏览器里的前端是旧版（${shown}），与服务端 v=${config.FRONTEND_VERSION} 不一致，` +
    '旧版解析不了新协议。请在地址栏打开 ' +
    `http://${config.HOST}:${config.PORT}/?r=1 回车，` +
    '或者按 Ctrl+Shift+R 硬刷新（普通刷新也行，首页每次都会带新的资产版本号）。'
  );
}

/** 读取 POST 的 JSON 请求体。 */
function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8');
      if (raw.trim() === '') {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(raw));
      } catch (err) {
        reject(new Error(`请求体不是合法 JSON: ${err && err.message ? err.message : err}`));
      }
    });
    req.on('error', reject);
  });
}

/**
 * 从请求里取出这一轮授予 agent 的权限。
 *
 * undefined = 请求里没这个字段 → 用这个 agent 自己的默认权限。
 * 数组（哪怕是空的）= 请求里明确给了 → 就用这份，空数组表示一项都不要。
 *
 * 写坏了直接报错：不是数组、或者有目录里没有的 id，都不静默处理 ——
 * 悄悄丢掉一个不认识的权限，等于悄悄少授了一项能力。
 */
function parsePermissions(raw) {
  if (raw === undefined || raw === null) return undefined;
  if (Array.isArray(raw)) return normalizePermissions(raw);
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    // 空串 = 明确表示"一项权限都不要"。
    if (trimmed === '') return [];
    return normalizePermissions(
      trimmed
        .split(',')
        .map((id) => id.trim())
        .filter((id) => id !== ''),
    );
  }
  throw new Error('permissions 必须是一个权限 id 的数组');
}

/**
 * 从请求里取出一次运行需要的东西：问题、agent 身份、上游结果、前端版本号、
 * 以及用户自己改写的系统提示词与权限。
 *
 * GET 走查询串（直接调用、测试用）；POST 走 JSON 请求体 —— 交接的上游分析结果
 * 与自定义提示词都可能很长，放 URL 里既难看又容易被截断。
 *
 * instructions 有两种状态，必须分开：
 *   - undefined：请求里没这个字段 → 用这个 agent 自己的默认提示词；
 *   - 字符串：请求里带了 → 就用这一份，空串表示"明确要没有系统提示词"。
 * 混成一种的话，"把提示词框清空"会被悄悄换成默认提示词。
 *
 * permissions 同理：undefined = 用这个 agent 的默认权限，
 * 数组 = 明确指定（空数组 = 一项都不要）。
 *
 * handoffFrom 是上游 agent 的名字：交接不再只有 A → B 这一条路，
 * 下游看到的来源名字得跟着实际发送方走。
 */
async function parseRunRequest(req, url) {
  if (req.method === 'POST') {
    const body = await readJsonBody(req);
    return {
      prompt: typeof body.prompt === 'string' ? body.prompt : '',
      profileId: typeof body.profile === 'string' ? body.profile : undefined,
      handoff: typeof body.handoff === 'string' ? body.handoff : '',
      handoffFrom: typeof body.handoffFrom === 'string' ? body.handoffFrom : undefined,
      instructions: typeof body.instructions === 'string' ? body.instructions : undefined,
      // 交接来源的身份：用来判断"发送方要不要先重置自己的会话"。
      handoffFromId: typeof body.handoffFromId === 'string' ? body.handoffFromId : '',
      permissions: parsePermissions(body.permissions),
      clientVersion: typeof body.v === 'string' ? body.v : null,
      round: parseRoundMeta(body),
    };
  }
  const rawInstructions = url.searchParams.get('instructions');
  const rawHandoffFrom = url.searchParams.get('handoffFrom');
  const rawPermissions = url.searchParams.get('permissions');
  const rawStepIndex = url.searchParams.get('stepIndex');
  const rawTotalSteps = url.searchParams.get('totalSteps');
  const rawStepTitle = url.searchParams.get('stepTitle');
  const rawStepInstruction = url.searchParams.get('stepInstruction');
  return {
    prompt: url.searchParams.get('prompt') || '',
    profileId: url.searchParams.get('profile') || undefined,
    handoff: '',
    handoffFrom: rawHandoffFrom === null ? undefined : rawHandoffFrom,
    instructions: rawInstructions === null ? undefined : rawInstructions,
    handoffFromId: url.searchParams.get('handoffFromId') || '',
    permissions: rawPermissions === null ? undefined : parsePermissions(rawPermissions),
    clientVersion: url.searchParams.get('v'),
    round: parseRoundMeta({
      roundId: url.searchParams.get('roundId'),
      stepIndex: rawStepIndex === null ? undefined : Number(rawStepIndex),
      stepTitle: rawStepTitle,
      stepInstruction: rawStepInstruction,
      totalSteps: rawTotalSteps === null ? undefined : Number(rawTotalSteps),
      roundPrompt: url.searchParams.get('roundPrompt'),
    }),
  };
}

/**
 * 从请求里取出这一轮在多步轮次中的位置。
 *
 * 这些字段只是"这次跑的是哪一步"的说明，服务端原样记进对话记录 ——
 * 页面刷新后靠它把记录重新排成"第几步"。
 *
 * 字段没给是预期内的缺失（直接调接口的单次运行）；给了但类型不对是配置坏了 ——
 * 直接报错，不猜它想说什么。
 */
function parseRoundMeta(raw) {
  const meta = {
    roundId: null,
    stepIndex: null,
    stepTitle: null,
    stepInstruction: null,
    totalSteps: null,
    roundPrompt: null,
  };
  if (raw.roundId !== undefined && raw.roundId !== null) {
    if (typeof raw.roundId !== 'string' || raw.roundId === '') {
      throw new Error('roundId 必须是一个非空字符串');
    }
    meta.roundId = raw.roundId;
  }
  if (raw.stepIndex !== undefined && raw.stepIndex !== null) {
    if (typeof raw.stepIndex !== 'number' || !Number.isInteger(raw.stepIndex) || raw.stepIndex < 0) {
      throw new Error('stepIndex 必须是一个不小于 0 的整数');
    }
    meta.stepIndex = raw.stepIndex;
  }
  if (raw.stepTitle !== undefined && raw.stepTitle !== null) {
    if (typeof raw.stepTitle !== 'string') {
      throw new Error('stepTitle 必须是字符串');
    }
    meta.stepTitle = raw.stepTitle;
  }
  if (raw.stepInstruction !== undefined && raw.stepInstruction !== null) {
    if (typeof raw.stepInstruction !== 'string') {
      throw new Error('stepInstruction 必须是字符串');
    }
    meta.stepInstruction = raw.stepInstruction;
  }
  if (raw.totalSteps !== undefined && raw.totalSteps !== null) {
    if (typeof raw.totalSteps !== 'number' || !Number.isInteger(raw.totalSteps) || raw.totalSteps < 0) {
      throw new Error('totalSteps 必须是一个不小于 0 的整数');
    }
    meta.totalSteps = raw.totalSteps;
  }
  if (raw.roundPrompt !== undefined && raw.roundPrompt !== null) {
    if (typeof raw.roundPrompt !== 'string') {
      throw new Error('roundPrompt 必须是字符串');
    }
    meta.roundPrompt = raw.roundPrompt;
  }
  return meta;
}

/**
 * 新建一个 agent。
 *
 * 页面上点「新建 Agent」就调这里。系统提示词默认是空的 —— 新 agent 的一切
 * 都由用户自己在页面上写。身份只活在进程内存里（见 config.js 的说明）。
 */
async function handleCreateAgent(req, res) {
  if (req.method !== 'POST') {
    res.writeHead(405, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: '新建 agent 只接受 POST' }));
    return;
  }

  let body;
  try {
    body = await readJsonBody(req);
  } catch (err) {
    res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: err && err.message ? err.message : String(err) }));
    return;
  }

  if (body === null || typeof body !== 'object' || Array.isArray(body)) {
    res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: '请求体必须是一个 JSON 对象' }));
    return;
  }

  // 名字与提示词都必须是字符串（缺省除外）。
  // 类型不对就报错，不猜它想干什么。
  if (body.name !== undefined && typeof body.name !== 'string') {
    res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: 'name 必须是字符串' }));
    return;
  }
  if (body.instructions !== undefined && typeof body.instructions !== 'string') {
    res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: 'instructions 必须是字符串' }));
    return;
  }
  // 权限：给了就必须是目录里有的 id。给不认识的 id 直接拒 ——
  // 存成"没这个权限"等于悄悄替用户决定少授一项能力。
  if (body.permissions !== undefined && body.permissions !== null) {
    if (!Array.isArray(body.permissions)) {
      res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: 'permissions 必须是一个权限 id 的数组' }));
      return;
    }
    try {
      normalizePermissions(body.permissions);
    } catch (err) {
      res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(
        JSON.stringify({ error: err && err.message ? err.message : String(err) }),
      );
      return;
    }
  }

  const profile = registerProfile({
    name: body.name === undefined ? '' : body.name,
    instructions: body.instructions === undefined ? '' : body.instructions,
    permissions: body.permissions === undefined ? undefined : body.permissions,
  });

  res.writeHead(201, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(profile));
}

/**
 * 删掉一个 agent 身份。
 *
 * 只有运行时新建的 agent 能从服务端删掉：配置里写死的（agent-a / agent-b / agent-math）
 * 是配置里写死的，页面上删它们的只是面板，身份还在。
 *
 * 状态码：
 *   - 200：删掉了；
 *   - 400：是固定的那两个之一，不许删；
 *   - 404：这个身份压根不存在。
 */
async function handleDeleteAgent(req, res, profileId) {
  if (req.method !== 'DELETE') {
    res.writeHead(405, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: '删除 agent 只接受 DELETE' }));
    return;
  }

  const known = listProfiles().find((profile) => profile.id === profileId);
  if (!known) {
    res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: `未知的 agent 身份: ${profileId}` }));
    return;
  }
  if (known.builtin) {
    res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: `配置里写死的 agent 不能从服务端删除: ${profileId}` }));
    return;
  }

  deleteProfile(profileId);
  res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify({ ok: true, id: profileId }));
}

/**
 * 存下某个 agent 的配置。
 *
 * PUT /api/agents/<id>/config
 *
 * 请求体里只放要改的字段：
 *   instructions / permissions / steps / handoffTargets
 * 传 null 表示"抹掉用户这一项的改动，回到代码里的默认"。
 *
 * 状态码：
 *   - 200：存好了，返回合并后的生效配置；
 *   - 400：字段写坏了（说明是哪个字段）；
 *   - 404：这个身份压根不存在；
 *   - 405：方法不对。
 */
async function handleSaveAgentConfig(req, res, profileId) {
  if (req.method !== 'PUT') {
    res.writeHead(405, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: '保存配置只接受 PUT' }));
    return;
  }

  const known = listProfiles().find((profile) => profile.id === profileId);
  if (!known) {
    res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: `未知的 agent 身份: ${profileId}` }));
    return;
  }

  let body;
  try {
    body = await readJsonBody(req);
  } catch (err) {
    res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: err && err.message ? err.message : String(err) }));
    return;
  }
  if (body === null || typeof body !== 'object' || Array.isArray(body)) {
    res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: '请求体必须是一个 JSON 对象' }));
    return;
  }

  try {
    const profile = saveProfileConfig(profileId, body);
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify(profile));
  } catch (err) {
    res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: err && err.message ? err.message : String(err) }));
  }
}

/**
 * 读出某个 agent 的对话记录。
 *
 * GET /api/agents/<id>/chats
 *
 * 记录按写入顺序返回：每一项是一次运行（一次请求）的完整分片，
 * 页面拿它把时间轴重新画出来。
 *
 * DELETE /api/agents/<id>/chats 把这个 agent 的记录清空 ——
 * 页面上的「清空历史」用它。别的身份没有记录时也照样返回 200：
 * "本来就没有"不是错误。
 */
async function handleListAgentChats(req, res, profileId) {
  if (req.method !== 'GET' && req.method !== 'DELETE') {
    res.writeHead(405, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: '对话记录只接受 GET（读）与 DELETE（清空）' }));
    return;
  }

  const known = listProfiles().find((profile) => profile.id === profileId);
  if (!known) {
    res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: `未知的 agent 身份: ${profileId}` }));
    return;
  }

  try {
    if (req.method === 'DELETE') {
      const removed = clearChatRecords(profileId);
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ ok: true, agentId: profileId, removed }));
      return;
    }
    const records = listChatRecords(profileId);
    res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ agentId: profileId, count: records.length, records }));
  } catch (err) {
    // 记录读不出来（文件坏了）：把原因说清楚，别假装这一栏没有历史。
    res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: err && err.message ? err.message : String(err) }));
  }
}

/**
 * 一轮运行落到记录里的状态。
 *
 * 判据是「这条流实际说明了什么」，不是「函数有没有抛异常」：
 *   - 客户端断开 → aborted；
 *   - 流里出现过 error 分片 → error（哪怕 handler 本身正常返回）；
 *   - 其余 → done。
 * 之前只看异常，出现过「时间轴上是错误、历史里却写着已完成」的对不上。
 */
export function runStatus({ aborted = false, chunks = [] } = {}) {
  if (aborted) return 'aborted';
  if (chunks.some((chunk) => chunk && chunk.type === 'error')) return 'error';
  return 'done';
}

async function handleRun(req, res, url, mode, { viaPublicPath = false } = {}) {
  let request;
  try {
    request = await parseRunRequest(req, url);
  } catch (err) {
    res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: err && err.message ? err.message : String(err) }));
    return;
  }

  // 运行环境是硬约束：主机或工作目录不对就拒绝启动，不让 agent 在别处读写。
  const envProblem = config.agentEnvProblem();
  if (envProblem) {
    res.writeHead(503, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: envProblem }));
    return;
  }

  const { handoff, handoffFrom, clientVersion, instructions, handoffFromId } = request;

  // 交接：发送方的配置里有两个开关，都在"下游拿到消息开始干活之前"生效 ——
  //   resetSessionOnHandoff         → 清掉发送方自己的会话（它下一轮从头开始）；
  //   resetReceiverSessionOnHandoff → 清掉接收方（也就是这一轮要跑的这一栏）的会话，
  //                                   让它只看这一条消息，不被自己上一轮的上下文带偏。
  if (handoff.trim() !== '' && handoffFromId) {
    try {
      const sender = resolveProfile(handoffFromId);
      if (sender.resetSessionOnHandoff) {
        resetAgentSessions(handoffFromId);
      }
      if (sender.resetReceiverSessionOnHandoff) {
        resetAgentSessions(profile.id);
      }
    } catch (err) {
      // 发送方身份认不出来（它被删了？）：不挡这一轮，交接照发。
      process.stderr.write(
        `交接来源 ${handoffFromId} 认不出来，没能重置它的会话：${(err && err.message) || String(err)}\n`,
      );
    }
  }
  let prompt = request.prompt.trim();
  // 没有要求、但有上游的交接结果：照样能跑 ——
  // 上游的分析结果本身就是下游要接着做的事，用户没写要求不该把交接挡掉。
  if (prompt === '' && handoff.trim() === '') {
    res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(JSON.stringify({ error: 'prompt 不能为空' }));
    return;
  }

  // 多步轮次的第一步：本轮输入由服务端拼在最前面。
  // 前端不再把它塞进 prompt —— 否则同一个请求里 prompt 与 roundPrompt 各有一份，
  // 而且后面几步也会各自重复一遍。后面几步靠同一个 pi 会话带着上下文。
  if (
    request.round &&
    request.round.stepIndex === 0 &&
    typeof request.round.roundPrompt === 'string' &&
    request.round.roundPrompt.trim() !== ''
  ) {
    prompt = `【本轮输入】\n${request.round.roundPrompt.trim()}\n\n${prompt}`;
  }

  // 页面上填的 IDEA 文本：随每次运行一起发，优先于仓库里的 idea.txt。
  // 读不出来就不跑 —— 用户以为文本生效了、实际没有，比报错更糟。
  let ideaText = '';
  try {
    ideaText = readIdea().text.trim();
  } catch (err) {
    res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(
      JSON.stringify({
        error: `读不到页面上的 idea 文本：${err && err.message ? err.message : String(err)}`,
      }),
    );
    return;
  }
  if (ideaText !== '') {
    prompt = `【idea 文本（页面填写，优先于 idea.txt）】\n${ideaText}\n\n${prompt}`;
  }

  // 版本闸门：前端必须自报家门。旧版前端解析不了新协议，
  // 与其让它把分片画成看不懂的卡片、还白跑一次真实调用，不如直接拦下并说清怎么修。
  //
  // 这一关必须排在身份校验前面：旧版前端硬编码的是旧的 agent 身份
  // （比如 profile=assistant），先验身份的话用户只会看到一句
  // "未知的 agent 身份: assistant"，根本猜不到真正的原因是前端该刷新了。
  if (clientVersion !== config.FRONTEND_VERSION) {
    res.writeHead(200, { ...UI_MESSAGE_STREAM_HEADERS });
    res.flushHeaders();
    res.write(
      `data: ${JSON.stringify({
        type: 'error',
        errorText: staleFrontendMessage(clientVersion),
      })}\n\n`,
    );
    res.write('data: [DONE]\n\n');
    res.end();
    return;
  }

  // 指定用哪个 agent（页面上左右两栏各一个）。身份认不出来就直接拒，
  // 不退回默认那个 —— 否则请求会悄悄跑成另一个 agent，结果没法信。
  let profile;
  let effectivePermissions;
  try {
    profile = resolveProfile(request.profileId);
    // 这一轮真正生效的权限：请求里明确给了就用给的，没给就用这个 agent 的默认。
    // 请求里那份在解析阶段就校验过了（不认识的 id 早就 400 了）。
    effectivePermissions = resolvePermissions(profile, request.permissions);
    // 公网入口不给「检索文献」：那个工具要出网，只留给本机直连。
    if (viaPublicPath) {
      effectivePermissions = effectivePermissions.filter((id) => id !== 'web');
    }
  } catch (err) {
    res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
    res.end(
      JSON.stringify({
        error: err && err.message ? err.message : String(err),
        hint: '如果这是浏览器里缓存的旧版前端，按 Ctrl+Shift+R 硬刷新后重试。',
      }),
    );
    return;
  }

  res.writeHead(200, { ...UI_MESSAGE_STREAM_HEADERS });
  res.flushHeaders();

  const controller = new AbortController();
  // 只有客户端真的断开了才中止；正常写完不算。
  res.on('close', () => {
    if (!res.writableFinished) controller.abort();
  });

  const keepAlive = setInterval(() => {
    if (!res.writableEnded) res.write(': keep-alive\n\n');
  }, KEEP_ALIVE_MS);
  keepAlive.unref();

  const instant = url.searchParams.get('instant') === '1';
  const chunkStream =
    mode === 'mock'
      ? runMockChunks(prompt, {
          profileId: profile.id,
          handoff,
          handoffFrom,
          instructions,
          permissions: request.permissions,
          signal: controller.signal,
          delay: instant ? 0 : 140,
        })
      : runLiveChunks(prompt, {
          profileId: profile.id,
          handoff,
          handoffFrom,
          instructions,
          permissions: request.permissions,
          signal: controller.signal,
          // 这一轮的第几步：服务端据此复用同一个 pi 会话（步骤之间不用重发上文）。
          round: request.round,
        });

  // 这一次运行的分片：边转发边收着，结束后整轮写进该 agent 的对话记录。
  // 收的是"原样"的分片（不带前端加的请求前缀）—— 回放时每个请求各自一份
  // store，id 不会撞。
  const chunks = [];
  const startedAt = Date.now();
  let status = 'done';

  try {
    for await (const chunk of chunkStream) {
      if (controller.signal.aborted) break;
      chunks.push(chunk);
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    }
    status = runStatus({ aborted: controller.signal.aborted, chunks });
  } catch (err) {
    status = 'error';
    const errorText = err && err.message ? err.message : String(err);
    if (!res.writableEnded) {
      res.write(`data: ${JSON.stringify({ type: 'error', errorText })}\n\n`);
    }
  } finally {
    clearInterval(keepAlive);

    // 落盘：这一轮写进该 agent 的对话记录，刷新页面后还在。
    // 写失败要把话说出来 —— 记录没存下来，用户刷新后会以为这一轮没发生过。
    const finishedAt = Date.now();
    const record = {
      runId: crypto.randomUUID(),
      agentId: profile.id,
      agentName: profile.name,
      prompt,
      roundId: request.round.roundId,
      stepIndex: request.round.stepIndex,
      stepTitle: request.round.stepTitle,
      stepInstruction: request.round.stepInstruction,
      totalSteps: request.round.totalSteps,
      roundPrompt: request.round.roundPrompt,
      handoffFrom: request.handoffFrom === undefined ? null : request.handoffFrom,
      // 上游交来的原文：刷新页面后要能把那张交接卡片原样画回来。
      handoff: handoff.trim() === '' ? null : handoff,
      // 这一轮真正生效的权限：记下来才知道当时这一轮能干什么。
      permissions: effectivePermissions,
      // 这一轮真正生效的系统提示词：记下来才知道当时这一轮是怎么跑的。
      instructions: profile.instructions,
      startedAt,
      finishedAt,
      durationMs: finishedAt - startedAt,
      status,
      chunks,
    };
    try {
      appendChatRecord(profile.id, record);
    } catch (err) {
      status = 'error';
      const errorText = `这一轮的对话记录没能存下来：${
        err && err.message ? err.message : String(err)
      }`;
      process.stderr.write(`${errorText}\n`);
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ type: 'error', errorText })}\n\n`);
      }
    }

    if (!res.writableEnded) {
      res.write('data: [DONE]\n\n');
      res.end();
    }
  }
}

export function createServer({
  mode = config.MODE,
  publicPath = config.DEFAULT_PUBLIC_PATH,
  dataDir = undefined,
} = {}) {
  // 数据目录：正式服务用默认的 .data/；测试用仓库内的临时目录，
  // 免得测试把用户自己配好的东西冲掉。
  if (dataDir !== undefined) {
    setDataDir(dataDir);
  }
  // 路径前缀是访问门槛。空字符串 = 不启用（本地开发）；非空 = 所有请求都得带上它。
  const prefix = publicPath ? `/${publicPath}` : '';

  return http.createServer((req, res) => {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    logRequest(req, url);

    // 这次请求是不是从外面进来的。
    //
    // 判据是 Host：本机直连的 Host 是 127.0.0.1:<端口> 或 localhost:<端口>，
    // 隧道过来的请求 Host 是隧道域名（Cloudflare 会把它改写掉，客户端伪造不了）。
    // 访问门槛（路径前缀）挡的是"谁进得来"，这一条挡的是"进了来的人能用什么"：
    // 文献检索要出网，只留给本机直连的那个入口。
    const localPort = req.socket.localPort;
    const requestHost = String(req.headers.host || '').toLowerCase();
    const viaPublicPath = !(
      requestHost === `127.0.0.1:${localPort}` || requestHost === `localhost:${localPort}`
    );

    if (prefix) {
      // 只写了前缀、没写尾斜杠：补上斜杠，省得后面的相对路径全错位。
      if (url.pathname === prefix) {
        res.writeHead(301, { Location: prefix + '/' });
        res.end();
        return;
      }
      // 没带前缀：一律 404，且不说破是谁挡的 —— 门槛不该替访问者指路。
      if (!url.pathname.startsWith(prefix + '/')) {
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('not found');
        return;
      }
      // 剥掉前缀，剩下的一律按裸路径处理。
      url.pathname = '/' + url.pathname.slice(prefix.length + 1);
    }

    if (url.pathname === '/api/health') {
      res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(
        JSON.stringify({
          ok: true,
          mode,
          model: config.MODEL_ID,
          provider: config.PROVIDER_NAME,
          sandboxRoot: config.SANDBOX_ROOT,
          // 所有 agent 的工作目录与运行主机（硬约束，页面与测试都读这里）。
          workdir: config.currentWorkdir(),
          requiredHost: config.currentHost(),
          host: os.hostname(),
          envProblem: config.agentEnvProblem(),
          // 全局设置（面板的「全局设置」里可改）：运行环境主机与工作目录。
          settings: config.getGlobalSettings(),
          defaultSettings: config.DEFAULT_GLOBAL_SETTINGS,
          // 组：交接目标既可以是 agent，也可以是组（组里每个人都收到同一条消息）。
          groups: config.listGroups(),
          frontendVersion: config.FRONTEND_VERSION,
          protocolVersion: config.PROTOCOL_VERSION,
          // 权限目录：页面上的权限区按它铺复选框。
          // 公网入口（带访问门槛前缀进来的）不给「检索文献」：出网的能力
          // 只留在本机直连的那个入口上。
          permissionCatalog: viaPublicPath
            ? config.PERMISSION_CATALOG.filter((item) => item.id !== 'web')
            : config.PERMISSION_CATALOG,
          agents: listProfiles(),
        }),
      );
      return;
    }

    // 页面上的 IDEA 文本：全局一份（两个 agent 共用）。
    if (url.pathname === '/api/idea') {
      if (req.method === 'GET') {
        try {
          // fileText 是仓库 idea.txt 的内容：页面第一次打开时拿它当草稿。
          res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
          const ideaDoc = readIdeaDoc();
          const mainLine = readMainLine();
          const baseline = readBaseline();
          const verification = readVerification();
          const math = readMath();
          res.end(JSON.stringify({
            ...readIdea(),
            fileText: readIdeaFile(),
            doc: ideaDoc.doc,
            docError: ideaDoc.error,
            ideaPath: ideaDoc.path,
            mainLine: mainLine.doc,
            mainLineError: mainLine.error,
            baseline: baseline.doc,
            baselineError: baseline.error,
            verification: verification.doc,
            verificationPath: verification.path || null,
            verificationError: verification.error,
            math: math.doc,
            mathPath: math.path || null,
            mathError: math.error,
          }));
        } catch (err) {
          res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(
            JSON.stringify({ error: err && err.message ? err.message : String(err) }),
          );
        }
        return;
      }
      if (req.method === 'PUT') {
        readJsonBody(req)
          .then((body) => {
            if (!body || typeof body !== 'object' || typeof body.text !== 'string') {
              res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
              res.end(JSON.stringify({ error: 'idea 必须是一个带 text 字符串的对象' }));
              return;
            }
            const saved = writeIdea(body.text);
            res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(JSON.stringify(saved));
          })
          .catch((err) => {
            if (res.writableEnded) return;
            res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(
              JSON.stringify({ error: err && err.message ? err.message : String(err) }),
            );
          });
        return;
      }
      res.writeHead(405, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: '/api/idea 只接受 GET 与 PUT' }));
      return;
    }

    // 重置会话：把 agent 拉回"没说过话"的状态。
    //
    // 一次重置做两件事：清掉服务端存的对话记录，并丢掉内存里挂着的模型会话
    // （pi 的会话带着前面的对话，不丢掉它就还是接着上次说）。页面上的时间轴
    // 由前端自己清 —— 服务端管不了浏览器里的东西。
    if (url.pathname === '/api/session/reset') {
      if (req.method !== 'POST') {
        res.writeHead(405, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: '重置会话只接受 POST' }));
        return;
      }
      readJsonBody(req)
        .then((body) => {
          const agentId = body && typeof body.agentId === 'string' ? body.agentId.trim() : '';
          const all = Boolean(body && body.all === true);
          if (!all && agentId === '') {
            res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(JSON.stringify({ error: '要指定 agentId，或者 all=true' }));
            return;
          }
          const known = new Set(listProfiles().map((profile) => profile.id));
          if (!all && !known.has(agentId)) {
            res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(JSON.stringify({ error: `没有这个 agent 身份：${agentId}` }));
            return;
          }
          const targets = all ? [...known] : [agentId];
          let disposed = 0;
          for (const id of targets) {
            config.clearChatRecords(id);
            disposed += resetAgentSessions(id);
          }
          res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(JSON.stringify({ ok: true, agents: targets, disposedSessions: disposed }));
        })
        .catch((err) => {
          res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(
            JSON.stringify({ error: err && err.message ? err.message : String(err) }),
          );
        });
      return;
    }

    // Group：把若干 agent 放在一组里，交接时可以整组发给。
    if (url.pathname === '/api/groups' || url.pathname.startsWith('/api/groups/')) {
      const rest = url.pathname.slice('/api/groups'.length).replace(/^\//, '');
      const groupId = rest ? decodeURIComponent(rest) : '';
      if (req.method === 'GET' && !groupId) {
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ groups: config.listGroups() }));
        return;
      }
      if (req.method === 'POST' && !groupId) {
        readJsonBody(req)
          .then((body) => {
            let group;
            try {
              group = config.createGroup(body || {});
            } catch (err) {
              res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
              res.end(
                JSON.stringify({ error: err && err.message ? err.message : String(err) }),
              );
              return;
            }
            res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(JSON.stringify({ group, groups: config.listGroups() }));
          })
          .catch((err) => {
            res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(
              JSON.stringify({ error: err && err.message ? err.message : String(err) }),
            );
          });
        return;
      }
      if (req.method === 'PUT' && groupId) {
        readJsonBody(req)
          .then((body) => {
            let group;
            try {
              group = config.updateGroup(groupId, body || {});
            } catch (err) {
              res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
              res.end(
                JSON.stringify({ error: err && err.message ? err.message : String(err) }),
              );
              return;
            }
            res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(JSON.stringify({ group, groups: config.listGroups() }));
          })
          .catch((err) => {
            res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(
              JSON.stringify({ error: err && err.message ? err.message : String(err) }),
            );
          });
        return;
      }
      if (req.method === 'DELETE' && groupId) {
        try {
          const result = config.deleteGroup(groupId);
          res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(JSON.stringify({ ...result, groups: config.listGroups() }));
        } catch (err) {
          res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(
            JSON.stringify({ error: err && err.message ? err.message : String(err) }),
          );
        }
        return;
      }
      res.writeHead(405, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: '组只接受 GET / POST / PUT / DELETE' }));
      return;
    }

    // 全局设置：运行环境主机与工作目录。面板上可改，改完对所有 agent 生效。
    if (url.pathname === '/api/settings') {
      if (req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(
          JSON.stringify({
            settings: config.getGlobalSettings(),
            defaults: config.DEFAULT_GLOBAL_SETTINGS,
            envProblem: config.agentEnvProblem(),
            host: os.hostname(),
          }),
        );
        return;
      }
      if (req.method === 'PUT') {
        readJsonBody(req)
          .then((body) => {
            let settings;
            try {
              settings = config.setGlobalSettings(body || {});
            } catch (err) {
              res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
              res.end(
                JSON.stringify({ error: err && err.message ? err.message : String(err) }),
              );
              return;
            }
            res.writeHead(200, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(
              JSON.stringify({
                settings,
                envProblem: config.agentEnvProblem(),
                host: os.hostname(),
              }),
            );
          })
          .catch((err) => {
            res.writeHead(400, { 'Content-Type': 'application/json; charset=utf-8' });
            res.end(
              JSON.stringify({ error: err && err.message ? err.message : String(err) }),
            );
          });
        return;
      }
      res.writeHead(405, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: '设置只接受 GET 与 PUT' }));
      return;
    }

    if (url.pathname === '/api/agents') {
      handleCreateAgent(req, res).catch((err) => {
        if (!res.writableEnded) {
          res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(
            JSON.stringify({ error: err && err.message ? err.message : String(err) }),
          );
        }
      });
      return;
    }

    // /api/agents/<id>[/config|/chats]
    if (url.pathname.startsWith('/api/agents/')) {
      const rest = url.pathname.slice('/api/agents/'.length);
      const slash = rest.indexOf('/');
      const profileId = decodeURIComponent(slash < 0 ? rest : rest.slice(0, slash));
      const sub = slash < 0 ? '' : rest.slice(slash + 1);

      const fail = (err) => {
        if (!res.writableEnded) {
          res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
          res.end(
            JSON.stringify({ error: err && err.message ? err.message : String(err) }),
          );
        }
      };

      if (sub === 'config') {
        handleSaveAgentConfig(req, res, profileId).catch(fail);
        return;
      }
      if (sub === 'chats') {
        handleListAgentChats(req, res, profileId).catch(fail);
        return;
      }
      if (sub !== '') {
        res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' });
        res.end(JSON.stringify({ error: `没有这个子资源: ${sub}` }));
        return;
      }
      // DELETE /api/agents/<id>
      handleDeleteAgent(req, res, profileId).catch(fail);
      return;
    }

    if (url.pathname === '/api/run') {
      handleRun(req, res, url, mode, { viaPublicPath }).catch((err) => {
        if (!res.writableEnded) {
          res.write(
            `data: ${JSON.stringify({
              type: 'error',
              errorText: err && err.message ? err.message : String(err),
            })}\n\n`,
          );
          res.end();
        }
      });
      return;
    }

    serveStatic(req, res, url, prefix);
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  createServer().listen(config.PORT, config.HOST, () => {
    const problem = config.agentEnvProblem();
    process.stdout.write(
      `agent-vis (${config.MODE}) listening on http://${config.HOST}:${config.PORT}\n` +
        `  agent workdir ${config.AGENT_WORKDIR} on ${os.hostname()}`
        + ` (required host ${config.AGENT_HOST})`
        + (problem ? `\n  REFUSING RUNS: ${problem}` : '\n'),
    );
  });
}

export { config };
