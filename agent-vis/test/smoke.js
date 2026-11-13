import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import http from 'node:http';
import { spawnSync } from 'node:child_process';

import { createServer, runStatus } from '../server.js';
import { config } from '../server.js';
import { createStore, applyChunk, foldText, roundEnded, handoffRequested, ROUND_END_TOOL, ROUND_HANDOFF_TOOL } from '../public/protocol.js';
import { scanIdeaFiles, archivedDirectionNames } from '../idea-store.js';
import { runMockChunks } from '../mock-agent.js';
import {
  PERMISSION_CATALOG,
  buildHandoffPrompt,
  buildPermissionPrompt,
  normalizePermissions,
  resolvePermissions,
} from '../config.js';
import { PI_TOOLS, TOOL_IMPLS, toolsForPermissions } from '../live-agent.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));

/**
 * 测试用的数据目录。
 *
 * 配置与对话记录现在会落盘；测试跑在自己的目录里（仓库内的 .out/），
 * 不然每次跑测试都会往用户自己的 .data/ 里写东西。
 */
const SCRATCH_DATA = path.join(HERE, '..', '.out', 'smoke-data');

/**
 * 离线协议一致性测试：不联网、不调用 LLM。
 *
 * 覆盖三件事：
 *   1. 静态资源与路径安全；
 *   2. 服务端输出的是标准 AI SDK v7 UI message stream（header + 帧格式 + [DONE]）；
 *   3. 前端 reducer 能把这一串分片还原成时间轴模型。
 *
 * 真实 agent 的端到端检查在 test/browser.js（真浏览器 + 真模型）。
 */

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed += 1;
    process.stdout.write(`  ✓ ${name}\n`);
  } catch (err) {
    failures.push({ name, err });
    process.stdout.write(`  ✗ ${name}\n    ${err && err.message}\n`);
  }
}

function listen(server) {
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve(server.address().port));
  });
}

function close(server) {
  return new Promise((resolve) => server.close(resolve));
}

/** 读取 SSE 响应并按帧拆开。 */
async function readSse(url, { signal, method = 'GET', payload } = {}) {
  // mock 模式的 140ms/块延迟是给人眼看的动画；测试要的只是协议与内容，
  // 直接把延迟关掉（instant=1），几十条断言就从分钟级掉到秒级。
  let target = url;
  if (target.includes('/api/run') && !target.includes('instant=')) {
    target += target.includes('?') ? '&instant=1' : '?instant=1';
  }
  const init = { method, signal };
  if (payload !== undefined) {
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(payload);
  }
  const res = await fetch(target, init);
  const body = await res.text();
  const frames = [];
  for (const block of body.split('\n\n')) {
    const dataLines = block
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trim());
    if (dataLines.length > 0) frames.push(dataLines.join('\n'));
  }
  return { res, body, frames };
}

function parseChunks(frames) {
  return frames
    .filter((frame) => frame !== '[DONE]')
    .map((frame) => JSON.parse(frame));
}

/** 从首页 HTML 里截出一个既有面板的片段（从 <section 到对应的 </section>）。 */
function extractPanel(html, id) {
  const at = html.indexOf(`id="${id}"`);
  if (at < 0) return null;
  const open = html.lastIndexOf('<section', at);
  const end = html.indexOf('</section>', at);
  if (open < 0 || end < 0) return null;
  return html.slice(open, end);
}

/** 从首页 HTML 里截出模板的**内容**（不含 <template> 本身）。 */
function extractTemplate(html, id) {
  const at = html.indexOf(`<template id="${id}">`);
  if (at < 0) return null;
  const openEnd = html.indexOf('>', at) + 1;
  const end = html.indexOf('</template>', openEnd);
  if (end < 0) return null;
  return html.slice(openEnd, end);
}

/**
 * 页面的"骨架"：只看标签名与 data-role，不看文字与属性值。
 *
 * 用它比对新建 agent 的模板与既有面板是不是同一种形式 ——
 * 只比对 data-role 的集合会漏掉标签结构本身的差异。
 */
function skeleton(html) {
  return (html.match(/<[a-zA-Z][^>]*>/g) || [])
    .map((tag) => {
      const name = tag.match(/^<([a-zA-Z][\w-]*)/)[1];
      const role = tag.match(/data-role="([^"]+)"/);
      return role ? `${name}[${role[1]}]` : name;
    })
    .join(' ');
}

async function main() {
  process.stdout.write('agent-vis 离线协议测试\n');

  // publicPath: '' —— 测试要的是不带访问门槛的裸服务。仓库里的
  // .local-public-path 是给公网入口用的，本地跑测试不该被它挡住。
  fs.rmSync(SCRATCH_DATA, { recursive: true, force: true });
  const server = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
  const port = await listen(server);
  const base = `http://127.0.0.1:${port}`;

  /** PUT 一个 agent 的配置，返回 {status, json}。 */
  const putAgentConfig = (id, patch) =>
    fetch(`${base}/api/agents/${id}/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    }).then(async (res) => ({ status: res.status, json: await res.json().catch(() => null) }));

  await test('静态资源返回正确的 content-type', async () => {
    for (const [route, type] of [
      ['/', 'text/html'],
      ['/app.js', 'text/javascript'],
      ['/protocol.js', 'text/javascript'],
      ['/run-request.js', 'text/javascript'],
      ['/style.css', 'text/css'],
    ]) {
      const res = await fetch(`${base}${route}`);
      assert.equal(res.status, 200, `${route} 状态码`);
      assert.match(res.headers.get('content-type'), new RegExp(type));
    }
  });

  await test('页面引用的每个模块都真的发得出去', async () => {
    // 页面里多 import 一个模块、却忘了把它放进 public/ 的话，
    // 浏览器会静默地不执行整个 app.js —— 面板一个都不显示，也没有任何报错。
    // （这个坑踩过一次：run-request.js 放在了 public/ 外面。）
    const appJs = fs.readFileSync(path.join(HERE, '..', 'public', 'app.js'), 'utf8');
    const specs = [...appJs.matchAll(/from\s+'(\.\/[^']+)'/g)].map((m) => m[1]);
    assert.ok(specs.length > 0, '没找到页面里的 import');
    for (const spec of specs) {
      const route = spec.replace(/^\.\//, '').split('?')[0];
      const res = await fetch(`${base}/${route}`);
      assert.equal(res.status, 200, `页面要用的 ${spec} 发不出去（${res.status}）`);
    }
  });

  await test('配置脚本语法没错（它坏了，AGENT IDEA 的配置就刷不进去）', () => {
    // apply-idea-schema-config.mjs 是两栏配置的唯一出处：它语法坏了，
    // 面板上的配置就悄悄停在旧版本，而且没有任何提示。
    const script = path.join(HERE, '..', 'apply-idea-schema-config.mjs');
    // node --check 只做语法检查，不执行脚本（脚本会真的去改面板配置）。
    const res = spawnSync(process.execPath, ['--check', script], { encoding: 'utf8' });
    assert.equal(res.status, 0, `语法检查没过：${res.stderr}`);
  });

  await test('循环闸的配置：模式、判据与每轮重置会话', async () => {
    const put = (patch) =>
      fetch(`${base}/api/agents/agent-a/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      }).then(async (res) => ({ status: res.status, json: await res.json().catch(() => null) }));

    // 闸的默认值：不设就是不设闸。
    let res = await put({ loopGate: { mode: 'ask', prompt: '还有没做完的才继续' } });
    assert.equal(res.status, 200);
    assert.deepEqual(res.json.loopGate, { mode: 'ask', prompt: '还有没做完的才继续' });

    res = await put({ loopResetSession: true });
    assert.equal(res.status, 200);
    assert.equal(res.json.loopResetSession, true);

    // 不认识的模式要报错，而不是默默当成关。
    res = await put({ loopGate: { mode: 'whatever', prompt: '' } });
    assert.equal(res.status, 400);
    assert.match(String(res.json.error), /loopGate.mode/);

    // 还原。
    await put({ loopGate: null, loopResetSession: null });
  });

  await test('重置会话时可以选择留下对话记录', async () => {
    const res = await fetch(`${base}/api/session/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agentId: 'agent-a', keepRecords: true }),
    });
    assert.equal(res.status, 200);
    const doc = await res.json();
    assert.equal(doc.ok, true);
    assert.deepEqual(doc.agents, ['agent-a']);
  });

  await test('循环次数可以是 0（不限次数）', async () => {
    const put = (patch) =>
      fetch(`${base}/api/agents/agent-a/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      }).then(async (res) => ({ status: res.status, json: await res.json().catch(() => null) }));

    let res = await put({ loop: { enabled: true, count: 0, intervalSeconds: 0 } });
    assert.equal(res.status, 200);
    assert.equal(res.json.loop.count, 0, '0 = 不限次数');
    assert.equal(res.json.loop.enabled, true);

    // 负数与超过上限的还是要拒。
    res = await put({ loop: { enabled: true, count: -1, intervalSeconds: 0 } });
    assert.equal(res.status, 400);
    res = await put({ loop: { enabled: true, count: 1000, intervalSeconds: 0 } });
    assert.equal(res.status, 400);

    await put({ loop: null });
  });

  await test('AGENT MATH 配好的是：不限次数 + 改了被审的文件才继续 + 每轮重置会话', async () => {
    const health = await (await fetch(`${base}/api/health`)).json();
    const math = (health.agents || []).find((a) => a.id === 'agent-math');
    assert.ok(math, 'AGENT MATH 不在身份表里');
    assert.equal(math.loop.enabled, true);
    assert.equal(math.loop.count, 0, '次数应为 0（不限）');
    assert.equal(math.loopGate.mode, 'files', '闸应当是客观的：看文件有没有改');
    assert.equal(math.loopGate.prompt, '', 'files 模式不用判据文本');
    // 它被审的只有那一份 math.json：状态行与 idea 每轮都变，拿它们当依据等于永远"有进展"。
    assert.deepEqual(math.loopWatchFiles, ['ideas/*/math.json']);
    assert.equal(math.loopResetSession, true);
  });

  await test('AGENT FIX 也配成：改了候选树代码才继续，没改就自己结束', async () => {
    const health = await (await fetch(`${base}/api/health`)).json();
    const fix = (health.agents || []).find((a) => a.id === 'agent-fix');
    assert.ok(fix, 'AGENT FIX 不在身份表里');
    assert.equal(fix.loop.enabled, true);
    assert.equal(fix.loop.count, 0, '次数应为 0（不限，由"有没有改"决定停不停）');
    assert.equal(fix.loopGate.mode, 'files');
    const watches = fix.loopWatchFiles || [];
    assert.ok(watches.length > 0, '要有被审文件的模式');
    assert.ok(
      watches.some((p) => p.includes('stage2_RQ-VAE/')),
      `被审的应当是候选树里的代码：${JSON.stringify(watches)}`,
    );
    assert.ok(
      !watches.some((p) => p.includes('TIGER_RQ-VAE')),
      '冻结基线不该进"有没有进展"的判断',
    );
    // 指纹接口按这些模式算，且不含基线。
    const fp = await (await fetch(`${base}/api/ideas/fingerprint?agentId=agent-fix`)).json();
    const names = Object.keys(fp.files || {});
    assert.ok(names.length > 0, 'AGENT FIX 的指纹不该是空的');
    assert.ok(!names.some((n) => n.startsWith('stage2_RQ-VAE/TIGER_RQ-VAE/')), '基线混进指纹了');
  });

  await test('重置会话就是把对话记录一起清掉（不留"只清上下文"这一说）', async () => {
    // 先给 agent-a 记一条对话，再重置：记录应当被清掉。
    const runRes = await fetch(`${base}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ v: config.FRONTEND_VERSION, profile: 'agent-a', prompt: '记一句' }),
    });
    assert.equal(runRes.status, 200);
    await runRes.text();
    const before = await (await fetch(`${base}/api/agents/agent-a/chats`)).json();
    assert.ok((before.records || []).length > 0, '重置之前应当有记录');

    const res = await fetch(`${base}/api/session/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agentId: 'agent-a' }),
    });
    assert.equal(res.status, 200);
    const doc = await res.json();
    assert.equal(doc.ok, true);
    assert.deepEqual(doc.agents, ['agent-a']);

    const after = await (await fetch(`${base}/api/agents/agent-a/chats`)).json();
    assert.equal((after.records || []).length, 0, '重置之后记录应当被清空');
  });

  await test('循环闸的指纹：agent 算自己被审的文件', async () => {
    const all = await (await fetch(`${base}/api/ideas/fingerprint`)).json();
    assert.ok(Object.keys(all.files || {}).length > 1, '默认要覆盖一摊文件');

    const math = await (await fetch(`${base}/api/ideas/fingerprint?agentId=agent-math`)).json();
    const names = Object.keys(math.files || {});
    assert.equal(names.length, 1, `AGENT MATH 只该看一份 math.json，实际 ${names.length} 份`);
    assert.match(names[0], /^ideas\/.+\/math\.json$/, `看的不是 math.json：${names[0]}`);
    assert.equal(typeof math.fingerprint, 'string');
  });

  await test('未知路由返回 404', async () => {
    const res = await fetch(`${base}/no-such-file`);
    assert.equal(res.status, 404);
  });

  await test('路径穿越无法读到源码', async () => {
    const res = await fetch(`${base}/../server.js`);
    assert.notEqual(res.status, 200);
    const text = await res.text();
    assert.ok(!text.includes('createServer'), '响应体泄露了服务端源码');
  });

  await test('/api/health 报告模式与模型', async () => {
    const res = await fetch(`${base}/api/health`);
    assert.equal(res.status, 200);
    const info = await res.json();
    assert.equal(info.ok, true);
    assert.equal(info.mode, 'mock');
    assert.equal(typeof info.model, 'string');
    assert.equal(info.frontendVersion, config.FRONTEND_VERSION);
    assert.equal(info.protocolVersion, config.PROTOCOL_VERSION);
  });

  await test('前端 app.js 内置的版本号与服务端一致', async () => {
    // 改了 public/ 却忘了同步版本号时，靠这条测试兜住：
    // 否则旧标签页会拿旧前端去解新协议，画出一堆看不懂的卡片。
    const source = fs.readFileSync(path.join(HERE, '..', 'public', 'app.js'), 'utf8');
    const match = source.match(/const FRONTEND_VERSION = '([^']+)'/);
    assert.ok(match, 'app.js 里找不到 FRONTEND_VERSION');
    assert.equal(match[1], config.FRONTEND_VERSION);
  });

  await test('首页给资产 URL 加一次性版本号，JS 不会被浏览器缓存', async () => {
    // 浏览器缓存旧版 app.js 是"只显示 input-start"的根因。
    // 每次请求首页都换一个 nonce，URL 变了浏览器就必须重新下载。
    const first = await (await fetch(`${base}/`)).text();
    const second = await (await fetch(`${base}/`)).text();

    const a = first.match(/\/app\.js\?v=([a-zA-Z0-9]+)/);
    const b = second.match(/\/app\.js\?v=([a-zA-Z0-9]+)/);
    assert.ok(a, '首页没有给 app.js 加版本参数');
    assert.ok(b, '首页没有给 app.js 加版本参数');
    assert.notEqual(a[1], b[1], '两次请求用了同一个资产版本号，缓存照样命中');

    assert.ok(first.includes('/style.css?v='), '样式表没有版本参数');
  });

  await test('首页带自愈守卫：JS 起不来就换地址重载一次', async () => {
    const html = await (await fetch(`${base}/`)).text();
    assert.ok(html.includes('__AGENT_VIS_FRONTEND__'), '首页缺少启动标记检查');
    assert.ok(html.includes('location.replace'), '首页缺少强制重载逻辑');
  });

  await test('旧版前端不带版本号时被拦下，并给出能照做的提示', async () => {
    // 旧前端解析不了新协议，会把分片画成 "input-start" 这种看不懂的卡片。
    // 服务端必须认出它、并且不再浪费一次真实的 agent 调用。
    const { res, frames } = await readSse(`${base}/api/run?prompt=stale-probe`);
    assert.equal(res.status, 200);

    const chunks = parseChunks(frames);
    const types = chunks.map((c) => c.type);
    assert.ok(!types.includes('start'), '旧版前端不该真的把 agent 跑起来');

    const err = chunks.find((c) => c.type === 'error');
    assert.ok(err, '没有给出错误提示');
    assert.match(err.errorText, /刷新/, '提示里没说要刷新');
    assert.ok(
      err.errorText.includes(config.FRONTEND_VERSION),
      '提示里没有当前服务端版本号',
    );
  });

  await test('版本号对得上的客户端正常跑', async () => {
    const { frames } = await readSse(
      `${base}/api/run?prompt=hello&v=${config.FRONTEND_VERSION}`,
    );
    const types = parseChunks(frames).map((c) => c.type);
    assert.equal(types[0], 'start', '版本号正确时不该被拦下');
  });

  await test('版本号过期的客户端同样被拦下', async () => {
    const { frames } = await readSse(`${base}/api/run?prompt=stale-probe&v=1999-01-01.0`);
    const chunks = parseChunks(frames);
    assert.ok(!chunks.some((c) => c.type === 'start'), '过期前端不该把 agent 跑起来');
    assert.ok(chunks.some((c) => c.type === 'error'), '过期前端应收到错误提示');
  });

  await test('前端 app.js 会暴露启动标记并在运行请求里带上版本号', async () => {
    const source = fs.readFileSync(path.join(HERE, '..', 'public', 'app.js'), 'utf8');
    assert.ok(
      source.includes('window.__AGENT_VIS_FRONTEND__'),
      'app.js 没有设置启动标记，首页守卫会误判成旧版',
    );
    assert.match(source, /v=\$\{FRONTEND_VERSION\}/, 'app.js 的运行请求没带版本号');
  });

  await test('页面左右各有一个 agent 面板', async () => {
    const html = await (await fetch(`${base}/`)).text();
    assert.ok(html.includes('id="panelA"'), '缺少左侧面板');
    assert.ok(html.includes('id="panelB"'), '缺少右侧面板');

    const panelA = extractPanel(html, 'panelA');
    const panelB = extractPanel(html, 'panelB');
    assert.ok(panelA && panelB, '左右两个面板的片段没截出来');

    // 两个面板各自要有输入框、运行按钮、时间轴、统计区。
    // 只数这两个面板里的 —— 首页里还有一份给新建 agent 用的模板，
    // 它不是此刻在跑的面板。
    const both = panelA + panelB;
    for (const role of ['prompt', 'runBtn', 'timeline', 'metaList']) {
      const count = (both.match(new RegExp(`data-role="${role}"`, 'g')) || []).length;
      assert.equal(count, 2, `${role} 的数量应为 2，实际 ${count}`);
    }

    const ids = [...html.matchAll(/data-profile="([^"]+)"/g)].map((m) => m[1]);
    assert.equal(ids.length, 2, `静态面板应声明两个 agent 身份，实际 ${ids.length}`);
    assert.notEqual(ids[0], ids[1], '两个静态面板必须是不同的 agent');
  });

  await test('/api/health 列出五个 agent 的身份', async () => {
    const info = await (await fetch(`${base}/api/health`)).json();
    assert.ok(Array.isArray(info.agents), 'health 没列出 agents');
    assert.equal(info.agents.length, 5, `应为五个 agent，实际 ${info.agents.length}`);
    for (const agent of info.agents) {
      assert.equal(typeof agent.id, 'string');
      assert.equal(typeof agent.name, 'string');
      assert.ok(agent.name.length > 0);
    }
    const unique = new Set(info.agents.map((agent) => agent.id));
    assert.equal(unique.size, info.agents.length, 'agent 身份必须互不相同');
  });

  await test('两个内置 agent 的提示词已经拆成固定步骤', async () => {
    const info = await (await fetch(`${base}/api/health`)).json();
    const byId = new Map(info.agents.map((agent) => [agent.id, agent]));
    const a = byId.get('agent-a');
    const b = byId.get('agent-b');
    assert.ok(a && b, 'health 里没有两个内置 agent');

    for (const agent of [a, b]) {
      assert.ok(Array.isArray(agent.steps), `${agent.id} 没有 steps`);
      assert.equal(
        agent.steps.length,
        agent.id === 'agent-a' ? 6 : 4,
        `${agent.id} 的步骤数不对: ${agent.steps.length}`,
      );
      agent.steps.forEach((step, index) => {
        assert.equal(typeof step.title, 'string');
        assert.ok(step.title.trim().length > 0, `${agent.id} 第 ${index} 步没有标题`);
        assert.equal(typeof step.instruction, 'string');
        assert.ok(step.instruction.trim().length > 0, `${agent.id} 第 ${index} 步没有指令`);
        assert.deepEqual(Object.keys(step).sort(), ['instruction', 'title'], `${agent.id} 第 ${index} 步带了多余字段`);
      });
    }

    // 步骤必须是提示词里那份流程拆出来的：用到的工具要对得上。
    const aText = a.steps.map((step) => step.instruction).join('\n');
    assert.ok(aText.includes('math.json'), 'AGENT Write 该读 math.json 里的公式');
    assert.ok(aText.includes('copyBaselineTree'), 'AGENT Write 该用 copyBaselineTree 建候选树');
    assert.ok(aText.includes('用 write'), 'AGENT Write 该用 write 写机制代码');
    assert.ok(aText.includes('tests/'), 'AGENT Write 该写测试');

    const bText = b.steps.map((step) => step.instruction).join('\n');
    assert.ok(bText.includes('listIssues'), 'AGENT B 的第一步该用 listIssues');
    assert.ok(bText.includes('gitLog'), 'AGENT B 该用 gitLog 读最近的 commit');
    assert.ok(bText.includes('idea.txt'), 'AGENT B 该核对 idea.txt');
  });

  await test('可以指定用哪个 agent 跑', async () => {
    const info = await (await fetch(`${base}/api/health`)).json();
    const second = info.agents[1];
    const { frames } = await readSse(
      `${base}/api/run?prompt=hi&v=${config.FRONTEND_VERSION}&profile=${second.id}`,
    );
    const chunks = parseChunks(frames);
    assert.equal(chunks[0].type, 'start', '指定 agent 后应正常开始');

    const text = chunks
      .filter((c) => c.type === 'text-delta')
      .map((c) => c.delta)
      .join('');
    assert.ok(text.includes(second.name), `回答里没认出第二个 agent: ${text.slice(0, 120)}`);
  });

  await test('不存在的 agent 身份直接报错，不退回默认', async () => {
    const res = await fetch(
      `${base}/api/run?prompt=hi&v=${config.FRONTEND_VERSION}&profile=no-such-agent`,
    );
    assert.equal(res.status, 400, '未知 agent 应该被拒掉');
  });

  await test('五个 agent 各自有一份独立的人设', async () => {
    assert.ok(Array.isArray(config.AGENT_PROFILES), 'config 没导出 AGENT_PROFILES');
    assert.equal(config.AGENT_PROFILES.length, 5);
    const texts = new Set(config.AGENT_PROFILES.map((profile) => profile.instructions));
    assert.equal(texts.size, config.AGENT_PROFILES.length,
                 '每个 agent 的人设说明必须不同，否则几栏做的是同一件事');
  });

  await test('五个 agent 的名字分别是 Write / B / MATH / STAGE2 / FIX', async () => {
    const info = await (await fetch(`${base}/api/health`)).json();
    assert.deepEqual(
      info.agents.map((agent) => agent.name),
      ['AGENT Write', 'AGENT B', 'AGENT MATH', 'AGENT STAGE2', 'AGENT FIX'],
      '页面上三栏的 agent 名字不对',
    );
  });

  await test('五个 agent 分别是实现者、控制器、公式审计员、训练员与核对员', async () => {
    const info = await (await fetch(`${base}/api/health`)).json();
    assert.deepEqual(
      info.agents.map((agent) => agent.role),
      ['MECHANISM_BUILDER', 'ISSUE_CONTROLLER', 'MATH_AUDITOR', 'STAGE2_RUNNER', 'MECHANISM_CHECKER'],
      '两个 agent 的角色不是"执行者 / 控制器"这一对',
    );
    assert.ok(
      config.AGENT_PROFILES[0].instructions.includes('MECHANISM_BUILDER'),
      'AGENT Write 的人设里没写明它是 MECHANISM_BUILDER',
    );
    assert.ok(
      config.AGENT_PROFILES[1].instructions.includes('ISSUE_CONTROLLER'),
      'AGENT B 的人设里没写明它是 ISSUE_CONTROLLER',
    );
  });

  await test('AGENT MATH 拿读与写权限，只核对公式', async () => {
    const info = await (await fetch(`${base}/api/health`)).json();
    const math = info.agents.find((agent) => agent.id === 'agent-math');
    assert.ok(math, 'health 里没有 agent-math');
    assert.deepEqual(math.permissions, ['read', 'write'],
                     `AGENT MATH 的权限不对: ${JSON.stringify(math.permissions)}`);
    assert.equal(math.builtin, true, 'AGENT MATH 是配置里写死的身份');
    assert.equal(math.steps.length, 5, `AGENT MATH 的步骤数不对: ${math.steps.length}`);
    const text = math.steps.map((step) => step.instruction).join('\n');
    assert.ok(text.includes('math.json'), 'AGENT MATH 的步骤里没点名 math.json');
    assert.ok(text.includes('用 edit'), 'AGENT MATH 改公式该用 edit');
    assert.ok(text.includes('PASS') && text.includes('FAIL'),
              'AGENT MATH 要对每条公式给出 PASS/FAIL 结论');
    assert.ok(math.instructions.includes('MATH_AUDITOR') &&
              math.instructions.includes('math.json'),
              'AGENT MATH 的人设没写明它审的是 math.json');
    assert.equal(math.defaultPrompt.includes('math.json'), true,
                 'AGENT MATH 的输入框预填指令该指向 math.json');
  });

  await test('每个 agent 的工作目录与运行主机是硬约束', async () => {
    const info = await (await fetch(`${base}/api/health`)).json();
    assert.equal(info.workdir, config.AGENT_WORKDIR, `workdir 不对: ${info.workdir}`);
    assert.equal(info.requiredHost, config.AGENT_HOST, `requiredHost 不对: ${info.requiredHost}`);
    assert.equal(typeof info.host, 'string');
    assert.ok(info.host.length > 0, 'health 没报当前主机名');
    assert.equal(info.envProblem, null, `本机不该有环境问题: ${info.envProblem}`);
    // 目录就是那个仓库：三个标记目录都在。
    assert.equal(config.agentWorkdirOk(), true,
                 `${config.AGENT_WORKDIR} 下缺仓库目录`);
    // 主机名判定：本名、带域名后缀都算对，别的主机不算。
    assert.equal(config.agentHostOk(config.AGENT_HOST), true);
    assert.equal(config.agentHostOk(`${config.AGENT_HOST}.massive.org.au`), true);
    assert.equal(config.agentHostOk('m3g105'), false, '别的主机不能算通过');
    assert.equal(config.agentEnvProblem(), null);
  });

  await test('环境不对时运行请求被拒绝，而不是跑到别处去', async () => {
    // 判定函数带上假的主机名/目录：必须给出可执行的原因，而不是静默放行。
    const wrongHost = config.agentEnvProblem({ hostname: 'm3g999' });
    assert.equal(typeof wrongHost, 'string');
    assert.ok(wrongHost.includes(config.AGENT_HOST),
              `原因里要点名要求的主机: ${wrongHost}`);
    const wrongDir = config.agentEnvProblem({ workdir: '/proc' });
    assert.equal(typeof wrongDir, 'string');
    assert.ok(wrongDir.includes(config.AGENT_WORKDIR) || wrongDir.includes('/proc'),
              `原因里要点名目录: ${wrongDir}`);
    // 本机（真主机名 + 真目录）必须通过。
    assert.equal(config.agentEnvProblem(), null);
  });

  await test('重置会话：清掉记录与模型上下文', async () => {
    // 先跑一轮，让 agent-a 真的有对话记录。
    await readSse(`${base}/api/run?prompt=记一句&v=${config.FRONTEND_VERSION}&profile=agent-a`);
    const before = await (await fetch(`${base}/api/agents/agent-a/chats`)).json();
    assert.ok((before.records || []).length > 0, '跑完一轮却没有任何记录');

    // 身份不存在要报错，别静默什么都不做。
    const unknown = await fetch(`${base}/api/session/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agentId: 'no-such-agent' }),
    });
    assert.equal(unknown.status, 404, `状态码 ${unknown.status}`);
    const empty = await fetch(`${base}/api/session/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    assert.equal(empty.status, 400, `状态码 ${empty.status}`);

    const reset = await fetch(`${base}/api/session/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agentId: 'agent-a' }),
    });
    assert.equal(reset.status, 200, `状态码 ${reset.status}`);
    const payload = await reset.json();
    assert.deepEqual(payload.agents, ['agent-a']);
    const after = await (await fetch(`${base}/api/agents/agent-a/chats`)).json();
    assert.equal((after.records || []).length, 0, '重置之后记录应该没了');

    // 全部重置：把所有身份都清一遍。
    const all = await fetch(`${base}/api/session/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ all: true }),
    });
    assert.equal(all.status, 200, `状态码 ${all.status}`);
    const allPayload = await all.json();
    assert.ok(Array.isArray(allPayload.agents) && allPayload.agents.length >= 3,
              `全部重置应该覆盖所有身份: ${JSON.stringify(allPayload.agents)}`);
  });

  await test('交接时要不要重置发送方的会话，可以逐个 agent 配', async () => {
    const put = (body) =>
      fetch(`${base}/api/agents/agent-a/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

    // 默认是关的。
    const before = await (await fetch(`${base}/api/health`)).json();
    const a0 = (before.agents || []).find((agent) => agent.id === 'agent-a');
    assert.equal(a0.resetSessionOnHandoff, false, '默认不该重置');

    // 勾上：存下来、读得回。
    const saved = await (await put({ resetSessionOnHandoff: true })).json();
    assert.equal(saved.resetSessionOnHandoff, true, '没存上');
    const after = await (await fetch(`${base}/api/health`)).json();
    const a1 = (after.agents || []).find((agent) => agent.id === 'agent-a');
    assert.equal(a1.resetSessionOnHandoff, true, '服务端没记住');

    // 不是布尔值要拒。
    const bad = await put({ resetSessionOnHandoff: 'yes' });
    assert.equal(bad.status, 400, '非布尔值应该被拒');

    // 恢复默认。
    await put({ resetSessionOnHandoff: null });
    const restored = await (await fetch(`${base}/api/health`)).json();
    const a2 = (restored.agents || []).find((agent) => agent.id === 'agent-a');
    assert.equal(a2.resetSessionOnHandoff, false, '恢复默认后不该再重置');
  });

  await test('交接时的两个重置开关：发送方自己 / 接收方', async () => {
    const put = (body) =>
      fetch(`${base}/api/agents/agent-a/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    const get = async () => {
      const info = await (await fetch(`${base}/api/health`)).json();
      return (info.agents || []).find((agent) => agent.id === 'agent-a');
    };

    // 两个默认都关。
    const before = await get();
    assert.equal(before.resetSessionOnHandoff, false, '默认不重置发送方');
    assert.equal(before.resetReceiverSessionOnHandoff, false, '默认不重置接收方');

    // 能各开各的。
    const onlySender = await (await put({ resetSessionOnHandoff: true })).json();
    assert.equal(onlySender.resetSessionOnHandoff, true);
    assert.equal(onlySender.resetReceiverSessionOnHandoff, false);
    const onlyReceiver = await (await put({
      resetSessionOnHandoff: false,
      resetReceiverSessionOnHandoff: true,
    })).json();
    assert.equal(onlyReceiver.resetSessionOnHandoff, false);
    assert.equal(onlyReceiver.resetReceiverSessionOnHandoff, true);

    // 非布尔值要拒。
    assert.equal(
      (await put({ resetReceiverSessionOnHandoff: 'yes' })).status,
      400,
      '非布尔值应该被拒',
    );

    // 恢复默认。
    await put({ resetSessionOnHandoff: null, resetReceiverSessionOnHandoff: null });
    const restored = await get();
    assert.equal(restored.resetSessionOnHandoff, false);
    assert.equal(restored.resetReceiverSessionOnHandoff, false);
  });

  await test('交接条件是一句话，交给最后一步的模型判定', async () => {
    const put = (body) =>
      fetch(`${base}/api/agents/agent-a/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

    // 默认没有条件（空串）= 无条件交接。
    const before = await (await fetch(`${base}/api/health`)).json();
    assert.equal(
      (before.agents || []).find((agent) => agent.id === 'agent-a').handoffCondition,
      '',
      '默认应该是无条件',
    );

    const sentence = '只有对方要求返修时才交回去；判定可行就不交。';
    const saved = await (await put({ handoffCondition: sentence })).json();
    assert.equal(saved.handoffCondition, sentence, '没存上');
    const after = await (await fetch(`${base}/api/health`)).json();
    assert.equal(
      (after.agents || []).find((agent) => agent.id === 'agent-a').handoffCondition,
      sentence,
      '服务端没记住条件',
    );

    // 条件只能是一句话，不接受对象（老的关键词形状要拒）。
    assert.equal(
      (await put({ handoffCondition: { match: 'x', action: 'skip' } })).status,
      400,
      '对象形状应该被拒',
    );

    await put({ handoffCondition: null });
    const restored = await (await fetch(`${base}/api/health`)).json();
    assert.equal(
      (restored.agents || []).find((agent) => agent.id === 'agent-a').handoffCondition,
      '',
      '恢复默认后应该是无条件',
    );
  });

  await test('交接工具名前后端一致', async () => {
    assert.equal(ROUND_HANDOFF_TOOL, config.ROUND_HANDOFF_TOOL,
                 'protocol.js 与 config.js 的交接工具名必须一致');
    // 顺带确认识别函数认得这个工具调用。
    const store = createStore();
    store.steps.push({ parts: [{ kind: 'tool', toolName: ROUND_HANDOFF_TOOL, state: 'input-available' }] });
    assert.equal(handoffRequested(store), true, 'handoffRequested 没认出交接调用');
  });

  await test('允许回传给上游是一栏一配的开关', async () => {
    const put = (body) =>
      fetch(`${base}/api/agents/agent-a/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

    const before = await (await fetch(`${base}/api/health`)).json();
    assert.equal(
      (before.agents || []).find((agent) => agent.id === 'agent-a').allowHandoffBack,
      false,
      '默认不许回传',
    );

    const saved = await (await put({ allowHandoffBack: true })).json();
    assert.equal(saved.allowHandoffBack, true, '没存上');
    const after = await (await fetch(`${base}/api/health`)).json();
    assert.equal(
      (after.agents || []).find((agent) => agent.id === 'agent-a').allowHandoffBack,
      true,
      '服务端没记住',
    );

    assert.equal((await put({ allowHandoffBack: 'yes' })).status, 400, '非布尔值应该被拒');

    await put({ allowHandoffBack: null });
    const restored = await (await fetch(`${base}/api/health`)).json();
    assert.equal(
      (restored.agents || []).find((agent) => agent.id === 'agent-a').allowHandoffBack,
      false,
      '恢复默认后不该再允许回传',
    );
  });

  await test('新的 idea 不许和归档的重复', async () => {
    const root = path.join(HERE, '..', '.out', `dedup-${process.pid}`);
    const ideas = path.join(root, 'ideas');
    const write = (dir, name) => {
      fs.mkdirSync(path.join(ideas, dir), { recursive: true });
      fs.writeFileSync(
        path.join(ideas, dir, name),
        JSON.stringify({ sections: { '1_goal': { statement: dir } }, result: { status: 'not_run' } }),
      );
    };
    write('kept', 'kept_idea.json');
    write('dup', 'dup_idea.json');
    write(path.join('_archive', 'dup'), 'dup_idea.json');
    try {
      // 归档里有的方向名，扫待做方向时要排掉。
      const archived = archivedDirectionNames(ideas);
      assert.ok(archived.has('dup'), '归档里的方向名没被认出来');
      const found = scanIdeaFiles(ideas).map((entry) => path.basename(entry.full));
      assert.deepEqual(found, ['kept_idea.json'], `重复的方向不该被当成待做: ${found}`);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  await test('公网入口不给文献检索，本机直连才有', async () => {
    // 本机直连（没有访问门槛前缀）：目录里有 web。
    const local = await (await fetch(`${base}/api/health`)).json();
    assert.ok(
      (local.permissionCatalog || []).some((item) => item.id === 'web'),
      '本机直连应当给文献检索',
    );

    // 从外面进来的请求（Host 不是本机）：出网的能力不给。
    const scratch = path.join(HERE, '..', '.out', `gated-${process.pid}`);
    const gated = createServer({ mode: 'mock', publicPath: '', dataDir: scratch });
    await new Promise((resolve) => gated.listen(0, '127.0.0.1', resolve));
    try {
      const gatedBase = `http://127.0.0.1:${gated.address().port}`;
      // fetch 不让改 Host，所以这里直接用 http 发一个带 Host 的请求。
      const info = await new Promise((resolve, reject) => {
        const request = http.request(
          {
            host: '127.0.0.1',
            port: gated.address().port,
            path: '/api/health',
            headers: { Host: 'outside.example.com' },
          },
          (response) => {
            let body = '';
            response.on('data', (chunk) => {
              body += chunk;
            });
            response.on('end', () => {
              try {
                resolve(JSON.parse(body));
              } catch (err) {
                reject(err);
              }
            });
          },
        );
        request.on('error', reject);
        request.end();
      });
      assert.ok(
        !(info.permissionCatalog || []).some((item) => item.id === 'web'),
        '外面进来的请求不该给文献检索',
      );
      assert.ok(
        (info.permissionCatalog || []).some((item) => item.id === 'read'),
        '别的权限照给',
      );
    } finally {
      gated.closeAllConnections();
      await new Promise((resolve) => gated.close(resolve));
      fs.rmSync(scratch, { recursive: true, force: true });
    }
  });

  await test('检索文献的工具与权限都在', async () => {
    const web = PERMISSION_CATALOG.find((item) => item.id === 'web');
    assert.ok(web, '权限目录里没有 web');
    assert.deepEqual(web.tools, ['webSearch']);
    assert.ok(web.description.length > 0, 'web 权限要说得清自己管什么');
    const tools = toolsForPermissions(['read', 'web']).map((tool) => tool.name).sort();
    assert.deepEqual(tools, ['checkIdea', 'checkJson', 'find', 'grep', 'ls', 'read', 'webSearch']);
  });

  await test('全局设置可读可改，改完立刻对运行生效', async () => {
    const before = await (await fetch(`${base}/api/settings`)).json();
    assert.ok(before.settings, 'settings 没返回');
    assert.equal(typeof before.settings.agentHost, 'string');
    assert.equal(typeof before.settings.agentWorkdir, 'string');
    assert.ok(before.settings.agentWorkdir.startsWith('/'), '工作目录该是绝对路径');

    // 改成不合法的值必须被拒：主机名不能空，工作目录必须是绝对路径。
    const blankHost = await fetch(`${base}/api/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agentHost: '   ' }),
    });
    assert.equal(blankHost.status, 400, '空主机名应该被拒');
    const relativeDir = await fetch(`${base}/api/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agentWorkdir: 'relative/dir' }),
    });
    assert.equal(relativeDir.status, 400, '相对路径应该被拒');

    // 存一个会挡住运行的目录：运行请求必须被拒，而且原因要点名那个目录。
    const original = before.settings;
    const bad = await (await fetch(`${base}/api/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agentWorkdir: '/tmp' }),
    })).json();
    assert.equal(bad.settings.agentWorkdir, '/tmp', '工作目录没存上');
    assert.ok(bad.envProblem, '换一个目录后应该报告环境问题');
    assert.ok(bad.envProblem.includes('/tmp'), `原因里要点名目录: ${bad.envProblem}`);
    const refused = await fetch(
      `${base}/api/run?prompt=hi&v=${config.FRONTEND_VERSION}&profile=agent-a`,
    );
    assert.equal(refused.status, 503, '环境不对时运行请求应该被拒');

    // 换回来，后面的用例还要在这个目录里跑。
    const restored = await (await fetch(`${base}/api/settings`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(original),
    })).json();
    assert.equal(restored.settings.agentWorkdir, original.agentWorkdir);
    assert.equal(restored.envProblem, null, `恢复后不该还有环境问题: ${restored.envProblem}`);
  });

  await test('AGENT FIX 既审也改：能写代码，但只能改候选树', async () => {
    const info = await (await fetch(`${base}/api/health`)).json();
    const check = info.agents.find((agent) => agent.id === 'agent-fix');
    assert.ok(check, 'health 里没有 agent-fix');
    // 它要能把写错的地方改对，所以得有写权限；也只到写为止。
    assert.deepEqual(check.permissions, ['read', 'write'],
                     `AGENT FIX 的权限不对: ${JSON.stringify(check.permissions)}`);
    assert.equal(check.steps.length, 5, `步骤数不对: ${check.steps.length}`);
    const text = check.steps.map((step) => step.instruction).join('\n');
    assert.ok(text.includes('改代码与测试') || text.includes('write'),
              '步骤里没有改代码这一步');
    assert.ok(check.instructions.includes('4. 改：把 FAIL 的条目在候选树里改对'),
              '人设里没写明它会把写错的地方改对');
    // 边界写清楚：只动候选树，规格与冻结基线不许动。
    assert.ok(check.instructions.includes('冻结基线') && check.instructions.includes('不许动'),
              '人设里没写明改动边界');
  });

  await test('AGENT STAGE2 只跑训练：能启动、能看状态、能暂停', async () => {
    const info = await (await fetch(`${base}/api/health`)).json();
    const runner = info.agents.find((agent) => agent.id === 'agent-stage2');
    assert.ok(runner, 'health 里没有 agent-stage2');
    assert.deepEqual(runner.permissions, ['read', 'train'],
                     `AGENT STAGE2 的权限不对: ${JSON.stringify(runner.permissions)}`);
    assert.equal(runner.steps.length, 5, `步骤数不对: ${runner.steps.length}`);
    const text = runner.steps.map((step) => step.instruction).join('\n');
    assert.ok(text.includes('stage2Start'), '它该会启动训练');
    assert.ok(text.includes('stage2Status'), '它该会看训练状态');
    assert.ok(text.includes('stage2Stop'), '发现问题时它该会暂停训练');
    assert.ok(text.includes('每分钟'), '它的步骤里没写明每分钟检查一次');
    assert.deepEqual(runner.handoffTargets, ['agent-fix'],
                     '出问题时它该把活交给 AGENT FIX');
    // 训练这一项权限带的就是这三个工具。
    const train = (info.permissionCatalog || []).find((item) => item.id === 'train');
    assert.ok(train, '权限目录里没有 train');
    assert.deepEqual(train.tools, ['stage2Start', 'stage2Status', 'stage2Stop']);
  });

  await test('Group：建组、改组、整组交接、删组', async () => {
    const post = (body) =>
      fetch(`${base}/api/groups`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    const created = await (await post({ name: '修复组', members: ['agent-fix', 'agent-math'] })).json();
    assert.ok(created.group, '组没建起来');
    const groupId = created.group.id;
    assert.equal(created.group.name, '修复组');
    assert.deepEqual(created.group.members, ['agent-fix', 'agent-math']);

    // 组名不能空，成员必须是存在的身份。
    assert.equal((await post({ name: '  ', members: ['agent-a'] })).status, 400, '空组名应该被拒');
    assert.equal((await post({ name: 'x', members: [] })).status, 400, '空成员应该被拒');
    assert.equal((await post({ name: 'x', members: ['no-such-agent'] })).status, 400, '不存在的成员应该被拒');

    // 改组：成员可以换。
    const updated = await (await fetch(`${base}/api/groups/${groupId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ members: ['agent-fix'] }),
    })).json();
    assert.deepEqual(updated.group.members, ['agent-fix']);

    // 交接目标可以指向一个组。
    const put = await fetch(`${base}/api/agents/agent-a/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ handoffTargets: [`group:${groupId}`] }),
    });
    assert.equal(put.status, 200, '把组设成交接目标应该被接受');
    const health = await (await fetch(`${base}/api/health`)).json();
    const a = health.agents.find((agent) => agent.id === 'agent-a');
    assert.ok(a.handoffTargets.includes(`group:${groupId}`), '组没进交接目标');
    assert.ok(
      (health.groups || []).some((group) => group.id === groupId),
      'health 里没有这个组',
    );

    // 删组：各 agent 的交接目标里那个组引用也要被摘掉。
    const removed = await fetch(`${base}/api/groups/${groupId}`, { method: 'DELETE' });
    assert.equal(removed.status, 200);
    const after = await (await fetch(`${base}/api/health`)).json();
    const a2 = after.agents.find((agent) => agent.id === 'agent-a');
    assert.ok(!a2.handoffTargets.includes(`group:${groupId}`), '删组后目标里不该还留着它');
    assert.equal((after.groups || []).length, 0, '组应该被删干净了');
  });

  await test('旧版前端被版本闸门拦下，拿到刷新提示而不是 400', async () => {
    // 旧前端硬编码了旧的 agent 身份（profile=assistant）。如果先校验身份、
    // 后校验版本，用户看到的就是一句莫名其妙的 "未知的 agent 身份"，
    // 根本猜不到要刷新页面。版本闸门必须排在身份校验前面。
    const { res, frames } = await readSse(
      `${base}/api/run?prompt=hi&v=2026-11-10.4&profile=assistant`,
    );
    assert.equal(res.status, 200, '旧版前端不该拿到 400，它应该收到可执行的提示');

    const chunks = parseChunks(frames);
    assert.ok(!chunks.some((c) => c.type === 'start'), '旧版前端不该把 agent 跑起来');

    const err = chunks.find((c) => c.type === 'error');
    assert.ok(err, '没有给出错误提示');
    assert.match(err.errorText, /刷新/, `提示里没说要刷新: ${err.errorText}`);
    assert.ok(
      err.errorText.includes(config.FRONTEND_VERSION),
      '提示里没有当前服务端版本号',
    );
  });

  await test('首页带上服务端当前版本号，页面守卫能据此比对', async () => {
    // 旧版的守卫只检查"启动标记在不在"，旧版 JS 也会设置标记，于是放行、
    // 一直用旧 JS。守卫得比对版本号，得先把期望值写进首页。
    const html = await (await fetch(`${base}/`)).text();
    assert.ok(
      html.includes(config.FRONTEND_VERSION),
      '首页里没有服务端当前版本号，守卫无从比对',
    );
    assert.ok(
      !html.includes('__FRONTEND_VERSION__'),
      '版本号占位符没有被替换',
    );
  });

  await test('POST /api/run 把上游分析结果交给下游 agent', async () => {
    const { res, frames } = await readSse(`${base}/api/run`, {
      method: 'POST',
      payload: {
        prompt: '把要点整理成三条行动项',
        v: config.FRONTEND_VERSION,
        profile: 'agent-b',
        handoff: 'AGENT A 的分析结论：正式训练入口在 stage3_T5Train/train_HG-Rec.py。',
      },
    });
    assert.equal(res.status, 200);

    const chunks = parseChunks(frames);
    assert.equal(chunks[0].type, 'start', '交接请求应正常开始');

    const text = chunks
      .filter((c) => c.type === 'text-delta')
      .map((c) => c.delta)
      .join('');
    assert.ok(
      text.includes('AGENT A 的分析结论'),
      `下游没拿到上游的分析结果: ${text.slice(0, 200)}`,
    );
    assert.ok(
      text.includes('三条行动项'),
      `下游没拿到给它的要求: ${text.slice(0, 200)}`,
    );
  });

  await test('POST /api/run 只有交接内容、没有要求时照样能跑', async () => {
    const { res, frames } = await readSse(`${base}/api/run`, {
      method: 'POST',
      payload: {
        prompt: '',
        v: config.FRONTEND_VERSION,
        profile: 'agent-b',
        handoff: 'AGENT A 的分析结论：正式训练入口在 stage3_T5Train/train_HG-Rec.py。',
      },
    });
    assert.equal(res.status, 200, '只带交接内容时不该被拒掉');

    const text = parseChunks(frames)
      .filter((c) => c.type === 'text-delta')
      .map((c) => c.delta)
      .join('');
    assert.ok(
      text.includes('AGENT A 的分析结论'),
      `下游没拿到上游的分析结果: ${text.slice(0, 200)}`,
    );
  });

  await test('交接消息里没有要求时，不留一个空的「给你的要求」段', () => {
    const withEmpty = buildHandoffPrompt('', '上游的一段分析', 'AGENT B');
    assert.ok(withEmpty.includes('上游的一段分析'), '上游结果丢了');
    assert.ok(!withEmpty.includes('【给你的要求】'), '不该出现空的「给你的要求」段');

    const withRequirement = buildHandoffPrompt('请给三条建议', '上游的一段分析', 'AGENT B');
    assert.ok(withRequirement.includes('【给你的要求】'), '有要求时该有这一段');
    assert.ok(withRequirement.includes('请给三条建议'), '要求丢了');
  });

  await test('POST /api/run 没有交接内容时，下游不会凭空收到上游结果', async () => {
    const { frames } = await readSse(`${base}/api/run`, {
      method: 'POST',
      payload: { prompt: 'hi', v: config.FRONTEND_VERSION, profile: 'agent-b' },
    });
    const text = parseChunks(frames)
      .filter((c) => c.type === 'text-delta')
      .map((c) => c.delta)
      .join('');
    assert.ok(!text.includes('AGENT A 的分析结论'), '没交接就不该出现上游结果');
  });

  await test('POST /api/run 缺 prompt 直接 400', async () => {
    const res = await fetch(`${base}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: '   ',
        v: config.FRONTEND_VERSION,
        profile: 'agent-b',
      }),
    });
    assert.equal(res.status, 400, '空 prompt 应被拒掉');
  });

  await test('POST /api/run 身份不认识直接 400，不退回默认', async () => {
    const res = await fetch(`${base}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: 'hi',
        v: config.FRONTEND_VERSION,
        profile: 'no-such-agent',
      }),
    });
    assert.equal(res.status, 400, '未知 agent 应该被拒掉');
  });

  await test('POST /api/run 版本对不上照样被拦下', async () => {
    const { res, frames } = await readSse(`${base}/api/run`, {
      method: 'POST',
      payload: { prompt: 'hi', v: '1999-01-01.0', profile: 'agent-b' },
    });
    assert.equal(res.status, 200);
    const chunks = parseChunks(frames);
    assert.ok(!chunks.some((c) => c.type === 'start'), '过期前端不该把 agent 跑起来');
    assert.ok(chunks.some((c) => c.type === 'error'), '过期前端应收到错误提示');
  });

  await test('空 prompt 返回 400 而不是静默运行', async () => {
    const res = await fetch(`${base}/api/run?prompt=`);
    assert.equal(res.status, 400);
  });

  await test('SSE 响应带 UI message stream 头', async () => {
    const { res } = await readSse(`${base}/api/run?prompt=hello&v=${config.FRONTEND_VERSION}`);
    assert.equal(res.status, 200);
    assert.equal(res.headers.get('x-vercel-ai-ui-message-stream'), 'v1');
    assert.match(res.headers.get('content-type'), /text\/event-stream/);
  });

  await test('SSE 以 data: [DONE] 结束', async () => {
    const { frames } = await readSse(`${base}/api/run?prompt=hello&v=${config.FRONTEND_VERSION}`);
    assert.ok(frames.length > 0, '没有任何帧');
    assert.equal(frames[frames.length - 1], '[DONE]');
  });

  await test('记录里的运行状态跟着流里的实际情况走', async () => {
    assert.equal(runStatus({ chunks: [{ type: 'start' }, { type: 'finish' }] }), 'done');
    // 流里出现 error 分片就不是 done —— 否则刷新后历史写着"已完成"，那一轮却什么都没产出。
    assert.equal(runStatus({ chunks: [{ type: 'start' }, { type: 'error' }] }), 'error');
    // 客户端断开优先：断线不是 agent 的失败。
    assert.equal(
      runStatus({ aborted: true, chunks: [{ type: 'error' }] }),
      'aborted',
    );
  });

  await test('帧序列覆盖 start/step/tool/text/finish', async () => {
    const { frames } = await readSse(`${base}/api/run?prompt=hello&v=${config.FRONTEND_VERSION}`);
    const chunks = parseChunks(frames);
    const types = chunks.map((c) => c.type);
    assert.equal(types[0], 'start');
    assert.ok(types.includes('start-step'), '缺少 start-step');
    assert.ok(types.includes('tool-input-start'), '缺少 tool-input-start');
    assert.ok(types.includes('tool-input-available'), '缺少 tool-input-available');
    assert.ok(types.includes('tool-output-available'), '缺少 tool-output-available');
    assert.ok(types.includes('text-delta'), '缺少 text-delta');
    assert.ok(types.includes('finish-step'), '缺少 finish-step');
    assert.equal(types[types.length - 2], 'finish');
  });

  await test('reducer 把分片还原成时间轴', async () => {
    const { frames } = await readSse(`${base}/api/run?prompt=hello&v=${config.FRONTEND_VERSION}`);
    const store = createStore();
    for (const chunk of parseChunks(frames)) applyChunk(store, chunk);

    assert.equal(store.steps.length, 2, '步骤数');
    assert.equal(store.messageId !== null, true, 'messageId');

    const tools = store.steps.flatMap((s) => s.parts).filter((p) => p.kind === 'tool');
    assert.equal(tools.length, 1, '工具调用数');
    assert.equal(tools[0].state, 'output-available');
    assert.deepEqual(tools[0].input, { path: 'package.json' });

    const texts = store.steps.flatMap((s) => s.parts).filter((p) => p.kind === 'text');
    assert.equal(texts.length, 1);
    assert.ok(texts[0].text.length > 0, '文本为空');

    assert.equal(store.finishReason, 'stop');
    assert.equal(store.usage.totalTokens, 2159);
    assert.equal(store.steps[0].closed, true, '第一步应已闭合');
  });

  await test('reducer 记录工具失败', async () => {
    const store = createStore();
    applyChunk(store, { type: 'start-step' });
    applyChunk(store, { type: 'tool-input-start', toolCallId: 'c1', toolName: 'readFile' });
    applyChunk(store, { type: 'tool-input-available', toolCallId: 'c1', toolName: 'readFile', input: { path: 'x' } });
    applyChunk(store, { type: 'tool-output-error', toolCallId: 'c1', errorText: '路径不存在: x' });
    const part = store.steps[0].parts[0];
    assert.equal(part.state, 'output-error');
    assert.equal(part.errorText, '路径不存在: x');
  });

  await test('reducer 记录 error 与 abort', async () => {
    const store = createStore();
    applyChunk(store, { type: 'error', errorText: 'boom' });
    applyChunk(store, { type: 'abort', reason: 'user cancelled' });
    assert.equal(store.error, 'boom');
    assert.equal(store.aborted, 'user cancelled');
  });

  await test('reducer 累积流式 text-delta 与 reasoning-delta', async () => {
    const store = createStore();
    applyChunk(store, { type: 'start-step' });
    applyChunk(store, { type: 'text-start', id: 't1' });
    applyChunk(store, { type: 'text-delta', id: 't1', delta: 'Hel' });
    applyChunk(store, { type: 'text-delta', id: 't1', delta: 'lo' });
    applyChunk(store, { type: 'text-end', id: 't1' });
    applyChunk(store, { type: 'reasoning-start', id: 'r1' });
    applyChunk(store, { type: 'reasoning-delta', id: 'r1', delta: 'think' });
    applyChunk(store, { type: 'reasoning-end', id: 'r1' });
    const parts = store.steps[0].parts;
    assert.equal(parts.find((p) => p.kind === 'text').text, 'Hello');
    assert.equal(parts.find((p) => p.kind === 'reasoning').text, 'think');
  });

  await test('mock agent 的分片本身就是协议合法的', async () => {
    const store = createStore();
    let count = 0;
    for await (const chunk of runMockChunks('hi')) {
      count += 1;
      applyChunk(store, chunk);
    }
    assert.ok(count > 8, '分片太少');
    assert.equal(store.steps.length, 2);
    assert.equal(store.error, null);
  });

  await test('foldText 把长输出折叠、短输出原样返回', async () => {
    const short = foldText('hello');
    assert.equal(short.collapsed, false);
    assert.equal(short.full, 'hello');
    assert.equal(short.preview, 'hello');

    const long = foldText('x'.repeat(1200));
    assert.equal(long.collapsed, true);
    assert.equal(long.preview.length, 600);
    assert.equal(long.full.length, 1200);
    assert.equal(long.chars, 1200);
    assert.equal(long.lines, 1);

    const obj = foldText({ a: 1 });
    assert.equal(obj.collapsed, false);
    assert.ok(obj.full.includes('"a": 1'));
  });

  await test('roundEnded 只认 endRound 工具调用，不认文字', async () => {
    assert.equal(config.ROUND_END_TOOL, ROUND_END_TOOL, '配置与协议里的控件名不一致');

    const store = createStore();
    applyChunk(store, { type: 'start-step' });
    applyChunk(store, { type: 'text-delta', id: 't', delta: '没有 Open Issue，本轮结束。' });
    assert.equal(roundEnded(store), false, '文字里的"本轮结束"不该被当成信号');

    applyChunk(store, { type: 'tool-input-start', toolCallId: 'e1', toolName: ROUND_END_TOOL });
    assert.equal(roundEnded(store), false, '参数还没到就不算调用');

    applyChunk(store, {
      type: 'tool-input-available',
      toolCallId: 'e1',
      toolName: ROUND_END_TOOL,
      input: { reason: '没有 Open Issue' },
    });
    assert.equal(roundEnded(store), true);
  });

  await test('reducer 记录认不出的分片类型', async () => {
    const store = createStore();
    applyChunk(store, { type: 'text-start', id: 't1' });
    applyChunk(store, { type: 'totally-unknown-chunk', payload: 1 });
    applyChunk(store, { type: 'totally-unknown-chunk', payload: 2 });
    assert.deepEqual(store.unknownTypes, ['totally-unknown-chunk']);
  });

  await test('reducer 处理审批、来源、附件、steering 分片', async () => {
    const store = createStore();
    applyChunk(store, { type: 'start-step' });
    applyChunk(store, { type: 'tool-input-start', toolCallId: 'c9', toolName: 'write' });
    applyChunk(store, {
      type: 'tool-approval-request',
      toolCallId: 'c9',
      approvalId: 'a9',
      reason: '需要人工确认',
    });
    const tool = store.steps[0].parts.find((p) => p.kind === 'tool');
    assert.equal(tool.state, 'approval-requested');
    assert.equal(tool.approvalId, 'a9');

    applyChunk(store, { type: 'tool-approval-response', approvalId: 'a9', approved: false });
    assert.equal(tool.state, 'approval-answered');
    assert.equal(tool.approved, false);

    applyChunk(store, { type: 'source-url', sourceId: 's1', url: 'https://x.dev' });
    applyChunk(store, { type: 'file', url: 'data:image/png;base64,AAA', mediaType: 'image/png' });
    applyChunk(store, { type: 'custom', kind: 'openai.compaction' });
    applyChunk(store, { type: 'steering-message', id: 'st1', parts: [] });
    const kinds = store.steps[0].parts.map((p) => p.kind);
    assert.ok(kinds.includes('source-url'));
    assert.ok(kinds.includes('file'));
    assert.ok(kinds.includes('custom'));
    assert.equal(store.steering.length, 1);
  });

  await test('客户端中途断开不会弄坏服务端', async () => {
    const controller = new AbortController();
    // 这条要的是"流还在吐的时候断开"，所以显式 instant=0 把 mock 的延迟留回来；
    // 关掉延迟的话整条响应会在 abort 之前就跑完，测不到断开。
    const pending = readSse(
      `${base}/api/run?prompt=hello&v=${config.FRONTEND_VERSION}&instant=0`,
      { signal: controller.signal },
    );
    setTimeout(() => controller.abort(), 30);
    await assert.rejects(pending, (err) => err.name === 'AbortError');

    // 服务端仍能继续服务下一个请求
    const { frames } = await readSse(`${base}/api/run?prompt=again&v=${config.FRONTEND_VERSION}`);
    assert.equal(frames[frames.length - 1], '[DONE]');
  });

  await test('PUT 配置：名字能改、能恢复默认、非法名字被拒', async () => {
    const put = putAgentConfig;

    const renamed = await put('agent-a', { name: '  首席执行者  ' });
    assert.equal(renamed.status, 200, JSON.stringify(renamed.json));
    assert.equal(renamed.json.name, '首席执行者');
    assert.equal(renamed.json.defaultName, 'AGENT Write');
    assert.equal(renamed.json.customized.name, true);

    const health = await (await fetch(`${base}/api/health`)).json();
    assert.equal(
      health.agents.find((agent) => agent.id === 'agent-a').name,
      '首席执行者',
      'health 里还是旧名字',
    );

    const blank = await put('agent-a', { name: '   ' });
    assert.equal(blank.status, 400, '空名字应该被拒');
    const long = await put('agent-a', { name: 'x'.repeat(41) });
    assert.equal(long.status, 400, '超长名字应该被拒');

    const restored = await put('agent-a', { name: null });
    assert.equal(restored.status, 200);
    assert.equal(restored.json.name, 'AGENT Write');
    assert.equal(restored.json.customized.name, false);
  });

  await test('PUT 配置：交接延迟能设、能清、非法值被拒', async () => {
    const set = await putAgentConfig('agent-a', { handoffDelaySeconds: 5 });
    assert.equal(set.status, 200, JSON.stringify(set.json));
    assert.equal(set.json.handoffDelaySeconds, 5);
    assert.equal(set.json.customized.handoffDelay, true);

    const health = await (await fetch(`${base}/api/health`)).json();
    assert.equal(
      health.agents.find((agent) => agent.id === 'agent-a').handoffDelaySeconds,
      5,
      'health 里没有延迟',
    );

    assert.equal((await putAgentConfig('agent-a', { handoffDelaySeconds: -1 })).status, 400);
    assert.equal((await putAgentConfig('agent-a', { handoffDelaySeconds: 1.5 })).status, 400);
    assert.equal((await putAgentConfig('agent-a', { handoffDelaySeconds: 3601 })).status, 400);
    assert.equal((await putAgentConfig('agent-a', { handoffDelaySeconds: '5' })).status, 400);

    const cleared = await putAgentConfig('agent-a', { handoffDelaySeconds: null });
    assert.equal(cleared.status, 200);
    assert.equal(cleared.json.handoffDelaySeconds, 0);
    assert.equal(cleared.json.customized.handoffDelay, false);
  });

  await test('循环配置：次数与间隔能存能取，坏值被拒', async () => {
    const putLoop = (loop) =>
      fetch(`${base}/api/agents/agent-a/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ loop }),
      });

    let res = await putLoop({ enabled: true, count: 3, intervalSeconds: 7 });
    const savedText = await res.text();
    assert.equal(res.status, 200, savedText);
    const saved = JSON.parse(savedText);
    assert.deepEqual(saved.loop, { enabled: true, count: 3, intervalSeconds: 7 });
    assert.equal(saved.customized.loop, true);
    assert.deepEqual(saved.defaultLoop, { enabled: false, count: 2, intervalSeconds: 0 });

    const health = await (await fetch(`${base}/api/health`)).json();
    const agentA = health.agents.find((agent) => agent.id === 'agent-a');
    assert.deepEqual(agentA.loop, { enabled: true, count: 3, intervalSeconds: 7 });

    for (const bad of [
      { enabled: 'yes', count: 2, intervalSeconds: 0 },
      { enabled: true, count: -1, intervalSeconds: 0 },
      { enabled: true, count: 101, intervalSeconds: 0 },
      { enabled: true, count: 2, intervalSeconds: -1 },
      { enabled: true, count: 2, intervalSeconds: 3601 },
      { enabled: true, count: 2.5, intervalSeconds: 0 },
      5,
    ]) {
      assert.equal((await putLoop(bad)).status, 400, `坏配置应被拒：${JSON.stringify(bad)}`);
    }

    res = await putLoop(null);
    assert.equal(res.status, 200);
    assert.equal((await res.json()).customized.loop, false, '置空之后回到默认关闭');
  });

  await test('IDEA 文本：能存能取、非法值被拒、随运行发给 agent', async () => {
    const ideaText = '页面提供的机制定义：E16×H16 混合几何，结构信号必须进同一条通路。';
    const put = (body) =>
      fetch(`${base}/api/idea`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

    let res = await fetch(`${base}/api/idea`);
    assert.equal(res.status, 200);
    const initial = await res.json();
    assert.equal(typeof initial.text, 'string');
    assert.equal(initial.text, '', '还没保存过，页面那份应该是空的');
    assert.equal(initial.updatedAt, null);
    // 字段契约的版本与核对结果要一起给页面：IDEA 视图靠它说准"结构合格"。
    const health = await (await fetch(`${base}/api/health`)).json();
    assert.equal(health.ideaSchemaVersion, 'genrec.idea.v6');
    if (initial.doc) {
      assert.ok(initial.ideaVerdict, '有稿子时必须给契约核对结果');
      assert.equal(typeof initial.ideaVerdict.ok, 'boolean');
      assert.equal(initial.ideaVerdict.version, 'genrec.idea.v6');
      assert.ok(Array.isArray(initial.ideaVerdict.errors));
    }
    // 仓库里的机制文件要能拿到：窗口第一次打开时靠它当草稿。
    // 「一个方向都没有」（都归档了）也是合法状态 —— 这时页面要明确说清楚，
    // 而不是装作有内容，也不是 500。
    assert.equal(typeof initial.fileText, 'string');
    if (initial.doc) {
      assert.ok(initial.fileText.length > 0, '机制文件的内容没拿到');
    } else {
      // 两种都合法：一个方向都没有（fileText 空、错误里点名 *_idea.json）；
      // 或者文件正在被写/写坏了（fileText 有内容、错误里是解析失败）。
      assert.equal(typeof initial.docError, 'string');
      if (initial.fileText === '') {
        assert.ok(
          initial.docError.includes('*_idea.json'),
          `没有方向时要说清楚找的是什么: ${initial.docError}`,
        );
      } else {
        assert.ok(
          initial.docError.includes('解析失败') || initial.docError.includes('不是 JSON'),
          `文件读得出但解析不了时要报解析错误: ${initial.docError}`,
        );
      }
    }
    // 主线在 ideas/main_line.json，整份方向共用一份；机制文件只写这一轮的差异。
    assert.equal(typeof initial.mainLine, 'object', 'mainLine 没返回');
    assert.equal(typeof initial.mainLine.statement, 'string');
    assert.ok(initial.mainLine.statement.length > 0, '主线的 statement 是空的');
    if (initial.mainLine.must_hold !== undefined) {
      assert.ok(Array.isArray(initial.mainLine.must_hold) && initial.mainLine.must_hold.length > 0,
                '主线的 must_hold 要么不给，要么是非空数组');
    }
    assert.equal(initial.mainLineError, null);
    // 基线事实共用一份，机制文件不再抄：页面从 baseline 块读。
    assert.equal(typeof initial.baseline, 'object', 'baseline 没返回');
    assert.equal(typeof initial.baseline.tree, 'string');
    assert.equal(initial.baselineError, null);
    // 验证计划是可选的：方向目录没有 verification.json 时，其余块照常返回，
    // verification 为 null 并带一句原因，而不是整页报错。
    // 公式单独一份：方向目录下的 math.json，可选；缺了整块不显示。
    const hasMath = initial.math !== null && initial.math !== undefined;
    if (hasMath) {
      assert.equal(typeof initial.math, 'object', 'math 应该是对象');
      // 老形状是 {symbols, formulas, ...}；新方向可能用自己的键（notation /
      // structural_potential ……）。两种都接受，但内容不能是空的。
      assert.ok(Object.keys(initial.math).length > 0, 'math 不该是空对象');
      if (initial.math.formulas !== undefined) {
        assert.ok(Array.isArray(initial.math.formulas) && initial.math.formulas.length > 0,
                  '给 formulas 就要是非空数组');
        assert.ok(initial.math.formulas.every((f) => typeof f.formula === 'string' && f.formula.trim() !== ''),
                  '每条公式都要有 formula 文本');
      }
      assert.equal(initial.mathError, null);
    } else {
      assert.equal(typeof initial.mathError, 'string', '缺 math.json 时要给出原因');
      assert.ok(
        initial.mathError.includes('math.json') || initial.mathError.includes('*_idea.json'),
        `原因里要点名文件（缺公式文件，或者连方向都没有），实际是 ${JSON.stringify(initial.mathError)}`,
      );
    }
    const hasVerification = initial.verification !== null && initial.verification !== undefined;
    if (hasVerification) {
      assert.equal(typeof initial.verification, 'object', 'verification 应该是对象');
      assert.ok(Array.isArray(initial.verification.checks) && initial.verification.checks.length > 0,
                'verification.checks 应该是非空数组');
      assert.equal(initial.verificationError, null);
    } else {
      assert.equal(typeof initial.verificationError, 'string', '缺 verification.json 时要给出原因');
      assert.ok(
        initial.verificationError.includes('verification.json') ||
          initial.verificationError.includes('*_idea.json'),
        `原因里要点名文件（缺验证计划，或者连方向都没有），实际是 ${JSON.stringify(initial.verificationError)}`,
      );
    }

    res = await put({ text: ideaText });
    assert.equal(res.status, 200, await res.text());
    const saved = await (await fetch(`${base}/api/idea`)).json();
    assert.equal(saved.text, ideaText);
    assert.equal(typeof saved.updatedAt, 'number');

    assert.equal((await put({})).status, 400, '缺 text 应该被拒');
    assert.equal((await put({ text: 123 })).status, 400, 'text 不是字符串应该被拒');
    assert.equal((await put({ text: 'x'.repeat(200001) })).status, 400, '超长应该被拒');

    // 随运行发送：mock 会把收到的 prompt 原样念出来。
    const { frames } = await readSse(`${base}/api/run?prompt=hello&v=${config.FRONTEND_VERSION}`);
    const spoken = parseChunks(frames)
      .filter((chunk) => chunk.type === 'text-delta')
      .map((chunk) => chunk.delta)
      .join('');
    assert.ok(spoken.includes('页面提供的机制定义'), 'prompt 里没有带上 idea 文本');

    // 清空，别影响后面的用例。
    assert.equal((await put({ text: '' })).status, 200);
  });

  // ---------- 新建 agent ----------
  // 这一段必须放在最后：新建的 agent 会进服务端的 agent 列表，
  // 前面那些"服务端有两个 agent"的断言得先跑完。
  process.stdout.write('\n  新建 agent · 系统提示词默认是空的\n');

  const homeHtml = await (await fetch(`${base}/`)).text();

  await test('首页有「新建 Agent」按钮', async () => {
    assert.ok(
      homeHtml.includes('data-role="newAgentBtn"'),
      '首页里没有新建 agent 的按钮',
    );
  });

  await test('主页面有 agent 侧边栏，配置不铺在聊天页上', async () => {
    // 主页面不该直接摊开完整配置：聊天区与配置区分成两块，
    // 配置区单独占一个视图（点侧边栏的「配置」进去）。
    assert.ok(homeHtml.includes('data-role="agentList"'), '首页里没有 agent 列表（侧边栏）');
    const chatAt = homeHtml.indexOf('class="chat-part"');
    const configAt = homeHtml.indexOf('class="config-part"');
    assert.ok(chatAt >= 0 && configAt > chatAt, '首页里没有把面板分成 chat-part / config-part');
    const timelineAt = homeHtml.indexOf('data-role="timeline"');
    assert.ok(
      timelineAt > chatAt && timelineAt < configAt,
      '时间轴不在聊天区里 —— 聊天框与配置区没有分开',
    );
    const instrAt = homeHtml.indexOf('data-role="instructionsBox"');
    assert.ok(instrAt > configAt, '系统提示词的配置不在配置区里');
    assert.ok(homeHtml.includes('data-role="backToChat"'), '配置区里没有回到聊天的入口');
  });

  await test('新建 agent 用的面板模板与既有面板同构', async () => {
    // 模板与 AGENT A / AGENT B 必须是同一种形式，差别只在提示词与预填内容。
    // 逐个标签比对骨架：只看 data-role 的集合会漏掉标签结构本身的差异。
    const panelA = extractPanel(homeHtml, 'panelA');
    const tpl = extractTemplate(homeHtml, 'panelTemplate');
    assert.ok(panelA, '首页里找不到 panelA');
    assert.ok(tpl, '首页里找不到给新建 agent 用的面板模板');
    assert.equal(
      skeleton(tpl),
      skeleton(panelA),
      '新建 agent 的模板与左栏结构不一致',
    );
  });

  let created = null;

  await test('POST /api/agents 造出一个系统提示词为空的 agent', async () => {
    const res = await fetch(`${base}/api/agents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    assert.equal(res.status, 201, `状态码 ${res.status}`);
    const profile = await res.json();
    assert.equal(typeof profile.id, 'string');
    assert.ok(profile.id.length > 0, '新 agent 没有 id');
    assert.ok(typeof profile.name === 'string' && profile.name.length > 0, '新 agent 没有名字');
    assert.equal(profile.instructions, '', '新 agent 的系统提示词应该是空的');
    assert.deepEqual(profile.steps, [], '新 agent 没有默认步骤');
    created = profile;
  });

  await test('新建的 agent 出现在 /api/health 里', async () => {
    const info = await (await fetch(`${base}/api/health`)).json();
    const found = (info.agents || []).find((agent) => agent.id === created.id);
    assert.ok(found, 'health 里没有新 agent');
    assert.equal(found.instructions, '', 'health 里新 agent 的提示词应该还是空的');
  });

  await test('新建的 agent 能真的跑起来', async () => {
    const { frames } = await readSse(
      `${base}/api/run?prompt=hi&v=${config.FRONTEND_VERSION}&profile=${created.id}`,
    );
    const chunks = parseChunks(frames);
    assert.equal(chunks[0].type, 'start', '新 agent 没有跑起来');
    const text = chunks
      .filter((c) => c.type === 'text-delta')
      .map((c) => c.delta)
      .join('');
    assert.ok(text.includes(created.name), `回答里没认出新 agent: ${text.slice(0, 120)}`);
  });

  await test('POST /api/agents 参数不合法直接 400', async () => {
    for (const body of [{ name: 123 }, { name: ['x'] }, { instructions: { a: 1 } }]) {
      const res = await fetch(`${base}/api/agents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      assert.equal(res.status, 400, `${JSON.stringify(body)} 应该被拒掉`);
    }
  });

  await test('POST /api/run 带空的 instructions 表示明确不要系统提示词', async () => {
    // 字段缺省 = 用这个 agent 的默认提示词；字段存在但为空 = 明确不要。
    // 这两件事必须分得开，否则"把提示词框清空"会被悄悄换成默认提示词。
    const { frames } = await readSse(`${base}/api/run`, {
      method: 'POST',
      payload: {
        prompt: 'hi',
        v: config.FRONTEND_VERSION,
        profile: 'agent-a',
        instructions: '',
      },
    });
    const text = parseChunks(frames)
      .filter((c) => c.type === 'text-delta')
      .map((c) => c.delta)
      .join('');
    assert.ok(
      text.includes('没有系统提示词'),
      `回答里没说清没有系统提示词: ${text.slice(0, 160)}`,
    );
  });

  // ---------- 权限 ----------
  process.stdout.write('\n  权限 · 每个 agent 能做什么由配置说了算\n');

  await test('权限目录里的每一项都绑着真实存在的工具', async () => {
    assert.ok(Array.isArray(PERMISSION_CATALOG), '权限目录必须是数组');
    assert.ok(PERMISSION_CATALOG.length > 0, '权限目录不能是空的');
    for (const item of PERMISSION_CATALOG) {
      assert.ok(typeof item.id === 'string' && item.id !== '', '权限项缺 id');
      assert.ok(typeof item.label === 'string' && item.label !== '', `权限 ${item.id} 缺名称`);
      assert.ok(
        typeof item.description === 'string' && item.description !== '',
        `权限 ${item.id} 缺说明`,
      );
      assert.ok(
        Array.isArray(item.tools) && item.tools.length > 0,
        `权限 ${item.id} 没有对应工具`,
      );
      for (const name of item.tools) {
        assert.ok(PI_TOOLS[name], `权限 ${item.id} 指向了不存在的工具 ${name}`);
      }
    }
  });

  const toolNames = (perms) => toolsForPermissions(perms).map((tool) => tool.name).sort();

  await test('按权限裁剪出来的工具集正好是目录里写的那些', async () => {
    assert.deepEqual(toolNames([]), []);
    assert.deepEqual(toolNames(['read']), ['checkIdea', 'checkJson', 'find', 'grep', 'ls', 'read']);
    // 写权限带三个工具：写文本文件、只改几处，以及从冻结基线复制一份候选树。
    assert.deepEqual(toolNames(['write']), ['copyBaselineTree', 'edit', 'write']);
    assert.deepEqual(toolNames(['delete']), ['deleteFile']);
    assert.deepEqual(toolNames(['execute']), ['gitLog']);
    assert.deepEqual(toolNames(['network']), ['listIssues', 'readIssue']);
    assert.deepEqual(toolNames(['read', 'execute', 'network']), [
      'checkIdea',
      'checkJson',
      'find',
      'gitLog',
      'grep',
      'listIssues',
      'ls',
      'read',
      'readIssue',
    ]);
  });

  await test('copyBaselineTree：从冻结基线复制候选树，只改 ARM_NAME', async () => {
    // 路径按工作目录（/home/... 那棵树）算，不是按进程 cwd —— 两者是同一棵树，
    // 但字符串不同，相对路径必须相对工作目录算才不会"越界"。
    const scratch = path.join(config.SANDBOX_ROOT, 'agent-vis', '.out', `copy-tree-${process.pid}`);
    fs.rmSync(scratch, { recursive: true, force: true });
    const baseline = path.join(scratch, 'stage2_RQ-VAE', 'TIGER_RQ-VAE');
    fs.mkdirSync(baseline, { recursive: true });
    fs.writeFileSync(
      path.join(baseline, 'train_rqvae.py'),
      'ARM_NAME = "TIGER_RQ-VAE"\nRESULT_DIR = REPO_ROOT / "results" / "stage2_RQ-VAE" / ARM_NAME\n',
    );
    fs.mkdirSync(path.join(baseline, '__pycache__'), { recursive: true });
    fs.writeFileSync(path.join(baseline, '__pycache__', 'junk.pyc'), 'x');
    const from = path.relative(config.SANDBOX_ROOT, baseline);

    const out = await TOOL_IMPLS.copyBaselineTree({ arm: 'TIGER_probe_rq', from });
    assert.ok(out.path.endsWith('TIGER_probe_rq'), `路径不对: ${out.path}`);
    assert.equal(out.rewritten, 1, `ARM_NAME 应该改一处: ${out.rewritten}`);
    const copied = fs.readFileSync(path.join(baseline, '..', 'TIGER_probe_rq', 'train_rqvae.py'), 'utf8');
    assert.ok(copied.includes('ARM_NAME = "TIGER_probe_rq"'), `ARM_NAME 没改: ${copied}`);
    assert.ok(copied.includes('RESULT_DIR'), '别的内容不该被改');
    assert.equal(
      fs.existsSync(path.join(baseline, '..', 'TIGER_probe_rq', '__pycache__')),
      false,
      '__pycache__ 不该被复制',
    );

    // 目标已存在不覆盖；名字不合法直接报错。
    await assert.rejects(
      () => TOOL_IMPLS.copyBaselineTree({ arm: 'TIGER_probe_rq', from }),
      /已经存在/,
      '已有的候选树不该被覆盖',
    );
    await assert.rejects(
      () => TOOL_IMPLS.copyBaselineTree({ arm: '../escape', from }),
      /arm 名字不对/,
      '名字里带路径分隔符应该被拒',
    );
    fs.rmSync(scratch, { recursive: true, force: true });
  });

  await test('权限目录里就是 read / write 这类基础权限', async () => {
    const ids = PERMISSION_CATALOG.map((item) => item.id);
    for (const id of ['read', 'write']) {
      assert.ok(ids.includes(id), `权限目录里没有 ${id}：${ids.join('、')}`);
    }
    // 每一项都得说清它管的是哪一类动作。
    for (const item of PERMISSION_CATALOG) {
      assert.ok(item.label.length > 0 && item.description.length > 0, `${item.id} 说不清自己管什么`);
    }
  });

  await test('不认识的权限 id 直接报错，不静默忽略', async () => {
    assert.throws(() => normalizePermissions(['read', 'nope']), /nope/);
    assert.throws(() => normalizePermissions('read'), /数组/);
    assert.throws(
      () => resolvePermissions({ permissions: ['read'] }, ['bad.id']),
      /bad\.id/,
    );
  });

  await test('没给权限就用这个 agent 自己的默认权限，给了空数组就是一项都不要', async () => {
    const profile = { permissions: ['read'] };
    assert.deepEqual(resolvePermissions(profile, undefined), ['read']);
    assert.deepEqual(resolvePermissions(profile, []), []);
    assert.deepEqual(resolvePermissions(profile, ['write']), ['write']);
  });

  await test('权限说明写进系统提示词：授了什么、没授什么', async () => {
    const text = buildPermissionPrompt(['read', 'write']);
    assert.match(text, /read/);
    assert.match(text, /write/);
    assert.match(text, /write/);
    // 没授的权限一个字都不该露出来 —— 露出来模型就会去用没给它的工具。
    assert.doesNotMatch(text, /gitLog/);
    assert.doesNotMatch(text, /execute/);
  });

  await test('写权限的工具真的能把文件写进仓库', async () => {
    const rel = 'agent-vis/.out-permission-write-test.txt';
    const abs = path.join(config.REPO_ROOT, rel);
    const content = '权限测试写入\n';
    try {
      await PI_TOOLS.write.execute(
        'test-write',
        { path: rel, content },
        undefined,
        undefined,
        undefined,
      );
      assert.equal(fs.readFileSync(abs, 'utf8'), content, '文件内容不对');
    } finally {
      fs.rmSync(abs, { force: true });
    }
  });

  await test('写权限也不许越界、不许写 .git 目录', async () => {
    const call = (input) =>
      PI_TOOLS.write.execute('test-write', input, undefined, undefined, undefined);
    await assert.rejects(() => call({ path: '../outside.txt', content: 'x' }), /越界/);
    await assert.rejects(
      () => call({ path: '.git/config', content: 'x' }),
      /\.git/,
    );
    await assert.rejects(
      () => call({ path: 'agent-vis/.out-too-big.txt', content: 'x'.repeat(2 * 1024 * 1024) }),
      /太大/,
    );
  });

  await test('删除权限能删掉仓库里的文件', async () => {
    const rel = 'agent-vis/.out-permission-delete-test.txt';
    const abs = path.join(config.REPO_ROOT, rel);
    fs.writeFileSync(abs, '待删\n', 'utf8');
    try {
      const result = await PI_TOOLS.deleteFile.execute(
        'test-delete',
        { path: rel },
        undefined,
        undefined,
        undefined,
      );
      assert.equal(result.details.path, rel, `删的是 ${result.details.path}`);
      assert.equal(fs.existsSync(abs), false, '文件还在');
    } finally {
      fs.rmSync(abs, { force: true });
    }
  });

  await test('删除权限不许删 .git、不许删仓库根、不许删不存在的东西', async () => {
    const call = (input) =>
      PI_TOOLS.deleteFile.execute('test-delete', input, undefined, undefined, undefined);
    await assert.rejects(() => call({ path: '.git/config' }), /\.git/);
    await assert.rejects(() => call({ path: '.' }), /仓库根/);
    await assert.rejects(
      () => call({ path: 'agent-vis/.out-not-here.txt' }),
      /不存在/,
    );
  });

  await test('health 里带着权限目录与每个 agent 的默认权限', async () => {
    const info = await (await fetch(`${base}/api/health`)).json();
    assert.ok(Array.isArray(info.permissionCatalog), 'health 里没有权限目录');
    assert.ok(info.permissionCatalog.length > 0, '权限目录是空的');
    const ids = new Set(info.permissionCatalog.map((item) => item.id));
    for (const agent of info.agents) {
      assert.ok(Array.isArray(agent.permissions), `${agent.id} 没有权限字段`);
      for (const id of agent.permissions) {
        assert.ok(ids.has(id), `${agent.id} 的权限 ${id} 不在目录里`);
      }
    }
  });

  await test('请求里带不认识的权限 id 时直接拒绝', async () => {
    const { res, body } = await readSse(
      `${base}/api/run?prompt=hi&v=${config.FRONTEND_VERSION}` +
        `&profile=agent-a&permissions=read,bogus`,
    );
    assert.equal(res.status, 400, `状态码 ${res.status}`);
    assert.ok(body.includes('bogus'), `错误信息里没有指出是哪个权限：${body.slice(0, 200)}`);
  });

  await test('请求里带的权限决定这一轮拿到的工具', async () => {
    const { res, frames } = await readSse(`${base}/api/run`, {
      method: 'POST',
      payload: {
        prompt: 'hi',
        v: config.FRONTEND_VERSION,
        profile: 'agent-a',
        permissions: ['read'],
      },
    });
    assert.equal(res.status, 200, `状态码 ${res.status}`);
    const text = parseChunks(frames)
      .filter((c) => c.type === 'text-delta')
      .map((c) => c.delta)
      .join('');
    assert.ok(text.includes('read'), `回答里没写明授了什么权限：${text.slice(0, 200)}`);
    assert.ok(!text.includes('execute'), `回答里写了没授的权限：${text.slice(0, 200)}`);
  });

  await test('新建 agent 时可以带上它的权限', async () => {
    const res = await fetch(`${base}/api/agents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: 'AGENT X',
        instructions: 'x',
        permissions: ['read', 'write'],
      }),
    });
    assert.equal(res.status, 201, `状态码 ${res.status}`);
    const profile = await res.json();
    assert.deepEqual(profile.permissions, ['read', 'write']);
  });

  await test('新建 agent 时给不认识的权限 id 直接拒绝', async () => {
    const res = await fetch(`${base}/api/agents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: 'AGENT Y', permissions: ['read', 'weird.id'] }),
    });
    assert.equal(res.status, 400, `状态码 ${res.status}`);
    const body = await res.text();
    assert.ok(body.includes('weird.id'), `错误信息里没有指出是哪个权限：${body.slice(0, 200)}`);
  });

  // ---------- 删除 agent ----------
  process.stdout.write('\n  删除 agent · 身份要从服务端的表里拿掉\n');

  await test('DELETE /api/agents/<id> 删掉运行时新建的 agent', async () => {
    const createdRes = await fetch(`${base}/api/agents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const profile = await createdRes.json();

    const res = await fetch(`${base}/api/agents/${profile.id}`, { method: 'DELETE' });
    assert.equal(res.status, 200, `状态码 ${res.status}`);

    const info = await (await fetch(`${base}/api/health`)).json();
    assert.ok(
      !info.agents.some((agent) => agent.id === profile.id),
      'health 里还有这个 agent',
    );

    const run = await fetch(
      `${base}/api/run?prompt=hi&v=${config.FRONTEND_VERSION}&profile=${profile.id}`,
    );
    assert.equal(run.status, 400, '删掉的身份不该还能跑');
  });

  await test('配置里写死的五个 agent 不能从服务端删除', async () => {
    for (const id of ['agent-a', 'agent-b', 'agent-math', 'agent-stage2', 'agent-fix']) {
      const res = await fetch(`${base}/api/agents/${id}`, { method: 'DELETE' });
      assert.equal(res.status, 400, `${id} 应该被拒掉`);
    }
  });

  await test('DELETE 不认识的 agent 直接 404', async () => {
    const res = await fetch(`${base}/api/agents/custom-999`, { method: 'DELETE' });
    assert.equal(res.status, 404, `状态码 ${res.status}`);
  });

  await test('新建 agent 只接受 POST，GET 不新建', async () => {
    const before = (await (await fetch(`${base}/api/health`)).json()).agents.length;
    const res = await fetch(`${base}/api/agents`);
    assert.equal(res.status, 405, `状态码 ${res.status}`);
    const after = (await (await fetch(`${base}/api/health`)).json()).agents.length;
    assert.equal(after, before, 'GET 不该新建出 agent');
  });

  await test('health 说得出每个 agent 能不能被删', async () => {
    const createdRes = await fetch(`${base}/api/agents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    const profile = await createdRes.json();
    const info = await (await fetch(`${base}/api/health`)).json();
    const fixed = info.agents.find((agent) => agent.id === 'agent-a');
    const custom = info.agents.find((agent) => agent.id === profile.id);
    assert.equal(fixed.builtin, true, '固定的两个 agent 应该是 builtin');
    assert.equal(custom.builtin, false, '新建的 agent 不该是 builtin');
  });

  await test('交接消息里的来源名字跟着发送方走', async () => {
    // 交接不再只有 A → B 这一条路，来源名字必须跟着实际发送方走。
    const { frames } = await readSse(`${base}/api/run`, {
      method: 'POST',
      payload: {
        prompt: 'hi',
        v: config.FRONTEND_VERSION,
        profile: 'agent-b',
        handoff: '上游给的一段结论。',
        handoffFrom: 'AGENT C',
      },
    });
    const text = parseChunks(frames)
      .filter((c) => c.type === 'text-delta')
      .map((c) => c.delta)
      .join('');
    assert.ok(text.includes('AGENT C'), `回答里没认出上游是 AGENT C: ${text.slice(0, 160)}`);
  });

  // ---------- 公网反代：访问门槛（路径前缀）----------
  // 这个服务能调真实模型、能读仓库里的文件。挂到公网上时必须有一道门槛，
  // 否则任何拿到链接的人都能用掉模型额度、读走仓库里的凭证。
  process.stdout.write('\n  公网反代 · 路径前缀是访问门槛\n');

  const gated = createServer({ mode: 'mock', publicPath: 'test-token', dataDir: SCRATCH_DATA });
  const gatedPort = await listen(gated);
  const gatedBase = `http://127.0.0.1:${gatedPort}`;

  await test('没带前缀的路径一律被挡下（不说是谁挡的）', async () => {
    for (const route of ['/', '/api/health', '/app.js', '/style.css', '/api/agents']) {
      const res = await fetch(`${gatedBase}${route}`);
      assert.equal(res.status, 404, `${route} 状态码 ${res.status}`);
      const body = await res.text();
      assert.ok(!body.includes('test-token'), `${route} 的响应里漏了前缀：${body.slice(0, 120)}`);
    }
  });

  await test('带前缀的路径正常工作', async () => {
    const res = await fetch(`${gatedBase}/test-token/api/health`);
    assert.equal(res.status, 200, `状态码 ${res.status}`);
    const info = await res.json();
    assert.equal(info.ok, true);
  });

  await test('只写前缀（没有尾斜杠）会跳到带尾斜杠的地址', async () => {
    const res = await fetch(`${gatedBase}/test-token`, { redirect: 'manual' });
    assert.equal(res.status, 301, `状态码 ${res.status}`);
    assert.equal(res.headers.get('location'), '/test-token/');
  });

  await test('静态资源也在前缀下', async () => {
    const res = await fetch(`${gatedBase}/test-token/app.js`);
    assert.equal(res.status, 200, `状态码 ${res.status}`);
    assert.match(res.headers.get('content-type'), /text\/javascript/);
  });

  await test('HTML 里注入了 API_BASE，前端知道请求该往哪发', async () => {
    const res = await fetch(`${gatedBase}/test-token/`);
    assert.equal(res.status, 200, `状态码 ${res.status}`);
    const html = await res.text();
    assert.match(html, /__AGENT_VIS_API_BASE__\s*=\s*['"]\/test-token['"]/, '没注入 API_BASE');
    assert.ok(!html.includes('__API_BASE_TOKEN__'), '占位符没被替换');
  });

  await close(gated);

  await close(server);

  process.stdout.write(`\n${passed} 通过, ${failures.length} 失败\n`);
  if (failures.length > 0) {
    for (const { name, err } of failures) {
      process.stdout.write(`\n--- ${name} ---\n${err && err.stack}\n`);
    }
    process.exitCode = 1;
  }
}

main().catch((err) => {
  process.stderr.write(`${err && err.stack}\n`);
  process.exit(1);
});
