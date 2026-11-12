import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { createServer, config } from '../server.js';

/**
 * 浏览器端回归测试：真实 Chromium 里验证前端。
 *
 * 覆盖四件事：
 *   1. 页面上的 agent 面板各自独立 —— 各跑各的，互不串内容，还能同时跑；
 *   2. AGENT A 跑完后自动把分析结果交给 AGENT B（交接）；
 *   3. AGENT B 单独跑时不带交接；
 *   4. 浏览器缓存了旧版 JS 时，首页守卫会换一次性地址重载并自愈。
 *
 * 需要 chromium-browser 与（live 模式下）~/.config/opencode/commandcode.key。
 * 离线协议检查在 test/smoke.js，真实模型/工具在 test/tools.js 里单独查。
 */

const HERE = path.dirname(fileURLToPath(import.meta.url));

/** 测试用的数据目录：配置与对话记录落在仓库内的 .out/ 下，不碰用户自己的 .data/。 */
const SCRATCH_DATA = path.join(HERE, '..', '.out', 'browser-data');
const OUT_DIR = path.join(HERE, '..', '.out');
const CDP_PORT = 9334;
const WIDTH = 1600;
const HEIGHT = 1200;

const PANEL_A = 'panelA';
const PANEL_B = 'panelB';

/** 给 AGENT A（Issue 执行者）的指令。 */
const PROMPT_A =
  '用 listFiles 列出 agent-vis 目录下的文件，然后用一句话说明这个目录是做什么的。';
/** 给 AGENT B（Issue 控制器）的指令 —— 交接时这条指令会跟着 A 的分析结果一起发出去。 */
const PROMPT_B = '请基于上游给出的分析结论，给出三条最值得先动手的改进建议。';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function request(url, method = 'GET', body = null) {
  return new Promise((resolve, reject) => {
    const payload = body === null ? null : Buffer.from(body, 'utf8');
    const headers = payload ? { 'Content-Type': 'application/json' } : {};
    const req = http.request(url, { method, headers }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString();
        try {
          resolve(JSON.parse(text));
        } catch (err) {
          reject(new Error(`${method} ${url} -> ${res.statusCode}: ${text.slice(0, 200)}`));
        }
      });
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

async function waitForDevTools() {
  for (let i = 0; i < 60; i++) {
    try {
      await request(`http://127.0.0.1:${CDP_PORT}/json/version`);
      return;
    } catch {
      await sleep(250);
    }
  }
  throw new Error('devtools endpoint never came up');
}

class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.handlers = [];
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result);
        return;
      }
      for (const handler of this.handlers) handler(msg);
    });
  }

  on(fn) {
    this.handlers.push(fn);
  }

  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression) {
    const res = await this.send('Runtime.evaluate', { expression, returnByValue: true });
    if (res.exceptionDetails) {
      throw new Error(`页面里执行出错: ${JSON.stringify(res.exceptionDetails).slice(0, 300)}`);
    }
    return res.result && res.result.value;
  }
}

async function openPage(cdp, url) {
  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');
  await cdp.send('Emulation.setDeviceMetricsOverride', {
    width: WIDTH,
    height: HEIGHT,
    deviceScaleFactor: 1,
    mobile: false,
  });
  await cdp.send('Page.navigate', { url });
}

async function waitFor(cdp, expression, tries = 120, gap = 250) {
  for (let i = 0; i < tries; i++) {
    const value = await cdp.evaluate(expression);
    if (value) return value;
    await sleep(gap);
  }
  return null;
}

let passed = 0;
const failures = [];

function check(name, ok, detail) {
  if (ok) {
    passed += 1;
    process.stdout.write(`  ✓ ${name}\n`);
  } else {
    failures.push({ name, detail });
    process.stdout.write(`  ✗ ${name}${detail ? ' — ' + detail : ''}\n`);
  }
}

/**
 * 在指定面板里求值一个表达式，返回它的值。
 *
 * 注意这里必须写 `return`：函数体里最后一个表达式语句不会自动成为返回值
 * （完成值语义只在 eval / 脚本顶层成立）。
 */
function inPanel(panelId, expression) {
  return (
    '(function () {' +
    `var panel = document.getElementById('${panelId}');` +
    `return ${expression};` +
    '})()'
  );
}

/**
 * 在指定面板里执行若干语句，不需要返回值。
 *
 * 触发事件这类动作必须走这里：写成 `return a; b;` 的话 b 是死代码，
 * 事件根本发不出去（之前右面板一直不跑就是这个原因）。
 */
function runInPanel(panelId, statements) {
  return (
    '(function () {' +
    `var panel = document.getElementById('${panelId}');` +
    statements +
    '})()'
  );
}

function panelCards(panelId) {
  return inPanel(
    panelId,
    "Array.from(panel.querySelectorAll('.part')).map(function (p) {" +
      "return {badge: p.querySelector('.badge') && p.querySelector('.badge').textContent," +
      "state: p.querySelector('.state') && p.querySelector('.state').textContent};})",
  );
}

const PANEL_IDS_EXPR =
  "Array.from(document.querySelectorAll('.panel')).map(function (p) { return p.id; })";

/**
 * 点「新建 Agent」，返回新出现的那个面板 id。
 *
 * 不能按「最后一个面板」取：页面上除了 A/B 还有内置的 AGENT MATH（从模板挂载）
 * 与之前场景留下的自定义 agent，新建的那个不一定排在末尾。按前后 id 集合求差。
 */
async function createAgentAndWait(cdp) {
  const before = await cdp.evaluate(PANEL_IDS_EXPR);
  await cdp.evaluate(
    "(function () {" +
      "var b = document.querySelector('[data-role=newAgentBtn]');" +
      'if (b) b.click();' +
      '})()',
  );
  for (let i = 0; i < 120; i++) {
    const now = await cdp.evaluate(PANEL_IDS_EXPR);
    if (now.length > before.length) {
      const added = now.filter((id) => !before.includes(id));
      if (added.length > 0) return added[added.length - 1];
    }
    await sleep(250);
  }
  return '';
}

function panelDone(panelId) {
  return inPanel(panelId, "panel.querySelector('.dot.done') ? 'done' : ''");
}

/**
 * 在一个面板里跑一轮，并等它真的跑完。
 *
 * 只等 .dot.done 是不够的：上一轮已经完成时它立刻为真，这一轮还没开始就返回了。
 * 所以先等"新块出现"，再等完成。
 */
async function runPromptOnce(cdp, panelId, prompt, tries = 600) {
  const before = await cdp.evaluate(
    inPanel(panelId, "panel.querySelectorAll('.run-block').length"),
  );
  await cdp.evaluate(submitPrompt(panelId, prompt));
  await waitFor(
    cdp,
    inPanel(panelId, `panel.querySelectorAll('.run-block').length > ${before} ? 'new' : ''`),
    200,
    250,
  );
  await waitFor(cdp, panelDone(panelId), tries, 250);
}

function panelName(panelId) {
  return inPanel(panelId, "panel.querySelector('[data-role=name]').textContent");
}

function panelError(panelId) {
  return inPanel(panelId, "(panel.querySelector('.err-banner') || {}).textContent || ''");
}

/**
 * 面板里最后一块（最近一轮）。
 *
 * 历史存在服务端，刷新页面就回来了 —— 时间轴上会堆着好几轮。
 * "这一轮有没有交接卡片"这类问题只看最后一块。
 */
function lastRunBlock(panelId) {
  return inPanel(
    panelId,
    "(function () {" +
      "var blocks = panel.querySelectorAll('.run-block');" +
      'return blocks[blocks.length - 1] || null;' +
      '})()',
  );
}

/** 交接卡片的数量（只看最后一块）。 */
function panelHandoffCards(panelId) {
  return inPanel(
    panelId,
    "(function () {" +
      "var blocks = panel.querySelectorAll('.run-block');" +
      'var last = blocks[blocks.length - 1];' +
      'if (!last) return 0;' +
      "return last.querySelectorAll('.handoff-card').length;" +
      '})()',
  );
}

/** 交接卡片里的文字（只看最后一块）。 */
function panelHandoffText(panelId) {
  return inPanel(
    panelId,
    "(function () {" +
      "var blocks = panel.querySelectorAll('.run-block');" +
      'var last = blocks[blocks.length - 1];' +
      'if (!last) return "";' +
      "var c = last.querySelector('.handoff-card');" +
      'return c ? c.textContent : "";' +
      '})()',
  );
}

/**
 * 这一栏最后一段文字回答里的最长一行 —— 当作"分析结果"的指纹。
 *
 * 取最长行而不是第一行：模型常以 markdown 标题开头（"## 结论"），
 * 拿标题当指纹验证不出什么。
 * 只看 .badge.text 的卡片：工具输出的卡片也是 .part-body，不能混进来。
 */
function panelFinalAnswer(panelId) {
  return inPanel(
    panelId,
    "(function () {" +
      "var blocks = panel.querySelectorAll('.run-block');" +
      "var last = blocks[blocks.length - 1];" +
      "if (!last) return '';" +
      "var cards = Array.from(last.querySelectorAll('.part'));" +
      "var textCards = cards.filter(function (p) { return p.querySelector('.badge.text'); });" +
      "if (!textCards.length) return '';" +
      "var body = textCards[textCards.length - 1].querySelector('.part-body');" +
      "if (!body) return '';" +
      "var lines = body.textContent.split('\\n').map(function (s) { return s.trim(); }).filter(Boolean);" +
      "if (!lines.length) return '';" +
      "var longest = lines[0];" +
      "lines.forEach(function (l) { if (l.length > longest.length) longest = l; });" +
      "return longest;" +
      '})()',
  );
}

/**
 * 最后那一轮的步骤卡状态。
 *
 * 时间轴上会堆着好几轮，卡片按 data-round 认轮次：只看最后一轮的。
 */
function lastRoundStepStates(panelId) {
  return inPanel(
    panelId,
    "(function () {" +
      "var cards = Array.from(panel.querySelectorAll('.round-step-card'));" +
      'if (!cards.length) return [];' +
      "var lastRound = cards[cards.length - 1].getAttribute('data-round');" +
      'return cards.filter(function (c) { return c.getAttribute("data-round") === lastRound; })' +
      ".map(function (c) { return c.querySelector('.round-step-state').textContent; });" +
      '})()',
  );
}

/**
 * 面板里最后一块的文字回答全文。
 *
 * 与记录里的原文逐字比：刷新页面之后画回来的东西必须跟当时跑出来的一样。
 */
function panelAnswerText(panelId) {
  return inPanel(
    panelId,
    "(function () {" +
      "var blocks = panel.querySelectorAll('.run-block');" +
      'var last = blocks[blocks.length - 1];' +
      "if (!last) return '';" +
      "var cards = Array.from(last.querySelectorAll('.part'));" +
      "var textCards = cards.filter(function (p) { return p.querySelector('.badge.text'); });" +
      'if (!textCards.length) return "";' +
      'return textCards.map(function (p) {' +
      "var b = p.querySelector('.part-body');" +
      'return b ? b.textContent : "";' +
      '}).join("\\n");' +
      '})()',
  );
}

/** 一条记录里的文字原文（把 text-delta 分片拼起来）。 */
function recordAnswerText(record) {
  const chunks = Array.isArray(record && record.chunks) ? record.chunks : [];
  return chunks
    .filter((chunk) => chunk && chunk.type === 'text-delta' && typeof chunk.delta === 'string')
    .map((chunk) => chunk.delta)
    .join('');
}

/** 比两段文字：忽略空白差异，但正文必须一样。 */
function sameText(a, b) {
  const strip = (text) => String(text).replace(/\s+/g, '');
  return strip(a) === strip(b);
}

/** 填输入框并提交。 */
function submitPrompt(panelId, prompt) {
  return runInPanel(
    panelId,
    "panel.querySelector('[data-role=prompt]').value = " +
      JSON.stringify(prompt) +
      "; panel.querySelector('[data-role=form]').dispatchEvent(new Event('submit', {cancelable:true}));",
  );
}

/** 点某一栏的「清空」按钮：清掉这一栏配的轮次步骤。 */
/**
 * 从一次 /api/run 请求里读出它带的权限。
 *
 * GET 把权限放在查询串里（逗号分隔），POST 放在 JSON body 的数组里 ——
 * 两种都要读得出来。
 */
function requestedPermissions(req) {
  if (!req) return null;
  if (req.method === 'POST') {
    try {
      const body = JSON.parse(req.body || '{}');
      return Array.isArray(body.permissions) ? body.permissions : [];
    } catch {
      return [];
    }
  }
  const at = req.url.indexOf('permissions=');
  if (at < 0) return [];
  const raw = decodeURIComponent(req.url.slice(at + 'permissions='.length).split('&')[0]);
  if (raw === '') return [];
  return raw.split(',').filter((id) => id !== '');
}

function clearSteps(panelId) {
  return runInPanel(panelId, "panel.querySelector('[data-role=clearSteps]').click();");
}

/**
 * 打开页面，并清掉两个面板上的默认步骤。
 *
 * AGENT A / AGENT B 现在默认带着提示词拆出来的固定步骤。别的场景测的是
 * "一次跑完"这条老路 —— 带着 4 步跑一遍会多发 4 次请求（live 模式下还是
 * 4 次真实请求），又慢又不稳，还会把交接请求的形状改掉。
 * 步骤编排本身由场景 L 测（那里专门看默认步骤）。
 */
async function openPageNoSteps(cdp, base) {
  await openPage(cdp, base);
  await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
  await cdp.evaluate(clearSteps(PANEL_A));
  await cdp.evaluate(clearSteps(PANEL_B));
  // 清空步骤是攒一会儿才发出去的：不等它落到服务端就走，
  // 下个场景打开页面看到的还是那 4 步。
  await waitForServer(async () => {
    const health = await request(`${base}/api/health`);
    return ['agent-a', 'agent-b'].every((id) => {
      const agent = (health.agents || []).find((a) => a.id === id);
      return agent && Array.isArray(agent.steps) && agent.steps.length === 0;
    });
  });
}

/**
 * 等服务端的某件事成立。
 *
 * 页面上的改动是攒一会儿才发出去的（打字的时候不该每敲一下发一次请求），
 * 所以断言"服务端存下来了"之前得等一等。
 */
async function waitForServer(fn, tries = 120, gap = 200) {
  for (let i = 0; i < tries; i++) {
    const value = await fn();
    if (value) return value;
    await sleep(gap);
  }
  return null;
}

/**
 * 等某个 agent 的对话记录攒到 n 条。
 *
 * 记录是请求跑完之后才写进去的：不等一下就查，会查到"上一轮"的记录，
 * 把结论算到错的轮次上。
 */
async function waitForChatCount(base, agentId, n, tries = 120) {
  let records = [];
  for (let i = 0; i < tries; i++) {
    const res = await request(`${base}/api/agents/${encodeURIComponent(agentId)}/chats`);
    records = res.records || [];
    if (records.length >= n) return records;
    await sleep(250);
  }
  return records;
}

/**
 * 把服务端的配置与对话记录恢复到出厂状态。
 *
 * 配置与聊天记录现在存在服务端（agent-vis/.data/）：上一个场景勾掉的
 * 交接目标、清空的步骤、跑过的对话，下一个场景打开页面都会看见。
 *
 * 每个场景组开始前先恢复出厂，测的才是这个场景本身 ——
 * 否则测到的可能是上一个场景留下来的状态。
 */
async function resetServerState(base) {
  const health = await request(`${base}/api/health`);
  for (const agent of health.agents || []) {
    const id = encodeURIComponent(agent.id);
    await request(`${base}/api/agents/${id}/chats`, 'DELETE');
    // 运行时新建的 agent 身份也删掉：页面上不该凭空多出几个。
    if (!agent.builtin) {
      await request(`${base}/api/agents/${id}`, 'DELETE');
    }
  }
  // 四个字段都传 null：抹掉用户改过的，回到代码里写死的默认。
  for (const id of ['agent-a', 'agent-b']) {
    await request(
      `${base}/api/agents/${id}/config`,
      'PUT',
      JSON.stringify({
        instructions: null,
        permissions: null,
        steps: null,
        handoffTargets: null,
      }),
    );
  }
}

/**
 * 打开一个"干净"的页面：先把服务端恢复出厂，再开页面。
 *
 * noSteps 为真时顺手清掉两个面板上的默认步骤 ——
 * 别的场景测的是"一次跑完"这条老路。
 */
async function openFresh(cdp, base, options = {}) {
  await resetServerState(base);
  if (options.noSteps) {
    await openPageNoSteps(cdp, base);
  } else {
    await openPage(cdp, base);
    await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
  }
}

async function main() {
  process.stdout.write('agent-vis 浏览器端回归测试\n');

  // publicPath: '' —— 测试要的是不带访问门槛的裸服务；公网入口的前缀
  // 写在仓库里的 .local-public-path，本地跑测试不该被它挡住。
  // 数据目录先清干净：上一次跑剩的配置与对话记录不能让这一次的结论失真。
  fs.rmSync(SCRATCH_DATA, { recursive: true, force: true });
  // 上一次跑被中途杀掉时，Chromium 会留在调试端口上；新的一轮会连到那具旧尸体，
  // 页面状态全是上一轮的。开跑前先把它清掉。
  try {
    execFileSync('pkill', ['-9', '-f', `remote-debugging-port=${CDP_PORT}`], { stdio: 'ignore' });
  } catch {
    // 没有匹配的进程时会返回非零，忽略即可。
  }
  const server = createServer({ mode: config.MODE, publicPath: '', dataDir: SCRATCH_DATA });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  process.stdout.write(`  被测服务: ${base} (${config.MODE})\n`);

  const chrome = spawn(
    'chromium-browser',
    [
      '--headless',
      '--disable-gpu',
      '--no-sandbox',
      '--disable-dev-shm-usage',
      '--hide-scrollbars',
      `--remote-debugging-port=${CDP_PORT}`,
      `--window-size=${WIDTH},${HEIGHT}`,
      'about:blank',
    ],
    // detached: 自成一个进程组。结束时按组杀，否则 chromium 拉起的
    // gpu/utility/renderer 子进程会变成孤儿，继续占着 9334 端口。
    { stdio: 'ignore', detached: true },
  );

  let cdp = null;
  try {
    await waitForDevTools();
    const target = await request(
      `http://127.0.0.1:${CDP_PORT}/json/new?${encodeURIComponent('about:blank')}`,
      'PUT',
    );
    const ws = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', resolve);
      ws.addEventListener('error', reject);
    });
    cdp = new Cdp(ws);

    // 页面里抛出来的异常：页面上没反应的时候，多半就是它。
    cdp.on((msg) => {
      if (msg.method === 'Runtime.exceptionThrown') {
        const d = msg.params.exceptionDetails || {};
        const desc = (d.exception && d.exception.description) || d.text || '';
        process.stdout.write(`  [页面异常] ${String(desc).slice(0, 300)}\n`);
      }
    });

    // 抓所有 POST /api/run 的请求体 —— 交接就发生在这里。
    const postBodies = [];
    // 所有 /api/run 请求（GET 与 POST 都要）：权限断言要看得见这一轮带了哪些权限。
    const runRequests = [];
    await cdp.send('Network.enable');
    cdp.on((msg) => {
      if (msg.method !== 'Network.requestWillBeSent') return;
      const req = msg.params.request;
      if (!req.url.includes('/api/run')) return;
      runRequests.push({ method: req.method, url: req.url, body: req.postData || '' });
      if (req.method === 'POST') {
        postBodies.push(req.postData || '');
      }
    });

    /**
     * 请求拦截器。模式为 null 时一律放行。
     *
     * 必须按 URL 分流：CDP 的 Fetch 拦截是全局的，一个 handler 收到所有
     * 被暂停的请求，不管 URL 就把 app.js 当成 /api/run 处理，浏览器会报
     * "Invalid InterceptionId"。关掉拦截后 in-flight 请求的 id 也会失效，
     * 所以错误一并吞掉。
     */
    let interceptMode = null;
    let appJsHits = 0;
    /** fail-second-run 模式：数到第几次 /api/run，好让指定的那一次失败。 */
    let runErrorHits = 0;
    /** hold-first-run 模式：挂起第一个 /api/run，好观察"这一步正在跑"的中间态。 */
    let holdRunId = null;
    cdp.on(async (msg) => {
      if (msg.method !== 'Fetch.requestPaused') return;
      const { requestId, request } = msg.params;
      const url = request.url;
      try {
        if (interceptMode === 'hold-first-run' && url.includes('/api/run')) {
          // 只挂起第一次：后面的步骤照常放行。
          if (holdRunId === null) {
            holdRunId = requestId;
            return;
          }
        }
        if (interceptMode === 'fail-second-run' && url.includes('/api/run')) {
          runErrorHits += 1;
          if (runErrorHits === 2) {
            await cdp.send('Fetch.fulfillRequest', {
              requestId,
              responseCode: 500,
              responseHeaders: [
                { name: 'Content-Type', value: 'application/json; charset=utf-8' },
              ],
              body: Buffer.from(JSON.stringify({ error: '第二步挂了' })).toString('base64'),
            });
            return;
          }
        }
        if (interceptMode === 'run-error' && url.includes('/api/run')) {
          await cdp.send('Fetch.fulfillRequest', {
            requestId,
            responseCode: 400,
            responseHeaders: [
              { name: 'Content-Type', value: 'application/json; charset=utf-8' },
            ],
            body: Buffer.from(
              JSON.stringify({
                error: '未知的 agent 身份: assistant',
                hint: '按 Ctrl+Shift+R 硬刷新后重试。',
              }),
            ).toString('base64'),
          });
          return;
        }
        if (interceptMode === 'stale-appjs' && url.includes('app.js')) {
          appJsHits += 1;
          if (appJsHits === 1) {
            // 给一份"旧版"：只把版本号改旧，其余照旧 ——
            // 旧版 JS 照样能启动、会设置启动标记，守卫必须靠比对版本号才发现它。
            const source = fs
              .readFileSync(path.join(HERE, '..', 'public', 'app.js'), 'utf8')
              .replace(
                /const FRONTEND_VERSION = '[^']+'/,
                "const FRONTEND_VERSION = '2026-11-10.4'",
              )
              .split('__ASSET_V__')
              .join('deadbeefdeadbeef');
            await cdp.send('Fetch.fulfillRequest', {
              requestId,
              responseCode: 200,
              responseHeaders: [
                { name: 'Content-Type', value: 'text/javascript; charset=utf-8' },
                { name: 'Cache-Control', value: 'no-store' },
              ],
              body: Buffer.from(source).toString('base64'),
            });
            return;
          }
        }
        if (interceptMode === 'slow-appjs' && url.includes('app.js')) {
          appJsHits += 1;
          if (appJsHits === 1) {
            // 故意让 app.js 迟到 1.5 秒：模拟网络慢 / 机器忙。
            // 守卫这时看到的只有"启动标记还没出现"，不能把它当成"旧版本"。
            await sleep(1500);
          }
        }
        await cdp.send('Fetch.continueRequest', { requestId });
      } catch {
        // 请求可能已经被导航取消或拦截被关掉，忽略即可。
      }
    });

    // --quick：只验证"时间轴上不画工具卡"（mock，离线，不开模型调用）就退出。
    // 改样式/渲染这类局部改动用它，别每次跑十几分钟的全量。
    if (process.argv.includes('--quick')) {
      const quickServer = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
      await new Promise((resolve) => quickServer.listen(0, '127.0.0.1', resolve));
      const quickBase = `http://127.0.0.1:${quickServer.address().port}`;
      await openFresh(cdp, quickBase, { noSteps: true });
      await runPromptOnce(cdp, PANEL_A, '快检：时间轴上不该出现任何工具卡');
      const quick = await cdp.evaluate(
        inPanel(
          PANEL_A,
          "(function () {" +
            "var tools = panel.querySelectorAll('.badge.tool').length;" +
            "var parts = panel.querySelectorAll('.part').length;" +
            "var texts = panel.querySelectorAll('.badge.text').length;" +
            "return {tools: tools, parts: parts, texts: texts};" +
            '})()',
        ),
      );
      check('快检：时间轴上没有工具卡', quick.tools === 0, `${quick.tools} 张`);
      check('快检：回答文字还在', quick.texts > 0, `${quick.texts} 张文字卡`);
      check('快检：时间轴不是空的', quick.parts > 0, `${quick.parts} 张卡`);

      // AGENT A 的默认交接目标是 B：等那次自动交接跑完再清场，
      // 否则它的记录会在清场之后落盘，把下面的轮次计数带偏。
      await waitFor(cdp, panelDone(PANEL_B), 300, 200);

      // 条件跳过：第一步判定"没有 Open Issue"（mock 剧本里的 endRound），
      // 后面两步不该发请求，进度条上标成跳过。
      await resetServerState(quickBase);
      await request(
        `${quickBase}/api/agents/agent-a/config`,
        'PUT',
        JSON.stringify({
          handoffTargets: [],
          steps: [
            {
              title: '查 Open Issue',
              instruction: '【模拟空轮】用 listIssues 查有没有 Open Issue。',
            },
            { title: '不该跑的一步', instruction: '这一步不该被发出去。' },
            { title: '也不该跑的一步', instruction: '这一步同样不该被发出去。' },
          ],
        }),
      );
      await openPage(cdp, quickBase);
      await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');

      // 步骤是页面加载之后从服务端拉的，等它到位再读。
      await waitFor(
        cdp,
        inPanel(PANEL_A, "panel.querySelectorAll('[data-step-chip]').length === 3 ? 'ready' : ''"),
        200,
        200,
      );
      const chips = await cdp.evaluate(
        inPanel(
          PANEL_A,
          "Array.from(panel.querySelectorAll('[data-step-chip]')).map(function (c) { return c.textContent; })",
        ),
      );
      check('快检：条件跳过的轮次配了 3 步', chips.length === 3, JSON.stringify(chips));

      const runsBefore = await cdp.evaluate(
        inPanel(PANEL_A, "panel.querySelectorAll('.run-block').length"),
      );
      const requestsBefore = runRequests.length;
      await cdp.evaluate(submitPrompt(PANEL_A, '快检：条件跳过'));
      await waitFor(
        cdp,
        inPanel(PANEL_A, "panel.querySelectorAll('.run-block').length > 0 ? 'block' : ''"),
        200,
        200,
      );
      await waitFor(cdp, panelDone(PANEL_A), 400, 200);
      await sleep(800);

      const states = await cdp.evaluate(
        inPanel(
          PANEL_A,
          "Array.from(panel.querySelectorAll('[data-step-chip]')).map(function (c) { return c.dataset.state; })",
        ),
      );
      const runsAfter = await cdp.evaluate(
        inPanel(PANEL_A, "panel.querySelectorAll('.run-block').length"),
      );
      check(
        '快检：第一步判定结束后，后面两步标成跳过',
        states[0] === 'filled' && states[1] === 'skipped' && states[2] === 'skipped',
        JSON.stringify(states),
      );
      check(
        '快检：跳过的步骤没有发请求',
        runRequests.length - requestsBefore === 1,
        `${runRequests.length - requestsBefore} 次请求`,
      );
      check(
        '快检：时间轴只多了一块',
        runsAfter - runsBefore === 1,
        `${runsAfter - runsBefore} 块`,
      );

      // IDEA 窗口：填进去、存到服务端、刷新后还在、随运行发给 agent。
      const IDEA_TEXT = '页面填的机制定义：结构信号必须进同一条通路。';
      await resetServerState(quickBase);
      await openPageNoSteps(cdp, quickBase);
      await cdp.evaluate(
        "(function(){var b=document.querySelector('[data-role=ideaBtn]'); if (b) b.click();})()",
      );
      const modalOpen = await cdp.evaluate(
        "(document.querySelector('[data-role=ideaModal]') && !document.querySelector('[data-role=ideaModal]').hidden) ? 'open' : ''",
      );
      check('快检：IDEA 窗口能打开', modalOpen === 'open', String(modalOpen));

      // 窗口第一次打开时用仓库 idea.txt 当草稿 —— 不该是一片空白。
      const ideaDraft = await waitFor(
        cdp,
        "(document.querySelector('[data-role=ideaInput]') && document.querySelector('[data-role=ideaInput]').value.trim().length > 0) ? document.querySelector('[data-role=ideaInput]').value : ''",
        100,
        100,
      );
      const ideaFileText = (await request(`${quickBase}/api/idea`)).fileText;
      check(
        '快检：IDEA 窗口先用 idea.txt 当草稿',
        Boolean(ideaDraft) && ideaDraft === ideaFileText,
        `${ideaDraft ? ideaDraft.length : 0} 字`,
      );

      await cdp.evaluate(
        `(function(){var t=document.querySelector('[data-role=ideaInput]');` +
          `t.value=${JSON.stringify(IDEA_TEXT)};` +
          `t.dispatchEvent(new Event('input',{bubbles:true}));` +
          `t.dispatchEvent(new Event('blur'));})()`,
      );
      await waitForServer(async () => (await request(`${quickBase}/api/idea`)).text === IDEA_TEXT);

      await openPageNoSteps(cdp, quickBase);
      await cdp.evaluate(
        "(function(){var b=document.querySelector('[data-role=ideaBtn]'); if (b) b.click();})()",
      );
      const ideaAfterReload = await waitFor(
        cdp,
        `(document.querySelector('[data-role=ideaInput]') && document.querySelector('[data-role=ideaInput]').value === ${JSON.stringify(IDEA_TEXT)}) ? 'ok' : ''`,
        100,
        100,
      );
      check('快检：IDEA 文本刷新后还在', ideaAfterReload === 'ok', String(ideaAfterReload));

      await cdp.evaluate(
        "(function(){var b=document.querySelector('[data-role=ideaClose]'); if (b) b.click();})()",
      );
      await cdp.evaluate(submitPrompt(PANEL_A, '快检：idea 随运行发送'));
      await waitFor(cdp, panelDone(PANEL_A), 400, 200);
      const chats = await request(`${quickBase}/api/agents/agent-a/chats`);
      const lastRecord = chats.records[chats.records.length - 1];
      check(
        '快检：运行时的 prompt 带上了 IDEA 文本',
        Boolean(lastRecord && lastRecord.prompt.includes(IDEA_TEXT)),
        lastRecord ? lastRecord.prompt.slice(0, 90) : '（没有记录）',
      );

      await cdp.evaluate(
        "(function(){var b=document.querySelector('[data-role=ideaBtn]'); if (b) b.click();})()",
      );
      await cdp.evaluate(
        "(function(){var b=document.querySelector('[data-role=ideaClear]'); if (b) b.click();})()",
      );
      await waitForServer(async () => (await request(`${quickBase}/api/idea`)).text === '');
      check('快检：IDEA 文本能清空', true, '');

      // 改名：配置页的输入框改完（change）存到服务端，页头立刻更新，刷新后还在。
      await resetServerState(quickBase);
      await openPage(cdp, quickBase);
      await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
      await waitFor(
        cdp,
        inPanel(PANEL_A, "panel.querySelector('[data-role=nameInput]') ? 'ok' : ''"),
        100,
        100,
      );
      await cdp.evaluate(
        runInPanel(
          PANEL_A,
          "var i = panel.querySelector('[data-role=nameInput]');" +
            "i.value = '快检执行者';" +
            "i.dispatchEvent(new Event('input', {bubbles: true}));" +
            "i.dispatchEvent(new Event('change', {bubbles: true}));",
        ),
      );
      await waitForServer(async () => {
        const health = await request(`${quickBase}/api/health`);
        return health.agents.find((agent) => agent.id === 'agent-a').name === '快检执行者';
      });
      const headerName = await cdp.evaluate(
        inPanel(PANEL_A, "panel.querySelector('[data-role=name]').textContent"),
      );
      check('快检：改名后页头立刻更新', headerName === '快检执行者', headerName);

      await openPage(cdp, quickBase);
      await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
      const afterReload = await waitFor(
        cdp,
        inPanel(
          PANEL_A,
          "panel.querySelector('[data-role=name]').textContent === '快检执行者' ? 'ok' : ''",
        ),
        100,
        100,
      );
      check('快检：刷新后名字还在', afterReload === 'ok', String(afterReload));

      await cdp.evaluate(
        runInPanel(
          PANEL_A,
          "var b = panel.querySelector('[data-role=resetName]'); if (b) b.click();",
        ),
      );
      await waitForServer(async () => {
        const health = await request(`${quickBase}/api/health`);
        return health.agents.find((agent) => agent.id === 'agent-a').name === 'AGENT A';
      });
      // 服务端先变、页面那边的 await 回来后才改标题：等页面追上，别抢跑。
      const restoredName = await waitFor(
        cdp,
        inPanel(
          PANEL_A,
          "panel.querySelector('[data-role=name]').textContent === 'AGENT A' ? 'AGENT A' : ''",
        ),
        100,
        100,
      );
      check('快检：恢复默认名字', restoredName === 'AGENT A', String(restoredName));

      // 交接延迟：A 配 2 秒延迟交给 B。B 不该立刻开跑，但到点必须开跑。
      await resetServerState(quickBase);
      await request(
        `${quickBase}/api/agents/agent-a/config`,
        'PUT',
        JSON.stringify({ handoffTargets: ['agent-b'], handoffDelaySeconds: 2 }),
      );
      await request(
        `${quickBase}/api/agents/agent-b/config`,
        'PUT',
        JSON.stringify({ handoffTargets: [] }),
      );
      await openPage(cdp, quickBase);
      await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
      await cdp.evaluate(submitPrompt(PANEL_A, '快检：延迟交接'));
      await waitFor(cdp, panelDone(PANEL_A), 400, 200);
      const doneAt = Date.now();

      await sleep(700);
      const earlyBlocks = await cdp.evaluate(
        inPanel(PANEL_B, "panel.querySelectorAll('.run-block').length"),
      );
      check('快检：延迟没到点时下游不会开跑', earlyBlocks === 0, `${earlyBlocks} 块`);

      const startedDelay = await waitFor(
        cdp,
        inPanel(PANEL_B, "panel.querySelectorAll('.run-block').length > 0 ? 'started' : ''"),
        200,
        100,
      );
      const waitedMs = Date.now() - doneAt;
      check(
        '快检：到点之后才发（不少于 1.5 秒）',
        startedDelay === 'started' && waitedMs >= 1500,
        `${waitedMs} ms`,
      );

      // 收尾：把目标清掉，别让后面的场景被这次交接带偏。
      await request(
        `${quickBase}/api/agents/agent-a/config`,
        'PUT',
        JSON.stringify({ handoffTargets: [] }),
      );

      // 交接环：A → B 与 B → A 都配好。现在完全允许来回 ——
      // 只有手动点「结束」才停，所以这里验证"会一直来回"和"停了就停"。
      await openFresh(cdp, quickBase, { noSteps: true });
      await request(
        `${quickBase}/api/agents/agent-a/config`,
        'PUT',
        JSON.stringify({ handoffTargets: ['agent-b'] }),
      );
      await request(
        `${quickBase}/api/agents/agent-b/config`,
        'PUT',
        JSON.stringify({ handoffTargets: ['agent-a'] }),
      );
      await openPageNoSteps(cdp, quickBase);
      await cdp.evaluate(submitPrompt(PANEL_A, '快检：交接环'));

      const runsExpression = (panelId) =>
        inPanel(panelId, "panel.querySelectorAll('.run-block').length");
      // A → B → A → B：两个面板都至少跑过两轮，说明来回没有被拦。
      const wentAround = await waitFor(
        cdp,
        `(${runsExpression(PANEL_A)}) >= 2 && (${runsExpression(PANEL_B)}) >= 2 ? 'looping' : ''`,
        600,
        250,
      );
      const aDuring = await cdp.evaluate(runsExpression(PANEL_A));
      const bDuring = await cdp.evaluate(runsExpression(PANEL_B));
      check(
        '快检：交接会一直来回（A→B→A→B）',
        wentAround === 'looping',
        `A=${aDuring} B=${bDuring}`,
      );

      // 手动停止：把还在跑的「结束」按钮点掉（轮流点几遍，避免正好错开）。
      for (let i = 0; i < 6; i += 1) {
        await cdp.evaluate(
          runInPanel(
            PANEL_A,
            "var b = panel.querySelector('[data-role=stopBtn]'); if (b && !b.disabled) b.click();",
          ),
        );
        await cdp.evaluate(
          runInPanel(
            PANEL_B,
            "var b = panel.querySelector('[data-role=stopBtn]'); if (b && !b.disabled) b.click();",
          ),
        );
        await sleep(300);
      }
      const aAfterStop = await cdp.evaluate(runsExpression(PANEL_A));
      const bAfterStop = await cdp.evaluate(runsExpression(PANEL_B));
      await sleep(2500);
      const aFinal = await cdp.evaluate(runsExpression(PANEL_A));
      const bFinal = await cdp.evaluate(runsExpression(PANEL_B));
      check(
        '快检：手动停止之后不再有新的交接',
        aFinal === aAfterStop && bFinal === bAfterStop,
        `停止时 A=${aAfterStop} B=${bAfterStop}，之后 A=${aFinal} B=${bFinal}`,
      );
      quickServer.closeAllConnections();
      await new Promise((resolve) => quickServer.close(resolve));
      return;
    }

    // ---------- 场景 A：左右两个 agent ----------
    process.stdout.write('\n  场景 A · 左右两个 agent 面板\n');
    await openFresh(cdp, base, { noSteps: true });

    const booted = await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
    check(
      '新前端启动并设置了启动标记',
      booted === config.FRONTEND_VERSION,
      `拿到 ${booted}`,
    );

    const href = await cdp.evaluate('location.href');
    check('没有触发自愈重载（地址里没有 r= 参数）', !href.includes('r='), href);

    const nameA = await waitFor(cdp, panelName(PANEL_A));
    const nameB = await waitFor(cdp, panelName(PANEL_B));
    check(
      '两个面板都拿到了自己的 agent 名',
      nameA === 'AGENT A' && nameB === 'AGENT B',
      `左 ${nameA} / 右 ${nameB}`,
    );
    check('左右是两个不同的 agent', nameA !== nameB, `${nameA} vs ${nameB}`);
    process.stdout.write(`      左: ${nameA} / 右: ${nameB}\n`);

    // ---------- 场景 B：A 跑完后自动交接给 B ----------
    process.stdout.write('\n  场景 B · AGENT A 跑完后自动交接给 AGENT B\n');

    await cdp.evaluate(runInPanel(PANEL_B, `panel.querySelector('[data-role=prompt]').value = ${JSON.stringify(PROMPT_B)};`));
    await cdp.evaluate(submitPrompt(PANEL_A, PROMPT_A));

    const doneA = await waitFor(cdp, panelDone(PANEL_A), 400, 400);
    check(
      'AGENT A 跑完',
      doneA === 'done',
      `状态 ${doneA} · 错误 ${await cdp.evaluate(panelError(PANEL_A))}`,
    );

    const answerA = await cdp.evaluate(panelFinalAnswer(PANEL_A));
    check('AGENT A 产出了文字分析结果', typeof answerA === 'string' && answerA.length > 0, `${answerA}`);

    // A 跑完之后 B 应该自动接手，不需要人点。
    const doneB = await waitFor(cdp, panelDone(PANEL_B), 500, 400);
    check(
      'AGENT A 跑完后 AGENT B 自动开跑并完成',
      doneB === 'done',
      `状态 ${doneB} · 错误 ${await cdp.evaluate(panelError(PANEL_B))}`,
    );

    const cardsA = await cdp.evaluate(panelCards(PANEL_A));
    check('AGENT A 画出了卡片', Array.isArray(cardsA) && cardsA.length > 0, `${cardsA && cardsA.length}`);

    const cardsB = await cdp.evaluate(panelCards(PANEL_B));
    check('AGENT B 有自己的卡片', Array.isArray(cardsB) && cardsB.length > 0, `${cardsB && cardsB.length}`);

    // 交接请求本身：必须同时带上 A 的分析结果和对 B 的要求。
    const handoffBodies = postBodies.filter((body) => body.includes('handoff'));
    check('AGENT B 是用 POST 带交接内容跑的', handoffBodies.length > 0, `POST 次数 ${handoffBodies.length}`);

    // 请求体是 JSON：指纹要按 JSON 里的样子比较（引号等会被转义）。
    const fingerprint = JSON.stringify(answerA.slice(0, 30)).slice(1, -1);
    const carriesAnswer = handoffBodies.some((body) => body.includes(fingerprint));
    check(
      '交给 AGENT B 的请求里带了 A 的分析结果',
      carriesAnswer,
      `A 的结果开头: ${answerA.slice(0, 30)}`,
    );
    check(
      '交给 AGENT B 的请求里带了对 B 的要求',
      handoffBodies.some((body) => body.includes(PROMPT_B)),
      `要求: ${PROMPT_B}`,
    );

    const handoffCards = await cdp.evaluate(panelHandoffCards(PANEL_B));
    check('AGENT B 的时间轴上显示了交接卡片', handoffCards === 1, `${handoffCards} 张`);

    const handoffText = await cdp.evaluate(panelHandoffText(PANEL_B));
    check(
      '交接卡片里有 AGENT A 的原文',
      answerA.length > 10 ? handoffText.includes(answerA.slice(0, 20)) : handoffText.includes(answerA),
      handoffText.slice(0, 120),
    );

    const stepsA = await cdp.evaluate(
      inPanel(PANEL_A, "Array.from(panel.querySelectorAll('.step-head')).length"),
    );
    const stepsB = await cdp.evaluate(
      inPanel(PANEL_B, "Array.from(panel.querySelectorAll('.step-head')).length"),
    );
    check('时间轴不再按 step 分段', stepsA === 0 && stepsB === 0, `A=${stepsA} B=${stepsB}`);

    const reasoningA = await cdp.evaluate(
      inPanel(PANEL_A, "Array.from(panel.querySelectorAll('.part-body.reasoning')).length"),
    );
    const reasoningB = await cdp.evaluate(
      inPanel(PANEL_B, "Array.from(panel.querySelectorAll('.part-body.reasoning')).length"),
    );
    check(
      '推理内容合成一块（不按段拆开）',
      reasoningA <= 1 && reasoningB <= 1,
      `A=${reasoningA} B=${reasoningB}`,
    );

    const reasoningTextA = await cdp.evaluate(
      inPanel(PANEL_A, "Array.from(panel.querySelectorAll('.part-body.reasoning')).map(function (n) { return n.textContent; }).join('')"),
    );
    check(
      '合并后的推理块里有内容',
      reasoningA === 0 || reasoningTextA.trim().length > 0,
      `推理字数=${reasoningTextA.length}`,
    );

    const TOOL_NAMES = [
      'listFiles',
      'readFile',
      'searchFiles',
      'listIssues',
      'readIssue',
      'gitLog',
      'writeFile',
      'deleteFile',
    ];

    for (const [label, panelId, cards] of [
      ['AGENT A', PANEL_A, cardsA],
      ['AGENT B', PANEL_B, cardsB],
    ]) {
      const badges = (cards || []).map((c) => c.badge);
      check(
        `${label}的徽标都是分片类型（没有 input-start 这类半成品）`,
        !badges.includes('input-start') &&
          badges.every((b) => b && b !== 'undefined' && b !== 'null'),
        `徽标: ${JSON.stringify(badges)}`,
      );
      check(
        `${label}的时间轴上没有任何工具卡`,
        !badges.some((b) => TOOL_NAMES.includes(b)),
        `徽标: ${JSON.stringify(badges)}`,
      );
      const toolCards = await cdp.evaluate(
        inPanel(panelId, "panel.querySelectorAll('.badge.tool').length"),
      );
      check(`${label}没有 .badge.tool 卡片`, toolCards === 0, `${toolCards} 张`);
    }

    const alertText = await cdp.evaluate("document.getElementById('alertBar').textContent");
    check('没有弹出告警条', alertText.trim() === '', alertText);

    const buildTag = await cdp.evaluate("document.getElementById('buildTag').textContent");
    check('标题栏显示构建号', buildTag.includes(config.FRONTEND_VERSION), buildTag);

    fs.mkdirSync(OUT_DIR, { recursive: true });
    const height = await cdp.evaluate('document.body.scrollHeight');
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: WIDTH,
      height: Math.min(height + 40, 4000),
      deviceScaleFactor: 1,
      mobile: false,
    });
    await sleep(400);
    const shot = await cdp.send('Page.captureScreenshot', {
      format: 'png',
      captureBeyondViewport: true,
    });
    const out = path.join(OUT_DIR, 'browser_a.png');
    fs.writeFileSync(out, Buffer.from(shot.data, 'base64'));
    process.stdout.write(`  截图: ${out}\n`);

    // ---------- 场景 C：B 单独跑（没有上游，不带交接） ----------
    process.stdout.write('\n  场景 C · AGENT B 单独跑，不带交接\n');
    await resetServerState(base);
    postBodies.length = 0;
    await cdp.evaluate(submitPrompt(PANEL_B, PROMPT_B));

    const doneB2 = await waitFor(cdp, panelDone(PANEL_B), 400, 400);
    check(
      'AGENT B 单独也能跑完',
      doneB2 === 'done',
      `状态 ${doneB2} · 错误 ${await cdp.evaluate(panelError(PANEL_B))}`,
    );
    const handoffCards2 = await cdp.evaluate(panelHandoffCards(PANEL_B));
    check('没有上游时 AGENT B 不显示交接卡片', handoffCards2 === 0, `${handoffCards2} 张`);
    check(
      '没有上游时走的是 GET，不带交接内容',
      postBodies.length === 0,
      `POST 次数 ${postBodies.length}`,
    );

    // ---------- 场景 S：下游没写要求，交接照样发起 ----------
    process.stdout.write('\n  场景 S · 下游没写要求时，交接照样发起\n');
    await openPageNoSteps(cdp, base);
    // 重新加载后下游的输入框是空的：用户只配了「把结果交给它」，
    // 没给下游写要求。交接是用户明确配好的，不该被一个空输入框悄悄丢掉。
    postBodies.length = 0;
    await cdp.evaluate(submitPrompt(PANEL_A, PROMPT_A));

    const doneA3 = await waitFor(cdp, panelDone(PANEL_A), 400, 400);
    check(
      'AGENT A 跑完',
      doneA3 === 'done',
      `状态 ${doneA3} · 错误 ${await cdp.evaluate(panelError(PANEL_A))}`,
    );

    const doneB4 = await waitFor(cdp, panelDone(PANEL_B), 500, 400);
    check(
      '下游没写要求也接了交接并跑完',
      doneB4 === 'done',
      `状态 ${doneB4} · 错误 ${await cdp.evaluate(panelError(PANEL_B))}`,
    );

    const handoffBodies3 = postBodies.filter((body) => body.includes('handoff'));
    check(
      '交接请求照样发出来了',
      handoffBodies3.length > 0,
      `POST 次数 ${postBodies.length}`,
    );

    const handoffCards3 = await cdp.evaluate(panelHandoffCards(PANEL_B));
    check(
      '下游的时间轴上显示了交接卡片',
      handoffCards3 === 1,
      `${handoffCards3} 张`,
    );

    // ---------- 场景 D：两个面板同时手动跑 ----------
    process.stdout.write('\n  场景 D · 两个面板同时手动跑\n');
    await resetServerState(base);
    await cdp.evaluate(submitPrompt(PANEL_A, PROMPT_A));
    await cdp.evaluate(submitPrompt(PANEL_B, PROMPT_B));

    const doneA2 = await waitFor(cdp, panelDone(PANEL_A), 400, 400);
    const doneB3 = await waitFor(cdp, panelDone(PANEL_B), 400, 400);
    check(
      '两个面板同时跑，都完成了',
      doneA2 === 'done' && doneB3 === 'done',
      `A=${doneA2} B=${doneB3}`,
    );

    // ---------- 场景 G：固定的显示窗口 ----------
    process.stdout.write('\n  场景 G · 时间轴是固定窗口，内容只在窗口里滚动\n');
    // 这条场景只关心"窗口固定 + 内部滚动"，跟模型说什么无关：
    // 用 mock 服务跑几轮离线内容，比让真模型写长文快一个数量级，也不花钱。
    const overflowServer = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
    await new Promise((resolve) => overflowServer.listen(0, '127.0.0.1', resolve));
    const overflowBase = `http://127.0.0.1:${overflowServer.address().port}`;
    await resetServerState(overflowBase);
    await openPageNoSteps(cdp, overflowBase);

    const overflowExpression = inPanel(
      PANEL_A,
      "(function () {" +
        "var tl = panel.querySelector('[data-role=timeline]');" +
        "return tl.scrollHeight > tl.clientHeight + 50 ? 'overflow' : '';" +
        '})()',
    );

    // 内容多到溢出窗口为止 —— 窗口没固定住的话，内容会把窗口一起撑高，
    // scrollHeight 永远等于 clientHeight，这条就等不到。
    let overflowed = '';
    for (let attempt = 0; attempt < 10 && overflowed !== 'overflow'; attempt += 1) {
      await runPromptOnce(cdp, PANEL_A, PROMPT_A);
      overflowed = await cdp.evaluate(overflowExpression);
    }
    check('内容超出后时间轴仍是一个固定高度的窗口', overflowed === 'overflow', `状态 ${overflowed}`);

    const win = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "(function () {" +
          "var tl = panel.querySelector('[data-role=timeline]');" +
          "return {" +
          "  overflowY: getComputedStyle(tl).overflowY," +
          "  scrollHeight: tl.scrollHeight," +
          "  clientHeight: tl.clientHeight," +
          "  bodyScroll: document.body.scrollHeight," +
          "  innerHeight: window.innerHeight," +
          '};' +
          '})()',
      ),
    );
    check(
      '窗口内部自己滚动（overflow-y 是 auto）',
      win.overflowY === 'auto',
      `overflow-y=${win.overflowY}`,
    );
    check(
      '内容溢出时窗口高度不变',
      win.scrollHeight > win.clientHeight,
      `scroll=${win.scrollHeight} client=${win.clientHeight}`,
    );
    check(
      '页面不被内容撑高',
      win.bodyScroll <= win.innerHeight + 40,
      `body=${win.bodyScroll} viewport=${win.innerHeight}`,
    );

    // 跑完后窗口应该停在最新一条上。
    await waitFor(cdp, panelDone(PANEL_A), 400, 400);
    const stick = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "(function () {" +
          "var tl = panel.querySelector('[data-role=timeline]');" +
          "return {gap: tl.scrollHeight - tl.scrollTop - tl.clientHeight, top: tl.scrollTop};" +
          '})()',
      ),
    );
    check('跑完后窗口停在最新一条上', stick.gap < 40, `离底部 ${stick.gap} px`);

    // 用户往上翻看历史时，新内容不该把他拽回底部。
    // 再跑一轮，在流式过程中把窗口滚到顶，看它会不会被拉回去。
    //
    // 先把视口压矮：内容很快就能撑破时间轴，滚到顶之后还有内容在来 ——
    // 视口高的时候内容要到最后一刻才溢出，滚完顶就没什么可等的了。
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: WIDTH,
      height: 420,
      deviceScaleFactor: 1,
      mobile: false,
    });

    // 数一数时间轴里发生过多少次内容变化（加卡片、写文字都算）。
    // 用它判断"新内容到了"，比数卡片准：文字是写进已有卡片里的。
    await cdp.evaluate(
      runInPanel(
        PANEL_A,
        'window.__timelineMutations = 0;' +
          'if (window.__timelineObserver) window.__timelineObserver.disconnect();' +
          'window.__timelineObserver = new MutationObserver(function () {' +
          '  window.__timelineMutations += 1;' +
          '});' +
          "window.__timelineObserver.observe(" +
          "  panel.querySelector('[data-role=timelineInner]')," +
          '  {childList: true, subtree: true, characterData: true});',
      ),
    );

    await cdp.evaluate(submitPrompt(PANEL_A, PROMPT_A));

    // 等内容开始进来。
    await waitFor(cdp, 'window.__timelineMutations > 0 ? "arrived" : ""', 200, 100);

    // 等时间轴真的能滚了（有内容溢出），这时候往回翻才翻得动。
    const scrollable = await waitFor(
      cdp,
      inPanel(
        PANEL_A,
        '(function () {' +
          "var tl = panel.querySelector('[data-role=timeline]');" +
          "return tl.scrollHeight > tl.clientHeight + 20 ? 'scrollable' : '';" +
          '})()',
      ),
      200,
      100,
    );
    check('时间轴里已经有能翻的内容', scrollable === 'scrollable', String(scrollable));

    // 用户往上翻看历史。
    await cdp.evaluate(
      runInPanel(
        PANEL_A,
        "var tl = panel.querySelector('[data-role=timeline]');" +
          'tl.scrollTop = 0;' +
          "tl.dispatchEvent(new Event('scroll'));",
      ),
    );
    const mutationsBefore = await cdp.evaluate('window.__timelineMutations');

    // 下一批内容还得继续到达 —— 用户翻到顶不能把流掐了。
    const grew = await waitFor(
      cdp,
      `window.__timelineMutations > ${mutationsBefore} ? 'grew' : ''`,
      160,
      100,
    );
    check(
      '窗口里确实又有新内容到达',
      grew === 'grew',
      `状态 ${grew}（之前 ${mutationsBefore} 次变化）`,
    );

    const afterScroll = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "(function () {" +
          "var tl = panel.querySelector('[data-role=timeline]');" +
          'return {top: tl.scrollTop, gap: tl.scrollHeight - tl.scrollTop - tl.clientHeight};' +
          '})()',
      ),
    );
    check(
      '用户往上翻看时不会被新内容拽回底部',
      afterScroll.top < 50,
      `scrollTop=${afterScroll.top}（离底部 ${afterScroll.gap} px）`,
    );

    // 视口恢复原样，后面的场景按正常高度跑。
    await cdp.send('Emulation.setDeviceMetricsOverride', {
      width: WIDTH,
      height: HEIGHT,
      deviceScaleFactor: 1,
      mobile: false,
    });
    overflowServer.closeAllConnections();
    await new Promise((resolve) => overflowServer.close(resolve));

    // ---------- 场景 H：结束按钮 ----------
    process.stdout.write('\n  场景 H · 「结束」按钮能中止正在跑的一轮\n');
    await resetServerState(base);
    await openPageNoSteps(cdp, base);

    const stopDisabledAtStart = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "(function () {" +
          "var b = panel.querySelector('[data-role=stopBtn]');" +
          "return b ? b.disabled : 'missing';" +
          '})()',
      ),
    );
    check(
      '没在跑时「结束」按钮是禁用的',
      stopDisabledAtStart === true,
      `disabled=${stopDisabledAtStart}`,
    );

    await cdp.evaluate(submitPrompt(PANEL_A, PROMPT_A));

    const stopEnabled = await waitFor(
      cdp,
      inPanel(
        PANEL_A,
        "(function () {" +
          "var b = panel.querySelector('[data-role=stopBtn]');" +
          "if (!b) return 'missing';" +
          "return b.disabled ? '' : 'ready';" +
          '})()',
      ),
      120,
      250,
    );
    check('跑起来之后「结束」按钮可以点了', stopEnabled === 'ready', `状态 ${stopEnabled}`);

    const cardsAtStop = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelectorAll('.part').length"),
    );
    await cdp.evaluate(
      runInPanel(
        PANEL_A,
        "var b = panel.querySelector('[data-role=stopBtn]');" +
          "if (b) b.click();",
      ),
    );

    const stopped = await waitFor(
      cdp,
      inPanel(PANEL_A, "panel.querySelector('.dot.stopped') ? 'stopped' : ''"),
      60,
      200,
    );
    check('点「结束」把这一轮标记成已中止', stopped === 'stopped', `状态 ${stopped}`);

    const stopText = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelector('[data-role=statusText]').textContent"),
    );
    check('状态栏写明是「已中止」', stopText === '已中止', `状态栏 ${stopText}`);

    const runEnabled = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelector('[data-role=runBtn]').disabled"),
    );
    check('中止之后可以重新运行', runEnabled === false, `运行按钮 disabled=${runEnabled}`);

    // 中止之后不该再有新分片进来 —— 连接断了，agent 那一侧也停了。
    await sleep(4000);
    const cardsAfterStop = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelectorAll('.part').length"),
    );
    check(
      '中止之后不再有新内容进来',
      cardsAfterStop === cardsAtStop,
      `中止时 ${cardsAtStop} 张，之后 ${cardsAfterStop} 张`,
    );

    // ---------- 场景 I：提示词可配置 ----------
    process.stdout.write('\n  场景 I · 两个 agent 的提示词是可编辑的配置项\n');
    await openFresh(cdp, base, { noSteps: true });

    const health = await request(`${base}/api/health`);
    const defaultA = (health.agents || []).find((a) => a.id === 'agent-a');
    const defaultB = (health.agents || []).find((a) => a.id === 'agent-b');
    check(
      '服务端把两个 agent 的默认提示词都给了出来',
      Boolean(defaultA && defaultB && defaultA.instructions && defaultB.instructions),
      `A 长度=${defaultA && defaultA.instructions ? defaultA.instructions.length : '无'}` +
        ` B 长度=${defaultB && defaultB.instructions ? defaultB.instructions.length : '无'}`,
    );

    const boxA = inPanel(PANEL_A, "panel.querySelector('[data-role=instructions]')");
    const filled = await waitFor(
      cdp,
      inPanel(
        PANEL_A,
        "(function () {" +
          "var t = panel.querySelector('[data-role=instructions]');" +
          "return t && t.value.length > 0 ? 'filled' : '';" +
          '})()',
      ),
      60,
      200,
    );
    check('AGENT A 的提示词框里装上了默认提示词', filled === 'filled', `状态 ${filled}`);

    const valueA = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "(function () {" +
          "var t = panel.querySelector('[data-role=instructions]');" +
          "return t ? t.value : '';" +
          '})()',
      ),
    );
    check(
      'AGENT A 提示词框里的内容与服务端默认一致',
      valueA === defaultA.instructions,
      `页面 ${String(valueA).slice(0, 40)} / 服务端 ${String(defaultA.instructions).slice(0, 40)}`,
    );

    const valueB = await cdp.evaluate(
      inPanel(
        PANEL_B,
        "(function () {" +
          "var t = panel.querySelector('[data-role=instructions]');" +
          "return t ? t.value : '';" +
          '})()',
      ),
    );
    check(
      'AGENT B 提示词框里的内容与服务端默认一致',
      valueB === defaultB.instructions,
      `页面 ${String(valueB).slice(0, 40)} / 服务端 ${String(defaultB.instructions).slice(0, 40)}`,
    );

    // 改提示词 → 存到服务端 → 这一轮用的就是新提示词。
    //
    // 提示词不再跟着请求发：它存在服务端，服务端按自己存的那份来跑。
    // 所以这里查的是服务端存的那份与这一轮记下来的那份，不是请求体。
    postBodies.length = 0;
    const CUSTOM = '你只需要回答一句话：自定义提示词生效。';
    await cdp.evaluate(
      runInPanel(
        PANEL_A,
        "var t = panel.querySelector('[data-role=instructions]');" +
          `t.value = ${JSON.stringify(CUSTOM)};` +
          "t.dispatchEvent(new Event('input', {bubbles:true}));",
      ),
    );
    await cdp.evaluate(submitPrompt(PANEL_A, '测试自定义提示词'));

    const recordsI = await waitForChatCount(base, 'agent-a', 1);
    const lastRecordI = recordsI[recordsI.length - 1] || {};
    check(
      '这一轮真正用的就是改过的提示词',
      lastRecordI.instructions === CUSTOM,
      `这一轮用的 ${String(lastRecordI.instructions).slice(0, 40)}`,
    );

    const healthI2 = await request(`${base}/api/health`);
    const agentAI = (healthI2.agents || []).find((a) => a.id === 'agent-a');
    check(
      '改过的提示词存到了服务端',
      Boolean(agentAI) && agentAI.instructions === CUSTOM,
      `服务端 ${agentAI ? String(agentAI.instructions).slice(0, 40) : '无'}`,
    );

    const sentInstructions = postBodies.filter((body) => body.includes('instructions'));
    check(
      '提示词不再跟着请求发（服务端才是唯一的一份）',
      sentInstructions.length === 0,
      `${sentInstructions.length} 条带 instructions 的请求`,
    );

    // 没改提示词时不该多传 instructions（默认走 GET）。
    postBodies.length = 0;
    await cdp.evaluate(
      runInPanel(
        PANEL_B,
        "var t = panel.querySelector('[data-role=instructions]');" +
          "var d = panel.querySelector('[data-role=resetInstructions]');" +
          'if (d) d.click();',
      ),
    );
    const restored = await cdp.evaluate(
      inPanel(
        PANEL_B,
        "(function () {" +
          "var t = panel.querySelector('[data-role=instructions]');" +
          "return t ? t.value : '';" +
          '})()',
      ),
    );
    check(
      '「恢复默认」把提示词改回服务端默认',
      restored === defaultB.instructions,
      `恢复后 ${String(restored).slice(0, 40)}`,
    );

    // ---------- 场景 J：新建 agent ----------
    process.stdout.write('\n  场景 J · 「新建 Agent」能造出一个新的 agent 面板\n');
    await openFresh(cdp, base, { noSteps: true });

    const panelsBefore = await cdp.evaluate("document.querySelectorAll('.panel').length");
    check('一开始页面上有三个面板', panelsBefore === 3, `${panelsBefore} 个`);

    const countsBeforeJ = await cdp.evaluate("document.querySelectorAll('.panel').length");
    const newPanelId = await createAgentAndWait(cdp);
    const countsAfterJ = await cdp.evaluate("document.querySelectorAll('.panel').length");
    check('点一下就多出一个面板', countsAfterJ === countsBeforeJ + 1, `${countsBeforeJ} → ${countsAfterJ}`);
    check('新面板有独立的 id', Boolean(newPanelId), `${newPanelId}`);

    const newInstructions = await cdp.evaluate(
      inPanel(
        newPanelId,
        "(function () {" +
          "var t = panel.querySelector('[data-role=instructions]');" +
          "return t ? t.value : 'missing';" +
          '})()',
      ),
    );
    check(
      '新 agent 的系统提示词默认是空的',
      newInstructions === '',
      `框里是 ${JSON.stringify(String(newInstructions).slice(0, 40))}`,
    );

    const newState = await cdp.evaluate(
      inPanel(
        newPanelId,
        "panel.querySelector('[data-role=instructionsState]').textContent",
      ),
    );
    check(
      '提示词状态写明它没有系统提示词',
      typeof newState === 'string' && newState.includes('没有系统提示词'),
      `状态 ${newState}`,
    );

    const newName = await cdp.evaluate(
      inPanel(newPanelId, "panel.querySelector('[data-role=name]').textContent"),
    );
    check(
      '新 agent 有自己的名字',
      typeof newName === 'string' &&
        newName.length > 0 &&
        newName !== 'AGENT A' &&
        newName !== 'AGENT B',
      `名字 ${newName}`,
    );

    const newPrompt = await cdp.evaluate(
      inPanel(newPanelId, "panel.querySelector('[data-role=prompt]').value"),
    );
    check('新 agent 的输入框也是空的', newPrompt === '', `输入框 ${JSON.stringify(newPrompt)}`);

    // 部件要和 AGENT A 完全一样 —— 形式相同是用户提的要求。
    const rolesOf = (panelId) =>
      cdp.evaluate(
        inPanel(
          panelId,
          "Array.from(panel.querySelectorAll('[data-role]')).map(function (e) {" +
            "return e.getAttribute('data-role');}).sort().join(',')",
        ),
      );
    const rolesNew = await rolesOf(newPanelId);
    const rolesA = await rolesOf(PANEL_A);
    check('新面板的部件与 AGENT A 一样', rolesNew === rolesA, `${rolesNew} / ${rolesA}`);

    // 给新 agent 写一份系统提示词，跑起来 —— 它得真的按这份提示词回答。
    // 复述用固定标记而不是一句自然语言：模型会把句子改写（"生效" → "已经生效"），
    // 这里要验的是"用的是这份提示词"，不是它的文风。
    const NEW_AGENT_MARKER = 'NEW_AGENT_MARKER_42';
    const NEW_INSTRUCTIONS =
      `只回答这个字符串，一个字都不要改：${NEW_AGENT_MARKER}。不要调用任何工具。`;
    postBodies.length = 0;
    await cdp.evaluate(
      runInPanel(
        newPanelId,
        "var t = panel.querySelector('[data-role=instructions]');" +
          `t.value = ${JSON.stringify(NEW_INSTRUCTIONS)};` +
          "t.dispatchEvent(new Event('input', {bubbles:true}));",
      ),
    );
    const newAgentId = await cdp.evaluate(
      inPanel(newPanelId, "panel.getAttribute('data-profile')"),
    );
    check('新 agent 在服务端有身份', Boolean(newAgentId), `${newAgentId}`);

    await cdp.evaluate(submitPrompt(newPanelId, '测试新建的 agent'));

    // 提示词不再跟着请求发：它存到了服务端，服务端按自己存的那份来跑。
    const recordsJ = await waitForChatCount(base, newAgentId, 1);
    const recordJ = recordsJ[recordsJ.length - 1] || {};
    check(
      '新 agent 跑的时候用的就是自己写的那份提示词',
      recordJ.instructions === NEW_INSTRUCTIONS,
      `这一轮用的 ${String(recordJ.instructions).slice(0, 40)}`,
    );

    const healthJ = await request(`${base}/api/health`);
    const agentJ = (healthJ.agents || []).find((a) => a.id === newAgentId);
    check(
      '新 agent 写的提示词存到了服务端',
      Boolean(agentJ) && agentJ.instructions === NEW_INSTRUCTIONS,
      `服务端 ${agentJ ? String(agentJ.instructions).slice(0, 40) : '无'}`,
    );

    const newBodies = postBodies.filter((body) => body.includes('instructions'));
    check(
      '新 agent 的请求也没带提示词（服务端才是唯一的一份）',
      newBodies.length === 0,
      `${newBodies.length} 条带 instructions 的请求`,
    );

    const newDone = await waitFor(cdp, panelDone(newPanelId), 400, 400);
    check(
      '新 agent 跑完了',
      newDone === 'done',
      `状态 ${newDone} · 错误 ${await cdp.evaluate(panelError(newPanelId))}`,
    );

    const newAnswer = await cdp.evaluate(panelFinalAnswer(newPanelId));
    check(
      '新 agent 按自己写的提示词回答',
      typeof newAnswer === 'string' && newAnswer.includes(NEW_AGENT_MARKER),
      String(newAnswer).slice(0, 140),
    );

    // ---------- 场景 K：删除按钮 + 多选交接配置 ----------
    process.stdout.write('\n  场景 K · 每个 agent 都能删、能多选交接目标\n');

    // 这一段跑在离线剧本模式的服务上：前端逻辑完全一样，但不必等真实模型。
    const mockServer = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
    await new Promise((resolve) => mockServer.listen(0, '127.0.0.1', resolve));
    const mockBase = `http://127.0.0.1:${mockServer.address().port}`;
    await openFresh(cdp, mockBase, { noSteps: true });

    const panelsAtStart = await cdp.evaluate("document.querySelectorAll('.panel').length");
    check('一开始页面上有三个面板', panelsAtStart === 3, `${panelsAtStart} 个`);

    const delButtons = await cdp.evaluate(
      "document.querySelectorAll('.panel [data-role=deleteBtn]').length",
    );
    check(
      '每个面板都有一个删除按钮',
      delButtons === panelsAtStart,
      `${delButtons} 个按钮 / ${panelsAtStart} 个面板`,
    );

    const targetAreas = await cdp.evaluate(
      "document.querySelectorAll('.panel [data-role=handoffTargets]').length",
    );
    check(
      '每个面板都有交接目标的选择区',
      targetAreas === panelsAtStart,
      `${targetAreas} 个`,
    );

    const readTargets = (panelId) =>
      cdp.evaluate(
        inPanel(
          panelId,
          "Array.from(panel.querySelectorAll('[data-role=handoffTargets] input[type=checkbox]')).map(function (i) {" +
            'return {value: i.value, checked: i.checked};})',
        ),
      );

    const aTargets0 = await readTargets(PANEL_A);
    check(
      'AGENT A 默认把结果交给 AGENT B',
      Array.isArray(aTargets0) &&
        aTargets0.some((t) => t.value === 'agent-b' && t.checked === true),
      JSON.stringify(aTargets0),
    );

    // 新建一个 agent：别的面板的交接目标列表里要出现它。
    const newPanelId2 = await createAgentAndWait(cdp);
    check('新 agent 的面板出来了', Boolean(newPanelId2), `${newPanelId2}`);
    const newProfileId = String(newPanelId2).replace(/^panel-/, '');

    await waitFor(
      cdp,
      inPanel(
        PANEL_A,
        `Array.from(panel.querySelectorAll('[data-role=handoffTargets] input[type=checkbox]')).some(function (i) { return i.value === '${newProfileId}'; }) ? 'ready' : ''`,
      ),
      60,
      200,
    );
    const optionsAfterCreate = await readTargets(PANEL_A);
    check(
      '新 agent 出现在 AGENT A 的交接目标里',
      Array.isArray(optionsAfterCreate) && optionsAfterCreate.length === 3,
      JSON.stringify(optionsAfterCreate),
    );

    // 多选：勾上新建的那个（AGENT B 默认已勾），其余目标保持不勾 ——
    // 页面上的目标不止两个，全点一遍会把不相干的目标也拉进来。
    await cdp.evaluate(
      inPanel(
        PANEL_A,
        "Array.from(panel.querySelectorAll('[data-role=handoffTargets] input[type=checkbox]')).forEach(function (i) {" +
          `if (i.value === '${newProfileId}' && !i.checked) i.click();` +
          '})',
      ),
    );
    const checkedNow = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "Array.from(panel.querySelectorAll('[data-role=handoffTargets] input[type=checkbox]')).filter(function (i) {" +
          'return i.checked;}).length',
      ),
    );
    check('交接目标可以多选（同时勾了两个）', checkedNow === 2, `勾了 ${checkedNow} 个`);

    // 源 agent 跑完后，两个被勾上的目标都该自动接手。
    await cdp.evaluate(
      runInPanel(PANEL_B, "panel.querySelector('[data-role=prompt]').value = '给 AGENT B 的要求';"),
    );
    await cdp.evaluate(
      runInPanel(newPanelId2, "panel.querySelector('[data-role=prompt]').value = '给新 agent 的要求';"),
    );

    postBodies.length = 0;
    await cdp.evaluate(submitPrompt(PANEL_A, '源 agent 的问题'));

    const doneSrc = await waitFor(cdp, panelDone(PANEL_A), 200, 300);
    check(
      '源 agent 跑完了',
      doneSrc === 'done',
      `状态 ${doneSrc} · 错误 ${await cdp.evaluate(panelError(PANEL_A))}`,
    );

    const doneT1 = await waitFor(cdp, panelDone(PANEL_B), 200, 300);
    const doneT2 = await waitFor(cdp, panelDone(newPanelId2), 200, 300);
    check(
      '两个被勾选的目标都自动跑完了',
      doneT1 === 'done' && doneT2 === 'done',
      `B=${doneT1} 新=${doneT2} · 错误 ${await cdp.evaluate(panelError(newPanelId2))}`,
    );

    const cardsT1 = await cdp.evaluate(panelHandoffCards(PANEL_B));
    const cardsT2 = await cdp.evaluate(panelHandoffCards(newPanelId2));
    check(
      '两个目标的时间轴上都出现了交接卡片',
      cardsT1 === 1 && cardsT2 === 1,
      `B=${cardsT1} 新=${cardsT2}`,
    );

    const cardText2 = await cdp.evaluate(panelHandoffText(newPanelId2));
    check(
      '交接卡片写明结果来自 AGENT A',
      typeof cardText2 === 'string' && cardText2.includes('AGENT A'),
      String(cardText2).slice(0, 120),
    );

    const handoffBodiesK = postBodies.filter((body) => body.includes('handoff'));
    check(
      '交接请求带上了来源名字 handoffFrom',
      handoffBodiesK.length > 0 &&
        handoffBodiesK.every((body) => body.includes('"handoffFrom":"AGENT A"')),
      handoffBodiesK.length ? handoffBodiesK[0].slice(0, 160) : '（没有交接请求）',
    );

    // 删除：第一次点只是进入确认状态，第二次才真删。
    await cdp.evaluate(
      runInPanel(newPanelId2, "var b = panel.querySelector('[data-role=deleteBtn]'); if (b) b.click();"),
    );
    const armedText = await cdp.evaluate(
      inPanel(newPanelId2, "panel.querySelector('[data-role=deleteBtn]').textContent"),
    );
    check('第一次点删除只是进入确认状态', String(armedText).includes('确认'), `按钮文字 ${armedText}`);
    const stillThere = await cdp.evaluate("document.querySelectorAll('.panel').length");
    check('确认之前面板还在', stillThere === 4, `${stillThere} 个`);

    await cdp.evaluate(
      runInPanel(newPanelId2, "var b = panel.querySelector('[data-role=deleteBtn]'); if (b) b.click();"),
    );
    const gone = await waitFor(
      cdp,
      "document.querySelectorAll('.panel').length === 3 ? 'gone' : ''",
      60,
      200,
    );
    check('再点一次面板被删掉了', gone === 'gone', `状态 ${gone}`);

    const healthAfter = await request(`${mockBase}/api/health`);
    check(
      '服务端也不再有这个 agent 身份',
      !(healthAfter.agents || []).some((a) => a.id === newProfileId),
      JSON.stringify((healthAfter.agents || []).map((a) => a.id)),
    );

    const optionsAfterDelete = await readTargets(PANEL_A);
    check(
      '被删掉的 agent 从别的面板的目标列表里消失了',
      Array.isArray(optionsAfterDelete) &&
        !optionsAfterDelete.some((t) => t.value === newProfileId),
      JSON.stringify(optionsAfterDelete),
    );

    // 回环保护：A → B 又 B → A 不能互相触发。
    await cdp.evaluate(
      runInPanel(
        PANEL_B,
        "Array.from(panel.querySelectorAll('[data-role=handoffTargets] input[type=checkbox]')).forEach(function (i) {" +
          "if (i.value === 'agent-a' && !i.checked) i.click();" +
          '})',
      ),
    );
    const bTargets = await readTargets(PANEL_B);
    check(
      'AGENT B 的交接目标里可以选 AGENT A',
      Array.isArray(bTargets) &&
        bTargets.some((t) => t.value === 'agent-a' && t.checked === true),
      JSON.stringify(bTargets),
    );

    postBodies.length = 0;
    await cdp.evaluate(submitPrompt(PANEL_A, '再来一轮'));

    let sawHandoff = false;
    for (let i = 0; i < 120; i++) {
      await sleep(250);
      if (postBodies.some((body) => body.includes('"handoffFrom":"AGENT A"'))) {
        sawHandoff = true;
        break;
      }
    }
    check('A 跑完后把结果交给了 B', sawHandoff, `POST 次数 ${postBodies.length}`);

    // 再等一会儿：B 如果把结果回传给 A，这里会多出一条 handoffFrom 是 AGENT B 的请求。
    await sleep(6000);
    const backToA = postBodies.filter((body) => body.includes('"handoffFrom":"AGENT B"'));
    check(
      'B 不会把结果回传给 A（没有来回触发）',
      backToA.length === 0,
      `回传请求 ${backToA.length} 条`,
    );

    // 固定的两个 agent 也能从页面上删掉（身份是配置里写死的，服务端仍保留）。
    await cdp.evaluate(
      runInPanel(PANEL_A, "var b = panel.querySelector('[data-role=deleteBtn]'); if (b) b.click();"),
    );
    await cdp.evaluate(
      runInPanel(PANEL_A, "var b = panel.querySelector('[data-role=deleteBtn]'); if (b) b.click();"),
    );
    const goneA = await waitFor(
      cdp,
      "document.querySelectorAll('.panel').length === 2 ? 'gone' : ''",
      60,
      200,
    );
    check('固定的两个 agent 也能从页面上删掉', goneA === 'gone', `状态 ${goneA}`);

    const healthAfterA = await request(`${mockBase}/api/health`);
    check(
      '但服务端仍保留固定的两个身份',
      ['agent-a', 'agent-b'].every((id) =>
        (healthAfterA.agents || []).some((a) => a.id === id),
      ),
      JSON.stringify((healthAfterA.agents || []).map((a) => a.id)),
    );

    mockServer.closeAllConnections();
    await new Promise((resolve) => mockServer.close(resolve));

    // ---------- 场景 L：轮次步骤 ----------
    process.stdout.write('\n  场景 L · 每个 agent 都能配自己的轮次步骤\n');

    // 同样跑在离线剧本模式上：步骤的顺序与传递要看得清，不必等真实模型。
    const mockServerL = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
    await new Promise((resolve) => mockServerL.listen(0, '127.0.0.1', resolve));
    const mockBaseL = `http://127.0.0.1:${mockServerL.address().port}`;
    await openFresh(cdp, mockBaseL);

    const stepAreas = await cdp.evaluate(
      "document.querySelectorAll('.panel [data-role=stepsList]').length",
    );
    check('每个面板都有轮次步骤的配置区', stepAreas === 3, `${stepAreas} 个`);

    // 默认步骤：AGENT A / AGENT B 的提示词拆成的固定步骤。
    // 服务端给出来，用户没配过之前页面就铺好。
    const defaultTitlesL = await cdp.evaluate(
      "['panelA','panelB'].map(function (id) {" +
        "var p = document.getElementById(id);" +
        'return Array.from(p.querySelectorAll(".step-title")).map(function (t) { return t.value; });' +
        '})',
    );
    check(
      'AGENT A 的步骤区按提示词铺好了 4 步',
      Array.isArray(defaultTitlesL[0]) && defaultTitlesL[0].length === 4,
      JSON.stringify(defaultTitlesL[0]),
    );
    check(
      'AGENT B 的步骤区也铺好了 4 步',
      Array.isArray(defaultTitlesL[1]) && defaultTitlesL[1].length === 4,
      JSON.stringify(defaultTitlesL[1]),
    );
    check(
      '默认步骤的标题写明这一步做什么',
      Array.isArray(defaultTitlesL[0]) &&
        defaultTitlesL[0].every((t) => String(t).trim().length > 0) &&
        String(defaultTitlesL[0][0]).includes('Open Issue'),
      JSON.stringify(defaultTitlesL[0]),
    );
    check(
      'AGENT B 的默认步骤也从查 Open Issue 开始',
      Array.isArray(defaultTitlesL[1]) && String(defaultTitlesL[1][0]).includes('Open Issue'),
      JSON.stringify(defaultTitlesL[1]),
    );
    check(
      '两个 agent 的默认步骤各不相同',
      Array.isArray(defaultTitlesL[0]) &&
        Array.isArray(defaultTitlesL[1]) &&
        JSON.stringify(defaultTitlesL[0]) !== JSON.stringify(defaultTitlesL[1]),
      `A ${JSON.stringify(defaultTitlesL[0])} / B ${JSON.stringify(defaultTitlesL[1])}`,
    );

    const defaultSummaryL = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelector('[data-role=stepsSummary]').textContent"),
    );
    check('步骤配置区写明默认有几步', String(defaultSummaryL).includes('4'), String(defaultSummaryL));

    // 步骤一多就超出面板了：面板是固定高度 + overflow:hidden，
    // 步骤列表自己没有滚动区的话，后面几步会被裁掉，谁也看不到、也滚不到。
    await cdp.evaluate(
      runInPanel(PANEL_A, "panel.querySelector('[data-role=stepsBox]').open = true;"),
    );
    const stepsScrollL = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "(function () {" +
          "var l = panel.querySelector('[data-role=stepsList]');" +
          "var s = getComputedStyle(l);" +
          "return {overflowY: s.overflowY, maxHeight: s.maxHeight," +
          "  overflow: l.scrollHeight > l.clientHeight + 1};" +
          '})()',
      ),
    );
    check(
      '步骤列表自己是一个滚动窗口（有上限、能滚）',
      stepsScrollL.overflowY === 'auto' && stepsScrollL.maxHeight !== 'none',
      JSON.stringify(stepsScrollL),
    );

    const lastStepVisibleL = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "(function () {" +
          "var l = panel.querySelector('[data-role=stepsList]');" +
          'l.scrollTop = l.scrollHeight;' +
          "var rows = l.querySelectorAll('.step-row');" +
          'var last = rows[rows.length - 1].getBoundingClientRect();' +
          'var box = l.getBoundingClientRect();' +
          'return {visible: last.bottom <= box.bottom + 1 && last.top >= box.top - 1};' +
          '})()',
      ),
    );
    check(
      '滚到列表底部时最后一个步骤看得见（没被裁掉）',
      lastStepVisibleL.visible === true,
      JSON.stringify(lastStepVisibleL),
    );

    // 用户清空之后默认步骤跟着一起没了；「恢复默认步骤」能把它找回来。
    await cdp.evaluate(clearSteps(PANEL_A));
    const clearedL = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelectorAll('.step-row').length"),
    );
    check('用户清空后默认步骤也一起没了', clearedL === 0, `${clearedL} 行`);

    const clearedSummaryL = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelector('[data-role=stepsSummary]').textContent"),
    );
    check(
      '清空之后摘要写明默认还有几步、怎么找回',
      String(clearedSummaryL).includes('4') &&
        String(clearedSummaryL).includes('恢复默认步骤'),
      String(clearedSummaryL),
    );

    await cdp.evaluate(
      runInPanel(PANEL_A, "panel.querySelector('[data-role=restoreSteps]').click();"),
    );
    const restoredL = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelectorAll('.step-row').length"),
    );
    check('「恢复默认步骤」能把 4 步找回来', restoredL === 4, `${restoredL} 行`);

    const STEP_TITLES = ['第一步 查', '第二步 改', '第三步 验'];

    // 下游 AGENT B 的默认步骤先清掉：这一段数的是 AGENT A 发了几次请求，
    // B 带着 4 个默认步骤跑会混进同一个计数里。
    await cdp.evaluate(clearSteps(PANEL_B));

    // 在 AGENT A 上配三步：查 / 改 / 验。先把默认步骤清掉 —— 这一段测的是
    // 用户自己配的步骤，不是默认步骤。
    await cdp.evaluate(clearSteps(PANEL_A));
    await cdp.evaluate(
      runInPanel(
        PANEL_A,
        "panel.querySelector('[data-role=stepsBox]').open = true;" +
          "var add = panel.querySelector('[data-role=addStep]');" +
          'add.click(); add.click(); add.click();',
      ),
    );

    await cdp.evaluate(
      runInPanel(
        PANEL_A,
        'var titles = ' +
          JSON.stringify(STEP_TITLES) +
          ';' +
          "var inputs = panel.querySelectorAll('.step-title');" +
          "var areas = panel.querySelectorAll('.step-instruction');" +
          'titles.forEach(function (t, i) {' +
          '  inputs[i].value = t;' +
          "  inputs[i].dispatchEvent(new Event('input', {bubbles:true}));" +
          "  areas[i].value = '做第 ' + i + ' 件事';" +
          "  areas[i].dispatchEvent(new Event('input', {bubbles:true}));" +
          '});',
      ),
    );

    const stepSummaryL = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelector('[data-role=stepsSummary]').textContent"),
    );
    check('步骤配置区写明一共几步', String(stepSummaryL).includes('3'), String(stepSummaryL));

    // 跑一轮：每一步各发一次请求，顺序固定。
    await cdp.evaluate(
      runInPanel(PANEL_B, "panel.querySelector('[data-role=prompt]').value = '请审查';")
    );
    postBodies.length = 0;
    await cdp.evaluate(submitPrompt(PANEL_A, '本轮的问题'));

    const doneL = await waitFor(cdp, panelDone(PANEL_A), 200, 300);
    check(
      '按步骤跑完整轮',
      doneL === 'done',
      `状态 ${doneL} · 错误 ${await cdp.evaluate(panelError(PANEL_A))}`,
    );

    // 交接是整轮跑完后才发起的：等下游也跑完，交接请求才算真的发出去了。
    const doneL_B = await waitFor(cdp, panelDone(PANEL_B), 200, 300);
    check('整轮跑完后结果交给了下游', doneL_B === 'done', `下游状态 ${doneL_B}`);

    const stepCardsL = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "Array.from(panel.querySelectorAll('.round-step-card')).map(function (c) {" +
          'return {name: c.querySelector(".round-step-name").textContent,' +
          'state: c.querySelector(".round-step-state").textContent};})',
      ),
    );
    check(
      '时间轴上有三张步骤卡',
      Array.isArray(stepCardsL) && stepCardsL.length === 3,
      JSON.stringify(stepCardsL),
    );
    check(
      '步骤卡按顺序写明各自的标题',
      Array.isArray(stepCardsL) &&
        STEP_TITLES.every((t, i) => String(stepCardsL[i] && stepCardsL[i].name).includes(t)),
      JSON.stringify(stepCardsL),
    );
    check(
      '每一步都真的产出了文字',
      Array.isArray(stepCardsL) && stepCardsL.every((c) => String(c.state) === '完成'),
      JSON.stringify(stepCardsL),
    );

    // 每一步的文字要各占一张卡：不同请求会给出同样的分片 id，
    // 不分开就会把几步的文字并进同一张卡，步骤就认不出自己的产出。
    const textCardsL = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "Array.from(panel.querySelectorAll('.part')).filter(function (p) {" +
          'return p.querySelector(".badge.text");}).length',
      ),
    );
    check('每一步的文字各占一张卡', textCardsL === 3, `${textCardsL} 张文字卡`);

    // 「总耗时」是这一轮**所有步骤**的耗时之和：每一步是一次请求，
    // 各有各的耗时，不该每换一步就从头算。
    const durationsL = await cdp.evaluate(
      inPanel(
        PANEL_A,
        '(function () {' +
          "  var blocks = Array.from(panel.querySelectorAll('.run-block'));" +
          '  var stepDurations = blocks.map(function (b) {' +
          "    var d = b.querySelector('.run-duration');" +
          '    return d ? parseFloat(d.textContent) : null;' +
          '  }).filter(function (v) { return typeof v === "number" && !isNaN(v); });' +
          "  var meta = panel.querySelector('[data-role=metaList]');" +
          '  var total = null;' +
          '  if (meta) {' +
          "    var dts = meta.querySelectorAll('dt');" +
          "    var dds = meta.querySelectorAll('dd');" +
          '    for (var i = 0; i < dts.length; i += 1) {' +
          "      if (dts[i].textContent.indexOf('总耗时') >= 0) total = parseFloat(dds[i].textContent);" +
          '    }' +
          '  }' +
          '  return { steps: stepDurations, total: total };' +
          '})()',
      ),
    );
    const lastThree = (durationsL.steps || []).slice(-3);
    const stepSum = lastThree.reduce((sum, v) => sum + v, 0);
    check(
      '总耗时是这一轮各步耗时之和',
      Array.isArray(durationsL.steps) &&
        durationsL.steps.length >= 3 &&
        typeof durationsL.total === 'number' &&
        Math.abs(durationsL.total - stepSum) <= 0.25,
      `总耗时 ${durationsL.total} / 各步 ${JSON.stringify(durationsL.steps)}`,
    );

    const parsedL = postBodies
      .map((body) => {
        try {
          return JSON.parse(body);
        } catch {
          return null;
        }
      })
      .filter(Boolean);
    const stepReqsL = parsedL.filter(
      (b) => typeof b.prompt === 'string' && b.prompt.includes('【第 '),
    );
    check('每一步各发一次请求', stepReqsL.length === 3, `实际 ${stepReqsL.length} 次`);

    check(
      '第一步只发自己的指令，本轮输入走单独字段（由服务端拼一次）',
      stepReqsL.length > 0 &&
        stepReqsL[0].prompt.includes('第一步 查') &&
        !stepReqsL[0].prompt.includes('本轮的问题') &&
        typeof stepReqsL[0].roundPrompt === 'string' &&
        stepReqsL[0].roundPrompt.includes('本轮的问题'),
      stepReqsL.length ? stepReqsL[0].prompt.slice(0, 200) : '（没有请求）',
    );
    check(
      '第二步只发自己的指令：不重发本轮输入，也不贴前面步骤的产出',
      stepReqsL.length > 1 &&
        stepReqsL[1].prompt.includes('第二步 改') &&
        !stepReqsL[1].prompt.includes('本轮的问题') &&
        !stepReqsL[1].prompt.includes('【前面步骤的产出】'),
      stepReqsL.length > 1 ? stepReqsL[1].prompt.slice(0, 300) : '（没有请求）',
    );
    check(
      '第三步同样只发自己的指令',
      stepReqsL.length > 2 &&
        stepReqsL[2].prompt.includes('第三步 验') &&
        !stepReqsL[2].prompt.includes('本轮的问题') &&
        !stepReqsL[2].prompt.includes('【前面步骤的产出】'),
      stepReqsL.length > 2 ? stepReqsL[2].prompt.slice(0, 300) : '（没有请求）',
    );

    // 三步共用一个 pi 会话：请求里必须带同一个 roundId，服务端据此复用会话。
    const roundIds = stepReqsL
      .map((body) => (typeof body.roundId === 'string' ? body.roundId : ''))
      .filter((id) => id !== '');
    check(
      '三步属于同一个 roundId（共用一个 pi 会话）',
      roundIds.length === stepReqsL.length && new Set(roundIds).size === 1,
      `roundId: ${JSON.stringify(roundIds)}`,
    );

    check(
      '本轮输入只在第一步的请求里出现一次',
      stepReqsL.length > 2 &&
        typeof stepReqsL[0].roundPrompt === 'string' &&
        stepReqsL[0].roundPrompt.includes('本轮的问题') &&
        stepReqsL.slice(1).every((body) => body.roundPrompt === undefined),
      `各步 roundPrompt: ${JSON.stringify(stepReqsL.map((b) => b.roundPrompt))}`,
    );

    const handoffL = parsedL.filter((b) => typeof b.handoff === 'string' && b.handoff !== '');
    check(
      '整轮跑完才交接一次，交的是最后一步的产出',
      handoffL.length === 1 &&
        handoffL[0].handoff.includes('第三步 验') &&
        handoffL[0].handoffFrom === 'AGENT A',
      `交接请求 ${handoffL.length} 条`,
    );
    check(
      '交接发生在所有步骤之后',
      parsedL.length > 0 && parsedL[parsedL.length - 1] === handoffL[0],
      `一共 ${parsedL.length} 条请求，交接是第 ${parsedL.indexOf(handoffL[0]) + 1} 条`,
    );

    // 第二步失败：整轮停在那里 —— 第三步不跑，也不交接。
    runErrorHits = 0;
    interceptMode = 'fail-second-run';
    await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*' }] });
    postBodies.length = 0;
    await cdp.evaluate(submitPrompt(PANEL_A, '再来一轮'));

    const errStateL = await waitFor(
      cdp,
      inPanel(PANEL_A, "panel.querySelector('.dot.error') ? 'error' : ''"),
      120,
      250,
    );
    check('第二步失败时整轮停住并标成出错', errStateL === 'error', `状态 ${errStateL}`);

    const cardsAfterFail = await cdp.evaluate(lastRoundStepStates(PANEL_A));
    check(
      '失败后只画了两张步骤卡（第三步没有跑）',
      Array.isArray(cardsAfterFail) && cardsAfterFail.length === 2,
      JSON.stringify(cardsAfterFail),
    );
    check(
      '失败的那一步标成失败',
      Array.isArray(cardsAfterFail) && String(cardsAfterFail[1]).includes('失败'),
      JSON.stringify(cardsAfterFail),
    );

    // 失败之后再等一会儿：万一实现是"失败也照样交接"，请求会在这段时间里飞过来。
    await sleep(2000);

    const handoffAfterFail = postBodies
      .map((body) => {
        try {
          return JSON.parse(body);
        } catch {
          return null;
        }
      })
      .filter(Boolean)
      .filter((b) => typeof b.handoff === 'string' && b.handoff !== '');
    check(
      '失败之后不交接（没有半截结果发出去）',
      handoffAfterFail.length === 0,
      `交接请求 ${handoffAfterFail.length} 条`,
    );

    interceptMode = null;
    await cdp.send('Fetch.disable');

    mockServerL.closeAllConnections();
    await new Promise((resolve) => mockServerL.close(resolve));

    // ---------- 场景 M：权限 ----------
    process.stdout.write('\n  场景 M · 每个 agent 都能配自己的权限\n');

    const mockServerM = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
    await new Promise((resolve) => mockServerM.listen(0, '127.0.0.1', resolve));
    const mockBaseM = `http://127.0.0.1:${mockServerM.address().port}`;
    await openFresh(cdp, mockBaseM, { noSteps: true });
    await waitFor(
      cdp,
      "document.querySelectorAll('.panel [data-role=permissionList]').length === 3",
    );

    const permAreasM = await cdp.evaluate(
      "document.querySelectorAll('.panel [data-role=permissionList]').length",
    );
    check('每个面板都有权限的配置区', permAreasM === 3, `${permAreasM} 个`);

    const permBoxesM = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "Array.from(panel.querySelectorAll('[data-role=permissionList] input[type=checkbox]'))" +
          '.map(function (b) { return {id: b.value, checked: b.checked}; })',
      ),
    );
    check(
      '权限区按服务端的权限目录铺出复选框',
      Array.isArray(permBoxesM) && permBoxesM.length === 5,
      JSON.stringify(permBoxesM),
    );
    check(
      '权限目录里就是 read / write 这类基础权限',
      Array.isArray(permBoxesM) &&
        ['read', 'write', 'delete', 'execute', 'network'].every((id) =>
          permBoxesM.some((b) => b.id === id),
        ),
      JSON.stringify(permBoxesM),
    );
    check(
      'AGENT A 默认授了三项权限',
      Array.isArray(permBoxesM) &&
        permBoxesM.filter((b) => b.checked).length === 3 &&
        ['read', 'execute', 'network'].every((id) =>
          permBoxesM.some((b) => b.id === id && b.checked),
        ),
      JSON.stringify(permBoxesM),
    );
    check(
      'AGENT A 默认没有写权限',
      Array.isArray(permBoxesM) &&
        permBoxesM.some((b) => b.id === 'write' && b.checked === false),
      JSON.stringify(permBoxesM),
    );

    const permSummaryM = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelector('[data-role=permissionsSummary]').textContent"),
    );
    check(
      '权限区摘要写明授了几项',
      String(permSummaryM).includes('3'),
      String(permSummaryM),
    );

    // 勾上「写」：这一轮跑起来时请求里必须带上它。
    await cdp.evaluate(
      runInPanel(
        PANEL_A,
        "panel.querySelector('[data-role=permissionsBox]').open = true;" +
          "var b = panel.querySelector('[data-role=permissionList] input[value=\"write\"]');" +
          "b.checked = true;" +
          "b.dispatchEvent(new Event('change', {bubbles: true}));",
      ),
    );

    const summaryAfterM = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelector('[data-role=permissionsSummary]').textContent"),
    );
    check(
      '勾上写权限后摘要变成四项',
      String(summaryAfterM).includes('4'),
      String(summaryAfterM),
    );

    // 权限存在服务端：这一轮真正生效的权限由服务端按存下来的那份来定。
    // 所以查的是服务端存的那份与这一轮记下来的那份，不是请求体。
    const beforeRunM = runRequests.length;
    await cdp.evaluate(submitPrompt(PANEL_A, '跑一轮看看权限'));
    const recordsM1 = await waitForChatCount(mockBaseM, 'agent-a', 1);
    const permsM1 = (recordsM1[recordsM1.length - 1] || {}).permissions || [];
    check(
      '这一轮真正生效的权限里有勾上的写权限',
      Array.isArray(permsM1) && permsM1.includes('write'),
      JSON.stringify(permsM1),
    );
    check(
      '这一轮真正生效的权限就是页面上勾的那些',
      Array.isArray(permsM1) &&
        permsM1.length === 4 &&
        ['read', 'execute', 'network', 'write'].every((id) => permsM1.includes(id)),
      JSON.stringify(permsM1),
    );

    const healthM1 = await request(`${mockBaseM}/api/health`);
    const agentAM1 = (healthM1.agents || []).find((a) => a.id === 'agent-a');
    check(
      '勾上的写权限存进了服务端',
      Boolean(agentAM1) && agentAM1.permissions.includes('write'),
      JSON.stringify(agentAM1 && agentAM1.permissions),
    );

    const reqM = runRequests[beforeRunM] || runRequests[runRequests.length - 1];
    const idsM = requestedPermissions(reqM);
    check(
      '权限不再跟着请求发（服务端才是唯一的一份）',
      Array.isArray(idsM) && idsM.length === 0,
      JSON.stringify(idsM),
    );

    // 取消一项：下一轮真正生效的权限里就不该再有它。
    await cdp.evaluate(
      runInPanel(
        PANEL_A,
        "var b = panel.querySelector('[data-role=permissionList] input[value=\"read\"]');" +
          "b.checked = false;" +
          "b.dispatchEvent(new Event('change', {bubbles: true}));",
      ),
    );
    await cdp.evaluate(submitPrompt(PANEL_A, '再跑一轮'));
    const recordsM2 = await waitForChatCount(mockBaseM, 'agent-a', 2);
    const permsM2 = (recordsM2[recordsM2.length - 1] || {}).permissions || [];
    check(
      '取消掉的权限不再生效',
      Array.isArray(permsM2) && !permsM2.includes('read'),
      JSON.stringify(permsM2),
    );
    check(
      '没取消的权限还在',
      Array.isArray(permsM2) && permsM2.includes('write') && permsM2.includes('network'),
      JSON.stringify(permsM2),
    );

    mockServerM.closeAllConnections();
    await new Promise((resolve) => mockServerM.close(resolve));

    // ---------- 场景 N：挂在路径前缀下（公网反代的访问门槛）----------
    process.stdout.write('\n  场景 N · 路径前缀下的访问门槛\n');

    const gatedServer = createServer({ mode: 'mock', publicPath: 'test-token', dataDir: SCRATCH_DATA });
    await new Promise((resolve) => gatedServer.listen(0, '127.0.0.1', resolve));
    const gatedBase = `http://127.0.0.1:${gatedServer.address().port}`;

    // 裸地址进不来。
    await openPage(cdp, gatedBase);
    await waitFor(cdp, "document.body ? 'ok' : ''", 40, 200);
    const bareHasPanel = await cdp.evaluate(
      "document.getElementById('panelA') ? 'yes' : 'no'",
    );
    check('没带前缀的地址看不到应用', bareHasPanel === 'no', String(bareHasPanel));

    // 带前缀的地址正常。
    await openPage(cdp, `${gatedBase}/test-token/`);
    const bootedN = await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
    check(
      '带前缀时前端照样启动',
      bootedN === config.FRONTEND_VERSION,
      `拿到 ${bootedN}`,
    );

    const beforeRunN = runRequests.length;
    await cdp.evaluate(submitPrompt(PANEL_A, '在前缀下跑一轮'));
    await waitFor(
      cdp,
      inPanel(PANEL_A, "panel.querySelectorAll('.part').length > 0 ? 'ok' : ''"),
      120,
      250,
    );
    const reqN = runRequests[beforeRunN] || runRequests[runRequests.length - 1];
    check(
      '请求走的是前缀下的路径',
      Boolean(reqN) && new URL(reqN.url).pathname.startsWith('/test-token/api/run'),
      String(reqN && reqN.url),
    );

    gatedServer.closeAllConnections();
    await new Promise((resolve) => gatedServer.close(resolve));

    // ---------- 场景 O：步骤进度按配好的步骤走 ----------
    process.stdout.write('\n  场景 O · 步骤进度显示配好的步骤\n');

    const mockServerO = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
    await new Promise((resolve) => mockServerO.listen(0, '127.0.0.1', resolve));
    const mockBaseO = `http://127.0.0.1:${mockServerO.address().port}`;
    await openFresh(cdp, mockBaseO);
    await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
    await waitFor(
      cdp,
      "document.querySelectorAll('.panel [data-role=stepBar]').length === 3",
    );

    const metaFieldsO = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "Array.from(panel.querySelectorAll('[data-role=metaList] dt')).map(function (dt) {" +
          "  return dt.textContent.trim();" +
          "})",
      ),
    );
    check(
      '运行信息里只剩总耗时',
      Array.isArray(metaFieldsO) &&
        metaFieldsO.length === 1 &&
        metaFieldsO[0] === '总耗时',
      JSON.stringify(metaFieldsO),
    );

    const stepChipsO = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "Array.from(panel.querySelectorAll('[data-role=stepBar] [data-step-chip]')).map(function (c) {" +
          "  return {text: c.textContent.trim(), state: c.dataset.state};" +
          "})",
      ),
    );
    // 期望值取自 AGENT A 面板自己的步骤配置，不写死标题：默认步骤改过一次，
    // 写死的标题会把"按配置铺开"这条检查变成检查一份过期的副本。
    const configuredStepsO = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "Array.from(panel.querySelectorAll('.step-title')).map(function (t) { return t.value; })",
      ),
    );
    check(
      '步骤进度按配好的四个步骤铺开',
      Array.isArray(stepChipsO) &&
        stepChipsO.length === configuredStepsO.length &&
        configuredStepsO.every(
          (title, i) => stepChipsO[i] && stepChipsO[i].text.includes(title),
        ),
      `chips ${JSON.stringify(stepChipsO)} / 配置 ${JSON.stringify(configuredStepsO)}`,
    );
    check(
      '还没开跑时四个步骤都是待开始',
      Array.isArray(stepChipsO) &&
        stepChipsO.every((c) => c.state === 'pending'),
      JSON.stringify(stepChipsO),
    );

    // 挂起第一个请求：这时第一步正在做，后面几步还没开始。
    interceptMode = 'hold-first-run';
    holdRunId = null;
    await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*' }] });
    await cdp.evaluate(submitPrompt(PANEL_A, '跑一轮看步骤进度'));

    const midStatesO = await waitFor(
      cdp,
      inPanel(
        PANEL_A,
        "(function () {" +
          "  var chips = Array.from(panel.querySelectorAll('[data-role=stepBar] [data-step-chip]'));" +
          "  if (chips.length < 4) return '';" +
          "  if (chips[0].dataset.state !== 'active') return '';" +
          "  return chips.map(function (c) { return c.dataset.state; }).join(',');" +
          "})()",
      ),
      120,
      250,
    );
    check(
      '正在做的那一步标成进行中，后面的还没做',
      midStatesO === 'active,pending,pending,pending',
      String(midStatesO),
    );

    // 放行，让整轮跑完。
    if (holdRunId !== null) {
      await cdp.send('Fetch.continueRequest', { requestId: holdRunId });
      holdRunId = null;
    }
    interceptMode = null;
    await cdp.send('Fetch.disable');

    const finalStatesO = await waitFor(
      cdp,
      inPanel(
        PANEL_A,
        "(function () {" +
          "  var chips = Array.from(panel.querySelectorAll('[data-role=stepBar] [data-step-chip]'));" +
          "  if (chips.length < 4) return '';" +
          "  var states = chips.map(function (c) { return c.dataset.state; });" +
          "  return states.every(function (s) { return s === 'filled' || s === 'failed';" +
          "  }) ? states.join(',') : '';" +
          "})()",
      ),
      200,
      250,
    );
    check(
      '跑完之后每个步骤都有结果（完成或失败）',
      finalStatesO === 'filled,filled,filled,filled',
      String(finalStatesO),
    );

    mockServerO.closeAllConnections();
    await new Promise((resolve) => mockServerO.close(resolve));

    // ---------- 场景 P：左侧 agent 列表，右侧只显示当前这个 ----------
    process.stdout.write('\n  场景 P · 侧边栏列出所有 agent，右侧只显示当前这个\n');

    const mockServerP = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
    await new Promise((resolve) => mockServerP.listen(0, '127.0.0.1', resolve));
    const mockBaseP = `http://127.0.0.1:${mockServerP.address().port}`;
    await openFresh(cdp, mockBaseP);
    await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
    // 加两个 agent：三个内置 + 两个新建，共五个，用来确认侧边栏把它们都列出来了。
    await cdp.evaluate("document.querySelector('.new-agent').click();");
    await sleep(300);
    await cdp.evaluate("document.querySelector('.new-agent').click();");
    await sleep(600);

    const layoutP = await cdp.evaluate(
      "(function () {" +
        "  var items = Array.from(document.querySelectorAll('[data-role=agentList] [data-agent-id]'));" +
        "  var panels = Array.from(document.querySelectorAll('.layout > .panel'));" +
        "  var visible = panels.filter(function (p) { return !p.hidden; });" +
        "  var rects = visible.map(function (p) { var r = p.getBoundingClientRect(); return r; });" +
        "  var overlap = 0;" +
        "  for (var a = 0; a < rects.length; a++) {" +
        "    for (var b = a + 1; b < rects.length; b++) {" +
        "      var ox = Math.min(rects[a].right, rects[b].right) - Math.max(rects[a].left, rects[b].left);" +
        "      var oy = Math.min(rects[a].bottom, rects[b].bottom) - Math.max(rects[a].top, rects[b].top);" +
        "      if (ox > 0 && oy > 0) overlap++;" +
        "    }" +
        "  }" +
        "  var clipped = visible.filter(function (p) {" +
        "    var r = p.getBoundingClientRect();" +
        "    var meta = p.querySelector('.meta').getBoundingClientRect();" +
        "    return meta.bottom > r.bottom + 1;" +
        "  }).length;" +
        "  return {" +
        "    sidebarCount: items.length," +
        "    panelCount: panels.length," +
        "    visibleCount: visible.length," +
        "    visibleId: visible.length ? visible[0].dataset.profile : ''," +
        "    overlap: overlap," +
        "    clipped: clipped," +
        "  };" +
        "})()",
    );
    check('侧边栏把五个 agent 都列出来了', layoutP.sidebarCount === 5, JSON.stringify(layoutP));
    check('页面上五个 agent 的面板都在', layoutP.panelCount === 5, JSON.stringify(layoutP));
    check(
      '右侧只显示当前这一个 agent 的聊天框',
      layoutP.visibleCount === 1,
      JSON.stringify(layoutP),
    );
    check(
      '可见的面板之间不重叠（不再互相挡住）',
      layoutP.overlap === 0 && layoutP.clipped === 0,
      JSON.stringify(layoutP),
    );

    mockServerP.closeAllConnections();
    await new Promise((resolve) => mockServerP.close(resolve));

    // ---------- 场景 Q：每个 agent 有独立的配置页 ----------
    process.stdout.write('\n  场景 Q · 每个 agent 有独立的配置页\n');

    const mockServerQ = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
    await new Promise((resolve) => mockServerQ.listen(0, '127.0.0.1', resolve));
    const mockBaseQ = `http://127.0.0.1:${mockServerQ.address().port}`;
    await openFresh(cdp, mockBaseQ);
    await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
    await waitFor(
      cdp,
      "document.querySelectorAll('[data-role=agentList] [data-agent-id]').length === 3",
    );

    const sideQ = await cdp.evaluate(
      "Array.from(document.querySelectorAll('[data-role=agentList] [data-agent-id]')).map(function (li) {" +
        "  return li.dataset.agentId + ':' + li.textContent.trim();}).join(' | ')",
    );
    check(
      '侧边栏列出 AGENT A 与 AGENT B',
      sideQ.includes('agent-a') && sideQ.includes('agent-b') &&
        sideQ.includes('AGENT A') && sideQ.includes('AGENT B'),
      sideQ,
    );

    // 主页面不该直接铺开完整配置：AGENT A 的聊天视图里看不到配置区。
    const chatViewQ = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "(function () {" +
          "  var cfg = panel.querySelector('.config-part');" +
          "  var chat = panel.querySelector('.chat-part');" +
          "  var cfgShown = cfg && getComputedStyle(cfg).display !== 'none';" +
          "  var chatShown = chat && getComputedStyle(chat).display !== 'none';" +
          "  return {cfgShown: !!cfgShown, chatShown: !!chatShown};" +
          "})()",
      ),
    );
    check(
      '聊天视图里不显示配置区，只显示聊天框',
      chatViewQ.chatShown === true && chatViewQ.cfgShown === false,
      JSON.stringify(chatViewQ),
    );

    // 点侧边栏里的 AGENT B：右侧换成 B 的聊天框，地址里也记着是哪个 agent。
    await cdp.evaluate(
      "document.querySelector('[data-role=agentList] [data-agent-id=\"agent-b\"] [data-role=pickAgent]').click();",
    );
    await sleep(300);
    const switchedQ = await cdp.evaluate(
      "(function () {" +
        "  var panels = Array.from(document.querySelectorAll('.layout > .panel'));" +
        "  var visible = panels.filter(function (p) { return !p.hidden; });" +
        "  return {" +
        "    visibleId: visible.length ? visible[0].dataset.profile : ''," +
        "    count: visible.length," +
        "    hash: location.hash," +
        "  };" +
        "})()",
    );
    check(
      '点侧边栏后右侧换成 AGENT B 的聊天框',
      switchedQ.visibleId === 'agent-b' && switchedQ.count === 1,
      JSON.stringify(switchedQ),
    );
    check('地址里记着当前看的是哪个 agent', switchedQ.hash.includes('agent-b'), switchedQ.hash);

    // 进配置页：配置区显示，聊天框不显示。
    await cdp.evaluate(
      "document.querySelector('[data-role=agentList] [data-agent-id=\"agent-b\"] [data-role=sideConfig]').click();",
    );
    await sleep(300);
    const configQ = await cdp.evaluate(
      inPanel(
        PANEL_B,
        "(function () {" +
          "  var cfg = panel.querySelector('.config-part');" +
          "  var chat = panel.querySelector('.chat-part');" +
          "  return {" +
          "    cfgShown: !!(cfg && getComputedStyle(cfg).display !== 'none')," +
          "    chatShown: !!(chat && getComputedStyle(chat).display !== 'none')," +
          "    hasInstructions: !!panel.querySelector('[data-role=instructionsBox]')," +
          "    hasPermissions: !!panel.querySelector('[data-role=permissionsBox]')," +
          "    hasHandoff: !!panel.querySelector('[data-role=handoffBox]')," +
          "    hasSteps: !!panel.querySelector('[data-role=stepsBox]')," +
          "  };" +
          "})()",
      ),
    );
    check(
      '配置页显示配置区、藏起聊天框',
      configQ.cfgShown === true && configQ.chatShown === false,
      JSON.stringify(configQ),
    );
    check(
      '配置页里四块配置都在（提示词 / 权限 / 交接 / 步骤）',
      configQ.hasInstructions && configQ.hasPermissions && configQ.hasHandoff && configQ.hasSteps,
      JSON.stringify(configQ),
    );

    // 回聊天：又变成聊天视图。
    await cdp.evaluate(
      inPanel(PANEL_B, "panel.querySelector('[data-role=backToChat]').click();"),
    );
    await sleep(300);
    const backQ = await cdp.evaluate(
      inPanel(
        PANEL_B,
        "(function () {" +
          "  var cfg = panel.querySelector('.config-part');" +
          "  var chat = panel.querySelector('.chat-part');" +
          "  return {" +
          "    cfgShown: !!(cfg && getComputedStyle(cfg).display !== 'none')," +
          "    chatShown: !!(chat && getComputedStyle(chat).display !== 'none')," +
          "  };" +
          "})()",
      ),
    );
    check(
      '点「回到聊天」又回到聊天视图',
      backQ.chatShown === true && backQ.cfgShown === false,
      JSON.stringify(backQ),
    );

    mockServerQ.closeAllConnections();
    await new Promise((resolve) => mockServerQ.close(resolve));

    // ---------- 场景 F：服务端返回 400 时，页面给出可读提示 ----------
    process.stdout.write('\n  场景 F · 服务端返回 400 时，页面给出可读提示\n');
    // 场景 K 里把面板删掉了，这一段要重新开一页：拿一个两个面板都在的页面。
    await openFresh(cdp, base, { noSteps: true });
    interceptMode = 'run-error';
    await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*' }] });

    await cdp.evaluate(submitPrompt(PANEL_A, '触发一次 400'));
    const errText = await waitFor(cdp, panelError(PANEL_A), 60, 200);
    check(
      '页面显示服务端给的可读提示，而不是干巴巴的 "HTTP 400"',
      typeof errText === 'string' &&
        errText.includes('未知的 agent 身份') &&
        !errText.includes('HTTP 400'),
      `${errText}`,
    );

    interceptMode = null;
    await cdp.send('Fetch.disable');

    // ---------- 场景 E：浏览器里跑的是旧版 JS ----------
    process.stdout.write('\n  场景 E · app.js 是旧版本（模拟浏览器缓存了旧版）\n');
    interceptMode = 'stale-appjs';
    appJsHits = 0;
    await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*' }] });

    await resetServerState(base);
    await openPage(cdp, base);
    // 只等"发生重载"：旧版 app.js 照样会设置启动标记，若把"已启动"也算作
    // 等待成功，这一轮会在守卫的 800ms 定时器触发前就返回，测不到自愈。
    const selfHealed = await waitFor(
      cdp,
      "location.href.includes('r=') ? 'reloading' : ''",
      60,
      200,
    );
    check('守卫察觉旧版 JS 并换了地址重载', selfHealed === 'reloading', `状态 ${selfHealed}`);

    const rebooted = await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null', 60, 200);
    check('重载后新前端正常启动', rebooted === config.FRONTEND_VERSION, `拿到 ${rebooted}`);

    const expected = await cdp.evaluate('window.__AGENT_VIS_EXPECTED__ || null');
    check(
      '页面里注入的期望版本与服务端一致',
      expected === config.FRONTEND_VERSION,
      `页面 ${expected} / 服务端 ${config.FRONTEND_VERSION}`,
    );

    interceptMode = null;
    await cdp.send('Fetch.disable');

    // ---------- 场景 R：app.js 迟到时守卫不能误判成旧版 ----------
    process.stdout.write('\n  场景 R · app.js 迟到时守卫不误判成旧版\n');
    interceptMode = 'slow-appjs';
    appJsHits = 0;
    await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*' }] });

    await resetServerState(base);
    await openPage(cdp, base);
    const lateBoot = await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null', 120, 200);
    check('迟到的 app.js 照样启动成功', lateBoot === config.FRONTEND_VERSION, `拿到 ${lateBoot}`);

    const lateHref = await cdp.evaluate('location.href');
    check(
      '守卫没把"还没到"当成"旧版本"（地址里没有 r= 参数）',
      !lateHref.includes('r='),
      lateHref,
    );

    interceptMode = null;
    await cdp.send('Fetch.disable');

    // ---------- 场景 T：配置存在服务端 ----------
    process.stdout.write('\n  场景 T · 配置存在服务端，刷新与重启之后都还在\n');

    const mockServerT = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
    await new Promise((resolve) => mockServerT.listen(0, '127.0.0.1', resolve));
    const mockBaseT = `http://127.0.0.1:${mockServerT.address().port}`;
    await openFresh(cdp, mockBaseT);

    // 改提示词 + 勾一项权限。
    //
    // 这两样都不再跟着请求发：它们存在服务端，服务端按自己存的那份来跑。
    const CUSTOM_T = '存在服务端的提示词：刷新与重启之后都该还在。';
    await cdp.evaluate(
      runInPanel(
        PANEL_A,
        "var t = panel.querySelector('[data-role=instructions]');" +
          `t.value = ${JSON.stringify(CUSTOM_T)};` +
          "t.dispatchEvent(new Event('input', {bubbles:true}));",
      ),
    );
    await cdp.evaluate(
      runInPanel(
        PANEL_A,
        "var b = panel.querySelector('[data-role=permissionList] input[value=\"write\"]');" +
          'b.checked = true;' +
          "b.dispatchEvent(new Event('change', {bubbles:true}));",
      ),
    );

    // 页面上的改动是攒一会儿才发出去的：等它真的落到服务端再断言。
    const storedT = await waitForServer(async () => {
      const health = await request(`${mockBaseT}/api/health`);
      const agent = (health.agents || []).find((a) => a.id === 'agent-a');
      if (!agent) return null;
      if (agent.instructions !== CUSTOM_T) return null;
      if (!Array.isArray(agent.permissions) || !agent.permissions.includes('write')) return null;
      return agent;
    });
    check(
      '改过的提示词与权限都落到了服务端',
      Boolean(storedT),
      storedT ? JSON.stringify(storedT.permissions) : '服务端还没存上',
    );

    // 刷新页面：配置得自己从服务端回来。
    await openPage(cdp, mockBaseT);
    await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
    // 配置是 /api/health 回来之后才铺上去的：等它铺好再断言。
    await waitFor(
      cdp,
      inPanel(
        PANEL_A,
        "(function () {" +
          "var t = panel.querySelector('[data-role=instructions]');" +
          "var b = panel.querySelector('[data-role=permissionList] input[value=\"write\"]');" +
          "return t && t.value.length > 0 && b ? 'ok' : '';" +
          '})()',
      ),
      120,
      250,
    );
    const shownT = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelector('[data-role=instructions]').value"),
    );
    check('刷新页面后提示词还在', shownT === CUSTOM_T, String(shownT).slice(0, 40));
    const permT = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "(function () {" +
          "var b = panel.querySelector('[data-role=permissionList] input[value=\"write\"]');" +
          'return b ? b.checked : null;' +
          '})()',
      ),
    );
    check('刷新页面后勾上的权限还在', permT === true, String(permT));

    // 服务重启：身份与配置都从盘上读回来。
    mockServerT.closeAllConnections();
    await new Promise((resolve) => mockServerT.close(resolve));
    const mockServerT2 = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
    await new Promise((resolve) => mockServerT2.listen(0, '127.0.0.1', resolve));
    const mockBaseT2 = `http://127.0.0.1:${mockServerT2.address().port}`;
    await openPage(cdp, mockBaseT2);
    await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
    await waitFor(
      cdp,
      inPanel(
        PANEL_A,
        "(function () {" +
          "var t = panel.querySelector('[data-role=instructions]');" +
          "var b = panel.querySelector('[data-role=permissionList] input[value=\"write\"]');" +
          "return t && t.value.length > 0 && b ? 'ok' : '';" +
          '})()',
      ),
      120,
      250,
    );
    const shownT2 = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelector('[data-role=instructions]').value"),
    );
    check('服务重启之后提示词还在', shownT2 === CUSTOM_T, String(shownT2).slice(0, 40));
    const permT2 = await cdp.evaluate(
      inPanel(
        PANEL_A,
        "(function () {" +
          "var b = panel.querySelector('[data-role=permissionList] input[value=\"write\"]');" +
          'return b ? b.checked : null;' +
          '})()',
      ),
    );
    check('服务重启之后勾上的权限还在', permT2 === true, String(permT2));

    mockServerT2.closeAllConnections();
    await new Promise((resolve) => mockServerT2.close(resolve));

    // ---------- 场景 U：对话记录存在服务端 ----------
    process.stdout.write('\n  场景 U · 对话记录存在服务端，刷新之后还在\n');

    const mockServerU = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
    await new Promise((resolve) => mockServerU.listen(0, '127.0.0.1', resolve));
    const mockBaseU = `http://127.0.0.1:${mockServerU.address().port}`;
    await openFresh(cdp, mockBaseU, { noSteps: true });

    // 跑一轮。
    await cdp.evaluate(submitPrompt(PANEL_A, '跑一轮留个记录'));
    const recordsU = await waitForChatCount(mockBaseU, 'agent-a', 1);
    check('这一轮被记到了服务端', recordsU.length >= 1, `${recordsU.length} 条`);
    const recordU = recordsU[recordsU.length - 1] || {};
    check(
      '记下来的这一轮跑完了',
      recordU.status === 'done',
      `状态 ${recordU.status}`,
    );
    const expectedTextU = recordAnswerText(recordU);
    check('记下来的这一轮有文字产出', expectedTextU.length > 0, `${expectedTextU.length} 字`);

    // 刷新页面：时间轴上那一轮还在，而且内容跟当时跑出来的一样。
    await openPage(cdp, mockBaseU);
    await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
    const replayedU = await waitFor(
      cdp,
      inPanel(PANEL_A, "panel.querySelectorAll('.run-block').length > 0 ? 'ok' : ''"),
      120,
      250,
    );
    check('刷新页面后那一轮回到了时间轴上', replayedU === 'ok', `状态 ${replayedU}`);
    const blocksU = await cdp.evaluate(
      inPanel(PANEL_A, "panel.querySelectorAll('.run-block').length"),
    );
    check('刷新页面后时间轴上正好是那一轮', blocksU === 1, `${blocksU} 块`);
    const shownTextU = await cdp.evaluate(panelAnswerText(PANEL_A));
    check(
      '刷新之后那一轮的内容跟当时跑出来的一样',
      sameText(shownTextU, expectedTextU),
      `页面 ${String(shownTextU).slice(0, 40)} / 记录 ${String(expectedTextU).slice(0, 40)}`,
    );
    const promptU = await cdp.evaluate(
      inPanel(PANEL_A, "(panel.querySelector('.run-prompt') || {}).textContent || ''"),
    );
    check(
      '刷新之后那一轮的提问也还在',
      promptU.includes('跑一轮留个记录'),
      String(promptU).slice(0, 40),
    );

    // 「清空历史」：服务端那份也要跟着清 —— 只清页面的话，刷新它又回来了。
    await cdp.evaluate(
      runInPanel(PANEL_A, "panel.querySelector('[data-role=clearHistory]').click();"),
    );
    const clearedU = await waitForServer(async () => {
      const res = await request(`${mockBaseU}/api/agents/agent-a/chats`);
      return (res.records || []).length === 0;
    });
    check('「清空历史」把服务端那份也清掉了', clearedU === true, String(clearedU));
    // 页面上的时间轴是服务端那份删掉之后才清的：等它清完再数。
    const clearedDomU = await waitFor(
      cdp,
      inPanel(PANEL_A, "panel.querySelectorAll('.run-block').length === 0 ? 'ok' : ''"),
      60,
      250,
    );
    check('「清空历史」把页面上的时间轴也清空了', clearedDomU === 'ok', `状态 ${clearedDomU}`);

    // 刷新之后它不该再回来。
    await openPage(cdp, mockBaseU);
    await waitFor(cdp, 'window.__AGENT_VIS_FRONTEND__ || null');
    const blocksReloadU = await waitFor(
      cdp,
      inPanel(PANEL_A, "panel.querySelectorAll('.run-block').length === 0 ? 'ok' : ''"),
      60,
      250,
    );
    check('刷新之后清掉的历史没有回来', blocksReloadU === 'ok', `状态 ${blocksReloadU}`);

    mockServerU.closeAllConnections();
    await new Promise((resolve) => mockServerU.close(resolve));

    // ---------- 场景 V：全局设置（运行环境主机与工作目录） ----------
    process.stdout.write('\n  场景 V · 全局设置面板改运行环境\n');
    const mockServerV = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH_DATA });
    await new Promise((resolve) => mockServerV.listen(0, '127.0.0.1', resolve));
    const mockBaseV = `http://127.0.0.1:${mockServerV.address().port}`;
    await openFresh(cdp, mockBaseV);

    // 侧栏底部一直写着当前生效的主机与工作目录。
    const sideV = await cdp.evaluate("document.getElementById('settingsMeta').textContent");
    check(
      '侧栏底部显示当前运行环境',
      typeof sideV === 'string' && sideV.includes('/') && sideV.length > 3,
      sideV,
    );

    await cdp.evaluate(
      "(function () {" +
        "var b = document.querySelector('[data-role=settingsBtn]');" +
        'if (b) b.click();' +
        '})()',
    );
    const dialogV = await waitFor(
      cdp,
      "document.querySelector('[data-role=settingsDialog]').open ? 'open' : ''",
      60,
      200,
    );
    check('点一下就打开全局设置', dialogV === 'open', `状态 ${dialogV}`);

    const fieldsV = await cdp.evaluate(
      '(function () {' +
        "  var h = document.querySelector('[data-role=settingsHost]');" +
        "  var w = document.querySelector('[data-role=settingsWorkdir]');" +
        '  return { host: h ? h.value : "", workdir: w ? w.value : "" };' +
        '})()',
    );
    check(
      '对话框里填着当前生效的主机与工作目录',
      typeof fieldsV.host === 'string' &&
        fieldsV.host.length > 0 &&
        String(fieldsV.workdir).startsWith('/'),
      JSON.stringify(fieldsV),
    );

    // 改成一个这台机器不是的主机：保存后应当立刻说出"跑不了"。
    await cdp.evaluate(
      '(function () {' +
        "  var h = document.querySelector('[data-role=settingsHost]');" +
        "  h.value = 'm3g999';" +
        "  h.dispatchEvent(new Event('input', {bubbles:true}));" +
        "  var b = document.querySelector('[data-role=settingsSave]');" +
        '  if (b) b.click();' +
        '})()',
    );
    const stateV = await waitFor(
      cdp,
      "document.querySelector('[data-role=settingsState]').textContent.indexOf('m3g999') >= 0 ? 'bad' : ''",
      60,
      200,
    );
    check('改成一个不对的主机后立刻报出来', stateV === 'bad', `状态 ${stateV}`);

    const sideAfterV = await cdp.evaluate("document.getElementById('settingsMeta').textContent");
    check(
      '侧栏底部跟着新设置走',
      typeof sideAfterV === 'string' && sideAfterV.includes('m3g999'),
      sideAfterV,
    );

    // 改回来，后面的场景还要在这个环境里跑。
    await cdp.evaluate(
      '(function () {' +
        "  var h = document.querySelector('[data-role=settingsHost]');" +
        `  h.value = ${JSON.stringify(fieldsV.host)};` +
        "  h.dispatchEvent(new Event('input', {bubbles:true}));" +
        "  var b = document.querySelector('[data-role=settingsSave]');" +
        '  if (b) b.click();' +
        '})()',
    );
    const okV = await waitFor(
      cdp,
      "document.querySelector('[data-role=settingsState]').textContent.indexOf('环境正常') >= 0 ? 'ok' : ''",
      60,
      200,
    );
    check('改回正确的主机后恢复正常', okV === 'ok', `状态 ${okV}`);

    mockServerV.closeAllConnections();
    await new Promise((resolve) => mockServerV.close(resolve));

    ws.close();
  } finally {
    // 先让浏览器自己退出：CDP 的 Browser.close 会把 chromium 及其所有
    // 子进程一并带走。注意不能 await —— 浏览器一关，连接就断了，
    // 那个 promise 可能永远不 settle，await 会把整个测试卡死在这里。
    if (cdp) {
      try {
        cdp.send('Browser.close').catch(() => {});
      } catch {
        // 浏览器可能已经自己退出了。
      }
    }
    await sleep(500);
    // chromium 会 fork 出脱离原进程组的子进程（crashpad 等），
    // 只 kill 进程组清不干净，所以再按调试端口特征扫一遍。
    try {
      execFileSync('pkill', ['-9', '-f', `remote-debugging-port=${CDP_PORT}`], {
        stdio: 'ignore',
      });
    } catch {
      // 没有匹配的进程时会返回非零，忽略即可。
    }
    try {
      process.kill(-chrome.pid, 'SIGKILL');
    } catch {
      // 已经退出了就不用管。
    }
    // 浏览器还留着 keep-alive 连接，server.close() 会一直等它们 ——
    // 先把连接全断了，再关服务，否则测试跑完却退不出来。
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }

  process.stdout.write(`\n${passed} 通过, ${failures.length} 失败\n`);
  if (failures.length > 0) {
    for (const { name, detail } of failures) {
      process.stdout.write(`\n--- ${name} ---\n${detail}\n`);
    }
    process.exitCode = 1;
  }
}

main().catch((err) => {
  process.stderr.write(`${err && err.stack}\n`);
  process.exit(1);
});
