import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * pi agent 接入的离线检查。
 *
 * 这里检查三件不依赖网络的事：
 *   1. 权限 → 工具表的裁剪（没授的工具不能出现在 pi 的工具列表里）；
 *   2. 工具实现自己的边界（越界、.git、删根目录）；
 *   3. pi 事件流 → UI message stream 分片的翻译（文本 / 工具 / usage / abort / 错误）。
 *
 * 真实 pi 会话（真的打模型、真的执行工具）在 test/browser.js 里跑（真浏览器 + 真模型）。
 */

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SCRATCH = path.join(HERE, '..', '.out', 'pi-test');

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

const {
  PI_TOOLS,
  TOOL_IMPLS,
  acquireRoundSession,
  buildSystemPromptParts,
  disposeRoundSession,
  roundSessionCount,
  runLiveChunks,
  sweepRoundSessions,
  toolsForPermissions,
  translateSession,
} = await import('../live-agent.js');
const { REPO_ROOT } = await import('../config.js');

async function collect(stream) {
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return chunks;
}

/**
 * 假的 pi 会话：只实现 translateSession 用到的那几个方法。
 *
 * script 里的每条事件都会在 prompt() 时按顺序发给订阅者 —— 与真实 pi 一样，
 * 事件是推给订阅者的，不是返回值。
 */
function createFakeSession(script, { promptError = null, hangUntilAbort = false } = {}) {
  const listeners = new Set();
  const calls = { disposed: 0, aborted: 0, prompts: [] };
  let release = null;
  return {
    calls,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    emit(event) {
      for (const listener of listeners) listener(event);
    },
    async prompt(prompt) {
      calls.prompts.push(prompt);
      for (const event of script) this.emit(event);
      if (promptError) throw promptError;
      if (hangUntilAbort) {
        await new Promise((resolve) => {
          release = resolve;
        });
      }
    },
    async abort() {
      calls.aborted += 1;
      if (release) release();
    },
    dispose() {
      calls.disposed += 1;
    },
  };
}

function assistantMessageUpdate(event) {
  return { type: 'message_update', assistantMessageEvent: event };
}

process.stdout.write('agent-vis pi agent 离线检查\n\n');

// ---------- 权限 → pi 工具表 ----------
process.stdout.write('  权限裁表\n');

await test('权限目录里的每个工具名都能在 pi 工具表里找到', async () => {
  const { PERMISSION_CATALOG } = await import('../config.js');
  for (const item of PERMISSION_CATALOG) {
    for (const name of item.tools) {
      assert.ok(PI_TOOLS[name], `权限 ${item.id} 指向了不存在的 pi 工具 ${name}`);
    }
  }
});

await test('read 权限只给只读工具（含 JSON 校验）', async () => {
  assert.deepEqual(
    toolsForPermissions(['read']).map((tool) => tool.name),
    ['listFiles', 'readFile', 'searchFiles', 'checkJson', 'checkIdea'],
  );
});

await test('没授权限就一个工具都不给', async () => {
  assert.deepEqual(toolsForPermissions([]), []);
});

await test('多项权限按请求顺序合并去重', async () => {
  const names = toolsForPermissions(['network', 'read', 'read']).map((tool) => tool.name);
  assert.deepEqual(names, [
    'listIssues',
    'readIssue',
    'listFiles',
    'readFile',
    'searchFiles',
    'checkJson',
    'checkIdea',
  ]);
});

await test('不认识的权限直接报错，不静默丢掉', async () => {
  assert.throws(() => toolsForPermissions(['fly']), /fly/);
});

await test('系统提示词里挂了中文推理约束，清空人设也不丢', async () => {
  const profile = { instructions: '你是 ISSUE_EXECUTOR。' };
  const parts = buildSystemPromptParts(profile, undefined, ['read']);
  assert.ok(parts.includes('你是 ISSUE_EXECUTOR。'), '默认人设丢了');
  assert.ok(
    parts.some((part) => part.includes('简体中文')),
    '没有要求用中文推理',
  );

  const cleared = buildSystemPromptParts(profile, '', []);
  assert.ok(!cleared.includes('你是 ISSUE_EXECUTOR。'), '清空人设没生效');
  assert.ok(
    cleared.some((part) => part.includes('简体中文')),
    '清空人设后把语言约束也丢了',
  );
  assert.ok(
    cleared.some((part) => part.includes('没有')),
    '清空人设后权限说明丢了',
  );
});

// ---------- 工具实现的边界 ----------
process.stdout.write('\n  工具边界\n');


fs.mkdirSync(path.join(SCRATCH, 'sub'), { recursive: true });
const relScratch = path.relative(REPO_ROOT, SCRATCH);
fs.writeFileSync(path.join(SCRATCH, 'a.txt'), 'line one\nline two\n');
fs.writeFileSync(path.join(SCRATCH, 'sub', 'b.txt'), 'nested needle\n');

await test('readFile 读到真实内容并按行截断', async () => {
  const result = await TOOL_IMPLS.readFile({ path: `${relScratch}/a.txt`, maxLines: 1 });
  assert.equal(result.content, 'line one');
  assert.equal(result.truncated, true);
  assert.equal(result.totalLines, 3);
});

await test('readFile 越出仓库根直接报错', async () => {
  await assert.rejects(() => TOOL_IMPLS.readFile({ path: '../outside.txt' }), /越界/);
});

await test('listFiles 跳过大目录，给出条目与大小', async () => {
  const result = await TOOL_IMPLS.listFiles({ dir: relScratch });
  assert.equal(result.count, result.items.length);
  const names = result.items.map((item) => item.name);
  assert.ok(names.includes('a.txt'));
  assert.ok(!names.includes('node_modules'));
  const file = result.items.find((item) => item.name === 'a.txt');
  assert.equal(file.type, 'file');
  assert.equal(typeof file.size, 'number');
});

await test('searchFiles 命中带文件名与行号', async () => {
  const result = await TOOL_IMPLS.searchFiles({ pattern: 'needle', dir: relScratch });
  assert.equal(result.matchCount, 1);
  assert.equal(result.matches[0].line, 1);
  assert.ok(result.matches[0].file.endsWith('b.txt'));
});

await test('writeFile 写进仓库并报告字节数', async () => {
  const result = await TOOL_IMPLS.writeFile({ path: `${relScratch}/new.txt`, content: '你好' });
  assert.equal(result.created, true);
  assert.equal(result.bytes, 6);
  assert.equal(fs.readFileSync(path.join(SCRATCH, 'new.txt'), 'utf8'), '你好');
});

await test('writeFile 不许写 .git 目录', async () => {
  await assert.rejects(
    () => TOOL_IMPLS.writeFile({ path: '.git/hooks/x', content: 'x' }),
    /\.git/,
  );
});

await test('deleteFile 不许删仓库根', async () => {
  await assert.rejects(() => TOOL_IMPLS.deleteFile({ path: '.' }), /仓库根/);
});

await test('deleteFile 删掉文件，删不存在的路径报错', async () => {
  const deleted = await TOOL_IMPLS.deleteFile({ path: `${relScratch}/new.txt` });
  assert.equal(deleted.deleted, true);
  await assert.rejects(
    () => TOOL_IMPLS.deleteFile({ path: `${relScratch}/new.txt` }),
    /不存在/,
  );
});

// ---------- pi 事件 → 协议分片 ----------
process.stdout.write('\n  pi 事件翻译\n');

const TEXT_RUN = [
  { type: 'agent_start' },
  { type: 'turn_start' },
  { type: 'message_start', message: { role: 'assistant' } },
  assistantMessageUpdate({ type: 'text_start', contentIndex: 0 }),
  assistantMessageUpdate({ type: 'text_delta', contentIndex: 0, delta: '你好' }),
  assistantMessageUpdate({ type: 'text_end', contentIndex: 0 }),
  {
    type: 'message_end',
    message: {
      role: 'assistant',
      stopReason: 'stop',
      usage: { input: 10, output: 3, reasoning: 0, totalTokens: 13 },
    },
  },
  { type: 'turn_end' },
  { type: 'agent_end' },
  { type: 'agent_settled' },
];

await test('文本被翻成 text-start/delta/end，并以 finish + data-usage 收尾', async () => {
  const session = createFakeSession(TEXT_RUN);
  const chunks = await collect(
    translateSession(session, { prompt: '打个招呼', messageId: 'm-1' }),
  );
  assert.deepEqual(
    chunks.map((chunk) => chunk.type),
    [
      'start',
      'start-step',
      'text-start',
      'text-delta',
      'text-end',
      'finish-step',
      'finish',
      'data-usage',
    ],
  );
  assert.equal(chunks[0].messageId, 'm-1');
  assert.equal(session.calls.prompts[0], '打个招呼');
  assert.equal(chunks.at(-1).data.totalTokens, 13);
  assert.equal(session.calls.disposed, 1);
  assert.equal(session.calls.aborted, 0);
});

const TOOL_RUN = [
  { type: 'turn_start' },
  { type: 'message_start', message: { role: 'assistant' } },
  assistantMessageUpdate({
    type: 'toolcall_start',
    contentIndex: 0,
    partial: { role: 'assistant', content: [{ type: 'toolCall', id: 'call_1', name: 'readFile' }] },
  }),
  assistantMessageUpdate({
    type: 'toolcall_delta',
    contentIndex: 0,
    delta: '{"path":"a.txt"}',
  }),
  assistantMessageUpdate({
    type: 'toolcall_end',
    contentIndex: 0,
    toolCall: { type: 'toolCall', id: 'call_1', name: 'readFile', arguments: { path: 'a.txt' } },
  }),
  {
    type: 'message_end',
    message: {
      role: 'assistant',
      stopReason: 'toolUse',
      usage: { input: 20, output: 5, reasoning: 2, totalTokens: 25 },
    },
  },
  { type: 'tool_execution_start', toolCallId: 'call_1', toolName: 'readFile', args: { path: 'a.txt' } },
  {
    type: 'tool_execution_end',
    toolCallId: 'call_1',
    toolName: 'readFile',
    isError: false,
    result: {
      content: [{ type: 'text', text: 'line one' }],
      details: { path: 'a.txt', totalLines: 2, truncated: true, content: 'line one' },
    },
  },
  { type: 'turn_end' },
  { type: 'turn_start' },
  { type: 'message_start', message: { role: 'assistant' } },
  assistantMessageUpdate({ type: 'text_start', contentIndex: 0 }),
  assistantMessageUpdate({ type: 'text_delta', contentIndex: 0, delta: '结论' }),
  assistantMessageUpdate({ type: 'text_end', contentIndex: 0 }),
  {
    type: 'message_end',
    message: {
      role: 'assistant',
      stopReason: 'stop',
      usage: { input: 30, output: 7, reasoning: 0, totalTokens: 37 },
    },
  },
  { type: 'turn_end' },
  { type: 'agent_settled' },
];

await test('工具调用：参数增量 → 可用参数 → 结构化输出', async () => {
  const chunks = await collect(translateSession(createFakeSession(TOOL_RUN), { prompt: 'x' }));
  const types = chunks.map((chunk) => chunk.type);
  const inputStart = chunks.find((chunk) => chunk.type === 'tool-input-start');
  assert.equal(inputStart.toolCallId, 'call_1');
  assert.equal(inputStart.toolName, 'readFile');

  const delta = chunks.find((chunk) => chunk.type === 'tool-input-delta');
  assert.equal(delta.inputTextDelta, '{"path":"a.txt"}');

  const available = chunks.find((chunk) => chunk.type === 'tool-input-available');
  assert.deepEqual(available.input, { path: 'a.txt' });

  const output = chunks.find((chunk) => chunk.type === 'tool-output-available');
  assert.equal(output.output.path, 'a.txt');
  assert.equal(output.output.content, 'line one');

  // 两轮 = 两组 start-step / finish-step
  assert.equal(types.filter((type) => type === 'start-step').length, 2);
  assert.equal(types.filter((type) => type === 'finish-step').length, 2);
});

await test('两轮的 usage 相加，步数按轮数计', async () => {
  const chunks = await collect(translateSession(createFakeSession(TOOL_RUN), { prompt: 'x' }));
  const usage = chunks.find((chunk) => chunk.type === 'data-usage');
  assert.deepEqual(usage.data, {
    inputTokens: 50,
    outputTokens: 12,
    reasoningTokens: 2,
    totalTokens: 62,
    steps: 2,
  });
  assert.equal(chunks.find((chunk) => chunk.type === 'finish').finishReason, 'stop');
});

await test('工具失败翻成 tool-output-error，不静默', async () => {
  const run = [
    { type: 'turn_start' },
    {
      type: 'tool_execution_end',
      toolCallId: 'call_9',
      toolName: 'readFile',
      isError: true,
      result: { content: [{ type: 'text', text: '路径越界（只允许仓库根之内）: ../x' }] },
    },
    { type: 'turn_end' },
    { type: 'agent_settled' },
  ];
  const chunks = await collect(translateSession(createFakeSession(run), { prompt: 'x' }));
  const err = chunks.find((chunk) => chunk.type === 'tool-output-error');
  assert.equal(err.toolCallId, 'call_9');
  assert.match(err.errorText, /越界/);
});

await test('think 分片翻成 reasoning-*', async () => {
  const run = [
    { type: 'turn_start' },
    assistantMessageUpdate({ type: 'thinking_start', contentIndex: 0 }),
    assistantMessageUpdate({ type: 'thinking_delta', contentIndex: 0, delta: '想一想' }),
    assistantMessageUpdate({ type: 'thinking_end', contentIndex: 0 }),
    { type: 'turn_end' },
    { type: 'agent_settled' },
  ];
  const types = (await collect(translateSession(createFakeSession(run), { prompt: 'x' }))).map(
    (chunk) => chunk.type,
  );
  assert.deepEqual(types.filter((type) => type.startsWith('reasoning')), [
    'reasoning-start',
    'reasoning-delta',
    'reasoning-end',
  ]);
});

await test('prompt 失败翻成 error 分片，且会话仍被释放', async () => {
  const session = createFakeSession([], { promptError: new Error('provider 挂了') });
  const chunks = await collect(translateSession(session, { prompt: 'x' }));
  const err = chunks.find((chunk) => chunk.type === 'error');
  assert.match(err.errorText, /provider 挂了/);
  assert.equal(session.calls.disposed, 1);
});

await test('客户端断开：调用 session.abort 并发出 abort 分片', async () => {
  const session = createFakeSession([], { hangUntilAbort: true });
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 10);
  const chunks = await collect(
    translateSession(session, { prompt: 'x', signal: controller.signal }),
  );
  assert.ok(session.calls.aborted >= 1, 'abort 没有被调用');
  assert.ok(
    chunks.some((chunk) => chunk.type === 'abort'),
    `没有 abort 分片: ${chunks.map((chunk) => chunk.type).join(', ')}`,
  );
  assert.equal(session.calls.disposed, 1);
});

function toolTurns(count) {
  const run = [];
  for (let i = 0; i < count; i += 1) {
    run.push({ type: 'turn_start' });
    run.push({
      type: 'message_end',
      message: {
        role: 'assistant',
        stopReason: 'toolUse',
        usage: { input: 1, output: 1, totalTokens: 2 },
      },
    });
    run.push({ type: 'turn_end' });
  }
  return run;
}

await test('没有步数上限：再长的工具链也让它自己收尾', async () => {
  const run = toolTurns(200);
  run.push({ type: 'turn_start' });
  run.push({
    type: 'message_end',
    message: {
      role: 'assistant',
      stopReason: 'stop',
      usage: { input: 1, output: 1, totalTokens: 2 },
    },
  });
  run.push({ type: 'turn_end' });
  run.push({ type: 'agent_settled' });

  const session = createFakeSession(run);
  const chunks = await collect(translateSession(session, { prompt: 'x' }));
  assert.ok(!chunks.some((chunk) => chunk.type === 'error'), '长工具链被中止了');
  assert.equal(session.calls.aborted, 0, '不该有中止');
  assert.ok(chunks.some((chunk) => chunk.type === 'finish'), '没有 finish');
  assert.equal(chunks.find((chunk) => chunk.type === 'data-usage').data.steps, 201);
  assert.equal(session.calls.disposed, 1);
});

// ---------- 多步轮次：pi 会话跨步复用 ----------
process.stdout.write('\n  多步轮次的会话复用\n');

const ROUND_PROFILE = { id: 'agent-a', name: 'AGENT A', instructions: '' };

/** 假会话工厂：记录造了几个会话、每个会话收到什么。 */
function fakeFactory() {
  const made = [];
  const factory = async (args) => {
    const session = createFakeSession([]);
    session.args = args;
    session.isStreaming = false;
    made.push(session);
    return session;
  };
  return { factory, made };
}

await test('同一轮的多步复用同一个 pi 会话', async () => {
  sweepRoundSessions(Number.MAX_SAFE_INTEGER);
  const { factory, made } = fakeFactory();
  const round = { roundId: 'round-x', stepIndex: 0, totalSteps: 4 };
  const first = await acquireRoundSession({
    round,
    profile: ROUND_PROFILE,
    instructions: '',
    permissionIds: ['read'],
    createSession: factory,
  });
  const second = await acquireRoundSession({
    round: { ...round, stepIndex: 1 },
    profile: ROUND_PROFILE,
    instructions: '',
    permissionIds: ['read'],
    createSession: factory,
  });
  assert.equal(first.reused, false);
  assert.equal(second.reused, true);
  assert.equal(made.length, 1, `造了 ${made.length} 个会话`);
  assert.equal(first.session, second.session);
  assert.equal(roundSessionCount(), 1);
  disposeRoundSession(first.key);
  assert.equal(roundSessionCount(), 0);
});

await test('提示词或权限变了就换会话，旧的那个要释放', async () => {
  sweepRoundSessions(Number.MAX_SAFE_INTEGER);
  const { factory, made } = fakeFactory();
  const round = { roundId: 'round-y', stepIndex: 0, totalSteps: 3 };
  const first = await acquireRoundSession({
    round,
    profile: ROUND_PROFILE,
    instructions: 'old',
    permissionIds: ['read'],
    createSession: factory,
  });
  const second = await acquireRoundSession({
    round: { ...round, stepIndex: 1 },
    profile: ROUND_PROFILE,
    instructions: 'new',
    permissionIds: ['read'],
    createSession: factory,
  });
  assert.notEqual(first.session, second.session);
  assert.equal(made.length, 2);
  assert.equal(made[0].calls.disposed, 1, '旧会话没有释放');
  disposeRoundSession(second.key);
});

await test('最后一步跑完就把会话释放，不留着占内存', async () => {
  sweepRoundSessions(Number.MAX_SAFE_INTEGER);
  const { factory, made } = fakeFactory();
  await collect(
    runLiveChunks('第二步的指令', {
      profileId: 'agent-a',
      permissions: ['read'],
      round: { roundId: 'round-z', stepIndex: 1, totalSteps: 2, roundPrompt: '本轮的问题' },
      createSession: factory,
    }),
  );
  assert.equal(made.length, 1);
  assert.equal(roundSessionCount(), 0, '整轮跑完会话还留着');
  assert.equal(made[0].calls.disposed, 1, '会话没有释放');
});

await test('只说 roundId、没给总步数时按单步处理：跑完就释放', async () => {
  sweepRoundSessions(Number.MAX_SAFE_INTEGER);
  const { factory, made } = fakeFactory();
  await collect(
    runLiveChunks('只有一步', {
      profileId: 'agent-a',
      permissions: ['read'],
      round: { roundId: 'round-solo', stepIndex: 0, roundPrompt: '本轮的问题' },
      createSession: factory,
    }),
  );
  assert.equal(roundSessionCount(), 0, '会话没被释放');
  assert.equal(made.length, 1);
  assert.equal(made[0].calls.disposed, 1);
});

await test('某一步出错也释放会话，不把半截上下文留给下一步', async () => {
  sweepRoundSessions(Number.MAX_SAFE_INTEGER);
  const made = [];
  const factory = async () => {
    const session = createFakeSession([], { promptError: new Error('炸了') });
    session.isStreaming = false;
    made.push(session);
    return session;
  };
  await collect(
    runLiveChunks('第一步的指令', {
      profileId: 'agent-a',
      permissions: ['read'],
      round: { roundId: 'round-e', stepIndex: 0, totalSteps: 4, roundPrompt: '本轮的问题' },
      createSession: factory,
    }),
  );
  assert.equal(roundSessionCount(), 0, '出错了会话还留着');
  assert.equal(made[0].calls.disposed, 1);
});

await test('会话丢了（过期/重启）时，后面那步兜底带上本轮输入', async () => {
  sweepRoundSessions(Number.MAX_SAFE_INTEGER);
  const { factory, made } = fakeFactory();
  await collect(
    runLiveChunks('第三步的指令', {
      profileId: 'agent-a',
      permissions: ['read'],
      round: { roundId: 'round-r', stepIndex: 2, totalSteps: 3, roundPrompt: '本轮的问题' },
      createSession: factory,
    }),
  );
  assert.equal(made.length, 1);
  const prompt = made[0].calls.prompts[0];
  assert.match(prompt, /本轮的问题/, '会话重建后没有把本轮输入补回去');
  assert.match(prompt, /第三步的指令/);
});

await test('本轮输入只在第一步发，后面几步用服务端记住的那份兜底', async () => {
  sweepRoundSessions(Number.MAX_SAFE_INTEGER);
  const made = [];
  const factory = async () => {
    const session = createFakeSession([]);
    session.isStreaming = false;
    made.push(session);
    return session;
  };
  const roundId = 'round-cache';
  await collect(
    runLiveChunks('第一步', {
      profileId: 'agent-a',
      permissions: ['read'],
      round: { roundId, stepIndex: 0, totalSteps: 3, roundPrompt: '本轮的问题' },
      createSession: factory,
    }),
  );
  assert.equal(made.length, 1);
  // 会话丢了（服务重启 / 回收），但本轮输入还留在服务端。
  disposeRoundSession(`${roundId}:agent-a`);
  await collect(
    runLiveChunks('第二步', {
      profileId: 'agent-a',
      permissions: ['read'],
      round: { roundId, stepIndex: 1, totalSteps: 3 },
      createSession: factory,
    }),
  );
  assert.equal(made.length, 2, '第二步应该新造会话');
  const prompt = made[1].calls.prompts[0];
  assert.match(prompt, /本轮的问题/, '兜底没有把本轮输入补回来');
  assert.match(prompt, /第二步/);
  disposeRoundSession(`${roundId}:agent-a`);
});

await test('闲置太久的会话会被扫掉', async () => {
  sweepRoundSessions(Number.MAX_SAFE_INTEGER);
  const { factory, made } = fakeFactory();
  await acquireRoundSession({
    round: { roundId: 'round-t', stepIndex: 0, totalSteps: 4 },
    profile: ROUND_PROFILE,
    instructions: '',
    permissionIds: ['read'],
    createSession: factory,
  });
  assert.equal(roundSessionCount(), 1);
  sweepRoundSessions(Date.now() + 31 * 60 * 1000);
  assert.equal(roundSessionCount(), 0);
  assert.equal(made[0].calls.disposed, 1);
});

await test('translateSession 可以选择不释放会话（轮次中间那几步）', async () => {
  const session = createFakeSession(TEXT_RUN);
  const stream = translateSession(session, { prompt: 'x', dispose: false });
  let status = null;
  while (true) {
    const step = await stream.next();
    if (step.done) {
      status = step.value;
      break;
    }
  }
  assert.equal(session.calls.disposed, 0, '不该在这一步释放');
  assert.equal(status.status, 'done');
});

await test('本步调用 endRound：会话立刻释放，不等轮次的最后一步', async () => {
  sweepRoundSessions(Number.MAX_SAFE_INTEGER);
  const made = [];
  const endRoundRun = [
    { type: 'turn_start' },
    {
      type: 'tool_execution_end',
      toolCallId: 'e1',
      toolName: 'endRound',
      isError: false,
      result: { content: [{ type: 'text', text: '本轮到此结束' }], details: { reason: '没有 Open Issue' } },
    },
    { type: 'turn_end' },
    { type: 'agent_settled' },
  ];
  const chunks = await collect(
    runLiveChunks('第一步：查 Open Issue', {
      profileId: 'agent-a',
      permissions: ['read'],
      round: { roundId: 'round-end', stepIndex: 0, totalSteps: 4, roundPrompt: '本轮的问题' },
      createSession: async () => {
        const session = createFakeSession(endRoundRun);
        session.isStreaming = false;
        made.push(session);
        return session;
      },
    }),
  );
  assert.ok(chunks.some((chunk) => chunk.type === 'tool-output-available'));
  assert.equal(roundSessionCount(), 0, '提前结束后会话还挂着（要等 TTL 才回收）');
  assert.equal(made[0].calls.disposed, 1);
});

await test('translateSession 把 endRound 报给上层', async () => {
  const session = createFakeSession([
    { type: 'turn_start' },
    {
      type: 'tool_execution_end',
      toolCallId: 'e2',
      toolName: 'endRound',
      isError: false,
      result: { content: [{ type: 'text', text: 'ok' }] },
    },
    { type: 'turn_end' },
    { type: 'agent_settled' },
  ]);
  const stream = translateSession(session, { prompt: 'x' });
  let outcome = null;
  while (true) {
    const step = await stream.next();
    if (step.done) {
      outcome = step.value;
      break;
    }
  }
  assert.deepEqual(outcome, { status: 'done', roundEnded: true });
});

fs.rmSync(SCRATCH, { recursive: true, force: true });

process.stdout.write(`\n${passed} passed, ${failures.length} failed\n`);
if (failures.length > 0) {
  for (const { name, err } of failures) {
    process.stdout.write(`\n${name}\n${err && err.stack ? err.stack : err}\n`);
  }
  process.exit(1);
}
