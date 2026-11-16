import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * agent-vis 作为 dsh 插件的接线检查（不启动真实 harness、不打模型）。
 *
 * 1. plugin.js 的插件形状与 apply：起 HTTP、挂上下文、卸载时清干净；
 * 2. createDshSession：scoped 工具注册、宿主工具 deny、提示词 section；
 * 3. 适配器把 dsh 的 assistant-stream / session 事件翻成协议分片。
 */

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const DATA = path.join(ROOT, '.out', 'plugin-test-data');

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

async function collect(stream) {
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return chunks;
}

/**
 * 假 harness：agents.create 走 setup，工具注册/restrict/提示词 section 全部留档，
 * 事件走 agentCtx.on 登记的监听器。
 */
function fakeHarness() {
  const created = [];
  const listeners = new Map();
  const disposers = [];
  let pendingIdle = [];

  const ctx = {
    logger: { info() {} },
    tools: {},
    effect(fn) {
      disposers.push(fn());
    },
    agents: {
      async create(options) {
        const agent = {
          id: options.sessionId,
          status: 'idle',
          messages: [],
          followup(message) {
            agent.messages.push(message);
            agent.status = 'running';
          },
          whenIdle() {
            return new Promise((resolve) => pendingIdle.push(resolve));
          },
          cancel() {
            agent.status = 'idle';
            for (const resolve of pendingIdle.splice(0)) resolve();
          },
        };
        const record = { agent, registrations: [], sections: [], restrictions: [], options };
        const agentCtx = {
          tools: {
            register(def) {
              record.registrations.push(def);
              return () => {};
            },
            definitions() {
              return [{ name: 'bash' }, { name: 'read' }, { name: 'edit' }];
            },
            restrict(filter) {
              record.restrictions.push(filter);
              return () => {};
            },
          },
          systemPrompt: {
            section(section) {
              record.sections.push(section);
              return () => {};
            },
          },
          on(name, fn) {
            if (!listeners.has(name)) listeners.set(name, []);
            listeners.get(name).push({ agent, fn });
            return () => {};
          },
        };
        await options.setup(agentCtx, agent);
        created.push(record);
        return {
          agent,
          async dispose() {
            agent.status = 'idle';
          },
        };
      },
    },
    /** 测试手动结束一次 whenIdle，并把 agent 状态收回 idle。 */
    finishTurn() {
      for (const record of created) record.agent.status = 'idle';
      for (const resolve of pendingIdle.splice(0)) resolve();
    },
  };
  return { ctx, created, listeners, disposers };
}

function fire(listeners, name, ...args) {
  for (const entry of listeners.get(name) || []) entry.fn(...args);
}

process.stdout.write('agent-vis dsh 插件接线检查\n\n');

const plugin = await import('../plugin.js');
const { createDshSession, translateSession, setHarnessContext } = await import('../live-agent.js');

await test('插件形状：name / inject / apply', () => {
  assert.equal(plugin.name, 'agent-vis');
  assert.deepEqual(plugin.inject, ['agents', 'tools', 'systemPrompt']);
  assert.equal(typeof plugin.apply, 'function');
});

await test('apply 起服务、返回真实端口、卸载时关端口并清上下文', async () => {
  fs.mkdirSync(DATA, { recursive: true });
  const { ctx, disposers } = fakeHarness();
  await plugin.apply(ctx, {
    host: '127.0.0.1',
    port: 0,
    publicPath: '',
    dataDir: DATA,
  });
  const mounted = plugin.mounted;
  assert.ok(mounted && mounted.port > 0, '没有拿到监听端口');

  const health = await fetch(`http://127.0.0.1:${mounted.port}/api/health`);
  assert.equal(health.status, 200);
  const body = await health.json();
  assert.ok(body.ok !== false);

  const { getHarnessContext } = await import('../live-agent.js');
  assert.equal(getHarnessContext(), ctx, '上下文没有挂上');

  for (const dispose of disposers.splice(0).reverse()) dispose();
  assert.equal(getHarnessContext(), null, '卸载后上下文没清');
  await assert.rejects(fetch(`http://127.0.0.1:${mounted.port}/api/health`));
  fs.rmSync(DATA, { recursive: true, force: true });
});

await test('createDshSession：scoped 工具 + 宿主工具 deny + 提示词 section', async () => {
  const { ctx, created } = fakeHarness();
  setHarnessContext(ctx);
  const session = await createDshSession(ctx, {
    profile: { id: 'agent-a', instructions: '你是测试' },
    instructions: undefined,
    permissionIds: ['read'],
  });
  assert.equal(created.length, 1);
  const record = created[0];
  const names = record.registrations.map((tool) => tool.name);
  assert.ok(names.includes('read'));
  assert.ok(names.includes('ls'));
  assert.ok(names.includes('endRound'));
  assert.ok(names.includes('requestHandoff'));
  assert.ok(!names.includes('bash'), '宿主工具不该被注册成 scoped 工具');
  assert.deepEqual(record.restrictions, [{ deny: ['bash', 'edit'] }]);
  assert.ok(record.sections.some((section) => section.text.includes('你是测试')));
  assert.ok(record.sections.some((section) => section.text.includes('简体中文')));
  assert.equal(record.options.agentOptions.provider, 'commandcode');
  assert.equal(record.options.meta.cwd !== undefined, true);

  // prompt 会走 agent.followup：消息是 user 角色、带 source。
  const running = session.prompt('你好');
  assert.equal(record.agent.messages.length, 1);
  assert.equal(record.agent.messages[0].role, 'user');
  assert.equal(record.agent.messages[0].source.kind, 'user');
  assert.equal(record.agent.messages[0].content[0].text, '你好');
  assert.equal(session.isStreaming, true);
  ctx.finishTurn();
  await running;
  assert.equal(session.isStreaming, false);
  session.dispose();
  setHarnessContext(null);
});

await test('适配器桥：文本 / 工具结果 / usage / endRound 全都能翻', async () => {
  const { ctx, created, listeners } = fakeHarness();
  setHarnessContext(ctx);
  const session = await createDshSession(ctx, {
    profile: { id: 'agent-a', instructions: '' },
    permissionIds: [],
  });
  const record = created[0];
  const fakeSession = { id: record.agent.id };

  const chunksPromise = collect(
    translateSession(session, { prompt: '跑一下', messageId: 'm-1' }),
  );

  // 等 prompt() 真正把 turn 挂起来，再按 dsh 的顺序推事件。
  await new Promise((resolve) => setTimeout(resolve, 0));
  fire(listeners, 'agent/assistant-stream', {
    agent: record.agent,
    frame: { type: 'start', attemptId: 'a', revision: 1, turn: 1, step: 1 },
  });
  fire(listeners, 'session/event', fakeSession, { type: 'step/start', data: { turn: 1, step: 1 } });
  fire(listeners, 'agent/assistant-stream', {
    agent: record.agent,
    frame: {
      type: 'chunk',
      attemptId: 'a',
      revision: 1,
      index: 0,
      time: 1,
      chunk: { type: 'block-start', index: 0, blockType: 'text' },
    },
  });
  fire(listeners, 'agent/assistant-stream', {
    agent: record.agent,
    frame: {
      type: 'chunk',
      attemptId: 'a',
      revision: 1,
      index: 1,
      time: 2,
      chunk: { type: 'text-delta', index: 0, text: '你好' },
    },
  });
  fire(listeners, 'agent/assistant-stream', {
    agent: record.agent,
    frame: {
      type: 'chunk',
      attemptId: 'a',
      revision: 1,
      index: 2,
      time: 3,
      chunk: { type: 'block-end', index: 0, block: { type: 'text', text: '你好' } },
    },
  });
  fire(listeners, 'session/event', fakeSession, {
    type: 'assistant/message',
    data: {
      turn: 1,
      step: 1,
      message: { role: 'assistant', content: [] },
      usage: { inputTokens: 5, outputTokens: 2, totalTokens: 7 },
    },
  });
  fire(listeners, 'session/event', fakeSession, {
    type: 'tool/call',
    data: { turn: 1, step: 1, callId: 'c1', name: 'endRound', arguments: '{"reason":"没有 Open Issue"}' },
  });
  fire(listeners, 'session/event', fakeSession, {
    type: 'tool/result',
    data: {
      turn: 1,
      step: 1,
      message: { role: 'tool', toolCallId: 'c1', content: [], source: { kind: 'tool', callId: 'c1' } },
      meta: { reason: '没有 Open Issue' },
    },
  });
  fire(listeners, 'session/event', fakeSession, {
    type: 'step/end',
    data: { turn: 1, step: 1 },
  });
  ctx.finishTurn();

  const chunks = await chunksPromise;
  const types = chunks.map((chunk) => chunk.type);
  assert.ok(types.includes('text-start'));
  assert.ok(types.includes('text-delta'));
  assert.ok(types.includes('text-end'));
  const output = chunks.find((chunk) => chunk.type === 'tool-output-available');
  assert.deepEqual(output.output, { reason: '没有 Open Issue' });
  const usage = chunks.find((chunk) => chunk.type === 'data-usage');
  assert.equal(usage.data.totalTokens, 7);
  assert.equal(usage.data.steps, 1);
  assert.equal(chunks.find((chunk) => chunk.type === 'finish').finishReason, 'stop');

  session.dispose();
  setHarnessContext(null);
});

await test('没有 harness 上下文时明确失败（不是静默跑空）', async () => {
  setHarnessContext(null);
  await assert.rejects(
    () => createDshSession(null, { profile: { id: 'x', instructions: '' }, permissionIds: [] }),
    /harness 上下文/,
  );
});

fs.rmSync(path.join(ROOT, '.out', 'plugin-test-data'), { recursive: true, force: true });

process.stdout.write(`\n${passed} passed, ${failures.length} failed\n`);
if (failures.length > 0) {
  for (const { name, err } of failures) {
    process.stdout.write(`\n${name}\n${err && err.stack ? err.stack : err}\n`);
  }
  process.exit(1);
}
