import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createServer } from '../server.js';
import { config } from '../server.js';
import { createStore, applyChunk } from '../public/protocol.js';

/**
 * 持久化接口测试：配置的存与取、对话记录的落盘与回放。
 *
 * 数据目录放在仓库内的 .out/ 下（不放 /tmp），跑完自己清掉 ——
 * 测试不该动用户自己配好的那一份。
 */

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SCRATCH = path.join(HERE, '..', '.out', 'api-persist-test');

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
  openedServers.add(server);
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve(server.address().port));
  });
}

function close(server) {
  return new Promise((resolve) => {
    // 连接池里的长连接会让 close() 等着 —— 先把它们掐掉。
    server.closeAllConnections();
    server.close(resolve);
  });
}

/** 这一轮测试起过的所有服务，退出前都要关掉。 */
const openedServers = new Set();

/** 读 SSE 响应并把分片拆出来。 */
async function readSse(url) {
  const res = await fetch(url);
  const body = await res.text();
  const chunks = [];
  for (const block of body.split('\n\n')) {
    for (const line of block.split('\n')) {
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      chunks.push(JSON.parse(payload));
    }
  }
  return { res, body, chunks };
}

function freshDir() {
  fs.rmSync(SCRATCH, { recursive: true, force: true });
  fs.mkdirSync(SCRATCH, { recursive: true });
  return SCRATCH;
}

async function putConfig(base, id, patch) {
  const res = await fetch(`${base}/api/agents/${encodeURIComponent(id)}/config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  const text = await res.text();
  let json = null;
  try {
    json = JSON.parse(text);
  } catch {
    json = null;
  }
  return { res, status: res.status, json, text };
}

async function getJson(base, route) {
  const res = await fetch(`${base}${route}`);
  const text = await res.text();
  let json = null;
  try {
    json = JSON.parse(text);
  } catch {
    json = null;
  }
  return { res, status: res.status, json, text };
}

async function main() {
  process.stdout.write('agent-vis 持久化接口测试\n');

  const dataDir = freshDir();
  const server = createServer({ mode: 'mock', publicPath: '', dataDir });
  const port = await listen(server);
  const base = `http://127.0.0.1:${port}`;

  // ---------- 配置：存进去、读得出来 ----------

  await test('PUT 配置：改过的提示词出现在 /api/health 里', async () => {
    const put = await putConfig(base, 'agent-a', { instructions: '你是测试用的提示词' });
    assert.equal(put.status, 200, put.text);
    assert.equal(put.json.instructions, '你是测试用的提示词');
    assert.equal(put.json.customized.instructions, true);

    const health = await getJson(base, '/api/health');
    const agent = health.json.agents.find((item) => item.id === 'agent-a');
    assert.equal(agent.instructions, '你是测试用的提示词');
    assert.equal(agent.customized.instructions, true);
  });

  await test('PUT 配置：权限存下来并标成用户改过的', async () => {
    const put = await putConfig(base, 'agent-a', { permissions: ['read', 'network'] });
    assert.equal(put.status, 200, put.text);
    assert.deepEqual(put.json.permissions, ['read', 'network']);
    assert.equal(put.json.customized.permissions, true);
  });

  await test('PUT 配置：轮次步骤存下来', async () => {
    const steps = [
      { title: '第一步', instruction: '读文件' },
      { title: '第二步', instruction: '写结论' },
    ];
    const put = await putConfig(base, 'agent-a', { steps });
    assert.equal(put.status, 200, put.text);
    assert.deepEqual(put.json.steps, steps);
    assert.equal(put.json.customized.steps, true);
  });

  await test('PUT 配置：交接目标存下来', async () => {
    const put = await putConfig(base, 'agent-a', { handoffTargets: ['agent-b'] });
    assert.equal(put.status, 200, put.text);
    assert.deepEqual(put.json.handoffTargets, ['agent-b']);
    assert.equal(put.json.customized.handoff, true);
  });

  await test('PUT 配置：改一个字段不影响别的字段', async () => {
    const before = (await getJson(base, '/api/health')).json.agents.find(
      (item) => item.id === 'agent-a',
    );
    await putConfig(base, 'agent-a', { instructions: '又改了一次' });
    const after = (await getJson(base, '/api/health')).json.agents.find(
      (item) => item.id === 'agent-a',
    );
    assert.deepEqual(after.permissions, before.permissions);
    assert.deepEqual(after.steps, before.steps);
    assert.deepEqual(after.handoffTargets, before.handoffTargets);
    assert.equal(after.instructions, '又改了一次');
  });

  await test('PUT 配置：字段写坏了直接拒绝，且不改已存的配置', async () => {
    const before = (await getJson(base, '/api/health')).json.agents.find(
      (item) => item.id === 'agent-a',
    );
    const bad = await putConfig(base, 'agent-a', { permissions: 'read' });
    assert.equal(bad.status, 400, bad.text);
    assert.ok(bad.json && typeof bad.json.error === 'string', '响应里要写明原因');

    const badId = await putConfig(base, 'agent-a', { permissions: ['nope'] });
    assert.equal(badId.status, 400, badId.text);

    const after = (await getJson(base, '/api/health')).json.agents.find(
      (item) => item.id === 'agent-a',
    );
    assert.deepEqual(after.permissions, before.permissions);
  });

  await test('PUT 配置：身份不存在返回 404', async () => {
    const put = await putConfig(base, 'agent-zzz', { instructions: 'x' });
    assert.equal(put.status, 404, put.text);
  });

  await test('PUT 配置：方法不对返回 405', async () => {
    const res = await fetch(`${base}/api/agents/agent-a/config`, { method: 'POST' });
    assert.equal(res.status, 405);
  });

  await test('PUT 配置：请求体不是合法 JSON 返回 400', async () => {
    const res = await fetch(`${base}/api/agents/agent-a/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: '{ 这不是 JSON',
    });
    assert.equal(res.status, 400);
  });

  await test('PUT 配置：值传 null 表示清掉用户的改动，回到默认', async () => {
    const put = await putConfig(base, 'agent-a', { instructions: null });
    assert.equal(put.status, 200, put.text);
    assert.equal(put.json.customized.instructions, false);
    assert.equal(put.json.instructions, config.AGENT_PROFILES[0].instructions);
  });

  await test('PUT 配置：新建的 agent 也能存配置', async () => {
    const created = await fetch(`${base}/api/agents`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: '测试 agent' }),
    });
    assert.equal(created.status, 201);
    const profile = await created.json();
    const put = await putConfig(base, profile.id, { instructions: '新 agent 的提示词' });
    assert.equal(put.status, 200, put.text);
    assert.equal(put.json.instructions, '新 agent 的提示词');
  });

  // ---------- 对话记录：跑一次落一条 ----------

  await test('GET 对话记录：还没跑过时是空数组', async () => {
    const got = await getJson(base, '/api/agents/agent-b/chats');
    assert.equal(got.status, 200, got.text);
    assert.deepEqual(got.json.records, []);
  });

  await test('GET 对话记录：身份不存在返回 404', async () => {
    const got = await getJson(base, '/api/agents/agent-zzz/chats');
    assert.equal(got.status, 404);
  });

  await test('跑一次之后对话记录落盘，并且能读回来', async () => {
    const { chunks } = await readSse(
      `${base}/api/run?prompt=${encodeURIComponent('第一个问题')}` +
        `&v=${config.FRONTEND_VERSION}&profile=agent-b&instant=1`,
    );
    assert.ok(chunks.length > 0, '这一跑没吐出分片');

    const got = await getJson(base, '/api/agents/agent-b/chats');
    assert.equal(got.status, 200, got.text);
    assert.equal(got.json.records.length, 1);
    const record = got.json.records[0];
    assert.equal(record.agentId, 'agent-b');
    assert.equal(record.prompt, '第一个问题');
    assert.equal(record.status, 'done');
    assert.ok(record.chunks.length > 0, '记录里没有分片');
    assert.ok(typeof record.startedAt === 'number');
    assert.ok(typeof record.durationMs === 'number');
  });

  await test('对话记录真的写到了盘上', async () => {
    const file = path.join(dataDir, 'chats', 'agent-b.jsonl');
    assert.ok(fs.existsSync(file), `${file} 不在`);
    const lines = fs.readFileSync(file, 'utf8').trim().split('\n');
    assert.equal(lines.length, 1);
    assert.equal(JSON.parse(lines[0]).prompt, '第一个问题');
  });

  await test('再跑一次，记录按顺序追加', async () => {
    await readSse(
      `${base}/api/run?prompt=${encodeURIComponent('第二个问题')}` +
        `&v=${config.FRONTEND_VERSION}&profile=agent-b&instant=1`,
    );
    const got = await getJson(base, '/api/agents/agent-b/chats');
    assert.equal(got.json.records.length, 2);
    assert.equal(got.json.records[0].prompt, '第一个问题');
    assert.equal(got.json.records[1].prompt, '第二个问题');
  });

  await test('不同 agent 的对话记录各存各的', async () => {
    const b = await getJson(base, '/api/agents/agent-b/chats');
    const a = await getJson(base, '/api/agents/agent-a/chats');
    assert.equal(b.json.records.length, 2);
    assert.equal(a.json.records.length, 0);
  });

  await test('记录里的分片能还原出时间轴（回放得到同样的步骤与文字）', async () => {
    const got = await getJson(base, '/api/agents/agent-b/chats');
    const record = got.json.records[0];
    const store = createStore();
    for (const chunk of record.chunks) applyChunk(store, chunk);
    assert.ok(store.steps.length >= 2, `步骤数不对：${store.steps.length}`);
    const texts = store.steps
      .flatMap((step) => step.parts)
      .filter((part) => part.kind === 'text')
      .map((part) => part.text)
      .join('');
    assert.ok(texts.includes('第一个问题'), '回放出来的文字里没有那一问');
  });

  await test('跑的时候带上的轮次信息也记下来', async () => {
    const body = {
      prompt: '第三步的提示词',
      v: config.FRONTEND_VERSION,
      profile: 'agent-b',
      roundId: 'round-1',
      stepIndex: 2,
      stepTitle: '第三步',
      totalSteps: 3,
      roundPrompt: '用户最初的问题',
    };
    const res = await fetch(`${base}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    assert.ok(text.includes('data:'), '没有吐分片');

    const got = await getJson(base, '/api/agents/agent-b/chats');
    const record = got.json.records[got.json.records.length - 1];
    assert.equal(record.roundId, 'round-1');
    assert.equal(record.stepIndex, 2);
    assert.equal(record.stepTitle, '第三步');
    assert.equal(record.totalSteps, 3);
    assert.equal(record.roundPrompt, '用户最初的问题');
  });

  // ---------- 重启之后还在 ----------

  await test('服务重启后配置与对话都还在', async () => {
    // 重启前先写一份配置：重启后它必须原样还在。
    const saved = await putConfig(base, 'agent-a', { instructions: '重启前写的提示词' });
    assert.equal(saved.status, 200, saved.text);

    await close(server);
    const server2 = createServer({ mode: 'mock', publicPath: '', dataDir });
    const port2 = await listen(server2);
    const base2 = `http://127.0.0.1:${port2}`;

    const health = await getJson(base2, '/api/health');
    const agent = health.json.agents.find((item) => item.id === 'agent-a');
    assert.equal(agent.instructions, '重启前写的提示词');
    assert.deepEqual(agent.permissions, ['read', 'network']);
    assert.equal(agent.customized.instructions, true);
    // 用户清空过的那一项仍然是"清空"：默认步骤没被铺回来。
    assert.equal(agent.customized.steps, true);
    assert.deepEqual(agent.handoffTargets, ['agent-b']);

    const chats = await getJson(base2, '/api/agents/agent-b/chats');
    assert.equal(chats.json.records.length, 3);

    await close(server2);
  });

  // ---------- 配置抹干净之后不留空壳 ----------

  await test('配置项全被抹掉之后，agents.json 里不留空壳', async () => {
    // 上一个用例把服务关了，这里自己起一个（同一个数据目录）。
    const server3 = createServer({ mode: 'mock', publicPath: '', dataDir });
    const port3 = await listen(server3);
    const base3 = `http://127.0.0.1:${port3}`;

    const saved = await putConfig(base3, 'agent-a', { instructions: '先写一笔' });
    assert.equal(saved.status, 200, saved.text);

    // 盘上这时确实有一份配置。
    const agentsFile = path.join(dataDir, 'agents.json');
    const before = JSON.parse(fs.readFileSync(agentsFile, 'utf8'));
    assert.ok(before.configs['agent-a'], '写完之后盘上应当有 agent-a 的配置');

    // 把这一项的用户改动全抹掉（null = 恢复默认）。
    const cleared = await putConfig(base3, 'agent-a', {
      instructions: null,
      permissions: null,
      steps: null,
      handoffTargets: null,
    });
    assert.equal(cleared.status, 200, cleared.text);

    // 抹干净之后：这一项不该在盘上留一个空对象 —— 那是没有信息量的壳，
    // 只会让 agents.json 越攒越脏。
    const after = JSON.parse(fs.readFileSync(agentsFile, 'utf8'));
    assert.equal(
      after.configs['agent-a'],
      undefined,
      `抹干净之后还留着空壳：${JSON.stringify(after.configs['agent-a'])}`,
    );

    // 生效值回到代码里的默认（不是空字符串）。
    const health = await getJson(base3, '/api/health');
    const agent = health.json.agents.find((item) => item.id === 'agent-a');
    assert.equal(agent.customized.instructions, false);
    assert.equal(agent.customized.permissions, false);
    assert.ok(agent.instructions.length > 0, '默认提示词应当还在');

    await close(server3);
  });

  for (const server of openedServers) {
    await close(server).catch(() => {});
  }
  fs.rmSync(SCRATCH, { recursive: true, force: true });

  process.stdout.write(`\n${passed} 通过, ${failures.length} 失败\n`);
  if (failures.length > 0) {
    for (const { name, err } of failures) {
      process.stdout.write(`\n--- ${name} ---\n${err && err.stack}\n`);
    }
    process.exitCode = 1;
  }
}

main().catch(async (err) => {
  process.stderr.write(`${err && err.stack}\n`);
  for (const server of openedServers) {
    await close(server).catch(() => {});
  }
  process.exit(1);
});
