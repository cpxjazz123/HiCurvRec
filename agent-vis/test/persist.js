import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createPersistence } from '../persistence.js';

/**
 * 落盘层的单元测试：配置与对话记录写进去、读回来是一回事。
 *
 * 临时目录放在仓库内的 .out/ 下（不放 /tmp），跑完自己清掉。
 */

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SCRATCH = path.join(HERE, '..', '.out', 'persist-test');

let passed = 0;
const failures = [];

async function test(name, fn) {
  const started = Date.now();
  try {
    await fn();
    passed += 1;
    process.stdout.write(`  ✓ ${name} (${Date.now() - started} ms)\n`);
  } catch (err) {
    failures.push({ name, err });
    process.stdout.write(`  ✗ ${name}\n    ${err && err.message}\n`);
  }
}

function freshDir() {
  fs.rmSync(SCRATCH, { recursive: true, force: true });
  fs.mkdirSync(SCRATCH, { recursive: true });
  return SCRATCH;
}

async function main() {
  process.stdout.write('agent-vis 落盘层测试\n');

  // ---------- 没有存过任何东西 ----------
  await test('没存过时读出来是空结构（预期内的缺失）', () => {
    const store = createPersistence({ dataDir: freshDir() });
    const state = store.loadAgents();
    assert.deepEqual(state, { version: 1, configs: {}, agents: {} });
  });

  await test('没有对话记录时读出来是空数组', () => {
    const store = createPersistence({ dataDir: freshDir() });
    assert.deepEqual(store.readChats('agent-a'), []);
  });

  // ---------- 写进去、读回来 ----------
  await test('配置写进去能原样读回来', () => {
    const store = createPersistence({ dataDir: freshDir() });
    const state = {
      version: 1,
      configs: { 'agent-a': { permissions: ['read', 'write'] } },
      agents: {
        'custom-1': {
          id: 'custom-1',
          name: 'AGENT C',
          role: 'CUSTOM',
          instructions: '',
          permissions: ['read'],
          steps: [],
          handoffTargets: [],
        },
      },
    };
    store.saveAgents(state);
    assert.deepEqual(store.loadAgents(), state);
  });

  await test('配置能覆盖着写（后写的为准）', () => {
    const store = createPersistence({ dataDir: freshDir() });
    store.saveAgents({ version: 1, configs: {}, agents: {} });
    store.saveAgents({ version: 1, configs: { 'agent-a': { instructions: 'x' } }, agents: {} });
    assert.deepEqual(store.loadAgents().configs, { 'agent-a': { instructions: 'x' } });
  });

  await test('写完不留临时文件', () => {
    const dir = freshDir();
    const store = createPersistence({ dataDir: dir });
    store.saveAgents({ version: 1, configs: {}, agents: {} });
    assert.ok(fs.existsSync(path.join(dir, 'agents.json')), 'agents.json 没写出来');
    const leftovers = fs.readdirSync(dir).filter((name) => name.endsWith('.tmp'));
    assert.deepEqual(leftovers, [], '留下了临时文件');
  });

  await test('对话记录按顺序追加、读回来顺序不变', () => {
    const store = createPersistence({ dataDir: freshDir() });
    store.appendChat('agent-a', { runId: 'r1', prompt: '第一个' });
    store.appendChat('agent-a', { runId: 'r2', prompt: '第二个' });
    store.appendChat('agent-b', { runId: 'r3', prompt: '别人的' });
    const a = store.readChats('agent-a');
    assert.equal(a.length, 2);
    assert.equal(a[0].runId, 'r1');
    assert.equal(a[1].runId, 'r2');
    assert.equal(store.readChats('agent-b').length, 1);
  });

  await test('对话记录带分片也能原样读回', () => {
    const store = createPersistence({ dataDir: freshDir() });
    const record = {
      runId: 'r1',
      roundId: 'round-1',
      stepIndex: 0,
      chunks: [{ type: 'text-delta', id: '0', delta: '你好' }],
    };
    store.appendChat('agent-a', record);
    assert.deepEqual(store.readChats('agent-a')[0], record);
  });

  // ---------- 写坏了的文件不许悄悄兜底 ----------
  await test('agents.json 是坏 JSON 时直接报错', () => {
    const dir = freshDir();
    fs.writeFileSync(path.join(dir, 'agents.json'), '{ 这不是 JSON', 'utf8');
    const store = createPersistence({ dataDir: dir });
    assert.throws(() => store.loadAgents(), /agents\.json/);
  });

  await test('agents.json 版本不认识时报错', () => {
    const dir = freshDir();
    fs.writeFileSync(
      path.join(dir, 'agents.json'),
      JSON.stringify({ version: 99, configs: {}, agents: {} }),
      'utf8',
    );
    const store = createPersistence({ dataDir: dir });
    assert.throws(() => store.loadAgents(), /版本/);
  });

  await test('agents.json 缺字段时报错', () => {
    const dir = freshDir();
    fs.writeFileSync(path.join(dir, 'agents.json'), JSON.stringify({ version: 1 }), 'utf8');
    const store = createPersistence({ dataDir: dir });
    assert.throws(() => store.loadAgents(), /configs/);
  });

  await test('对话记录里有坏行时报错，不当成空记录', () => {
    const dir = freshDir();
    const store = createPersistence({ dataDir: dir });
    store.appendChat('agent-a', { runId: 'r1' });
    fs.appendFileSync(path.join(dir, 'chats', 'agent-a.jsonl'), '{ 坏行\n', 'utf8');
    assert.throws(() => store.readChats('agent-a'), /第 2 行/);
  });

  // ---------- 目录不存在时自己建 ----------
  await test('数据目录不存在时自己建出来', () => {
    const dir = path.join(SCRATCH, 'not-there-yet', 'deeper');
    fs.rmSync(path.join(SCRATCH, 'not-there-yet'), { recursive: true, force: true });
    const store = createPersistence({ dataDir: dir });
    store.saveAgents({ version: 1, configs: {}, agents: {} });
    store.appendChat('agent-a', { runId: 'r1' });
    assert.ok(fs.existsSync(path.join(dir, 'agents.json')));
    assert.ok(fs.existsSync(path.join(dir, 'chats', 'agent-a.jsonl')));
  });

  fs.rmSync(SCRATCH, { recursive: true, force: true });

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
