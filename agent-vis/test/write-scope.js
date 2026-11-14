/**
 * writeScope 的自测。
 *
 * 并行写同一棵候选树的前提是"谁都不能越界写别人的文件" —— 所以这条约束必须是
 * 机器强制的：越界写入必须被拒、而且拒绝发生在落盘之前。
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import { scopeAllows, toolsForPermissions } from '../live-agent.js';
import { currentIdeaFiles, currentIdeaOwners, currentWorkdir, partitionWriterFiles, resolveProfile } from '../config.js';

const REPO = currentWorkdir();
const ROOT = 'agent-vis/.out/scope-fixture';
const ALLOWED = `${ROOT}/allowed`;
const abs = (rel) => path.join(REPO, rel);

function rm(rel) {
  fs.rmSync(abs(rel), { recursive: true, force: true });
}
function read(rel) {
  return fs.readFileSync(abs(rel), 'utf8');
}

let passed = 0;
async function test(name, fn) {
  try {
    await fn();
    passed += 1;
    console.log(`  ✓ ${name}`);
  } catch (err) {
    console.error(`  ✗ ${name}`);
    console.error(`    ${err && err.message ? err.message : err}`);
    process.exitCode = 1;
  }
}

rm(ROOT);
fs.mkdirSync(abs(ALLOWED), { recursive: true });
fs.writeFileSync(abs(`${ALLOWED}/mine.py`), 'a = 1\n');
fs.writeFileSync(abs(`${ROOT}/theirs.py`), 'b = 1\n');

const scoped = (perms, scope) => {
  const tools = toolsForPermissions(perms, { writeScope: scope });
  const byName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));
  const call = (name, args) => byName[name].execute(`call-${name}`, args);
  return { byName, call };
};

const scope = [`${ALLOWED}/**`];

await test('作用域内的文件可以写', async () => {
  const { call } = scoped(['write'], scope);
  await call('write', { path: `${ALLOWED}/mine.py`, content: 'a = 2\n' });
  assert.equal(read(`${ALLOWED}/mine.py`), 'a = 2\n');
});

await test('作用域内新建子目录里的文件也可以（父目录自动建）', async () => {
  const { call } = scoped(['write'], scope);
  await call('write', { path: `${ALLOWED}/deep/new.py`, content: 'x = 1\n' });
  assert.equal(read(`${ALLOWED}/deep/new.py`), 'x = 1\n');
});

await test('作用域外的写入被拒，且没有落盘', async () => {
  const { call } = scoped(['write'], scope);
  await assert.rejects(
    () => call('write', { path: `${ROOT}/theirs.py`, content: 'hacked\n' }),
    /写入越界/,
  );
  assert.equal(read(`${ROOT}/theirs.py`), 'b = 1\n', '被拒的写入不能改到文件');
});

await test('edit 同样受限：域内改得动、域外改不动', async () => {
  const { call } = scoped(['write'], scope);
  await call('edit', {
    path: `${ALLOWED}/mine.py`,
    edits: [{ oldText: 'a = 2', newText: 'a = 3' }],
  });
  assert.equal(read(`${ALLOWED}/mine.py`), 'a = 3\n');
  await assert.rejects(
    () =>
      call('edit', {
        path: `${ROOT}/theirs.py`,
        edits: [{ oldText: 'b = 1', newText: 'b = 2' }],
      }),
    /越界/,
  );
  assert.equal(read(`${ROOT}/theirs.py`), 'b = 1\n');
});

await test('deleteFile 同样受限', async () => {
  const { call } = scoped(['delete'], scope);
  await assert.rejects(() => call('deleteFile', { path: `${ROOT}/theirs.py` }), /越界/);
  assert.equal(fs.existsSync(abs(`${ROOT}/theirs.py`)), true);
  await call('deleteFile', { path: `${ALLOWED}/deep/new.py` });
  assert.equal(fs.existsSync(abs(`${ALLOWED}/deep/new.py`)), false);
});

await test('copyBaselineTree 的目标树必须在作用域里', async () => {
  const { call } = scoped(['write'], scope);
  await assert.rejects(() => call('copyBaselineTree', { arm: 'TIGER_scope_probe' }), /建树越界/);
  assert.equal(fs.existsSync(abs('stage2_RQ-VAE/TIGER_scope_probe')), false);
});

await test('没有作用域时行为不变（整仓库根都可写）', async () => {
  const { call } = scoped(['write'], []);
  await call('write', { path: `${ROOT}/theirs.py`, content: 'b = 9\n' });
  assert.equal(read(`${ROOT}/theirs.py`), 'b = 9\n');
});

await test('writer 能写自己的文件，但工具表里没有 copyBaselineTree（树由 AGENT IDEA 建）', () => {
  for (const id of ['agent-w1', 'agent-w2', 'agent-w3', 'agent-w4']) {
    const profile = resolveProfile(id);
    const names = toolsForPermissions(profile.permissions, {
      writeScope: profile.writeScope,
      denyTools: profile.denyTools,
    }).map((tool) => tool.name);
    assert.equal(names.includes('copyBaselineTree'), false, `${id} 不该有建树工具`);
    assert.equal(names.includes('write'), true, `${id} 仍要能写文件`);
    assert.equal(names.includes('edit'), true, `${id} 仍要能改文件`);
  }
});

await test('分片按 import 依赖聚类：互相依赖的文件落在同一栏', async () => {
  const dir = 'agent-vis/.out/partition-fixture';
  fs.mkdirSync(abs(dir), { recursive: true });
  fs.writeFileSync(abs(`${dir}/core.py`), 'X = 1\n');
  fs.writeFileSync(abs(`${dir}/consumer.py`), 'from core import X\n');
  fs.writeFileSync(abs(`${dir}/other.py`), 'Y = 2\n');
  try {
    const slices = partitionWriterFiles(
      [`${dir}/core.py`, `${dir}/consumer.py`, `${dir}/other.py`],
      4,
    );
    const withCore = slices.find((slice) => slice.includes(`${dir}/core.py`));
    assert.ok(withCore.includes(`${dir}/consumer.py`), 'core 与它的消费者必须同栏');
    assert.equal(slices.flat().length, 3, '覆盖全部实现文件');
  } finally {
    fs.rmSync(abs(dir), { recursive: true, force: true });
  }
});

await test('idea 写了 owners 时按角色分片（同角色同栏、测试除外）', () => {
  const t = 'stage2_RQ-VAE/TIGER_x_rq/';
  const files = [
    `${t}a.py`, `${t}b.py`, `${t}c.py`, `${t}d.py`,
    `${t}tests/test_a.py`,
  ];
  const owners = {
    [`${t}a.py`]: 'geometry',
    [`${t}b.py`]: 'geometry',
    [`${t}c.py`]: 'model',
    [`${t}d.py`]: 'trainer',
    [`${t}tests/test_a.py`]: 'geometry',
  };
  const slices = partitionWriterFiles(files, 4, owners);
  const flat = slices.flat();
  assert.equal(flat.some((f) => f.includes('/tests/')), false, '测试不分配');
  const withA = slices.find((slice) => slice.includes(`${t}a.py`));
  assert.ok(withA.includes(`${t}b.py`), '同角色必须同栏');
  assert.equal(withA.includes(`${t}c.py`), false, '不同角色不能混到一栏');
  assert.equal(flat.length, 4);
});

await test('没有 owners 时退回 import 依赖分片', () => {
  const t = 'stage2_RQ-VAE/TIGER_x_rq/';
  const slices = partitionWriterFiles([`${t}a.py`, `${t}b.py`], 4, {});
  assert.equal(slices.flat().length, 2);
});

await test('当前 idea 的 owners 覆盖全部实现文件（没有漏网）', () => {
  const owners = currentIdeaOwners();
  const files = currentIdeaFiles().filter((f) => !f.includes('/tests/'));
  const missing = files.filter((f) => !owners[f]);
  assert.deepEqual(missing, [], `这些文件没有 owner：${missing.join(', ')}`);
});

await test('writer 的运行命令只描述功能，不含具体文件名', () => {
  for (const id of ['agent-w1', 'agent-w2', 'agent-w3', 'agent-w4']) {
    const prompt = resolveProfile(id).defaultPrompt || '';
    assert.equal(prompt.includes('stage2_RQ-VAE/'), false, `${id} 的预填里出现了文件路径`);
    assert.ok(prompt.includes('assignedFiles'), `${id} 的预填要指向 assignedFiles`);
  }
});

await test('assignedFiles 列出的就是这一栏的写入范围', async () => {
  for (const id of ['agent-w1', 'agent-w2', 'agent-w3', 'agent-w4']) {
    const profile = resolveProfile(id);
    const tool = toolsForPermissions(profile.permissions, {
      writeScope: profile.writeScope,
      denyTools: profile.denyTools,
    }).find((item) => item.name === 'assignedFiles');
    assert.ok(tool, `${id} 没有 assignedFiles 工具`);
    const out = JSON.parse((await tool.execute('call-assigned', {})).content[0].text);
    assert.deepEqual(out.files, profile.writeScope);
  }
});

await test('四个 writer 的分片只含实现文件：覆盖、互斥、不含测试', () => {
  const slices = ['agent-w1', 'agent-w2', 'agent-w3', 'agent-w4']
    .map((id) => resolveProfile(id).writeScope);
  const flat = slices.flat();
  assert.equal(flat.length, new Set(flat).size, '同一份文件不能分给两栏');
  assert.equal(flat.some((file) => file.includes('/tests/')), false, '写手不负责测试文件');
  const declared = currentIdeaFiles().filter((file) => !file.includes('/tests/'));
  const covered = new Set(flat);
  for (const file of declared) assert.ok(covered.has(file), `声明文件没被任何一栏覆盖：${file}`);
});

// 建树是四个 writer 任意一个都能做的事：先复制冻结基线，再写自己的文件。
const PROBE = 'TIGER_wscope_probe_rq';
const probeScope = [`stage2_RQ-VAE/${PROBE}/probe.py`];

await test('树还没建时，作用域内的写入被拒并提示先建树', async () => {
  rm(`stage2_RQ-VAE/${PROBE}`);
  const { call } = scoped(['write'], probeScope);
  await assert.rejects(
    () => call('write', { path: `stage2_RQ-VAE/${PROBE}/probe.py`, content: 'x = 1\n' }),
    /树缺失/,
  );
  assert.equal(fs.existsSync(abs(`stage2_RQ-VAE/${PROBE}`)), false, '不能顺手造出空树');
});

await test('四个 writer 的 writeScope 互不重叠，且都落在候选树里', () => {
  const scopes = ['agent-w1', 'agent-w2', 'agent-w3', 'agent-w4'].map(
    (id) => resolveProfile(id).writeScope,
  );
  const flat = scopes.flat();
  assert.equal(new Set(flat).size, flat.length, '同一个文件不能有两个 owner');
  for (const entry of flat) assert.match(entry, /^stage2_RQ-VAE\/TIGER_af3_curvature_factor_mass_rq\//);
});

rm(ROOT);
rm(`stage2_RQ-VAE/${PROBE}`);
console.log(`writeScope: ${passed} 通过`);
if (process.exitCode) process.exit(process.exitCode);
