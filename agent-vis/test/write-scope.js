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
import { currentIdeaFiles, currentWorkdir, resolveProfile } from '../config.js';

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

await test('四个 writer 的分片由当前 idea 算出来：覆盖全部声明文件、互不重叠', () => {
  const slices = ['agent-w1', 'agent-w2', 'agent-w3', 'agent-w4']
    .map((id) => resolveProfile(id).writeScope);
  const flat = slices.flat();
  assert.equal(flat.length, new Set(flat).size, '同一份文件不能分给两栏');
  const declared = currentIdeaFiles();
  const covered = new Set(flat);
  for (const file of declared) assert.ok(covered.has(file), `声明文件没被任何一栏覆盖：${file}`);
  // 每个实现文件都配一份同名测试，且跟它的实现同一栏
  for (const slice of slices) {
    for (const file of slice.filter((f) => !f.includes('/tests/'))) {
      const dir = file.slice(0, file.lastIndexOf('/'));
      const stem = file.slice(file.lastIndexOf('/') + 1, -3);
      const test = `${dir}/tests/test_${stem}.py`;
      assert.ok(slice.includes(test), `${file} 的测试不在同一栏：${test}`);
    }
  }
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
