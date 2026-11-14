/**
 * treeDiff 的自测。
 *
 * 它替掉的是一段很贵的动作：逐个 ls/find、两侧 read、再人眼比对。
 * 所以它的每一种结论都必须真的有依据 —— 相同、改动（带 diff）、只在一边，
 * 以及"缓存与二进制不进表"这条边界。
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import { PI_TOOLS } from '../live-agent.js';
import { currentWorkdir } from '../config.js';

// 夹具放仓库内的临时目录（不放 /tmp）；工具按"仓库根相对路径"解析，所以两边都要换算。
const REPO = currentWorkdir();
const ROOT = 'agent-vis/.out/tree-diff-fixture';
const BASE = `${ROOT}/base`;
const TREE = `${ROOT}/tree`;
const abs = (rel) => path.join(REPO, rel);

function rm(rel) {
  fs.rmSync(abs(rel), { recursive: true, force: true });
}
function write(rel, text) {
  const target = abs(rel);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, text);
}

let passed = 0;
function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  ✓ ${name}`);
  } catch (err) {
    console.error(`  ✗ ${name}`);
    console.error(`    ${err && err.message ? err.message : err}`);
    process.exitCode = 1;
  }
}
async function testAsync(name, fn) {
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
write(`${BASE}/same.py`, 'x = 1\n');
write(`${TREE}/same.py`, 'x = 1\n');
write(`${BASE}/mod.py`, 'a = 1\n');
write(`${TREE}/mod.py`, 'a = 1\nb = 2\n');
write(`${BASE}/gone.py`, 'only baseline\n');
write(`${TREE}/new.py`, 'only tree\n');
write(`${BASE}/__pycache__/same.cpython-39.pyc`, 'binary-ish\n');
write(`${TREE}/__pycache__/same.cpython-39.pyc`, 'binary-ish\n');
write(`${BASE}/weights.npz`, 'not text\n');
write(`${TREE}/weights.npz`, 'not text\n');

const diff = async (args) => {
  const result = await PI_TOOLS.treeDiff.execute('call-1', args);
  return JSON.parse(result.content[0].text);
};

await testAsync('相同 / 改动 / 只在一边 分别归位', async () => {
  const out = await diff({ baseline: BASE, tree: TREE });
  assert.equal(out.ok, true, out.error);
  assert.deepEqual(out.summary, {
    identical: 1,
    changed: 1,
    onlyInTree: 1,
    onlyInBaseline: 1,
  });
  assert.deepEqual(out.identical, ['same.py']);
  assert.deepEqual(out.onlyInTree, ['new.py']);
  assert.deepEqual(out.onlyInBaseline, ['gone.py']);
  assert.equal(out.changed[0].file, 'mod.py');
  assert.equal(out.changed[0].added, 1);
  assert.equal(out.changed[0].removed, 0);
});

await testAsync('改动的文件带 unified diff', async () => {
  const out = await diff({ baseline: BASE, tree: TREE });
  const hunks = out.changed[0].hunks;
  assert.match(hunks, /\+b = 2/);
  assert.match(hunks, / a = 1/);
});

await testAsync('缓存目录与二进制不进表', async () => {
  const out = await diff({ baseline: BASE, tree: TREE });
  const all = [...out.identical, ...out.onlyInTree, ...out.onlyInBaseline,
    ...out.changed.map((entry) => entry.file)];
  assert.equal(all.some((name) => name.includes('__pycache__')), false);
  assert.equal(all.some((name) => name.endsWith('.npz')), false);
});

await testAsync('不给 baseline 时用冻结基线作默认值', async () => {
  const out = await diff({ tree: TREE });
  assert.equal(out.ok, true, out.error);
  assert.equal(out.baseline, 'stage2_RQ-VAE/TIGER_RQ-VAE');
});

await testAsync('路径不存在时返回错误而不是抛栈', async () => {
  const out = await diff({ baseline: `${ROOT}/no-such-dir`, tree: TREE });
  assert.equal(out.ok, false);
  assert.match(out.error, /不存在/);
});

await testAsync('越界的路径被拒', async () => {
  await assert.rejects(() => diff({ baseline: '../../../etc', tree: TREE }), /越界/);
});

const diffCall = async (args) => {
  const result = await PI_TOOLS.readDiff.execute('call-2', args);
  return JSON.parse(result.content[0].text);
};

await testAsync('readDiff 单文件：只回改动，不改的文件不进列表', async () => {
  const out = await diffCall({ baseline: BASE, tree: TREE, file: 'mod.py' });
  assert.equal(out.ok, true, out.error);
  assert.equal(out.files.length, 1);
  assert.equal(out.files[0].file, 'mod.py');
  assert.match(out.files[0].diff, /\+b = 2/);
  assert.equal(out.files[0].truncated, false);
  const same = await diffCall({ baseline: BASE, tree: TREE, file: 'same.py' });
  assert.deepEqual(same.files, []);
});

await testAsync('readDiff 全树：新增与删除的文件标 onlyIn', async () => {
  const out = await diffCall({ baseline: BASE, tree: TREE });
  const byFile = Object.fromEntries(out.files.map((entry) => [entry.file, entry]));
  assert.equal(byFile['new.py'].onlyIn, 'tree');
  assert.equal(byFile['gone.py'].onlyIn, 'baseline');
  assert.ok(byFile['mod.py'], '改动文件必须出现');
  assert.equal(byFile['same.py'], undefined, '相同文件不该出现');
});

rm(ROOT);
console.log(`treeDiff: ${passed} 通过`);
if (process.exitCode) process.exit(process.exitCode);
