/**
 * specCheck 的自测。
 *
 * 它接手的是 step 2 里"本来就该机器判"的那七类：文件齐缺、路径违规、身份、
 * 冻结超参、metrics 落地、空断言。所以每一条都必须真的能抓出来 ——
 * 抓不到的预检等于把模型的眼睛换成了一个更不可靠的眼睛。
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import { PI_TOOLS } from '../live-agent.js';
import { currentWorkdir } from '../config.js';

const REPO = currentWorkdir();
const ROOT = 'agent-vis/.out/spec-check-fixture';
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

const BASE = `${ROOT}/base`;
const TREE = `${ROOT}/tree`;
const IDEA = `${ROOT}/idea`;

rm(ROOT);
// 基线
write(`${BASE}/config/train_config.json`, JSON.stringify({ epochs: 3000, secret: 0.003 }, null, 2));
write(`${BASE}/train_config.py`, 'EPOCHS = 3000\nSK_EPSILON = 0.003\n');
write(`${BASE}/model/layers.py`, 'def sinkhorn(x):\n    return x\n');
// 候选：一个常量被改、一个 JSON 项被改、一个绝对路径、一个 parents[N]
write(`${TREE}/config/train_config.json`, JSON.stringify({ epochs: 3000, secret: 0.01 }, null, 2));
write(`${TREE}/train_config.py`, 'EPOCHS = 1200\nSK_EPSILON = 0.003\nNEW_CONST = 0.5\n');
write(`${TREE}/af3_item_potential.py`, 'DATA = "/home/someone/data.npy"\n');
write(`${TREE}/factor_share_geometry.py`, 'import pathlib\nROOT = pathlib.Path(__file__).resolve().parents[3]\n');
// 声明"改过"但其实没改的文件
write(`${BASE}/distances/distance.py`, 'X = 1\n');
write(`${TREE}/distances/distance.py`, 'X = 1\n');
// 跨臂引用：一行引用别的臂（错），一行引用本树与基线（对）
write(`${TREE}/model/layers.py`,
  'ARM_NAME = "TIGER_fixture"\n'
  + '# depends on stage2_RQ-VAE/TIGER_other_rq/af3_potential.py (external!)\n'
  + '# ok: stage2_RQ-VAE/TIGER_fixture/af3_potential.py and stage2_RQ-VAE/TIGER_RQ-VAE/train_config.py\n'
  + '\ndef sinkhorn(x):\n    return x\n');
// 测试：一个健康、一个无断言；metrics 名落地一个、缺一个
write(`${TREE}/tests/test_ok.py`, 'def test_ok():\n    assert good_metric == 1\n');
write(`${TREE}/tests/test_empty.py`, 'def test_nothing():\n    print("hi")\n');
write(`${TREE}/train_rqvae.py`, 'good_metric = 1\n');
// idea
write(`${IDEA}/fixture_idea.json`, JSON.stringify({
  arm: 'TIGER_fixture',
  sections: {
    '4_method': {
      new_files: [
        { path: `${TREE}/af3_item_potential.py`, what: 'a' },
        { path: `${TREE}/factor_share_geometry.py`, what: 'b' },
        { path: `${TREE}/missing_module.py`, what: 'c' },
      ],
      modified_files: [
        { path: `${TREE}/distances/distance.py`, what: 'declared changed but is not' },
      ],
    },
  },
}, null, 2));
write(`${IDEA}/math.json`, JSON.stringify({
  metrics: { good_metric: 'present', absent_metric: 'never implemented' },
  constants: [{ name: 'EPOCHS', value: 1200 }, { name: 'NOT_A_CONSTANT', value: 1 }],
  formulas: [{ name: 'sinkhorn' }, { name: 'never_named_formula' }],
}, null, 2));

const call = async (args) => {
  const result = await PI_TOOLS.specCheck.execute('call-1', args);
  return JSON.parse(result.content[0].text);
};

const out = await call({ tree: TREE, idea: IDEA, baseline: BASE });

await test('声明的 new_files 缺哪个就报哪个', () => {
  const missing = out.files.newFiles.map((entry) => entry.file);
  assert.deepEqual(missing, ['missing_module.py']);
  assert.equal(out.files.newFilesChecked, 3);
});

await test('声明改了但和基线逐字节相同 → 列出来', () => {
  assert.equal(out.files.modifiedFiles.length, 1);
  assert.equal(out.files.modifiedFiles[0].file, 'distances/distance.py');
  assert.equal(out.files.modifiedFiles[0].differsFromBaseline, false);
});

await test('绝对路径与 parents[N] 都能抓出来', () => {
  assert.ok(out.paths.absolute.total >= 1, JSON.stringify(out.paths.absolute));
  assert.match(out.paths.absolute.hits[0].where, /af3_item_potential\.py:1/);
  assert.ok(out.paths.parentsN.total >= 1, JSON.stringify(out.paths.parentsN));
});

await test('跨臂引用被点名，本树与基线的引用不算', () => {
  assert.equal(out.paths.crossArm.total, 1, JSON.stringify(out.paths.crossArm));
  assert.equal(out.paths.crossArm.hits[0].arm, 'TIGER_other_rq');
  assert.match(out.paths.crossArm.hits[0].where, /model\/layers\.py:2/);
});

await test('ARM_NAME 取值被抓出来', () => {
  assert.deepEqual(out.paths.armNames, [{ where: 'model/layers.py', value: 'TIGER_fixture' }]);
});

await test('冻结超参：JSON 项与 Python 常量分别报变化', () => {
  assert.deepEqual(out.frozen.json.changed.map((x) => x.key), ['secret']);
  assert.deepEqual(out.frozen.constants.changed.map((x) => x.key), ['EPOCHS']);
  assert.deepEqual(out.frozen.constants.added, ['NEW_CONST']);
});

await test('规格里的 metrics 没落地会被点名', () => {
  assert.deepEqual(out.metrics.missing, ['absent_metric']);
  assert.equal(out.metrics.declared, 2);
});

await test('规格声明的常量与公式名定位到文件:行', () => {
  const constants = Object.fromEntries(out.declarations.constants.map((e) => [e.name, e]));
  assert.match(constants.EPOCHS.where, /train_config\.py:\d+/);
  assert.equal(constants.NOT_A_CONSTANT.hits, 0);
  const formulas = Object.fromEntries(out.declarations.formulas.map((e) => [e.name, e]));
  assert.ok(formulas.sinkhorn.hits >= 1, JSON.stringify(formulas.sinkhorn));
  assert.equal(formulas.never_named_formula.hits, 0);
});

await test('无断言的测试文件进 suspicious', () => {
  const flagged = out.tests.suspicious.map((entry) => entry.file);
  assert.deepEqual(flagged, ['tests/test_empty.py']);
  assert.equal(out.tests.files, 2, '只统计有 def test_ 的文件');
});

await test('树不存在时返回错误而不是抛栈', async () => {
  const bad = await call({ tree: `${ROOT}/no-such-tree`, idea: IDEA, baseline: BASE });
  assert.equal(bad.ok, false);
  assert.match(bad.error, /不存在/);
});


// ---- 插件宿主：改动面只允许 mechanism.py + ARM_NAME --------------------------
const PB = `${ROOT}/plugin-base`;
const PT = `${ROOT}/plugin-tree`;
const PT2 = `${ROOT}/plugin-tree-extra-new`;
const PT3 = `${ROOT}/plugin-tree-extra-changed`;

write(`${PB}/mechanism_base.py`, '"""host marker"""\n');
write(`${PB}/train_rqvae.py`, 'ARM_NAME = "TIGER_RQ-VAE"\nprint("baseline")\n');
write(`${PB}/config/train_config.json`, JSON.stringify({ epochs: 3000 }, null, 2));
write(`${PB}/distances/distance.py`, 'X = 1\n');
write(`${PB}/model/layers.py`, 'Y = 1\n');
for (const dir of [PT, PT2, PT3]) {
  write(`${dir}/mechanism_base.py`, '"""host marker"""\n');
  write(`${dir}/train_rqvae.py`, 'ARM_NAME = "TIGER_rq_plugin"\nprint("baseline")\n');
  write(`${dir}/config/train_config.json`, JSON.stringify({ epochs: 3000 }, null, 2));
  write(`${dir}/distances/distance.py`, 'X = 1\n');
  write(`${dir}/mechanism.py`, 'def distance(a, b):\n    return a\n');
}
write(`${PT2}/extra_idea_file.py`, 'pass\n');
write(`${PT3}/distances/distance.py`, 'X = 2\n');

await test('插件宿主：只加 mechanism.py + 改 ARM_NAME → surface.ok', async () => {
  const out = await call({ tree: PT, baseline: PB, idea: IDEA });
  assert.ok(out.surface, 'surface 缺失');
  assert.equal(out.surface.pluginHost, true);
  assert.equal(out.surface.mechanism, true);
  assert.deepEqual(out.surface.unexpectedNew, []);
  assert.deepEqual(out.surface.unexpectedChanged, []);
  assert.equal(out.surface.armNameOnly.ok, true, JSON.stringify(out.surface.armNameOnly));
  assert.equal(out.surface.ok, true);
});

await test('插件宿主：多出一个新文件 → 被点名', async () => {
  const out = await call({ tree: PT2, baseline: PB, idea: IDEA });
  assert.deepEqual(out.surface.unexpectedNew, ['extra_idea_file.py']);
  assert.equal(out.surface.ok, false);
});

await test('插件宿主：改动别的文件 → 被点名', async () => {
  const out = await call({ tree: PT3, baseline: PB, idea: IDEA });
  assert.equal(out.surface.unexpectedChanged.length, 1);
  assert.equal(out.surface.unexpectedChanged[0].file, 'distances/distance.py');
  assert.equal(out.surface.ok, false);
});

await test('非插件基线：surface 为 null（沿用原规则）', async () => {
  assert.equal(out.surface, null);
});


// ---- touch：idea 直接声明"允许改哪些文件" ----------------------------------
const TD = `${ROOT}/touch-idea`;
const PT4 = `${ROOT}/touch-tree`;
const PT5 = `${ROOT}/touch-tree-extra`;
for (const dir of [PT4, PT5]) {
  write(`${dir}/mechanism_base.py`, '"""host marker"""\n');
  write(`${dir}/train_rqvae.py`, 'ARM_NAME = "TIGER_touch"\nprint("baseline")\n');
  write(`${dir}/distances/distance.py`, 'X = 2\n');   // 允许：touch 里写了它
  write(`${dir}/model/layers.py`, 'Y = 1\n');
}
write(`${PT5}/model/layers.py`, 'Y = 2\n');           // 不允许：不在 touch 里
function writeTouchIdea(dir, tree) {
  write(`${dir}/touch_idea.json`, JSON.stringify({
    schema: 'genrec.idea.v6',
    arm: 'TIGER_touch',
    sections: {
      '1_goal': { idea: 'touch smoke' },
      '4_method': {
        new_files: [],
        modified_files: [{ path: `${tree}/distances/distance.py`, what: 'only this file' }],
        touch: [`${tree}/distances/distance.py`],
      },
    },
  }, null, 2));
  write(`${dir}/math.json`, JSON.stringify({ metrics: {} }));
}
writeTouchIdea(TD, PT4);
writeTouchIdea(`${ROOT}/touch-idea-extra`, PT5);

await test('touch：只改清单里的文件 → surface.ok', async () => {
  const out = await call({ tree: PT4, baseline: PB, idea: TD });
  assert.equal(out.surface.mode, 'touch');
  assert.deepEqual(out.surface.allowed, ['distances/distance.py']);
  assert.deepEqual(out.surface.unexpectedChanged, []);
  assert.equal(out.surface.ok, true);
});

await test('touch：改了清单外的文件 → 被点名', async () => {
  const out = await call({ tree: PT5, baseline: PB, idea: `${ROOT}/touch-idea-extra` });
  assert.deepEqual(out.surface.unexpectedChanged.map((x) => x.file), ['model/layers.py']);
  assert.equal(out.surface.ok, false);
});

rm(ROOT);
console.log(`specCheck: ${passed} 通过`);
if (process.exitCode) process.exit(process.exitCode);
