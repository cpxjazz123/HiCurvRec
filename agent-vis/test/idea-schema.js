import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  IDEA_SCHEMA_VERSION,
  ideaSkeleton,
  validateIdeaDoc,
} from '../idea-schema.js';

/**
 * idea 字段契约（genrec.idea.v2）的离线检查。
 *
 * 这里守住的是"形状"这一层：模型写 idea 时不能自由发挥到把运行计划、
 * 自定门槛、状态塞进稿子里。内容好不好是 AGENT AGAINST 的活，这里不判。
 *
 * 反例用的是仓库里真实存在的稿子（AGENT IDEA 之前写的那份 transport idea），
 * 不是人造的：它当时真的把 13 次训练、多 seed 统计、自定 success threshold
 * 写进了 idea —— 那正是这套字段要挡住的东西。
 */

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.join(HERE, '..', '..');
/** 冻结基线树：用来分辨"新文件"与"基线里已有、这次要改的文件"。 */
const BASELINE_ROOT = path.join(REPO, 'stage2_RQ-VAE', 'TIGER_RQ-VAE');

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

/** 一份各字段都合规的样例（骨架填满）。 */
function goodDoc() {
  return {
    schema: IDEA_SCHEMA_VERSION,
    arm: 'TIGER_af3_curvature_transport_rq',
    sections: {
      '1_goal': {
        idea:
          '表示空间换成逐商品曲率的产品流形，结构势决定每个商品落在哪一段曲率上，' +
          '几何进该层的分配代价与残差递推两处；与已有的双曲量化工作相比，' +
          '曲率由商品侧的结构势给出，而不是固定值。',
      },
      '4_method': {
        new_files: [
          { path: 'af3_curvature_transport.py', what: '算曲率表并冻结，输出逐商品的曲率' },
          { path: 'model/product_geometry.py', what: '产品流形的指数映射与对数映射' },
        ],
        modified_files: [
          { path: 'model/layers.py', what: '把欧氏减法换成 Mobius 减 + Log' },
        ],
      },
    },
  };
}

/** 把错误表压成一行，便于断言。 */
function pathsOf(verdict) {
  return verdict.errors.map((e) => e.path);
}

await test('合规稿子通过', () => {
  const verdict = validateIdeaDoc(goodDoc());
  assert.equal(verdict.ok, true, `不该有错：${JSON.stringify(verdict.errors)}`);
});

/** 收集一份文档的键路径（数组下标不算键）。 */
function keyPaths(value, prefix = '', out = []) {
  if (Array.isArray(value)) {
    value.forEach((item) => keyPaths(item, prefix, out));
    return out;
  }
  if (!value || typeof value !== 'object') return out;
  for (const [key, child] of Object.entries(value)) {
    const here = prefix ? `${prefix}.${key}` : key;
    out.push(here);
    keyPaths(child, here, out);
  }
  return out;
}

await test('骨架的键形状与合规稿子一致（提示词与校验器不许漂）', () => {
  const skeleton = JSON.parse(ideaSkeleton());
  const fromSkeleton = new Set(keyPaths(skeleton));
  const fromGood = new Set(keyPaths(goodDoc()));
  const missing = [...fromGood].filter((key) => !fromSkeleton.has(key)).sort();
  const extra = [...fromSkeleton].filter((key) => !fromGood.has(key)).sort();
  assert.deepEqual(missing, [], '骨架少了这些键');
  assert.deepEqual(extra, [], '骨架多了这些键');
});

await test('顶层多一个键就判不合规（状态/日期/轮次流水不归 idea）', () => {
  const doc = goodDoc();
  doc.result = { status: 'not_run' };
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).includes('result'), '要指出 result 这个键');
});

await test('4_method.runs 不存在（运行计划归框架）', () => {
  const doc = goodDoc();
  doc.sections['4_method'].runs = '13 训 + 13 次 Stage3';
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).includes('sections.4_method.runs'));
});

await test('判据整段都不许出现在 idea 里（门槛是仓库契约，不是 idea 自己定）', () => {
  const doc = goodDoc();
  doc.sections['6_evaluation'] = {
    metric: 'stage3.test_recall@10',
    threshold: 0.0592,
    comparison_arm: 'TIGER_mixed_rq_F0',
    falsifier: '若不超过对照臂则判否',
  };
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).includes('sections.6_evaluation'));
});

await test('must_hold 少一条就判不合规', () => {
  const doc = goodDoc();
  doc.sections['1_goal'].must_hold = [{ clause: 'must_hold#1', how: 'x' }];
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).includes('sections.1_goal.must_hold'));
});

await test('new_files 的每一项都要有路径与用途', () => {
  const doc = goodDoc();
  doc.sections['4_method'].new_files = [{ path: '', what: '' }];
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  const paths = pathsOf(verdict);
  assert.ok(paths.includes('sections.4_method.new_files[0].path'));
  assert.ok(paths.includes('sections.4_method.new_files[0].what'));
});

await test('idea 里只留那三件事：多出来的字段一律判不合规', () => {
  const doc = goodDoc();
  doc.sections['1_goal'].af3_entry = { signal: 's_i', formula: 'x', enters_at: ['loss'] };
  doc.sections['1_goal'].must_hold = [{ clause: 'must_hold#1', how: 'x' }];
  doc.sections['1_goal'].novelty = { claim: 'x', inherited_from: [], closest_works: [] };
  doc.sections['4_method'].surfaces = [{ surface: 'loss', symbol: 'x', change: 'y' }];
  doc.sections['4_method'].numerics = { dtype: 'fp32', clamps: [], degenerate_to_baseline: 'x' };
  doc.sections['4_method'].sanity_checks = ['一句能机械判定的话'];
  doc.sections['6_evaluation'] = { metric: 'stage3.test_recall@10', threshold: 0.065 };
  doc.sections['5_references'] = [];
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  const paths = pathsOf(verdict);
  for (const key of [
    'sections.1_goal.af3_entry',
    'sections.1_goal.must_hold',
    'sections.1_goal.novelty',
    'sections.4_method.surfaces',
    'sections.4_method.numerics',
    'sections.4_method.sanity_checks',
    'sections.6_evaluation',
    'sections.5_references',
  ]) {
    assert.ok(paths.includes(key), `要指出 ${key}`);
  }
});

await test('参考文献整段不许出现在 idea 里（检索照做，但不写进文件）', () => {
  const doc = goodDoc();
  doc.sections['5_references'] = [
    { title: 'A', year: 2024, venue: 'NeurIPS', doi: '10.1/a', relation: 'same_family' },
  ];
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).includes('sections.5_references'));
});

await test('基线里已有的文件写进 new_files 判不合规（复制基线不算新建）', () => {
  const doc = goodDoc();
  doc.sections['4_method'].new_files.push({ path: 'model/layers.py', what: '照抄基线' });
  const verdict = validateIdeaDoc(doc, { baselineRoot: BASELINE_ROOT });
  assert.equal(verdict.ok, false);
  const hit = verdict.errors.find((e) => e.path.endsWith('new_files[2].path'));
  assert.ok(hit, '要指出那一条');
  assert.match(hit.why, /基线里已经有这个文件/);
});

await test('基线里没有的文件写进 modified_files 判不合规', () => {
  const doc = goodDoc();
  doc.sections['4_method'].modified_files.push({ path: 'totally_new_thing.py', what: '新的' });
  const verdict = validateIdeaDoc(doc, { baselineRoot: BASELINE_ROOT });
  assert.equal(verdict.ok, false);
  const hit = verdict.errors.find((e) => e.path.endsWith('modified_files[1].path'));
  assert.ok(hit, '要指出那一条');
  assert.match(hit.why, /基线里没有这个文件/);
});

await test('多种子计划在值上被认出来', () => {
  const doc = goodDoc();
  doc.sections['1_goal'].idea = '跑 seed 42/43/44 三次取均值';
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).includes('$'), '要有一条指向整份文档的错误');
});

await test('否定句"不做多种子"不会被误伤', () => {
  const doc = goodDoc();
  doc.sections['1_goal'].idea = '表示空间换成逐商品曲率的产品流形；不做多种子实验，也不训练消融臂。';
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, true, `不该有错：${JSON.stringify(verdict.errors)}`);
});

await test('真实那份 transport idea 判不合规（当时把训练计划写进了稿子）', () => {
  const file = path.join(
    REPO,
    'ideas',
    '_archive',
    'AF3_curvature_transport_idea',
    'AF3_curvature_transport_idea.json',
  );
  if (!fs.existsSync(file)) {
    process.stdout.write('    （跳过：稿子不在盘上）\n');
    return;
  }
  const doc = JSON.parse(fs.readFileSync(file, 'utf8'));
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false, '这份稿子按 v2 必须判不合规');
  const paths = pathsOf(verdict);
  assert.ok(paths.includes('date'), '要指出顶层多出来的键');
  assert.ok(paths.includes('sections.4_method.runs'), '要指出运行计划那一项');
  assert.ok(paths.includes('sections.1_goal.statement'), '要指出多出来的目标字段');
  assert.ok(paths.includes('sections.6_evaluation'), '要指出整段判据不该在 idea 里');
});

process.stdout.write(`\n${passed} 通过, ${failures.length} 失败\n`);
if (failures.length > 0) {
  for (const f of failures) {
    process.stdout.write(`  ✗ ${f.name}\n${f.err && f.err.stack ? f.err.stack : ''}\n`);
  }
  process.exitCode = 1;
}
