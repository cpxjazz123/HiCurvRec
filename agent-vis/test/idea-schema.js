import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  IDEA_EVAL_METRIC,
  IDEA_EVAL_THRESHOLD,
  IDEA_SCHEMA_VERSION,
  IDEA_SURFACES,
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
        mechanism: '表示空间换成逐商品曲率的产品流形，几何进分配代价与残差递推两处。',
        af3_entry: {
          signal: 'AF3 结构势 s_i（训练共现图上冻结）',
          formula: 'structure_potential',
          enters_at: ['assignment_cost', 'residual_update'],
        },
        one_variable: {
          changed: ['assignment_cost', 'residual_update'],
          unchanged: ['codebook', 'loss', 'encoder_output'],
        },
        must_hold: [
          { clause: 'must_hold#1', how: '结构只以逐商品标量进入分配代价，不做商品对比较。' },
          { clause: 'must_hold#2', how: '曲率由 s_i 闭式给出，并进分配代价与残差递推。' },
        ],
        novelty: {
          claim: '只主张残差在 Poincaré 球上做 Mobius 减后 Log 回原点切空间。',
          inherited_from: ['AF3_curvature_continuum_idea'],
          closest_works: [
            {
              title: 'Hyperbolic Residual Quantization',
              year: 2024,
              venue: 'NeurIPS',
              doi: '10.1234/abc',
              relation: 'same_family',
            },
          ],
        },
      },
      '4_method': {
        surfaces: [
          { surface: 'residual_update', symbol: 'ResidualTransport.forward', change: '欧氏减法换成 Mobius 减 + Log' },
        ],
        numerics: {
          dtype: 'fp32',
          clamps: [{ name: 'ARTANH_EPS', value: 1e-6, why: 'artanh 的定义域边界，取 IEEE 双精度下 1-1e-6' }],
          degenerate_to_baseline: 'TRANSPORT_MODE=euclidean 时距离与递推都退回 distances/distance.py',
        },
      },
      '5_references': [
        {
          title: 'Hyperbolic Residual Quantization',
          year: 2024,
          venue: 'NeurIPS',
          doi: '10.1234/abc',
          relation: 'same_family',
        },
      ],
      '6_evaluation': {
        metric: IDEA_EVAL_METRIC,
        threshold: IDEA_EVAL_THRESHOLD,
        comparison_arm: 'TIGER_mixed_rq_F0',
        falsifier: '若 test_R@10 不超过对照臂，则判否。',
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

await test('6_evaluation 自定门槛判不合规', () => {
  const doc = goodDoc();
  doc.sections['6_evaluation'].threshold = 0.0592;
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).includes('sections.6_evaluation.threshold'));
});

await test('指标名不是仓库契约就判不合规', () => {
  const doc = goodDoc();
  doc.sections['6_evaluation'].metric = 'stage2.l3_collision';
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).includes('sections.6_evaluation.metric'));
});

await test('must_hold 少一条就判不合规', () => {
  const doc = goodDoc();
  doc.sections['1_goal'].must_hold = [{ clause: 'must_hold#1', how: 'x' }];
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).includes('sections.1_goal.must_hold'));
});

await test('面名只能取那七个', () => {
  const doc = goodDoc();
  doc.sections['1_goal'].af3_entry.enters_at = ['whatever'];
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).some((p) => p.startsWith('sections.1_goal.af3_entry.enters_at')));
  assert.ok(IDEA_SURFACES.includes('assignment_cost'));
});

await test('idea 里不许有 check 类字段（验证计划归实现阶段）', () => {
  const doc = goodDoc();
  doc.sections['4_method'].sanity_checks = ['一句能机械判定的话'];
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).includes('sections.4_method.sanity_checks'));
});

await test('文献里的被引数判不合规（没有消费者）', () => {
  const doc = goodDoc();
  doc.sections['5_references'][0].cited_by = 42;
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).includes('sections.5_references[0].cited_by'));
});

await test('多种子计划在值上被认出来', () => {
  const doc = goodDoc();
  doc.sections['4_method'].sanity_checks = ['跑 seed 42/43/44 三次取均值'];
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, false);
  assert.ok(pathsOf(verdict).includes('$'), '要有一条指向整份文档的错误');
});

await test('否定句"不做多种子"不会被误伤', () => {
  const doc = goodDoc();
  doc.sections['1_goal'].mechanism = '表示空间换成逐商品曲率的产品流形；不做多种子实验，也不训练消融臂。';
  const verdict = validateIdeaDoc(doc);
  assert.equal(verdict.ok, true, `不该有错：${JSON.stringify(verdict.errors)}`);
});

await test('真实那份 transport idea 判不合规（当时把训练计划写进了稿子）', () => {
  const file = path.join(REPO, 'ideas', 'AF3_curvature_transport_idea', 'AF3_curvature_transport_idea.json');
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
  assert.ok(paths.includes('sections.6_evaluation.threshold'), '要指出自定门槛');
});

process.stdout.write(`\n${passed} 通过, ${failures.length} 失败\n`);
if (failures.length > 0) {
  for (const f of failures) {
    process.stdout.write(`  ✗ ${f.name}\n${f.err && f.err.stack ? f.err.stack : ''}\n`);
  }
  process.exitCode = 1;
}
