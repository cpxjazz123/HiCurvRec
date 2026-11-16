import assert from 'node:assert/strict';

/**
 * dsh-toolset 插件的接线检查：只注册工具与提示词，不起端口。
 */
let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    process.stdout.write(`  ✓ ${name}\n`);
  } catch (err) {
    failures.push({ name, err });
    process.stdout.write(`  ✗ ${name}\n    ${err && err.message}\n`);
  }
}

const plugin = await import('../dsh-toolset.js');

function fakeCtx() {
  const registrations = [];
  const sections = [];
  const logs = [];
  return {
    registrations,
    sections,
    logs,
    tools: {
      register(def) {
        registrations.push(def);
        return () => {};
      },
    },
    systemPrompt: {
      section(section) {
        sections.push(section);
        return () => {};
      },
    },
    logger: {
      info(...args) {
        logs.push(args);
      },
    },
  };
}

function names(ctx) {
  return ctx.registrations.map((tool) => tool.name).sort();
}

process.stdout.write('agent-vis dsh toolset 插件检查\n\n');

test('插件形状：name / inject / apply', () => {
  assert.equal(plugin.name, 'agent-vis-toolset');
  assert.deepEqual(plugin.inject, ['tools', 'systemPrompt']);
  assert.equal(typeof plugin.apply, 'function');
});

test('默认只给只读工具，且不覆盖宿主同名工具', () => {
  const ctx = fakeCtx();
  plugin.apply(ctx, {});
  assert.deepEqual(names(ctx), [
    'checkIdea',
    'checkJson',
    'readDiff',
    'specCheck',
    'treeDiff',
  ]);
  for (const hostTool of ['read', 'write', 'edit', 'ls', 'grep', 'find']) {
    assert.ok(!names(ctx).includes(hostTool), `不该覆盖宿主工具 ${hostTool}`);
  }
  assert.deepEqual(plugin.mounted.permissions, ['read']);
  assert.equal(ctx.sections.length, 1);
  assert.ok(ctx.sections[0].text.includes('简体中文'));
  assert.ok(ctx.sections[0].text.includes('仓库工具'));
});

test('network 加 issue 工具，web 加文献检索', () => {
  const ctx = fakeCtx();
  plugin.apply(ctx, { permissions: ['read', 'network', 'web'] });
  const list = names(ctx);
  assert.ok(list.includes('listIssues'));
  assert.ok(list.includes('readIssue'));
  assert.ok(list.includes('webSearch'));
});

test('write 只加建树（write/edit 由宿主提供），delete 加删除', () => {
  const writeCtx = fakeCtx();
  plugin.apply(writeCtx, { permissions: ['write'] });
  assert.deepEqual(names(writeCtx), ['copyBaselineTree']);

  const deleteCtx = fakeCtx();
  plugin.apply(deleteCtx, { permissions: ['delete'] });
  assert.deepEqual(names(deleteCtx), ['deleteFile']);
});

test('execute 加 gitLog 与 runTests', () => {
  const ctx = fakeCtx();
  plugin.apply(ctx, { permissions: ['execute'] });
  assert.deepEqual(names(ctx), ['gitLog', 'runTests']);
});

test('train 加 Stage2/Stage3 启停与状态工具', () => {
  const ctx = fakeCtx();
  plugin.apply(ctx, { permissions: ['train'] });
  assert.deepEqual(names(ctx), [
    'stage2Start',
    'stage2Status',
    'stage2Stop',
    'stage3Start',
    'stage3Status',
  ]);
});

test('不认识的权限直接报错', () => {
  assert.throws(() => plugin.apply(fakeCtx(), { permissions: ['fly'] }), /fly/);
});

test('apply 不返回对象（Cordis 返回值必须是 effect）', () => {
  const ctx = fakeCtx();
  const returned = plugin.apply(ctx, {});
  assert.equal(returned, undefined);
});

process.stdout.write(`\n${passed} passed, ${failures.length} failed\n`);
if (failures.length > 0) {
  for (const { name, err } of failures) {
    process.stdout.write(`\n${name}\n${err && err.stack ? err.stack : err}\n`);
  }
  process.exit(1);
}
