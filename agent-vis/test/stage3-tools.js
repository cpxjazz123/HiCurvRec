// The Stage3 runner must actually receive the Stage3 tools -- the run that found
// stage3Start missing from its tool table is the reason this file exists.
import assert from 'node:assert/strict';

import { PERMISSION_CATALOG, resolvePermissions, resolveProfile } from '../config.js';
import { DSH_TOOLS, toolsForPermissions } from '../live-agent.js';

let passed = 0;

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  ✓ ${name}`);
  } catch (error) {
    process.exitCode = 1;
    console.log(`  ✗ ${name}`);
    console.log(`    ${error && error.message ? error.message : error}`);
  }
}

test('train 权限同时包含 Stage2 与 Stage3 工具', () => {
  const train = PERMISSION_CATALOG.find((item) => item.id === 'train');
  assert.ok(train, 'permission catalog lacks train');
  for (const name of ['stage2Start', 'stage2Status', 'stage2Stop', 'stage3Start', 'stage3Status']) {
    assert.ok(train.tools.includes(name), `train 权限缺 ${name}`);
  }
});

test('两个 Stage3 工具都有实现', () => {
  for (const name of ['stage3Start', 'stage3Status']) {
    assert.equal(typeof DSH_TOOLS[name], 'object', `DSH_TOOLS 缺 ${name}`);
    assert.equal(DSH_TOOLS[name].name, name);
    assert.ok(DSH_TOOLS[name].parameters, `${name} 没有参数 schema`);
  }
});

test('AGENT STAGE3 拿到 Stage3 工具、拿不到写工具与 Stage2 启停之外的东西', () => {
  const profile = resolveProfile('agent-stage3');
  const ids = resolvePermissions(profile);
  const tools = toolsForPermissions(ids, { denyTools: profile.denyTools || [] }).map((tool) => tool.name);
  assert.ok(tools.includes('stage3Start'), `STAGE3 缺少 stage3Start：${tools}`);
  assert.ok(tools.includes('stage3Status'), `STAGE3 缺少 stage3Status：${tools}`);
  for (const forbidden of ['write', 'edit', 'deleteFile', 'copyBaselineTree']) {
    assert.ok(!tools.includes(forbidden), `STAGE3 不该有 ${forbidden}`);
  }
  assert.ok(!tools.includes('stage2Start'), 'STAGE3 不该有 stage2Start（denyTools 应挡住）');
});

test('AGENT STAGE2 仍然拿到 Stage2 工具', () => {
  const profile = resolveProfile('agent-stage2');
  const ids = resolvePermissions(profile);
  const tools = toolsForPermissions(ids, { denyTools: profile.denyTools || [] }).map((tool) => tool.name);
  for (const name of ['stage2Start', 'stage2Status', 'stage2Stop']) {
    assert.ok(tools.includes(name), `STAGE2 缺少 ${name}`);
  }
});

console.log(`stage3-tools: ${passed} 通过`);
if (process.exitCode) process.exit(process.exitCode);
