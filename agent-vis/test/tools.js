import assert from 'node:assert/strict';

import { TOOL_IMPLS } from '../live-agent.js';

/**
 * 真实只读工具的检查：GitHub Issue 与 git commit 的读取。
 *
 * 这两个工具靠外部命令（gh / git）实现，不可能离线造假，所以单独成一个**小**套件：
 * 只打两三个外部查询，不调用模型 —— 原来那套 test/live.js 把整轮真实运行也塞在这里，
 * 一次几分钟还烧 token，那部分已经由 test/browser.js（真实模型 + 真浏览器）覆盖。
 */

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

process.stdout.write('agent-vis 只读工具（真实 gh / git）\n');

await test('能列出 GitHub 上的 Open Issue', async () => {
  const issues = await TOOL_IMPLS.listIssues({ limit: 5 });
  assert.ok(Array.isArray(issues), 'listIssues 没返回数组');
  for (const issue of issues) {
    assert.equal(typeof issue.number, 'number');
    assert.equal(typeof issue.title, 'string');
    assert.equal(issue.state, 'OPEN');
  }
  process.stdout.write(`      当前 ${issues.length} 个 Open Issue\n`);
});

await test('能读到某个 Issue 的正文', async () => {
  const issues = await TOOL_IMPLS.listIssues({ limit: 1 });
  if (issues.length === 0) {
    process.stdout.write('      当前没有 Open Issue，跳过读正文\n');
    return;
  }
  const detail = await TOOL_IMPLS.readIssue({ number: issues[0].number });
  assert.equal(detail.number, issues[0].number);
  assert.equal(typeof detail.body, 'string');
  assert.ok(detail.body.length > 0, 'Issue 正文是空的');
});

await test('能读到最近的 commit 及其改动文件', async () => {
  const log = await TOOL_IMPLS.gitLog({ count: 2 });
  assert.ok(Array.isArray(log.commits), 'gitLog 没返回 commits 数组');
  assert.ok(log.commits.length >= 1, '一条 commit 都没读到');
  const first = log.commits[0];
  assert.ok(/^[0-9a-f]{40}$/.test(first.sha), `commit sha 不对: ${first.sha}`);
  assert.equal(typeof first.subject, 'string');
  assert.ok(Array.isArray(first.files), 'commit 没有改动文件列表');
  process.stdout.write(`      最近 commit: ${first.sha.slice(0, 8)} ${first.subject}\n`);
});

process.stdout.write(`\n${passed} passed, ${failures.length} failed\n`);
if (failures.length > 0) {
  for (const { name, err } of failures) {
    process.stdout.write(`\n${name}\n${err && err.stack ? err.stack : err}\n`);
  }
  process.exit(1);
}
