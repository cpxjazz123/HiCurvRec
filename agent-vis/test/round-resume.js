// The rule that decides whether a multi-step round can be picked up again after the
// page was reloaded mid-round. Offline: no server, no browser.
import assert from 'node:assert/strict';

import { findUnfinishedRound } from '../public/round-resume.js';

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

const step = (roundId, stepIndex, totalSteps, status = 'done', extra = {}) => ({
  roundId,
  stepIndex,
  totalSteps,
  status,
  startedAt: 1_700_000_000_000 + stepIndex * 1000,
  finishedAt: 1_700_000_000_000 + stepIndex * 1000 + 500,
  roundPrompt: '跑这一轮',
  prompt: `第 ${stepIndex} 步`,
  ...extra,
});

test('没有记录 → 没有没跑完的轮次', () => {
  assert.equal(findUnfinishedRound([]), null);
  assert.equal(findUnfinishedRound(null), null);
});

test('一步的轮次不算"没跑完"（那是重来，不是接上）', () => {
  assert.equal(findUnfinishedRound([step('r1', 0, 1)]), null);
});

test('没有 roundId 的记录（单次运行）不算', () => {
  assert.equal(findUnfinishedRound([{ stepIndex: 0, totalSteps: 3, status: 'done' }]), null);
});

test('跑完的轮次 → 没有没跑完的轮次', () => {
  const records = [step('r1', 0, 3), step('r1', 1, 3), step('r1', 2, 3)];
  assert.equal(findUnfinishedRound(records), null);
});

test('页面刷新打断的轮次 → 自动接上，从下一步开始', () => {
  const records = [step('r1', 0, 3), step('r1', 1, 3)];
  const found = findUnfinishedRound(records);
  assert.ok(found, '应该认出这轮没跑完');
  assert.equal(found.roundId, 'r1');
  assert.equal(found.nextIndex, 2);
  assert.equal(found.totalSteps, 3);
  assert.equal(found.auto, true);
  assert.equal(found.reason, 'interrupted');
  assert.equal(found.roundPrompt, '跑这一轮');
});

test('人为中止的一步 → 不自动接（等人决定）', () => {
  const records = [step('r1', 0, 3), step('r1', 1, 3, 'aborted')];
  const found = findUnfinishedRound(records);
  assert.ok(found);
  assert.equal(found.auto, false);
  assert.equal(found.reason, 'aborted');
  assert.equal(found.nextIndex, 2);
});

test('出错的一步 → 不自动接', () => {
  const records = [step('r1', 0, 3), step('r1', 1, 3, 'error')];
  const found = findUnfinishedRound(records);
  assert.ok(found);
  assert.equal(found.auto, false);
  assert.equal(found.reason, 'error');
});

test('多轮时只认走得最远的那一轮', () => {
  const records = [
    step('r1', 0, 3),
    step('r1', 1, 3),
    step('r2', 0, 2),
    step('r2', 1, 2),
  ];
  assert.equal(findUnfinishedRound(records), null, '最后一轮跑完了 → 没有待接的');

  const records2 = [
    step('r1', 0, 3),
    step('r1', 1, 3),
    step('r2', 0, 4),
  ];
  const found = findUnfinishedRound(records2);
  assert.equal(found.roundId, 'r2');
  assert.equal(found.nextIndex, 1);
});

test('同一轮里重复的步骤按 stepIndex 排序后取最后一条', () => {
  const records = [step('r1', 1, 3), step('r1', 0, 3)];
  const found = findUnfinishedRound(records);
  assert.equal(found.nextIndex, 2);
});

console.log(`round-resume: ${passed} 通过`);
if (process.exitCode) process.exit(process.exitCode);
