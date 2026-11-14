/**
 * copyBaselineTree 的两种形态。
 *
 * 默认：复制基线并把 ARM_NAME 改成新树名（训练产物目录由它推出来）。
 * renameArm:false：除目录名外与基线逐字节相同 —— 只建"对应的机制树目录"时用这个。
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import { PI_TOOLS } from '../live-agent.js';
import { currentWorkdir } from '../config.js';

const REPO = currentWorkdir();
const PRISTINE = 'TIGER_copytest_pristine_rq';
const RENAMED = 'TIGER_copytest_renamed_rq';
const abs = (rel) => path.join(REPO, rel);

function rm(rel) {
  fs.rmSync(abs(rel), { recursive: true, force: true });
}
const call = async (args) => {
  const result = await PI_TOOLS.copyBaselineTree.execute('call-copy', args);
  return JSON.parse(result.content[0].text);
};

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

rm(`stage2_RQ-VAE/${PRISTINE}`);
rm(`stage2_RQ-VAE/${RENAMED}`);

await test('renameArm:false —— 目录名换了，内容与基线逐字节相同', async () => {
  const out = await call({ arm: PRISTINE, renameArm: false });
  assert.equal(out.pristine, true);
  assert.equal(fs.existsSync(abs(`stage2_RQ-VAE/${PRISTINE}`)), true);
  const diff = JSON.parse(
    (await PI_TOOLS.treeDiff.execute('c', {
      baseline: 'stage2_RQ-VAE/TIGER_RQ-VAE',
      tree: `stage2_RQ-VAE/${PRISTINE}`,
    })).content[0].text,
  );
  assert.equal(diff.summary.changed, 0, '内容不许有任何改动');
  assert.equal(diff.summary.onlyInTree, 0);
  assert.equal(diff.summary.onlyInBaseline, 0);
  const armLine = fs
    .readFileSync(abs(`stage2_RQ-VAE/${PRISTINE}/train_rqvae.py`), 'utf8')
    .split('\n')
    .find((line) => line.startsWith('ARM_NAME'));
  assert.match(armLine, /TIGER_RQ-VAE/, 'renameArm:false 时 ARM_NAME 保持基线值');
});

await test('默认形态 —— ARM_NAME 被改成新树名', async () => {
  await call({ arm: RENAMED });
  const armLine = fs
    .readFileSync(abs(`stage2_RQ-VAE/${RENAMED}/train_rqvae.py`), 'utf8')
    .split('\n')
    .find((line) => line.startsWith('ARM_NAME'));
  assert.match(armLine, new RegExp(RENAMED));
});

await test('目标已存在时拒绝覆盖', async () => {
  await assert.rejects(() => call({ arm: PRISTINE, renameArm: false }), /已经存在/);
});

rm(`stage2_RQ-VAE/${PRISTINE}`);
rm(`stage2_RQ-VAE/${RENAMED}`);
console.log(`copyBaselineTree: ${passed} 通过`);
if (process.exitCode) process.exit(process.exitCode);
