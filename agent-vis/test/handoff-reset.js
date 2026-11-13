import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createServer } from '../server.js';
import { buildRunRequestBody } from '../public/run-request.js';

/**
 * 交接时的会话重置 —— 回归测试。
 *
 * 这个功能曾经**从来没有生效过**，而且两头都没人看见：
 *   1. 页面发的请求里没有发送方的身份（只有名字），服务端不知道该不该重置；
 *   2. 服务端拿到身份后，重置代码写在接收方身份解析之前，撞上 const 的暂时性死区，
 *      抛出的 ReferenceError 被 catch 吃掉，日志里只留下一句
 *      "交接来源认不出来" —— 看不出真正原因。
 *
 * 所以这里测两件事：页面发的身份在不在，服务端收到身份后会不会崩。
 */

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SCRATCH = path.join(HERE, '..', '.out', 'handoff-reset-test');

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

function listen(server) {
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => resolve(server.address().port));
  });
}

await test('页面把发送方的身份一起发出去（没有它，服务端不知道该不该重置）', () => {
  const body = buildRunRequestBody({
    prompt: '继续',
    version: 'v1',
    profile: 'agent-b',
    handoff: '上游的结果',
    handoffFrom: 'AGENT A',
    handoffFromId: 'agent-a',
    round: null,
  });
  assert.equal(body.handoffFrom, 'AGENT A');
  assert.equal(body.handoffFromId, 'agent-a', '请求体里必须有发送方的身份');
});

await test('没有上游时不带来源字段', () => {
  const body = buildRunRequestBody({ prompt: '跑', version: 'v1', profile: 'agent-a' });
  assert.equal(body.handoff, '');
  assert.equal('handoffFrom' in body, false);
  assert.equal('handoffFromId' in body, false);
});

await test('轮次里的那一步带上它在整轮里的位置', () => {
  const body = buildRunRequestBody({
    prompt: '第二步',
    version: 'v1',
    profile: 'agent-a',
    handoff: '',
    round: {
      roundId: 'r-1',
      stepIndex: 1,
      stepTitle: '第二步',
      stepInstruction: '做第二步',
      totalSteps: 3,
      roundPrompt: '本轮输入',
    },
  });
  assert.equal(body.roundId, 'r-1');
  assert.equal(body.stepIndex, 1);
  assert.equal(body.totalSteps, 3);
  assert.equal('roundPrompt' in body, false, '本轮输入只在第一步发');
});

await test('服务端收到发送方身份时不报错（重置代码曾经撞上暂时性死区）', async () => {
  fs.rmSync(SCRATCH, { recursive: true, force: true });
  const server = createServer({ mode: 'mock', publicPath: '', dataDir: SCRATCH });
  const port = await listen(server);
  const base = `http://127.0.0.1:${port}`;

  // 抓服务端往 stderr 写的东西：重置失败时它会写一句警告。
  const written = [];
  const originalWrite = process.stderr.write.bind(process.stderr);
  process.stderr.write = (chunk, ...rest) => {
    written.push(String(chunk));
    return originalWrite(chunk, ...rest);
  };
  try {
    // 发送方要打开"重置接收方会话"这个开关，重置那段代码才会真的走到 ——
    // 开关关着的时候，旧版代码里的那行会整段被跳过，测不出问题。
    const cfg = await fetch(`${base}/api/agents/agent-a/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ resetReceiverSessionOnHandoff: true }),
    });
    assert.equal(cfg.status, 200);

    const res = await fetch(`${base}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        v: (await import('../config.js')).FRONTEND_VERSION,
        profile: 'agent-b',
        prompt: '接着做',
        handoff: '上游交下来的结果',
        handoffFrom: 'AGENT A',
        handoffFromId: 'agent-a',
      }),
    });
    const text = await res.text();
    assert.equal(res.status, 200);
    assert.ok(!text.includes('"type":"error"'), `不该是错误流：${text.slice(0, 200)}`);
    const warnings = written.filter((line) => line.includes('认不出来'));
    assert.deepEqual(warnings, [], '重置代码不该抛错：那说明它又踩到暂时性死区了');
  } finally {
    process.stderr.write = originalWrite;
    server.close();
  }
});

process.stdout.write(`\n${passed} 通过, ${failures.length} 失败\n`);
if (failures.length > 0) {
  for (const f of failures) {
    process.stdout.write(`  ✗ ${f.name}\n${f.err && f.err.stack ? f.err.stack : ''}\n`);
  }
  process.exitCode = 1;
}
