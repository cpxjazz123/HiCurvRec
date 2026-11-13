/**
 * 离线剧本。
 *
 * 产出与真实 agent 完全同构的 UI message stream 分片，用来在没有网络时验证
 * 协议、前端渲染与测试。它不属于线上路径：真实路径见 live-agent.js。
 */
import fs from 'node:fs';
import path from 'node:path';

import { REPO_ROOT, handoffSourceLabel, resolvePermissions, resolveProfile } from './config.js';

const USAGE = {
  inputTokens: 1842,
  outputTokens: 317,
  reasoningTokens: 118,
  totalTokens: 2159,
  steps: 2,
};

function sleep(ms, signal) {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    if (signal) {
      signal.addEventListener(
        'abort',
        () => {
          clearTimeout(timer);
          resolve();
        },
        { once: true },
      );
    }
  });
}

/** 剧本读一个真实存在的本地文件，让离线演示也带着真实内容。 */
function readDemoFile() {
  const rel = 'agent-vis/package.json';
  const abs = path.join(REPO_ROOT, rel);
  const raw = fs.readFileSync(abs, 'utf8');
  const parsed = JSON.parse(raw);
  return {
    path: rel,
    bytes: raw.length,
    name: parsed.name,
    version: parsed.version,
    dependencies: parsed.dependencies,
  };
}

export async function* runMockChunks(
  prompt,
  { profileId, handoff, handoffFrom, instructions, permissions, signal, delay = 140 } = {},
) {
  const profile = resolveProfile(profileId);
  const wait = (ms) => sleep(ms, signal);
  const upstream = typeof handoff === 'string' ? handoff.trim() : '';
  // 上游的名字跟着实际发送方走 —— 交接不再只有 A → B 这一条路。
  const sourceLabel = handoffSourceLabel(handoffFrom);
  // instructions 缺省 = 用该 agent 的默认提示词；带空串 = 明确不要系统提示词。
  const hasCustom = typeof instructions === 'string';
  const custom = hasCustom ? instructions.trim() : '';
  // 权限：没给就用这个 agent 的默认权限，给了空数组就是一项也不要。
  const permissionIds = resolvePermissions(profile, permissions);

  yield { type: 'start', messageId: 'mock-message-1' };

  await wait(delay);
  yield { type: 'start-step' };

  await wait(delay);
  yield { type: 'reasoning-start', id: 'reasoning-0' };
  for (const piece of [
    `离线剧本模式：${profile.name} 先读一个真实文件确认内容，再回答。`,
  ]) {
    await wait(delay);
    yield { type: 'reasoning-delta', id: 'reasoning-0', delta: piece };
  }
  await wait(delay);
  yield { type: 'reasoning-end', id: 'reasoning-0' };

  const input = { path: 'package.json' };
  const raw = JSON.stringify(input);
  await wait(delay);
  yield { type: 'tool-input-start', toolCallId: 'call_mock_1', toolName: 'readFile' };
  await wait(delay);
  yield {
    type: 'tool-input-delta',
    toolCallId: 'call_mock_1',
    inputTextDelta: raw.slice(0, Math.ceil(raw.length / 2)),
  };
  await wait(delay);
  yield {
    type: 'tool-input-delta',
    toolCallId: 'call_mock_1',
    inputTextDelta: raw.slice(Math.ceil(raw.length / 2)),
  };
  await wait(delay);
  yield {
    type: 'tool-input-available',
    toolCallId: 'call_mock_1',
    toolName: 'readFile',
    input,
  };

  await wait(delay * 2);
  yield {
    type: 'tool-output-available',
    toolCallId: 'call_mock_1',
    output: readDemoFile(),
  };
  await wait(delay);
  yield { type: 'finish-step' };

  // 离线剧本里留一个"本步判定本轮结束"的入口：prompt 带这个标记就调用 endRound。
  // 用来在不起模型的情况下验证"后面的步骤会被跳过"。
  // 交接步骤（轮次的最后一步）：剧本里默认调用 requestHandoff。
  // 想在测试里演"判定不交接"，在条件里写 【模拟不交接】。
  if (prompt.includes('【交接步骤】')) {
    if (!prompt.includes('【模拟不交接】')) {
      await wait(delay);
      yield { type: 'tool-input-start', toolCallId: 'call_handoff', toolName: 'requestHandoff' };
      await wait(delay);
      yield {
        type: 'tool-input-available',
        toolCallId: 'call_handoff',
        toolName: 'requestHandoff',
        input: { summary: '离线剧本：把这一轮的结论交出去' },
      };
      await wait(delay);
      yield {
        type: 'tool-output-available',
        toolCallId: 'call_handoff',
        output: { summary: '离线剧本：把这一轮的结论交出去' },
      };
      await wait(delay);
    }
    yield { type: 'text-start', id: 'txt-handoff' };
    await wait(delay);
    yield {
      type: 'text-delta',
      id: 'txt-handoff',
      delta: prompt.includes('【模拟不交接】')
        ? '交接条件不满足：这一轮不交接。'
        : '已交接：这一轮的结论交给下游。',
    };
    await wait(delay);
    yield { type: 'text-end', id: 'txt-handoff' };
    await wait(delay);
    yield { type: 'finish-step' };
    return;
  }

  if (prompt.includes('【模拟空轮】')) {
    await wait(delay);
    yield { type: 'tool-input-start', toolCallId: 'call_end_round', toolName: 'endRound' };
    await wait(delay);
    yield {
      type: 'tool-input-available',
      toolCallId: 'call_end_round',
      toolName: 'endRound',
      input: { reason: '离线剧本：模拟没有 Open Issue' },
    };
    await wait(delay);
    yield {
      type: 'tool-output-available',
      toolCallId: 'call_end_round',
      output: { reason: '离线剧本：模拟没有 Open Issue' },
    };
    await wait(delay);
    yield { type: 'start-step' };
    await wait(delay);
    yield { type: 'text-start', id: 'txt-0' };
    await wait(delay);
    yield {
      type: 'text-delta',
      id: 'txt-0',
      delta: '离线剧本：没有 Open Issue，本轮到此结束。',
    };
    await wait(delay);
    yield { type: 'text-end', id: 'txt-0' };
    await wait(delay);
    yield { type: 'finish-step' };
    yield { type: 'finish', finishReason: 'stop' };
    yield { type: 'data-usage', data: USAGE };
    return;
  }

  await wait(delay);
  yield { type: 'start-step' };
  await wait(delay);
  yield { type: 'text-start', id: 'txt-0' };

  const answer = [
    `【${profile.name}】收到你的问题：「${prompt}」。`,
    upstream === ''
      ? '本次没有上游交接内容。'
      : `已收到 ${sourceLabel} 的分析结果：${upstream.slice(0, 200)}`,
    !hasCustom
      ? '本次使用默认系统提示词。'
      : custom === ''
        ? '本次没有系统提示词。'
        : `本次使用自定义系统提示词（${custom.length} 字）：${custom.slice(0, 120)}`,
    permissionIds.length === 0
      ? '本次授予的权限：一项也没有（既不能读也不能写）。'
      : `本次授予的权限：${permissionIds.join('、')}。没授予的工具调不到。`,
    '当前是离线剧本模式：上面的工具调用是真实读取本地文件得到的结果。',
    '把 config.js 里的 MODE 改成 live 并重启，就是真实模型跑出来的同一套协议。',
  ];
  for (const piece of answer) {
    await wait(delay);
    yield { type: 'text-delta', id: 'txt-0', delta: piece };
  }
  await wait(delay);
  yield { type: 'text-end', id: 'txt-0' };
  await wait(delay);
  yield { type: 'finish-step' };

  await wait(delay);
  yield { type: 'finish', finishReason: 'stop' };
  yield { type: 'data-usage', data: USAGE };
}
