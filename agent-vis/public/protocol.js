/**
 * AI SDK v7 UI message stream → 时间轴数据模型。
 *
 * 纯函数、无 DOM 依赖：浏览器与 Node 测试共用同一份实现，
 * 保证「后端发什么」和「前端画什么」不会各写一套。
 *
 * 协议参考：node_modules/ai/docs/04-ai-sdk-ui/50-stream-protocol.mdx
 */

/** 轮次提前结束的控件名：与 config.js 的 ROUND_END_TOOL 必须一致（smoke 会校验）。 */
export const ROUND_END_TOOL = 'endRound';

/**
 * 这一步有没有要求"本轮到此结束"。
 *
 * 只看工具调用，不看文字：文字里的"本轮结束"客户端读不懂，
 * 也不该去猜自然语言。工具调用是结构化的，调用过就是调用过。
 */
export function roundEnded(store) {
  return store.steps.some((step) =>
    step.parts.some(
      (part) =>
        part.kind === 'tool' &&
        part.toolName === ROUND_END_TOOL &&
        (part.state === 'input-available' || part.state === 'output-available'),
    ),
  );
}

export function createStore() {
  return {
    messageId: null,
    steps: [],
    usage: null,
    finishReason: null,
    error: null,
    aborted: null,
    data: [],
    steering: [],
    unknownTypes: [],
  };
}

function currentStep(store) {
  if (store.steps.length === 0) {
    store.steps.push({ index: 0, closed: false, parts: [] });
  }
  return store.steps[store.steps.length - 1];
}

function findPart(store, kind, key) {
  for (let i = store.steps.length - 1; i >= 0; i -= 1) {
    for (const part of store.steps[i].parts) {
      if (part.kind === kind && part.key === key) return part;
    }
  }
  return null;
}

function ensureTool(store, toolCallId, toolName) {
  const existing = findPart(store, 'tool', toolCallId);
  if (existing) {
    if (toolName && !existing.toolName) existing.toolName = toolName;
    return existing;
  }
  const part = {
    kind: 'tool',
    key: toolCallId,
    toolName: toolName || 'tool',
    state: 'input-streaming',
    inputRaw: '',
    input: undefined,
    output: undefined,
    errorText: undefined,
  };
  currentStep(store).parts.push(part);
  return part;
}

function ensureTextual(store, kind, id) {
  const existing = findPart(store, kind, id);
  if (existing) return existing;
  const part = { kind, key: id, text: '', closed: false };
  currentStep(store).parts.push(part);
  return part;
}

function findToolByApproval(store, approvalId) {
  for (let i = store.steps.length - 1; i >= 0; i -= 1) {
    for (const part of store.steps[i].parts) {
      if (part.kind === 'tool' && part.approvalId === approvalId) return part;
    }
  }
  return null;
}

function ensureSimplePart(store, kind, key, fields) {
  const existing = findPart(store, kind, key);
  if (existing) return existing;
  // kind / key 放在展开之后：fields 里就算带了同名键也覆盖不了分片身份。
  const part = { ...fields, kind, key };
  currentStep(store).parts.push(part);
  return part;
}

/**
 * 把一个 UI message chunk 并进 store。返回同一个 store，方便链式调用。
 */
export function applyChunk(store, chunk) {
  if (!chunk || typeof chunk.type !== 'string') return store;

  switch (chunk.type) {
    case 'start':
      store.messageId = chunk.messageId ?? null;
      break;

    case 'start-step':
      store.steps.push({ index: store.steps.length, closed: false, parts: [] });
      break;

    case 'finish-step':
      currentStep(store).closed = true;
      break;

    case 'reset-step':
      currentStep(store).parts = [];
      break;

    case 'text-start':
      ensureTextual(store, 'text', chunk.id).closed = false;
      break;

    case 'text-delta':
      ensureTextual(store, 'text', chunk.id).text += chunk.delta ?? '';
      break;

    case 'text-end':
      ensureTextual(store, 'text', chunk.id).closed = true;
      break;

    case 'reasoning-start':
      ensureTextual(store, 'reasoning', chunk.id).closed = false;
      break;

    case 'reasoning-delta':
      ensureTextual(store, 'reasoning', chunk.id).text += chunk.delta ?? '';
      break;

    case 'reasoning-end':
      ensureTextual(store, 'reasoning', chunk.id).closed = true;
      break;

    case 'tool-input-start':
      ensureTool(store, chunk.toolCallId, chunk.toolName);
      break;

    case 'tool-input-delta': {
      const part = ensureTool(store, chunk.toolCallId);
      part.inputRaw += chunk.inputTextDelta ?? '';
      break;
    }

    case 'tool-input-available': {
      const part = ensureTool(store, chunk.toolCallId, chunk.toolName);
      part.state = 'input-available';
      part.input = chunk.input;
      part.inputRaw = '';
      break;
    }

    case 'tool-output-available': {
      const part = ensureTool(store, chunk.toolCallId);
      part.state = 'output-available';
      part.output = chunk.output;
      break;
    }

    case 'tool-output-error': {
      const part = ensureTool(store, chunk.toolCallId);
      part.state = 'output-error';
      part.errorText = chunk.errorText ?? 'unknown error';
      break;
    }

    case 'tool-output-denied': {
      const part = ensureTool(store, chunk.toolCallId);
      part.state = 'output-denied';
      break;
    }

    case 'error':
      store.error = chunk.errorText ?? 'unknown error';
      break;

    case 'abort':
      store.aborted = chunk.reason ?? 'aborted';
      break;

    case 'finish':
      store.finishReason = chunk.finishReason ?? null;
      break;

    case 'reasoning-file':
      ensureSimplePart(store, 'reasoning-file', chunk.url ?? 'file', {
        url: chunk.url,
        mediaType: chunk.mediaType,
      });
      break;

    case 'source-url':
      ensureSimplePart(store, 'source-url', chunk.sourceId ?? chunk.url, {
        url: chunk.url,
        title: chunk.title,
      });
      break;

    case 'source-document':
      ensureSimplePart(store, 'source-document', chunk.sourceId ?? 'document', {
        url: chunk.url,
        mediaType: chunk.mediaType,
        title: chunk.title,
      });
      break;

    case 'file':
      ensureSimplePart(store, 'file', chunk.url ?? 'file', {
        url: chunk.url,
        mediaType: chunk.mediaType,
      });
      break;

    case 'custom':
      ensureSimplePart(store, 'custom', chunk.kind ?? 'custom', {
        providerKind: chunk.kind,
        providerMetadata: chunk.providerMetadata,
      });
      break;

    case 'tool-approval-request': {
      const part = ensureTool(store, chunk.toolCallId);
      part.state = 'approval-requested';
      part.approvalId = chunk.approvalId;
      part.approvalReason = chunk.reason;
      break;
    }

    case 'tool-approval-response': {
      const part = findToolByApproval(store, chunk.approvalId);
      if (part) {
        part.state = 'approval-answered';
        part.approved = chunk.approved;
        part.approvalReason = chunk.reason ?? part.approvalReason;
      }
      break;
    }

    case 'steering-message':
      store.steering.push(chunk);
      break;

    default:
      if (chunk.type.startsWith('data-')) {
        if (chunk.type === 'data-usage') store.usage = chunk.data ?? null;
        else store.data.push(chunk);
      } else if (!store.unknownTypes.includes(chunk.type)) {
        // 认不出来的分片要留下痕迹，不能静默丢掉——那正是"页面版本对不上"的症状。
        store.unknownTypes.push(chunk.type);
      }
  }

  return store;
}

/** 把 store 压成前端要的汇总数字。 */
export function summarize(store) {
  const parts = store.steps.flatMap((step) => step.parts);
  return {
    steps: store.steps.length,
    tools: parts.filter((p) => p.kind === 'tool').length,
    text: parts
      .filter((p) => p.kind === 'text')
      .map((p) => p.text)
      .join('\n'),
  };
}

/** 超过这个字符数的工具输出就折叠起来，避免一条结果把整页撑爆。 */
export const MAX_INLINE_CHARS = 600;

/**
 * 计算一段文本的折叠视图。纯函数，浏览器与测试共用。
 */
export function foldText(text, limit = MAX_INLINE_CHARS) {
  const value = typeof text === 'string' ? text : JSON.stringify(text, null, 2);
  return {
    collapsed: value.length > limit,
    preview: value.slice(0, limit),
    full: value,
    lines: value.split('\n').length,
    chars: value.length,
  };
}
