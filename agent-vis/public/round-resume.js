/**
 * 多步轮次被打断之后，从哪里接上。
 *
 * 多步轮次里每一步都是浏览器发的一次请求，服务端一次只跑一步。浏览器里的那个
 * for 循环一旦没了（页面刷新、标签关掉、网络断了），剩下的步骤就再也不会被
 * 发出去 —— 面板看上去"卡住"；如果那一步刚好启动了训练，训练还在后台跑，
 * 只是没人接着盯、也没人报结果。
 *
 * 这个模块只做一件事：从服务端的对话记录里认出「没跑完的轮次」以及该从第几步接上。
 * 规则与服务端记下来的字段一一对应：
 *   - 同一个 roundId 的记录属于同一轮；stepIndex 是这一步在轮里的位置；
 *   - totalSteps 是那一轮一共几步；
 *   - 最后一条 status === 'done' → 这一步本身是好的，只是下一步没发出去
 *     （页面被刷新或关掉），可以自动接上；
 *   - 最后一条 status 是 'aborted'（人为按了停）或 'error'（真出错）→ 不自动接：
 *     重跑之前该由人看一眼再决定。
 *
 * 单独成文件，是为了让这条规则能被离线测到（test/round-resume.js）。
 */

/**
 * 在对话记录里找出「没跑完的多步轮次」。
 *
 * @param {Array<object>} records 服务端存的对话记录（按时间顺序）
 * @returns {null|{roundId: string, totalSteps: number, nextIndex: number,
 *   roundPrompt: string, auto: boolean, reason: string}} null = 没有没跑完的轮次
 */
export function findUnfinishedRound(records) {
  if (!Array.isArray(records) || records.length === 0) return null;

  // 只有多步轮次才有得接：一步的轮次要么跑完了，要么没跑完（那不是"接上"，是重来）。
  const rounds = new Map();
  for (const record of records) {
    if (!record || typeof record !== 'object') continue;
    const roundId = typeof record.roundId === 'string' ? record.roundId : '';
    if (roundId === '') continue;
    if (!Number.isInteger(record.stepIndex)) continue;
    const totalSteps = Number.isInteger(record.totalSteps) ? record.totalSteps : 0;
    if (totalSteps <= 1) continue;
    const bucket = rounds.get(roundId) || { totalSteps, steps: [] };
    if (totalSteps > bucket.totalSteps) bucket.totalSteps = totalSteps;
    bucket.steps.push(record);
    rounds.set(roundId, bucket);
  }
  if (rounds.size === 0) return null;

  // 最后一轮 = 走得最远的那一轮；一样远就取更晚发的那一轮
  //（同一轮里步骤是顺序发的，所以"更晚"就是更近的那一轮）。
  let latest = null;
  for (const [roundId, bucket] of rounds) {
    bucket.steps.sort((a, b) => {
      if (a.stepIndex !== b.stepIndex) return a.stepIndex - b.stepIndex;
      return (a.startedAt || 0) - (b.startedAt || 0);
    });
    const last = bucket.steps[bucket.steps.length - 1];
    const farthest = last.stepIndex;
    const startedAt = last.startedAt || 0;
    if (
      !latest ||
      farthest > latest.farthest ||
      (farthest === latest.farthest && startedAt > latest.startedAt)
    ) {
      latest = { roundId, bucket, last, farthest, startedAt };
    }
  }
  if (!latest) return null;

  const { roundId, bucket, last } = latest;
  const nextIndex = last.stepIndex + 1;
  if (nextIndex >= bucket.totalSteps) return null; // 这一轮跑完了

  const status = typeof last.status === 'string' ? last.status : 'done';
  const roundPrompt =
    typeof last.roundPrompt === 'string' && last.roundPrompt !== ''
      ? last.roundPrompt
      : typeof last.prompt === 'string'
        ? last.prompt
        : '';

  if (status === 'aborted') {
    return {
      roundId,
      totalSteps: bucket.totalSteps,
      nextIndex,
      roundPrompt,
      auto: false,
      reason: 'aborted',
    };
  }
  if (status === 'error') {
    return {
      roundId,
      totalSteps: bucket.totalSteps,
      nextIndex,
      roundPrompt,
      auto: false,
      reason: 'error',
    };
  }
  return {
    roundId,
    totalSteps: bucket.totalSteps,
    nextIndex,
    roundPrompt,
    auto: true,
    reason: 'interrupted',
  };
}
