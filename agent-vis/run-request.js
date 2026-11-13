/**
 * 一次运行请求的请求体。
 *
 * 单独成文件，是为了让"页面到底发了什么"能被离线测到 ——
 * 之前"交接时页面没带发送方身份"就是这么丢的：页面在浏览器里，
 * 服务端又把它吞在 catch 里，两头都没人看见。
 *
 * 规则：
 *   - 有上游结果时，带上来源的名字 **和身份**（handoffFrom / handoffFromId）。
 *     只带名字不行：服务端要靠身份才知道该不该重置会话。
 *   - 轮次里的步骤带上它在整轮里的位置，服务端据此把同一步的会话接起来。
 */

/**
 * 拼出 POST /api/run 的请求体。
 *
 * @param {object} input
 * @param {string} input.prompt        这一步要模型做什么
 * @param {string} input.version       前端版本号（服务端靠它认出旧版页面）
 * @param {string} input.profile       这一栏的 agent 身份
 * @param {string} [input.handoff]     上游交下来的结果；空串表示没有上游
 * @param {string} [input.handoffFrom] 上游的名字
 * @param {string} [input.handoffFromId] 上游的身份（决定要不要重置会话）
 * @param {object} [input.round]       这一步在整轮里的位置
 * @returns {object} 请求体
 */
export function buildRunRequestBody({
  prompt,
  version,
  profile,
  handoff = '',
  handoffFrom = '',
  handoffFromId = '',
  round = null,
}) {
  const body = {
    prompt,
    v: version,
    profile,
    handoff,
  };
  // 来源名字：下游看到的"这是谁的结果"得跟着实际发送方走。
  if (handoff !== '') {
    body.handoffFrom = handoffFrom;
    // 来源身份：服务端靠它决定"要不要重置会话"（发送方自己 / 接收方）。
    // 只传名字不行 —— 名字会改、会重名，身份不会。少了这个字段，
    // 交接照跑，但接收方会带着上一轮的上下文继续，被自己带偏。
    if (handoffFromId !== '') body.handoffFromId = handoffFromId;
  }
  // 这一步在整轮里的位置：服务端原样记进对话记录。
  if (round) {
    body.roundId = round.roundId;
    body.stepIndex = round.stepIndex;
    body.stepTitle = round.stepTitle;
    body.stepInstruction = round.stepInstruction;
    body.totalSteps = round.totalSteps;
    // 本轮输入只在第一步发：服务端会记住它，会话丢失时用来自动补上下文。
    if (round.stepIndex === 0) body.roundPrompt = round.roundPrompt;
  }
  return body;
}
