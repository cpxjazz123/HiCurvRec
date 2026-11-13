/**
 * 交接时的会话重置规则。
 *
 * 单独成文件，是为了让"收到交接要不要清空这一栏"能被离线测到。
 *
 * 规则只有一条，但两端必须一致：
 *   - 服务端：发送方配置里 resetReceiverSessionOnHandoff 为真 → 清掉接收方的
 *     模型会话 **和** 它存着的对话记录；
 *   - 页面：同一条件下 → 把接收方面板上已经画出来的历史清掉。
 * 两端不一致就会出现"模型已经不记得了，屏幕上却还摆着上一轮"的错位。
 */

/**
 * 发送方把结果交过来时，接收方的会话会不会被重置。
 *
 * @param {object|null} senderProfile 发送方的配置（页面里 profileState 那一份）
 * @returns {boolean} 真 = 接收方会被重置（模型上下文与面板历史一起清）
 */
export function handoffResetsReceiver(senderProfile) {
  if (!senderProfile || typeof senderProfile !== 'object') return false;
  return senderProfile.resetReceiverSessionOnHandoff === true;
}
