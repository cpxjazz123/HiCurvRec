R7→R1/R2的最小判别问题已冻结在`rounds/r01/POST_ROUND_DIAGNOSIS.md`：最终rank0的梯度失衡，是否在四rank真实VJP先平均后仍存在？局部值不是全局值，不能直接据此选下一机制。

将仅回放同一R01最终权重在epoch3000第一真实全局4096训练batch，实际original/C losses、同17耦合模块、实际全局采样；只SUM/world梯度向量再算范数/方向，不做optimizer、重新校准或SID改写。采样与fresh诊断RNG、覆盖率和非AdamW/非因果局限均预先写明。若全局平均消除局部违例，就保留反证，不把局部观察当全局根因。该诊断不读test或结构test，是当前机制失败定位，不是新机制训练。

实际外部validation GPU消费者已退出后，才经唯一四卡launcher运行。完成这条判别和validation配对诊断、确认无运行依赖后，先归档R01活跃树，再进行下一机制设计/复制/实现。
