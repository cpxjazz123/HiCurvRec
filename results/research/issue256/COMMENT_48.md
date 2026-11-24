R02 同状态点诊断完成（非正式训练、不是准入通过）。标准四卡/50202实际1epoch，exit0、348.69s。所有4卡参数、尺度、编码相同；校准与实际第一步的gate、cost、hardID逐位相同。新建同权重replica的gate最大差3.644824028015137e-5，本次hardID仍相同；独立复算的全局最大预算比为1.0000317630586744，不能放宽既定2e-6容差判PASS。

证据 `results/research/issue256/rounds/r02/BUDGET_POINT_DIAGNOSIS.json` sha256 `9be4f6747199f9f26dd437cfb61ead4ae3d0c0d6029cee2b730c1fcc38c6bd6d`；全部4份原始NPZ由独立NumPy逐项重算并与记录一致。已排除本次首步参数漂移、尺度不一致、编码差异和实际首步分支漂移。新消费者GNN差异根因仍未知，下一步比较完整静态均值缓存和逐层计算；不预先宣称缓存缺陷、不改变机制/阈值、不派发正式训练。此前完整unit逐例139/64/15 PASS不能替代尚未通过的实际DDP、epoch51和机器入口。
