## C6 完成；继续精确执行优化，不改变估计器

训练估计器先在CONTRACT.md:4ca0c421b397b20e13c0ab970a7e69cdf9752b7a947fb4d8f5ee2986f843bb92 / COMMENT_007冻结，然后实现：seed42的全部24,326有效中心循环，每次从原中心全量occurrences和原完整component分别均匀无放回抽最多32个；辅助ID最少256，真实hard-q及同代soft反向。每个实际optimizer step结构loss均参与，无长期跳步。完整28,391,948目标保留、最终全参考评估保留。

独立中点恒等式/有限总体穷举和生产路径已验证：固定外部q的raw统计无偏，但Huber后及subset hard-assignment全目标有明确偏差；没有声称loss/梯度无偏。手算q=[0,1,3]的全参考值4/3，单参考raw[-4,0,8]、平均Huber11/3而完整Huber5/6；闭式q梯度[-4/3,2/3,2/3]。真实60epoch各rank每epoch结构更新/目标/参考coverage保存在失败benchmark原始epoch_boundaries，不把抽样覆盖称为全表覆盖。最终实际模块/DDP接线仍单列C8，未冒充完成。

为修C7，先冻结EXACT_GNN_OPTIMIZATION.md再实现精确首层仿射重排：只缓存原6维静态特征的方向均值/空邻域mask，所有学习权重每次实时求导；第二非线性层、全节点全边和readout不变。6个生产路径/独立全部参数与输入梯度/手算/缓存测试RED→GREEN，STATIC_MEANS_TESTS.log；消除checkpoint反传重播。实际显存和性能尚未凭此推断，已派发标准launcher三完整epoch warmed-epoch2 profiling。

floor冲突选择仅将同一GPU argsort优先级的host-list循环替换为device first-position reduction：3项RED→GREEN，另180个CPU正常/并列/竞争fixture与旧active copy逐ID/metadata完全一致，FLOOR_COMPATIBILITY.json；未改变Sinkhorn/greedy rounds。额外floor2试例暴露旧通用repair轮数界限，但正式冻结floor=1；未越界改变它或把floor2当本任务准入。GPU真实路径仍待profiling/benchmark。