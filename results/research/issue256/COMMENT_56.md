R02 CHECK_4：8 PASS / 1 FAIL / 8 NOT_RUN，Issue再次读取OPEN且正文SHA未变。C12记录完整suite中的原相同实例断言失败，而不是把新门控回归绿当作完整准入。`CHECK_4.json` sha256 `6fea45f8a992b47970fb5de2c1e5547350a36b798af9b75c61a8bb0ca06e6e20` 绑定17条原文、该失败代源码/输出、生命周期红和后续修复依据。

独立CPU NumPy reducer已从四卡原始NPZ重算全部112项end-to-end batch/精度层输出统计，逐项完全一致（exit0、2.92s），见 `BATCH_TRACE_RECONCILIATION.json`；固定token实验未保存返回weights，其部分不冒充已独立重算。继续bg_513当前源全部检查，不读推荐test，不正式派发。
