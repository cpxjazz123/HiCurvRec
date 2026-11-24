C2/C3/C11 的首个 TDD RED 已实际执行：`/home/wlia0047/ar57_scratch/wenyu/genrec_env_v2/bin/python3.9 -m unittest tools.issue253.test_three_contract -v`，3个测试、3个指定断言失败、exit=1。旧生产路径输出几何末轴5而非3、Query为[5,32]而非[3,32]；不是导入错误/崩溃/超时冒充RED。独立ambient参考先于实现完成。

证据：`results/research/issue253/THREE_EXPERT_RED_2.log`、`tools/issue253/reference.py`、`test_three_contract.py`。最初60秒空日志超时已单独诊断为解释器冷导入，import-only证据 `IMPORT_DIAGNOSE.log`；不作为数值RED。正式实现和其余验收尚未完成，继续三专家与精确计算复用。
