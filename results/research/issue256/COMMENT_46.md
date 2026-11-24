R02 检查轮2出现新的真实FAIL，继续停在R4：139候选+64框架+15控制器用例已通过唯一四卡launcher执行，逐例无重复/漏项/skip；但实际Trainer的初始全局预算复算4/4 rank报 `actual initial global coefficient is not maximal`。epoch51、性能和机器入口仍未运行，不能派发正式训练。

实际校准lambda=1.069727773028717e-6，在校准自身17组记录中的最大比例1.0000000472964818；独立重建点的最大性不一致。原因尚未证实。源码排查未发现host/buffer尺度精度漂移，不能把它编造成原因。

现在增加一次1epoch真实四卡辨别诊断（bg_506）：同权重、同batch/sample下捕获校准/真实首步/独立副本的encoded、gates、原始cost、hard IDs和全局VJP，先区分离散数值边界与状态/计算漂移；不改权重、容差或冻结协议。

证据 `results/research/issue256/rounds/r02/BUDGET_POINT_FAILURE.json` sha256 `3fdf738b6e5a8a12ec940b1fded46050676b7629bff6f9e51030088d8a388944`，完整新139/64/15凭据与失败原文已冻结在pre_budget_point/。
