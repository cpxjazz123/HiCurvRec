1. 检查是否有 open issue;如果有:**马上在当前 tick 开始实现 issue,严禁等待用户评论/授权/Gate A 文本 unlock**。
   - 一旦发现 open issue,立即: ① **新建 tasks/Issue<NN>_<任务名>/ 目录(目录名格式: `Issue` + issue 编号 + 要完成的任务,例:`Issue121_relational_knn_infonce`、`Issue120_pre_quant_residual_ref`)**; ② 写 stage3.py / stage4_beam20.py; ③ 修改 common/ 下训练/评估脚本加新机制 patch; ④ `nvidia-smi` 确认 GPU 空闲后立即 `python3 tasks/.../stage3.py` 启动训练(写 _TRAINING_PID); ⑤ R19 跨 issue 并行(一卡一实验); ⑥ 训练结束 → 跑 stage4_beam20 → 验证 4 Gate → commit + push → issue comment 含 4 Gate 答案 → close issue。
   - **目录名硬约束 (2026-08-11 用户新增)**: tasks/ 下新任务目录必须命名为 `Issue<NN>_<任务描述>`,NN 是 gitlab issue 编号,禁止 `vN_xxx_from_vN-1` 等无 issue 编号前缀的命名。**例外**: 历史 v* 目录(已存在的 v10_v11_v12...v23)不追溯重命名,仅本规则生效后新建的目录必须遵守。
   - 进入实施阶段后:完成该 issue,将完成该 issue 时修改的所有文件全部 commit 并 push,将完成该 issue 时的全部过程以 comment 形式记录到该 issue;确认 commit、push 和 comment 均已完成后,close issue。
   - **禁**: 等用户评论、问"是否启动?"、Gate A 文本 unlock 阻塞、任何"是否继续?" A-B 选项话术(R28)。
2. 如果没有 open issue,就不做任何工作。
