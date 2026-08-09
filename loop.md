1. 检查是否有 open issue；如果有：
   1.1. **Gate A (issue 文本授权检查)**: 只有当 issue 文本**明确要求"立即启动"**（出现以下任一关键字：`立即启动` / `launch now` / `🟢 GO` / `start immediately` / `R19 启动` / 移除 `🟡 待启动` 标记）时，才进入实施阶段；否则只执行接收（comment）+ 跟踪登记，保持 issue open 状态等待显式 go 信号。
   1.2. 进入实施阶段后：完成该 issue，将完成该 issue 时修改的所有文件全部 commit 并 push，将完成该 issue 时的全部过程以 comment 形式记录到该 issue；确认 commit、push 和 comment 均已完成后，close issue。
2. 如果没有 open issue，就不做任何工作。
3. **强约束**：任何 tick 都不能因 "issue 在" 而强制 launch GPU 训练 — 必须先通过 Gate A。