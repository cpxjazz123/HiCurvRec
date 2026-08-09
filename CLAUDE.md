# CLAUDE.md

> HG-Rec (Hyperbolic RQ-VAE + Differential-Length Codebook + T5) 流水线复现工作目录, 基线 Task #84 valid R@10=0.1267 / test R@10=0.1024 (Musical_Instruments 9922 items).

---

## R 规则 (28 条)

**R1**: 默认 `genrec_env`, KG 任务用 `deepke`, base anaconda 仅 zero-dep grep + MiniMax API。

**R2**: 禁止 fallback 逻辑 (默认值/回退/降级), 预期内缺失返回 None/空值, 预期外失败直接 raise。

**R4**: 修改 Python 脚本后必须立即 `python3 -m py_compile` 验证语法 (文档例外)。

**R5**: 任务硬约束 = 基线 HG-Rec Task #84 (valid R@10=0.1267, test R@10=0.1024), 仅 RQ-VAE 量化, Musical_Instruments (9922 items), 4 阶段流水线, seed=42。

**R7**: 启动新实验前必须 `nvidia-smi` 核对 (util<10%, mem<5GB), 选完全空闲 GPU, 禁等待已占卡或挤同一张卡。

**R10**: (已删除 2026-08-09 用户取消, 不再调用 `glab issue`, 不再要求 issue 闭环)

**R11**: AI 自主决策, 子步骤禁"等用户拍板"/"是否启动?"阻塞话术, 兜底顺序: CLAUDE.md > 上游 default > 论文 > 简单实用。

**R12**: 训练固定阶段强制存 checkpoint (epoch 末), 删旧 ckpt, 写 `_TRAINING_PID`。

**R13**: 禁 `EnterWorktree` + git worktree, 代码改共享 checkout, 临时文件用 `$CLAUDE_JOB_DIR/tmp`。

**R15**: (已删除 2026-08-09 跟随 R10, 不再处理 issue 闭环)

**R16**: (已删除 2026-08-09 跟随 R10)

**R17**: commit message 必含 `Gate <N> FAIL/PASS` + 失败原因, 前 Gate FAIL → 后 Gate STOP。

**R18**: 新 issue 必须跟历史做 4 维度对比 (D1 spec 摘录 / D2 实施核心 / D3 Gate 1 失败机制 / D4 引用文献), 任一不同 → 必须实验, 禁"路径同构" NO-GO。

**R19**: AI = 激进 owner, precheck PASS 立即启动 GPU 训练, 跨 issue 并行 (一卡一实验), 禁任何"是否启动?"询问。

**R20**: commit + issue comment 必须详细回答 4 Gate (≥3-5 行/Gate), close issue 前必发 comment。

**R21**: commit hash 必须明示 (禁 "pending"/"TBD"/"TODO"), comment 必须在 commit + push 之后发。

**R22**: tick 必须实际推进 (启动 precheck/Gate1/训练/评估/修复/验证 之一), 4 卡全占 → 换 GPU/nohup/缩减规模, 禁"等下一轮"/"等 owner 拍板"。

**R23**: tick 扫活跃训练, 7 信号任一触发立即终止 (val_R@10=0 跨 ≥2 ckpt / loss 不下降 / val loss 反向 / wrapper broken / ckpt 不存 / NaN-Inf / GPU 100% loss 不变), kill -9 + NO-GO + commit + push + close。

**R24**: tick 检查 in_progress 任务是否真在执行 (有 PID + file mtime 更新), 无活跃行为 → 立即 R11.5 决策 + 执行, 禁"等 owner 拍板"/"wrapper 复杂"/"等下一 tick"。

**R25**: issue 里的"方向A"/"方向B"指的就是 taskA/ + taskB/ (代码位置/入口/产物路径), 禁解读为 TIGER/LETTER/RecBole/phonism 等其他 lineage。

**R26**: tick 必须实际推进 issue (启动 precheck/Gate1/训练/评估/修复/验证 之一), 禁仅规划/§16 状态/commit + close。

**R27**: tick 必须 `python3 <existing_script>.py` 启动脚本产生 PID 活跃, 禁仅规划/读/写脚本/改 verdict。

**R28**: 任何决策/修复方案/启动判断禁"等用户拍板"/"是否启动?"/A-B 选项/"你选"/"请告诉我"/"要不要"话术, 必须直接给出推荐方案 + 立即执行 (唯一例外: 不可逆操作)。

**R29**: tick 输出必须 (a) ≥1 个 R26 动作 + (b) ≥1 个 R27 动作, 末尾明示"已执行 X" + 实际产物 (verdict 路径/commit hash/PID)。

**R30**: 脚本参数硬编码, 禁 `os.environ.get` 读超参, 所有超参 (epoch/batch_size/lr/seed/...) 必须硬编码进脚本 (常量或 argparse 默认), launch 脚本仅设 GPU/路径。

**R31**: 每个 stage 目录只允许一个主脚本 (如 `taskA_stage2.py`), 禁 fork `_v2.py/_v8.py` 多版本并存, 历史实验变体从 git 历史恢复。

**R32**: 运行脚本必须直接 `python3` 执行, 禁写 `.sh` 包装启动, GPU 选择用 `CUDA_VISIBLE_DEVICES=0 python3 -u ...` 内联 (唯一例外: DDP 多卡 `torchrun`)。

**R33**: verdict 文件必须放 `verdicts/<gitlab_iid>/<final_verdict>.<ext>` (1:1 映射 iid 1-90), 中间产物不落盘, orphan (internal #N > 90) → `verdicts/_misc/orphan/`。

**R34** (2026-08-09 用户新增): 每次迭代新版本之前,必须在 `/home/wlia0047/ar57/wenyu/GeneRec/tasks/` 下新建文件夹,文件夹名描述本任务。每个文件夹必须有且只有 4 个脚本:`stage1.py` / `stage2.py` / `stage3.py` / `stage4.py`(R31 强化: 每 stage 目录仅 1 个主脚本,禁 fork 版本)。

**R35** (2026-08-09 用户新增): 评估强约束 — 必须只使用单 checkpoint + `beam_search=20`。**绝对禁止 Borda Rank Fusion / 任何 ensemble 多 ckpt 融合**。任何 `stage4_*.py` 默认 `--beam_size 20`, 禁止 `--beam_size 30/50`。违反此规则直接 raise。

**R36** (2026-08-09 用户新增): 方法路径强约束 — **禁止通过调参形式 (LR/dropout/label_smoothing/weight_decay sweep) 提升指标**。**必须通过改善曲率框架** (curvature framework): Stage 2 κ 学习 (K / REC_LAYER_W / capmatch 上限)、Stage 3 κ frozen→learnable、新曲率正则项、Poincaré/Minkowski/Lorentz 曲率机制变更 等。违反此规则直接 raise `NotImplementedError("R36 禁调参, 必走曲率机制")`。

**R37** (2026-08-09 用户新增): **版本回滚硬约束** — 如果新版本 (vN) 的评估结果 (test_R@10) **比上一版本 (vN-1) 差**, **必须回到 vN-1 重新开始**, 从 vN-1 的配置基础再做创新。**禁止** 在比旧版本差的版本上进行二次创新 (二次叠加新曲率机制、组合多个失败尝试等)。失败即终止该 lineage, vN 的所有产物 (Stage 3 ckpt / Stage 4 raw_predictions / verdict) 仅留作记录, 不作为下一版本起点。新版本必须以"当前最优版本"为唯一基础。
   - **判定准则**: 比较 stage4_beam20 的 `eval_test.json` 中 `test_R@10` 字段
   - **执行流程**: vN 评估 → 若 `test_R@10 < vN-1.test_R@10` → 立即终止 → 新版本目录命名为 `vN_<新机制>_from_vN-1`, 完全复用 vN-1 配置 + 仅新加的曲率机制
   - **记录要求**: verdict 必须含 `R37 决策行` — "vN 比 vN-1 差 (-X.XXXX), 回退至 vN-1 重新创新"

**R38** (2026-08-09 用户新增): **训练期早停回退硬约束** — 如果新版本 (vN) 在训练期间 (Stage 3 进行中), 训练指标 (valid_R@10 / loss / 收敛速度) 已经明确**比上一版本 (vN-1) 差**, **不需要等到 Stage 3 自然结束**, **立即 kill 训练 + 写 R38 决策行 + 触发 R37 回退流程**。节省 GPU 时间, 避免无效 epoch 浪费。
   - **判定准则** (任一满足即触发):
     1. vN 训练 best valid_R@10 已落后 vN-1 (同 epoch 范围对比), 且连续 ≥30 epoch 无改进 (饱和平台)
     2. vN 训练 loss 下降速度明显慢于 vN-1 (同样 epoch 数 loss 高出 ≥0.05)
     3. vN 训练曲线出现 NaN/Inf/反向 (loss 上升)
   - **执行流程**: kill -9 PID → 写 `R38 决策行` ("vN mid-training regress, vN best_valid=X.XXXX < vN-1 best_valid=Y.YYYY (-Z.ZZZZ), 立即 kill + 回退至 vN-1 重新创新") → 复用 vN 已存的 HG_Rec_best.pth 跑 stage4 (留作记录) → 新版本目录命名为 `vN_<新机制>_from_vN-1`
   - **例外**: 若 vN 仍在快速提升期 (valid 持续上升, loss 持续下降), 即使当前落后 vN-1 也应等训练自然结束再判定 R37

**R39** (2026-08-10 用户新增): **Open Issue 立即实现硬约束** — 配合 loop.md §1 强化版,任何 open issue 一旦被 loop tick 发现,**必须立即实施**,严禁等待用户评论授权。强制动作序列 (R19+R26+R27 联动):
   1. 新建 `tasks/vN_<新机制>_from_vN-1/` 目录 (R34+R37);
   2. 写 `stage3.py` + `stage4_beam20.py`(基于 vN-1 base + 仅新机制 patch);
   3. 修改 `common/stage3/stage3_train_pure_t5_v85p_repro.py` 与 `common/stage4/stage4_eval_pure_t5_v85p_4layer.py` 加新机制 patch;
   4. `nvidia-smi` 确认 GPU 空闲 → `python3 -u tasks/vN_*/stage3.py` 立即 launch GPU 训练(写 `_TRAINING_PID` + product_dir);
   5. 训练结束 → `python3 -u tasks/vN_*/stage4_beam20.py` 评估 → 验证 4 Gate → commit + push → issue comment 含 4 Gate 答案 → close issue。
   **严禁**:
   - ① 等用户评论含 `立即启动` / `launch now` / `🟢 GO` / Gate A 文本 unlock 才 launch (loop.md §1 强化版禁 Gate A 文本 gate);
   - ② "是否启动?" / A-B 选项 / "请告诉我" 话术 (R28);
   - ③ 仅登记 issue 不实施 (R26);
   - ④ 任何 gate-text-only 阻塞 / "issue body 含 🟡 待启动" 解读为禁 launch (用户 2026-08-10 明确废除此 gate)。
   - 违反直接 raise `NotImplementedError("R39 禁等评论, 必须立即实现")`。
   - **判定**: tick 输出末尾必须含 "已启动 vN_*/stage3.py, PID=XXX, product_dir=YYY" + R29 强制双动作 (R26 + R27)。