## Issue #52 [方向A Gate1] Stage1 Lorentz 正式训练 — Gate 1 PASS

**Gate 1 (Stage1 训练): PASS** — 训练代码 commit `29f7bf4df3a23807f7100e2915426ebf0d86fe5f` (--train 分支), 证据 commit `de2bd102240729fb07e4dbb152d8274e7cb58017`。

### 训练配置 (R30 硬编码)
- TRAIN_BATCH_SIZE=64 / TRAIN_EPOCHS=3 / TRAIN_LR=0.0001 / TRAIN_SEED=42 / KL_TEMP=0.07 / L_AUG_WEIGHT=0.1 / DROPOUT_AUG=0.2 / C_ENC=1.0 / RHO_MAX=1.0 / MAX_SEQ_LEN=64
- 目标: batch 内关系分布 KL 蒸馏 (学生 Lorentz u_item vs 教师完整 sentence-t5-base mean-pool, 余弦/KL_TEMP) + L_AUG_WEIGHT×dropout 增强一致性 (两次学生前向, DROPOUT_AUG); 仅 input_proj + lorentz_block 可训 (float64 几何核心), T5 前 11 block 冻结。

### 训练结果
- train_curve_mean_loss_per_epoch: [1.134859, 0.955455, 0.944629]
- loss_decreasing=True (1.134859 → 0.955455 → 0.944629, 单调下降)
- params_finite=True (全部 trainable 参数 finite, 训练中梯度 NaN/Inf 即终止)
- ckpt: taskA/_history/taskA_stage1_issue52/stage1_lorentz_ckpt.pt (epoch 末保存, 删旧保新, R12)

### 9922 全量导出 (PC7 同协议)
- shape=[9922, 768] n_items=9922 | cast_err_f64_to_f32=1.8618631428268806e-09 | reload_max_abs_err=0.0 (<1e-6)
- item_ids_sha256: `a496c0bcea829344231e11ef4b3c7e1fcd5cf5ad4e16cbf809287993eaa8dfae` (与 Issue #51 canonical 的 item_ids_sha256 完全一致, 有序 ItemID 绑定确认)
- parquet sha256: `6fce0e93330962f7d41d20fed5c06ccfe8cfbc6f2dca7590e3e5ecedc4a83198` | item_emb_u32.npy 同步落盘 (9922, 768) float32, 全 finite
- teacher_student_recall10=0.4432170933279595 (教师自匹配 R@10 对照, 诊断指标非硬阈值 — 学生已部分保真教师关系结构)

**Gate 2/3/4: 未涉及** — 本 issue 仅 Stage1 训练 + 导出, 无 Stage2/3/4 产物; 下一步 = Stage2 训练 issue (item_emb_u32.npy 为输入)。

### 结论
- status=PASS → **Gate 1 PASS**; 产物就绪 (item_emb_u32.npy), 允许进入 Stage2。
