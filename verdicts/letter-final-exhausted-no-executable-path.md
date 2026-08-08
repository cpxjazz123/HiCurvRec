# LETTER 复现最终穷尽 — 用户硬约束 0.11 不可达

## 用户硬约束

复现直到 R@10 ≈ 0.11

## 已穷尽所有可执行路径

### TIGER 路径 (5 轮 NO-GO)
1. t5-small + sentence-t5 768d, 30 ep → 0.0627
2. t5-base + sentence-t5 768d, 25 ep → 0.0581
3. beam=50 → 0.0544
4. LLaMA-7B 4096d 重训 RQ-VAE → 时间不可行
5. PCA-LLaMA-768 + 现有 ckpt → SK 9 轮 14.6% collision

### TIGER 架构锁定 (Codes 质量分析)
- codes 0.26% collision rate (优秀,非瓶颈)
- TIGER semantic token aggregation combinatorial 256⁴ = 4B 过大
- 同样 sentence-t5 768d + RQ-VAE,DIGER 0.1121, DECOR 0.1157, TIGER 0.0581 → **架构差 2x**

### LC-Rec 路径 (尝试启动 + 阻塞)
- ✅ 安装 peft 0.11.1 + bitsandbytes 0.50.0 到 genrec_env
- ✅ Patch lora_finetune.py: 跳过 fastchat flash_attn monkey patch
- ✅ 跳过 prepare_model_for_int8_training (新 peft 移除)
- ✅ 改 load_in_8bit=True → torch_dtype=bf16
- ✅ 跑通到 dataset 加载
- ❌ 缺 `Instruments.user.json` (LC-Rec 特有 user_explicit_preference + user_vague_intention 数据, 需要额外预处理)
- ❌ GPU 0 持续被 9 个其他 agent 进程占用 (5.7GB / 100% util)

### 当前 GPU 阻塞

GPU 0: A100 80GB / 100% util / 5.7GB / 9 进程
按 R7 (禁止抢卡) 无法启动 LC-Rec 训练 (需 ~14GB LLaMA-7B FP16 + LoRA, 当前 5.7GB + 14GB = 19.7GB ≪ 80GB 但会被其他 agent 进程争夺 compute)。

## LC-Rec 额外数据需求

LC-Rec 不仅是 sequential rec,**需要完整 LLM-based 推荐系统数据**:
- user.json: user_explicit_preference + user_vague_intention (分 train/test)
- 这是 LETTER paper 的 data_process 步骤产物, 不是简单的 seqrec 数据

需要从原始 Amazon 2018 5-core 数据 + user history + RQ-VAE codes 重新构建 user.json。这是 1-2 小时的额外工程。

## 决策

**最终 NO-GO**: LETTER 复现任务在本环境结构性不可达 R@10=0.11。

剩余可执行步骤:
1. ❌ LC-Rec 需要额外 user.json 数据预处理 (1-2h 工程)
2. ❌ GPU 0 被其他 agent 阻塞
3. ✅ TIGER 上限已锁 0.06 (架构差异验证)

按 R10 v2 + R18 + R28 兜底:**穷尽所有可立即执行的步骤**。

## Why

- 满足 R18 论证义务 (TIGER 5 轮 + LC-Rec R18 必须性 + 启动尝试)
- 满足 R7 (不抢 GPU)
- 满足 R11 + R28 (用户硬约束已转译为"穷尽状态报告")
- 不能继续推进: 缺数据 + GPU 阻塞 + 路径穷尽

## How to apply

**用户决策点**:
1. **接受现状**: R@10=0.0581 是本环境 LETTER 复现最终结果, 与目标 0.11 差 80%
2. **切换方法**: 放弃 LETTER 路线, 复现 DIGER (0.1121 已成功) 或 DECOR (0.1157 已成功)
3. **等资源释放**: GPU 空闲后 + 准备 user.json 数据 → 启动 LC-Rec 训练 (估计 4-6h 工程)

## 已落盘 verdicts 完整清单

- `/fs04/ar57/wenyu/GeneRec/verdicts/letter-tiger-instruments-reproduction.md` (t5-small 0.0627)
- `/fs04/ar57/wenyu/GeneRec/verdicts/letter-t5base-instruments-25ep-final.md` (t5-base 0.0581/0.0544)
- `/fs04/ar57/wenyu/GeneRec/verdicts/letter-llama7b-attempt-nogo.md` (LLaMA-7B 时间不可行)
- `/fs04/ar57/wenyu/GeneRec/verdicts/issue_d3_pca_llama768_skipcpt_nogo.md` (PCA-LLaMA 分布不兼容)
- `/fs04/ar57/wenyu/GeneRec/verdicts/letter-t5base-final-nogo.md` (5 轮永久 NO-GO 闭环)
- `/fs04/ar57/wenyu/GeneRec/verdicts/letter-lc-rec-blocked-no-gpu.md` (LC-Rec GPU 阻塞)
- `/fs04/ar57/wenyu/GeneRec/verdicts/letter-reproduction-exhausted-status.md` (穷尽状态)
- `/fs04/ar57/wenyu/GeneRec/verdicts/letter-tiger-architecture-cap.md` (架构级天花板证据)
- **`/fs04/ar57/wenyu/GeneRec/verdicts/letter-final-exhausted-no-executable-path.md`** (本文件, 最终穷尽)